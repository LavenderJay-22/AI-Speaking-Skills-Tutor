import language_tool_python
import os
import sys
import json
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.whisper_model import FILLER_WORDS

# Load LLM functions only when needed to avoid circular imports
# query_ollama_deterministic is imported inside _check_grammar_llm()

MAX_ERRORS = 8

# Set true to print LanguageTool rule IDs and categories
DEBUG = False   

_grammar_tool = None

def load_tool():
    """Load LanguageTool once and reuse."""
    global _grammar_tool

    # Load LanguageTool grammar checker
    if _grammar_tool is None:
        print("Loading LanguageTool grammar checker...")
        _grammar_tool = language_tool_python.LanguageTool('en-US')
        print("LanguageTool loaded.")

    return _grammar_tool

# =====================================
# PASS 1 - LANGUAGETOOL
# Rule-based grammar checking
# =====================================

# Ignore non-grammar LanguageTool rules
SKIP_RULES = {
    "WHITESPACE_RULE",
    "COMMA_PARENTHESIS_WHITESPACE",
    "EN_QUOTES",
    "DOUBLE_PUNCTUATION",
    "UPPERCASE_SENTENCE_START",
    "MISSING_CAPITALIZATION",
    "CAPITALIZATION",
    "I_LOWERCASE",
    "EN_COMPOUNDS",
    "COMMA_COMPOUND_SENTENCE_2",
}

# Ignore non-grammar categories
SKIP_CATEGORIES = {
    "TYPOGRAPHY",
    "CASING",
    "PUNCTUATION",
    "TYPOS",
    "COMPOUNDING",
}

def _is_style_match(match) -> bool:
    """
    Catches hyphenation suggestions such as "high end" -> "high-end".
    These are written-style rules, not grammar, and their rule ids vary,
    so we also filter on the message text.
    """
    return "hyphen" in (match.message or "").lower()


def _check_grammar_languagetool(transcript: str) -> list:
    grammar_tool = load_tool()

    try:
        # Check the transcript
        raw_matches = grammar_tool.check(transcript)
    except Exception as e:
        print(f"LanguageTool check failed: {e}")
        return []

    if DEBUG:
        for match in raw_matches:
            print(f"[LT] rule_id={match.rule_id} category={match.category} msg={match.message}")

    # Remove non-grammar matches
    grammar_matches = [
        match for match in raw_matches
        if match.rule_id not in SKIP_RULES
        and match.category not in SKIP_CATEGORIES
        and not _is_style_match(match)
        and transcript[match.offset:match.offset + match.error_length].lower()
            not in FILLER_WORDS
    ]

    languagetool_errors = []
    for match in grammar_matches:
        wrong_text = transcript[match.offset:match.offset + match.error_length]
        suggested_text = match.replacements[0] if match.replacements else None

        languagetool_errors.append({
            "wrong": wrong_text,
            "suggestion": suggested_text,
            "message": match.message,
            "category": match.category,
            "offset": match.offset,
            "source": "languagetool"
        })

    return languagetool_errors

# ===========================================
# PASS 2 - LLM (contextual, per-sentence)
# Contextual grammar checking
# ===========================================

def _split_sentences(transcript: str) -> list:
    """Naive sentence splitter - good enough for spoken transcripts."""
    raw_sentences = re.split(r'(?<=[.!?])\s+', transcript.strip())
    return [sentence.strip() for sentence in raw_sentences if sentence.strip()]

def _find_offset_in_sentence(sentence: str, sentence_start: int, wrong_text: str) -> int:
    """
    Locate wrong_text inside ONE sentence (case-insensitive) and return
    its absolute offset in the transcript. Returns -1 if not found
    (e.g. the LLM paraphrased instead of quoting verbatim).
    """
    match_index = sentence.lower().find(wrong_text.lower())
    return -1 if match_index == -1 else sentence_start + match_index

def _extract_json_object(raw_response: str):
    """
    Pull a single JSON object out of the model's raw response.
    Tries a straight parse first, then falls back to the first {...}
    block anywhere in the text (local models sometimes wrap JSON in
    stray prose despite instructions say not to).
    """
    raw_response = raw_response.strip()

    try:
        return json.loads(raw_response)
    except (json.JSONDecodeError, ValueError):
        pass

    json_block_match = re.search(r"\{.*\}", raw_response, re.DOTALL)

    if json_block_match:
        try:
            return json.loads(json_block_match.group(0))
        except (json.JSONDecodeError, ValueError):
            pass

    return {}

def _norm(text: str) -> str:
    """Lowercase and strip punctuation, for comparing wrong vs suggestion."""
    return re.sub(r"[^\w\s']", "", text).strip().lower()

def _is_valid_llm_error(wrong_text: str, suggested_text: str, message: str, sentence: str) -> bool:
    """Reject empty, no-op, whole-sentence, or 'no error' results."""
    if not wrong_text or not suggested_text:
        return False
    if wrong_text.lower() in FILLER_WORDS:
        return False
    if _norm(wrong_text) == _norm(suggested_text):  # no-op "correction"
        return False
    if len(wrong_text.split()) > 5:  # model returned a long span
        return False
    if _norm(wrong_text) == _norm(sentence):  # whole sentence flagged
        return False
    if "no error" in message.lower():
        return False

    # Agreement fixes are single-word swaps (are <-> is, is <-> am).
    # Anything that inserts/deletes words or changes several is rejected.
    wrong_tokens = _norm(wrong_text).split()
    suggested_tokens = _norm(suggested_text).split()
    if len(wrong_tokens) != len(suggested_tokens):
        return False
    if sum(a != b for a, b in zip(wrong_tokens, suggested_tokens)) != 1:
        return False
    return True

def _check_grammar_llm(transcript: str) -> list:
    """
    Contextual grammar pass, sentence by sentence. Catches agreement
    errors LanguageTool has no rule for (e.g. singular "technology" +
    "are"). Treated as a second opinion merged against LanguageTool.
    """
    # Import LLM functions here to avoid circular imports
    from models.llm_fusion import (
        query_ollama_deterministic,
        clean_feedback,
        _repair_near_misses,
        _real_matches,
        _protected_words,
    )

    # Skip grammar analysis for very short transcripts
    if len(transcript.split()) < 10:
        return []

    llm_errors = []

    # Track offsets to prevent duplicate errors
    seen_offsets = set()
    search_position = 0

    # Words the speaker actually said are never treated as spelling errors.
    protected_words = _protected_words(transcript)

    for sentence in _split_sentences(transcript):
        sentence_start = transcript.find(sentence, search_position)
        if sentence_start == -1:
            continue
        search_position = sentence_start + len(sentence)
        if len(sentence.split()) < 3:
            continue

        prompt = f"""You are a strict grammar checker for transcribed speech.

SENTENCE:
{sentence}

Check ONLY this single sentence for subject-verb agreement and
verb-form errors (e.g. "technology are" should be "technology is",
"I is" should be "I am"). A sentence can contain more than one error -
report ALL of them.

Be very conservative when identifying grammar errors.

First identify the actual grammatical subject of the sentence.
Then determine whether the subject is singular or plural before
suggesting a verb correction.

For example:

Sentence: Physical activity stimulates the brain.

{{"errors": []}}

Sentence: Physical activity stimulate the brain.

{{"errors": [{{"wrong": "Physical activity stimulate", "suggestion": "Physical activity stimulates", "message": "The subject 'physical activity' is singular, so the verb should be 'stimulates'."}}]}}

Sentence: Good sleep helps you process information.

{{"errors": []}}

Sentence: Good sleep help you process information.

{{"errors": [{{"wrong": "Good sleep help", "suggestion": "Good sleep helps", "message": "The subject 'good sleep' is singular, so the verb should be 'helps'."}}]}}

Sentence: Sleep and exercise help you process information.

{{"errors": []}}

Sentence: Sleep and exercise helps you process information.

{{"errors": [{{"wrong": "Sleep and exercise helps", "suggestion": "Sleep and exercise help", "message": "The compound subject is plural, so the verb should be 'help'."}}]}}

Sentence: The quality of the roads are poor.

{{"errors": [{{"wrong": "quality of the roads are", "suggestion": "quality of the roads is", "message": "The main subject 'quality' is singular, so the verb should be 'is'."}}]}}

Sentence: Cats and dogs is popular pets.

{{"errors": [{{"wrong": "Cats and dogs is", "suggestion": "Cats and dogs are", "message": "A compound subject with 'and' is plural, so the verb should be 'are'."}}]}}

Sentence: Music are relaxing and books is fun.

{{"errors": [{{"wrong": "Music are", "suggestion": "Music is", "message": "'Music' is singular."}}, {{"wrong": "books is", "suggestion": "books are", "message": "'books' is plural."}}]}}

Sentence: That is what makes it special.

{{"errors": []}}

Do NOT flag filler words, punctuation, capitalisation, spelling,
hyphenation, or word choice/style.

IMPORTANT:
Many English nouns are singular non-count nouns even though they
refer to a general concept, activity, or amount. Do not assume that
a noun is plural simply because it represents many things.

Determine the grammatical number of the actual subject from its
meaning and context.

For example:

"Regular exercise has many benefits." -> correct
"Good sleep helps you concentrate." -> correct
"Physical activity stimulates the brain." -> correct
"Education has many benefits." -> correct
"Research shows that..." -> correct
"Information is important." -> correct

Do not change a correctly conjugated singular verb to a plural/base
form unless the subject is actually plural.

If you are uncertain whether something is a genuine grammar error,
do NOT report it. It is better to miss an uncertain error than to
provide an incorrect correction.

Now check the SENTENCE above.

Respond with ONLY this JSON (nothing else):
{{"errors": [{{"wrong": "<exact 2-5 word wrong phrase copied from the sentence>", "suggestion": "<corrected phrase>", "message": "<short explanation>"}}]}}

If the sentence has no errors, respond with exactly: {{"errors": []}}
"""
        try:
            raw_response = query_ollama_deterministic(prompt)
        except Exception as e:
            print(f"LLM grammar pass skipped for a sentence: {e}")
            continue

        parsed_response = _extract_json_object(raw_response)

        if not isinstance(parsed_response, dict):
            continue

        error_items = parsed_response.get("errors", [])

        if not isinstance(error_items, list):
            continue

        for error_item in error_items:
            if not isinstance(error_item, dict):
                continue

            wrong_text = str(error_item.get("wrong", "")).strip()
            suggested_text = str(error_item.get("suggestion", "") or "").strip()
            message = str(error_item.get("message", "")).strip()

            # Fix garbled words like 'dumpls' -> 'dumplings' in the correction.
            suggested_text = _repair_near_misses(suggested_text, sentence)

            if not _is_valid_llm_error(wrong_text, suggested_text, message, sentence):
                continue

            error_offset = _find_offset_in_sentence(sentence, sentence_start, wrong_text)

            if error_offset == -1 or error_offset in seen_offsets:
                continue

            seen_offsets.add(error_offset)

            # The validator guarantees exactly one word differs.
            wrong_tokens = _norm(wrong_text).split()
            suggested_tokens = _norm(suggested_text).split()
            old_word, new_word = next(
                (a, b) for a, b in zip(wrong_tokens, suggested_tokens) if a != b
            )

            # Safe wording used whenever the model's explanation can't
            # be trusted. It is built from the correction itself, so it
            # can never contain a typo.
            fallback_message = f"Use '{new_word}' instead of '{old_word}'."

            if message:
                # Repair garbled words using the whole transcript as the reference.
                message = _repair_near_misses(clean_feedback(message), transcript)

                # Words from the sentence and the correction are allowed.
                allowed_words = protected_words | _protected_words(wrong_text, suggested_text)

                # Any spelling error still left -> do NOT let the spell
                # checker guess; use the fallback.
                misspelt_matches = [
                    match for match in _real_matches(message, allowed_words)
                    if match.category == "TYPOS"
                ]

                # The explanation must actually mention the suggested word.
                # Otherwise it contradicts the correction shown above it
                mentions_fix = re.search(
                    rf"\b{re.escape(new_word)}\b", message, re.IGNORECASE
                )

                if misspelt_matches or not mentions_fix:
                    message = fallback_message
            else:
                message = fallback_message

            llm_errors.append({
                "wrong": transcript[error_offset:error_offset + len(wrong_text)],
                "suggestion": suggested_text,
                "message": message,
                "category": "GRAMMAR",
                "offset": error_offset,
                "source": "llm"
            })
    return llm_errors

# ========================================================
# MERGE
# Combine LanguageTool and LLM results
# ========================================================

def _spans_overlap(a_start, a_end, b_start, b_end) -> bool:
    return a_start < b_end and b_start < a_end

def _merge_errors(languagetool_errors: list, llm_errors: list) -> list:
    merged_errors = list(languagetool_errors)

    for llm_error in llm_errors:

        llm_start = llm_error["offset"]
        llm_end = llm_start + len(llm_error["wrong"])
        overlapping_errors = [
            error for error in merged_errors
            if _spans_overlap(
                llm_start, llm_end,
                error["offset"], error["offset"] + len(error["wrong"])
            )
        ]

        if not overlapping_errors:
            merged_errors.append(llm_error)
            continue

        can_replace = all(
            error["source"] == "languagetool"
            and llm_start <= error["offset"]
            and error["offset"] + len(error["wrong"]) <= llm_end
            and len(error["wrong"]) < len(llm_error["wrong"])
            for error in overlapping_errors
        )

        if can_replace:
            for error in overlapping_errors:
                merged_errors.remove(error)
            merged_errors.append(llm_error)

    merged_errors.sort(key=lambda error: error["offset"])
    return merged_errors

# ============================================================
# # RUN
# # Complete grammar analysis
# ============================================================
def check_grammar(transcript: str) -> dict:
    """
    Two-pass grammar check: LanguageTool (fast, rule-based) plus a
    per-sentence LLM pass (contextual, catches agreement errors LT has
    no rule for). Results are merged and deduped by character span.
    """

    if len(transcript.split()) < 10:
        return {
            "error_count": 0,
            "errors": [],
            "summary": "Transcript too short for grammar analysis."
        }

    print("Checking grammar...")

    try:
        languagetool_errors = _check_grammar_languagetool(transcript)
        llm_errors = _check_grammar_llm(transcript)
        merged_errors = _merge_errors(languagetool_errors, llm_errors)
        errors_to_show = merged_errors[:MAX_ERRORS]  # display cap
        error_count = len(errors_to_show)  # count matches what's shown

        if error_count == 0:
            summary = "No grammar errors detected - excellent accuracy."
        elif error_count <= 2:
            summary = f"{error_count} minor grammar issue(s) detected."
        elif error_count <= 5:
            summary = f"{error_count} grammar issues detected - some improvement needed."
        else:
            summary = f"{error_count} grammar issues detected - focus on accuracy."

        return {
            "error_count": error_count,
            "errors": errors_to_show,
            "summary": summary,
            "success": True
        }

    except Exception as e:
        print(f"Grammar check failed: {e}")
        return {
            "error_count": 0,
            "errors": [],
            "summary": "Grammar check could not be completed.",
            "success": False
        }

# ============================================================
# TEST 
# Run a sample grammar check
# ============================================================

if __name__ == "__main__":
    test_text = (
        "I feel education are high end AI driven nowadays. "
        "I also feel that technology are important. Technologies is really nice. "
        "All the books is good. I also feel that books is important. "
        "Knowledge are important and notes is essential. "
        "That's all I want to talk about education. Education."
    )
    result = check_grammar(test_text)
    print(f"Errors found: {result['error_count']}")
    print(f"Summary: {result['summary']}")
    for error in result['errors']:
        print(f"  [{error.get('source', '?')}] '{error['wrong']}' -> '{error['suggestion']}'")
        print(f"    {error['message']}")