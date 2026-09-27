import re
import requests
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OLLAMA_URL, OLLAMA_MODEL
from concurrent.futures import ThreadPoolExecutor

MIN_RESPONSE_WORDS = 10

# ============================================================
# OLLAMA HELPERS
# ============================================================

def is_ollama_available() -> bool:
    """
    Quick, cheap reachability check so the caller can fail in under
    a second with a clear message, instead of grinding through the
    whole analysis pipeline first only to fail on the first LLM call.
    """
    try:
        base_url = OLLAMA_URL.rsplit("/api/", 1)[0]
        requests.get(base_url, timeout=5)
        return True
    except Exception:
        return False

def _post_to_ollama(payload: dict) -> str:
    """
    Shared low-level call to the Ollama API. Single attempt with a
    generous ceiling - on CPU-only hardware a slow response is normal
    load, not a transient blip, so retrying just doubles the wait
    without helping. The per-step try/except in generate_report()
    catches a genuine timeout and falls back gracefully instead of
    crashing the whole report.
    """
    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=600  # 10 minutes - a safety net, not an expected wait
        )

        response.raise_for_status()
        response_data = response.json()
        return response_data.get("response", "").strip()

    except requests.exceptions.Timeout:
        raise TimeoutError(
            "The model took too long to respond (over 10 minutes)."
        )

    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Cannot connect to Ollama. "
            "Make sure ollama serve is running in a separate terminal."
        )

def query_ollama(prompt: str) -> str:
    """
    Send a prompt to Phi-3 via Ollama and return the response.
    Used for longer, free-form generation (questions, feedback prose).
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 1700,
            "num_ctx": 2048
        }
    }
    return _post_to_ollama(payload)

def query_ollama_single(prompt: str) -> str:
    """
    Send a prompt expecting a very short response
    (e.g. a 1-4 content quality rating). Stops on the first
    newline/period/comma to keep the answer minimal.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 10,
            "num_ctx": 2048,
            "stop": ["\n", ".", ","]
        }
    }
    return _post_to_ollama(payload)

def query_ollama_reasoned(prompt: str) -> str:
    """
    Send a prompt expecting a short reasoning line followed by
    a verdict line. Unlike query_ollama_single, this does NOT
    stop on the first newline, so the model has room to justify
    itself before committing to YES/NO. This tends to be far
    more reliable on borderline judgment calls than a forced
    single-token answer.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 60,
            "num_ctx": 2048,
            "stop": ["\n\n"]
        }
    }
    return _post_to_ollama(payload)

def query_ollama_deterministic(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 300,
            "num_ctx": 2048
        }
    }
    return _post_to_ollama(payload)

# ============================================================
# ERROR-FREE QUESTION GENERATION
# Generates a question, checks it with LanguageTool, and regenerates 
# if errors are found. If every attempt still has errors,
# the best attempt is auto-corrected using LanguageTool's 
# own suggestions.
# ============================================================

import language_tool_python

_grammar_tool = None

def _get_lt_tool():
    """Create the LanguageTool checker once and reuse it."""
    global _grammar_tool
    if _grammar_tool is None:
        _grammar_tool = language_tool_python.LanguageTool("en-US")
    return _grammar_tool

def _clean_question(prompt: str, keep_words: str = "", max_attempts: int = 3) -> str:
    """
    Generate a question and return one free of spelling/grammar errors.
    keep_words: words (e.g. the topic) that auto-correction must not
    change, so a valid topic word is never "fixed" into something else.
    """
    grammar_tool = _get_lt_tool()

    best_question = None
    best_error_count = None

    for attempt_number in range(1, max_attempts + 1):
        question = query_ollama(prompt)
        if "?" in question:
            question = question.split("?")[0].strip() + "?"
        question = question.strip().strip('"')
        error_count = len(grammar_tool.check(question))
        print(f"  Question attempt {attempt_number}: {error_count} error(s)")

        if error_count == 0:
            return question
        if best_error_count is None or error_count < best_error_count:
            best_question = question
            best_error_count = error_count

    # Every attempt had errors: apply LanguageTool's suggestions
    fixed_question = grammar_tool.correct(best_question)

    # Never accept a "fix" that rewrites the topic word
    for word in keep_words.lower().split():
        if word in best_question.lower() and word not in fixed_question.lower():
            return best_question
    return fixed_question

# ============================================================
# ERROR-FREE FEEDBACK TEXT
# Checks a piece of generated text with LanguageTool and applies
# its suggested fixes. 
# ============================================================

import difflib

def _protected_words(*texts):
    """Lowercase words from the question/answer that must not be 'fixed'."""
    return set(re.findall(r"[a-z']+", " ".join(texts).lower()))

def _match_span(match):
    """Start/end of a LanguageTool match (works across library versions)."""
    match_length = getattr(match, "errorLength", None)
    if match_length is None:
        match_length = getattr(match, "error_length", 0)
    return match.offset, match.offset + match_length

def _real_matches(text, protected_words):
    """LanguageTool matches, ignoring words taken from the question/answer."""
    real_matches = []
    for match in _get_lt_tool().check(text):
        match_start, match_end = _match_span(match)
        flagged_text = text[match_start:match_end].strip().lower()
        if flagged_text in protected_words:
            continue
        real_matches.append(match)
    return real_matches

def _apply_suggestions(text, matches):
    """Apply the top suggestion for each match, last one first."""
    for match in sorted(matches, key=lambda match: match.offset, reverse=True):
        if not match.replacements:
            continue
        match_start, match_end = _match_span(match)
        text = text[:match_start] + match.replacements[0] + text[match_end:]
    return text

def _fix_text(text, protected_words, max_passes=2):
    """See the block comment above. Returns (text, ok)."""
    current_text = text
    for _ in range(max_passes):
        matches = _real_matches(current_text, protected_words)
        if not matches:
            break
        current_text = _apply_suggestions(current_text, matches)
    if _real_matches(current_text, protected_words):
        return text, False
    similarity_ratio = difflib.SequenceMatcher(
        None, text.lower(), current_text.lower()
    ).ratio()
    if similarity_ratio < 0.85:
        return text, False
    return current_text, True

# ============================================================
# ECHOED-INSTRUCTION DETECTION
# ============================================================

_INSTRUCTION_MARKERS = [
    "one sentence only",
    "using only information",
    "only information actually present",
    "actually present in the answer",
    "one genuinely different",
    "one different actionable",
    "one specific actionable",
    "one delivery positive",
    "one specific piece of advice",
    "one specific preparation strategy",
    "state clearly that the student",
    "maximum 12 words",
]

_INSTRUCTION_START = re.compile(
    r"^(describe|give|identify|write)\s+one\b",
    re.IGNORECASE
)

def _looks_like_instruction(value, instruction):
    """True if a parsed section is the instruction echoed back."""
    lower_value = value.lower()
    if any(marker in lower_value for marker in _INSTRUCTION_MARKERS):
        return True
    if _INSTRUCTION_START.match(value.strip()):
        return True
    similarity_ratio = difflib.SequenceMatcher(
        None, lower_value, instruction.lower()
    ).ratio()
    return similarity_ratio >= 0.6

# ============================================================
# NEAR-MISS REPAIR
# Fixes real-but-wrong words the spell checker can't catch, e.g.
# 'dumps' or 'dumpls' written for 'dumplings'.
# ============================================================

def _repair_near_misses(text, source_text):
    """
    A word is replaced only if it:
      - does not appear in the question/answer itself,
      - shares its first 4 letters and its last letter with a longer
        word (7+ letters) that DOES appear there,
      - is at least 3 letters shorter than that word,
      - is not just its beginning (so 'explain' vs 'explained' is safe),
      - is very similar to it.
    """
    source_words = set(re.findall(r"[a-z]+", source_text.lower()))
    candidate_target_words = [word for word in source_words if len(word) >= 7]

    def fix(word_match):
        word = word_match.group(0)
        lower_word = word.lower()
        if lower_word in source_words or len(lower_word) < 4:
            return word
        for target_word in candidate_target_words:
            if (
                len(target_word) - len(lower_word) >= 3
                and target_word[:4] == lower_word[:4]
                and target_word[-1] == lower_word[-1]
                and not target_word.startswith(lower_word)
                and difflib.SequenceMatcher(None, lower_word, target_word).ratio() >= 0.7
            ):
                print(f"  Repaired '{word}' -> '{target_word}'")
                return target_word.capitalize() if word[0].isupper() else target_word
        return word
    return re.sub(r"[A-Za-z]+", fix, text)

# ============================================================
# ERROR-FREE FEEDBACK GENERATION
# Generates the feedback lines, checks each one, and regenerates
# until every line is clean (max_attempts). A line that comes out
# clean is kept, so later attempts only need to fix the others.
# Lines still bad after the last attempt get LanguageTool's
# suggested fixes, or "" (so the caller's fallback text is used).
# ============================================================

def _parse_feedback_lines(raw_response, keys):
    """Pull 'WELL_1: ...' style lines out of the model's reply."""
    parsed_sections = {key: "" for key in keys}
    for line in raw_response.split("\n"):
        line = line.strip()
        for key in keys:
            key_prefix = key.upper() + ":"
            if line.startswith(key_prefix):
                parsed_sections[key] = line[len(key_prefix):].strip()
                break
    return parsed_sections

def _generate_clean_sections(
    prompt, instructions, source_text, skip=(), max_attempts=3
):
    """
    instructions: {section_key: the instruction text given to the model}
    source_text:  question + student answer (words allowed as-is)
    skip:         section keys the caller fills in itself
    """
    keys = list(instructions.keys())
    protected_words = _protected_words(source_text)
    clean_sections = {}   # key -> text with zero errors
    best_attempt_by_key = {}   # key -> (error_count, text) of the least-bad attempt
    for attempt_number in range(1, max_attempts + 1):
        print(f"Generating feedback prose (attempt {attempt_number})...")
        parsed_sections = _parse_feedback_lines(query_ollama(prompt), keys)
        for key in keys:
            if key in skip or key in clean_sections:
                continue
            value = parsed_sections[key].strip()
            if not value:
                continue
            if _looks_like_instruction(value, instructions[key]):
                print(f"  '{key}' echoed the instruction - discarded.")
                continue

            cleaned_text = _repair_near_misses(clean_feedback(value), source_text)
            error_count = len(_real_matches(cleaned_text, protected_words))
            print(f"  '{key}': {error_count} error(s)")
            if error_count == 0:
                clean_sections[key] = cleaned_text
            elif key not in best_attempt_by_key or error_count < best_attempt_by_key[key][0]:
                best_attempt_by_key[key] = (error_count, cleaned_text)
        pending_keys = [key for key in keys if key not in skip and key not in clean_sections]
        if not pending_keys:
            break
        if attempt_number < max_attempts:
            print(f"  Retrying for: {pending_keys}")
    result_sections = {key: "" for key in keys}

    for key in keys:
        if key in clean_sections:
            result_sections[key] = clean_sections[key]
        elif key in best_attempt_by_key:
            fixed_text, fix_ok = _fix_text(best_attempt_by_key[key][1], protected_words)
            if fix_ok:
                result_sections[key] = fixed_text
            else:
                print(f"  '{key}' could not be cleaned - using fallback text.")
    return result_sections

# ============================================================
# QUESTION GENERATION
# ============================================================

def generate_question(
    topic: str,
    difficulty: str = "intermediate"
) -> str:
    """
    Generate a speaking question based on topic and difficulty.
    Styled like an IELTS or university oral exam question.
    """

    prompt = f"""
You are an IELTS speaking examiner.

Generate exactly ONE speaking question about:

TOPIC: {topic}
DIFFICULTY: {difficulty}

Rules:

- Maximum 20 words.
- Must be one clear question.
- Do not create multiple questions.
- Do not use sub-questions.
- Use natural everyday English.
- Focus on opinions, experiences, preferences, or general knowledge.
- Do not use technical jargon.
- Make sure the question is clearly about the requested topic.
- Write only the question.
- Nothing else.
"""

    print(f"Generating question about {topic}...")
    question = _clean_question(prompt, keep_words=topic)
    print("Question generated.")
    return question

# ============================================================
# FOLLOW-UP QUESTION GENERATION
# ============================================================

def generate_followup_question(
    previous_question: str,
    focus: str,
    detail: str = ""
) -> str:
    """
    Generate a natural follow-up speaking question that continues
    the same conversation as previous_question, but is specifically
    designed to make the student address whatever weak area 'focus'
    describes (e.g. missed coverage, shallow answer, off-topic).
    """
    detail_line = f"SPECIFIC ASPECT TO COVER: {detail}\n" if detail else ""

    prompt = f"""
You are an IELTS speaking examiner continuing a conversation
with a student.

PREVIOUS QUESTION: {previous_question}
FOCUS FOR THIS FOLLOW-UP: {focus}
{detail_line}
Generate exactly ONE follow-up speaking question that:

- Feels like a natural continuation of the same conversation,
  not a brand new unrelated topic.
- Specifically requires the student to address the focus above.
- Maximum 20 words.
- Is one clear question, no sub-questions.
- Uses natural everyday English.
- Does not mention scores, mistakes, or that this is a "follow-up".
- Write only the question, nothing else.
"""

    print(f"Generating follow-up question (focus: {focus})...")
    question = _clean_question(prompt)
    print("Follow-up question generated.")
    return question

# ============================================================
# CEFR ASSESSMENT
# ============================================================

def assess_cefr_level(transcript: str) -> dict:
    """
    Assess CEFR level consistently using language quality only.

    The assessment is based on:
    - vocabulary range
    - grammatical accuracy
    - sentence complexity
    - fluency

    Content relevance is NOT considered.
    Uses temperature 0.0 for consistent results.
    """

    prompt = f"""You are a strict and consistent CEFR English examiner.

Assess the student's spoken English using ONLY the language shown in the response.

DO NOT consider:
- whether the student answered the question
- whether the ideas are good or bad
- the topic
- the student's knowledge
- the length of the answer by itself

Assess ONLY:
1. Vocabulary range and precision
2. Grammar accuracy
3. Sentence structure and complexity
4. Fluency and ability to express ideas clearly

IMPORTANT:
- Do not give C1 or C2 unless the language clearly demonstrates those characteristics.
- Do not give B2 simply because the response is understandable.
- Grammar errors must lower the level when they are frequent or noticeable.
- Do not infer a higher level from a few advanced words.
- Judge the overall language performance consistently.

Use these exact criteria:

A1:
- Very basic words and phrases
- Very limited sentence structures
- Frequent basic grammar problems
- Difficulty expressing complete ideas

A2:
- Simple vocabulary related to familiar topics
- Mostly simple sentences
- Noticeable grammar limitations
- Can communicate basic ideas but with limited flexibility

B1:
- Generally clear communication
- Mostly everyday vocabulary with some variety
- Mainly simple sentence structures with some attempts at longer sentences
- Some grammar errors may occur
- Can explain opinions and experiences clearly enough

B2:
- Good range of vocabulary
- Can express ideas clearly and in detail
- Uses a mixture of simple and complex sentence structures
- Grammar is generally accurate, with occasional errors
- Shows flexibility in expressing ideas

C1:
- Wide and flexible vocabulary
- Precise word choice
- Consistently accurate grammar
- Frequent and natural use of complex structures
- Ideas are expressed fluently and flexibly

C2:
- Near-native command of English
- Highly precise and sophisticated vocabulary
- Very high grammatical accuracy
- Complex structures are used naturally and effortlessly
- Extremely fluent and flexible expression

STUDENT RESPONSE:
{transcript[:500]}

Choose ONE level only.

Write exactly two lines:

LEVEL: [A1/A2/B1/B2/C1/C2]
REASON: [one sentence explaining the level using specific evidence from the student's language]

Do not mention the topic.
Do not mention whether the answer was on topic.
Do not give advice.
Do not provide a second possible level.
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 120,
            "num_ctx": 2048
        }
    }

    raw_response = _post_to_ollama(payload)
    cefr_level = None
    cefr_reason = None

    # --------------------------------------------------------
    # PARSE LEVEL
    # --------------------------------------------------------

    for line in raw_response.splitlines():
        line = line.strip()
        if line.upper().startswith("LEVEL:"):
            extracted_level = line.split(":", 1)[1].strip().upper()
            if extracted_level in ["A1", "A2", "B1", "B2", "C1", "C2"]:
                cefr_level = extracted_level
        elif line.upper().startswith("REASON:"):
            cefr_reason = line.split(":", 1)[1].strip()

    # --------------------------------------------------------
    # FALLBACK LEVEL DETECTION
    # --------------------------------------------------------

    if not cefr_level:
        for level_option in ["C2", "C1", "B2", "B1", "A2", "A1"]:
            if re.search(rf"\b{level_option}\b", raw_response.upper()):
                cefr_level = level_option
                break

    # --------------------------------------------------------
    # FALLBACKS
    # --------------------------------------------------------

    if not cefr_level:
        print("CEFR: no level parsed from model output, falling back to B1")
        cefr_level = "B1"

    if not cefr_reason:
        cefr_reason = (
            "The response demonstrates generally clear communication "
            "with a developing range of vocabulary and sentence structures."
        )

    cefr_reason = clean_feedback(cefr_reason)

    # Keep the CEFR reason to one sentence
    reason_sentences = re.split(r'(?<=[.!?])\s+', cefr_reason.strip())

    if reason_sentences:
        cefr_reason = reason_sentences[0]
    return {
        "level": cefr_level,
        "reason": cefr_reason
    }

# ============================================================
# TOPIC RELEVANCE (reasoning-before-verdict, best-of-3 voting)
# ============================================================

def _extract_verdict(raw_response: str) -> str:
    """
    Pull YES or NO out of a reasoned response. Looks for a line
    starting with VERDICT: first; falls back to scanning the
    whole response for a standalone YES/NO token.
    """
    for line in raw_response.splitlines():
        line = line.strip()
        if line.upper().startswith("VERDICT:"):
            verdict_value = line.split(":", 1)[1].strip().upper()
            if verdict_value.startswith("YES"):
                return "YES"
            if verdict_value.startswith("NO"):
                return "NO"
    fallback_match = re.search(r'\b(YES|NO)\b', raw_response.upper())
    if fallback_match:
        return fallback_match.group(1)
    return "NO"

def check_topic(
    question: str,
    transcript: str
) -> dict:
    """
    Determine whether the student's answer directly relates
    to the question.
    Three independent checks are performed, each allowed to
    write one short line of reasoning before its verdict.
    Majority (2 of 3) decides the result, rather than requiring
    unanimous agreement.
    """

    prompt = f"""
You are a strict speaking examiner.

Your ONLY task is to decide whether the student's answer is relevant
to the question.

QUESTION:
{question}

STUDENT ANSWER:
{transcript[:500]}

Compare the SPECIFIC MAIN SUBJECT of the question with the
SPECIFIC MAIN SUBJECT of the student's answer.

IMPORTANT RULES:

- Do not judge grammar.
- Do not judge vocabulary.
- Do not judge fluency.
- Do not judge how detailed the answer is.
- Do not judge whether the answer is good.
- Only determine whether the subjects match.
- If the question asks for a personal experience, example, or story
  about a subject, ANY specific example of that subject counts as
  on-topic. Do not require the answer to restate the subject in
  general terms - a concrete instance of the topic IS the topic.
- Narrowing from a general subject to one specific example of that
  same subject is NOT a topic change.
- If the question asks how a subject INFLUENCED, SHAPED, AFFECTED,
  or CHANGED the student (their development, choices, outlook, or
  behavior), an answer that narrates a concrete instance of that
  subject AND draws any connecting insight, lesson, or effect - even
  briefly, even in one sentence - counts as on-topic. Do not require
  the answer to explicitly restate "personal development" or "life
  choices" - a lesson learned or behavior described as a result of
  the subject IS the influence being asked about.

The answer must actually discuss the subject asked about.

Examples:

Question: What is your favorite way to prepare a meal?
Answer: I usually bake pasta with vegetables.
Reasoning: The answer describes a way of preparing a meal, matching the question.
VERDICT: YES

Question: What is your favorite way to prepare a meal?
Answer: My best friend helped me during a difficult time.
Reasoning: The answer is about friendship, not meal preparation.
VERDICT: NO

Question: Can you share a personal experience that highlights the importance of friendship in your life?
Answer: I have a close friend from university who I lean on during stressful times, and we always support each other.
Reasoning: The question asks for a personal experience about friendship, and the answer gives exactly that - a specific friend and how the friendship matters.
VERDICT: YES

Question: What are the effects of climate change?
Answer: Rising temperatures can affect oceans and wildlife.
Reasoning: The answer discusses climate effects, matching the question.
VERDICT: YES

Question: What are the effects of climate change?
Answer: My family always supports me when I have problems.
Reasoning: The answer is about family support, unrelated to climate change.
VERDICT: NO

Question: How has friendship influenced your personal development and life choices?
Answer: My university friend and I always support each other, and this friendship taught me the importance of having someone who understands and supports you.
Reasoning: The answer gives a concrete friendship and states a lesson it taught the student, which is the influence being asked about.
VERDICT: YES

Be strict about genuine subject mismatches, but do not penalise the
student for answering with a specific, concrete example when the
question invites one.

Write exactly two lines:

Reasoning: [one short sentence]
VERDICT: [YES or NO]
"""

    print("Checking topic relevance (best-of-3)...")

    def _one_call(_):
        raw_response = query_ollama_reasoned(prompt)
        verdict = _extract_verdict(raw_response)
        print(f"  Call: {verdict}")
        return verdict

    with ThreadPoolExecutor(max_workers=3) as executor:
        verdicts = list(executor.map(_one_call, range(3)))

    yes_count = verdicts.count("YES")
    no_count = verdicts.count("NO")
    on_topic = yes_count > no_count

    print(
        f"Final topic votes -> YES: {yes_count}, NO: {no_count} "
        f"=> {'ON TOPIC' if on_topic else 'OFF TOPIC'}"
    )

    # --------------------------------------------------------
    # TOPIC MISMATCH
    # --------------------------------------------------------

    if on_topic:
        topic_mismatch = "None"
    else:
        extract_prompt = f"""
You are a speaking examiner.

QUESTION:
{question}

STUDENT ANSWER:
{transcript[:500]}

Identify:

1. The main subject of the question.
2. The main subject of the student's answer.

Write exactly ONE sentence:

You were asked about [QUESTION TOPIC] but you spoke about [ANSWER TOPIC].

Rules:

- QUESTION TOPIC must be 2 to 5 words.
- ANSWER TOPIC must be 2 to 5 words.
- Use ONLY information clearly present in the question.
- Use ONLY information clearly present in the student's answer.
- Do not invent topics.
- Do not guess what the student intended.
- Do not give advice.
- Do not mention any unrelated topic.
- Do not add explanations.
"""

        extracted_response = query_ollama(extract_prompt)
        topic_mismatch = None
        for line in extracted_response.strip().split("\n"):
            line = line.strip()
            if (
                "you were asked about" in line.lower()
                and
                "but you spoke about" in line.lower()
            ):
                topic_mismatch = line
                break
        if not topic_mismatch:
            topic_mismatch = (
                "The response did not address the subject "
                "of the question."
            )
    return {
        "on_topic": on_topic,
        "topic_mismatch": topic_mismatch
    }

# ============================================================
# QUESTION COVERAGE
# ============================================================

def check_coverage(
    question: str,
    transcript: str
) -> str:
    """
    Check whether the student addressed all aspects
    of the question.

    Returns:
        None if fully covered.
        String describing missing aspects otherwise.
    """

    prompt = f"""
You are a speaking examiner.

QUESTION:
{question}

STUDENT ANSWER:
{transcript[:500]}

Determine whether the student directly answered the question.

IMPORTANT:

- Identify what the question specifically asks for.
- Check whether the answer actually addresses it.
- Do not invent information.
- A vague statement does not count as a specific answer.
- Do not judge grammar.
- Do not judge pronunciation.
- Do not judge fluency.
- Do not provide advice.

If the student addressed everything, write exactly:

FULLY COVERED

If something important was missed, write exactly:

MISSING: [specific missing aspect]

Nothing else.
"""

    response = query_ollama(prompt)
    first_line = (
        response.strip()
        .split("\n")[0]
        .strip()
    )
    if first_line.upper().startswith("FULLY COVERED"):
        return None
    if "MISSING:" in first_line.upper():
        missing_aspect = first_line.split(":", 1)[1].strip()
        return missing_aspect
    return None

# ============================================================
# CONTENT QUALITY
# ============================================================

def assess_content_quality(
    question: str,
    transcript: str
) -> int:
    """
    Rate answer quality from 1 to 4.

    Only called when the answer is on-topic.
    """

    prompt = f"""
You are a speaking examiner.

QUESTION:
{question}

STUDENT ANSWER:
{transcript[:500]}

Rate the quality and depth of the answer.

1 = very shallow, almost no explanation
2 = some explanation but limited detail
3 = good explanation with useful detail or examples
4 = thorough, detailed, well-structured answer

IMPORTANT:

- Only assess content quality.
- Do not assess grammar.
- Do not assess pronunciation.
- Do not assess eye contact.
- Do not penalise the student simply because the answer is short.
- Do not invent information.
- Write only one number:

1
2
3
or
4
"""

    response = query_ollama_single(prompt)
    first_word = (
        response.strip().split()[0]
        if response.strip()
        else "2"
    )

    try:
        quality_rating = int(first_word[0])
        if quality_rating < 1 or quality_rating > 4:
            quality_rating = 2
    except Exception:
        quality_rating = 2
    return quality_rating

# ============================================================
# SCORE CALCULATION
# ============================================================

def calculate_scores(
    on_topic: bool,
    filler_count: int,
    word_count: int,
    quality_rating: int = 2,
    gaze_data: dict = None,
    yamnet_summary: dict = None
) -> dict:
    """
    Calculate all scores in Python.

    Phi-3 does NOT decide the scores.
    """
    # --------------------------------------------------------
    # DELIVERY SCORE
    # --------------------------------------------------------

    if filler_count <= 5:
        delivery_score = 80
    elif filler_count <= 15:
        delivery_score = round(65 - ((filler_count - 6)* (25 / 9)))
    else:
        delivery_score = max(0, 39 - (filler_count - 16) * 2)

    # --------------------------------------------------------
    # YAMNET SILENCE / AUDIO PENALTY
    # --------------------------------------------------------

    if yamnet_summary:
        silence_ratio = yamnet_summary.get("silence_ratio", 0)
        notable_events = yamnet_summary.get("notable_audio_events", 0)
        if silence_ratio > 0.3:
            delivery_score = max(0, delivery_score - 10)
        if notable_events > 3:
            delivery_score = max(0, delivery_score - 5)

    # --------------------------------------------------------
    # CONTENT SCORE
    # --------------------------------------------------------

    if not on_topic:
        if word_count < 30:
            content_score = 3
        else:
            content_score = 8
    else:
        if quality_rating == 1:
            content_score = 45
        elif quality_rating == 2:
            content_score = 62
        elif quality_rating == 3:
            content_score = 78
        elif quality_rating == 4:
            content_score = 90
        else:
            content_score = 62

    # --------------------------------------------------------
    # VISUAL SCORE
    # --------------------------------------------------------

    if gaze_data:
        eye_contact_ratio = gaze_data.get("eye_contact_ratio", 0.0)
        visual_score = round(eye_contact_ratio * 100)
    else:
        visual_score = 75

    # --------------------------------------------------------
    # OVERALL SCORE
    # --------------------------------------------------------

    overall_score = round((content_score * 0.5) + (delivery_score * 0.3) + (visual_score * 0.2))
    return {
        "content_score": content_score,
        "delivery_score": delivery_score,
        "visual_score": visual_score,
        "overall_score": overall_score
    }

# ============================================================
# FEEDBACK GENERATION
# ============================================================

def generate_feedback(
    question: str,
    transcript: str,
    on_topic: bool,
    filler_count: int,
    filler_severity: str,
    filler_list: str,
    missing_coverage: str = None,
    gaze_data: dict = None
) -> dict:
    """
    Ask Phi-3 only for short prose sections.
    Python handles all factual scoring and detected metrics.
    """

    # --------------------------------------------------------
    # FILLER FEEDBACK
    # --------------------------------------------------------

    if filler_count == 0:
        filler_bullet = (
            "No filler words were detected - excellent "
            "control of spoken language."
        )
    elif filler_count == 1:
        filler_bullet = (
            f"One filler word was detected - {filler_list} - "
            "try replacing it with a brief deliberate pause "
            "to maintain confident flow."
        )
    else:
        filler_bullet = (
            f"{filler_count} filler words were detected - "
            f"{filler_list} - practice pausing briefly instead "
            "of filling silences to sound more polished."
        )

    # --------------------------------------------------------
    # EYE CONTACT FEEDBACK
    # --------------------------------------------------------

    if gaze_data and "error" not in gaze_data:
        eye_contact_ratio = gaze_data.get("eye_contact_ratio", 0)
        if eye_contact_ratio >= 0.6:
            eye_contact_well = (
                "Eye contact with the camera was strong and "
                "consistent throughout, which conveys "
                "confidence and engagement."
            )
            eye_contact_feedback = None
            eye_contact_improvement = None
        elif eye_contact_ratio >= 0.4:
            eye_contact_well = None
            eye_contact_feedback = (
                "Eye contact was moderate, so maintaining "
                "more consistent eye contact with the camera "
                "would help build stronger audience connection."
            )
            eye_contact_improvement = (
                "Practice looking directly at the camera "
                "regularly while recording."
            )
        else:
            eye_contact_well = None
            eye_contact_feedback = (
                "Eye contact was limited, so looking at the "
                "camera more consistently would improve "
                "audience connection and confidence."
            )
            eye_contact_improvement = (
                "Place a small visual marker near the camera "
                "lens to help maintain eye contact."
            )
    else:
        eye_contact_well = None
        eye_contact_feedback = None
        eye_contact_improvement = None

    # --------------------------------------------------------
    # COVERAGE FEEDBACK
    # --------------------------------------------------------

    if missing_coverage:
        coverage_bullet = (
            f"The following part of the question was not "
            f"addressed: {missing_coverage}."
        )
    else:
        coverage_bullet = None
    if coverage_bullet:
        coverage_bullet, coverage_ok = _fix_text(coverage_bullet, _protected_words(question, transcript))
        if not coverage_ok:
            coverage_bullet = "Part of the question was not fully addressed."

    # --------------------------------------------------------
    # INSTRUCTIONS TO PHI-3
    # --------------------------------------------------------

    if on_topic:
        well_1_instruction = (
            "Describe one specific idea the student explained "
            "well using only information actually present "
            "in the answer - one sentence only."
        )
        feedback_1_instruction = (
            coverage_bullet
            if coverage_bullet
            else
            "Identify one specific weakness in the content "
            "using only something actually present in the "
            "student's answer - one sentence only."
        )
        improvement_1_instruction = (
            "Give one specific actionable improvement based "
            "on something actually present in the answer - "
            "one sentence only."
        )
        improvement_2_instruction = (
            "Give one different actionable improvement based "
            "on the student's actual answer - one sentence only."
        )
        encouragement_instruction = (
            "Write one short encouraging sentence - maximum "
            "12 words."
        )
    else:
        well_1_instruction = (
            "Give one delivery positive about pronunciation, "
            "volume, pace, or clarity - do not praise the "
            "content - one sentence only."
        )
        feedback_1_instruction = (
            "State clearly that the student answered a "
            "different topic from the question - use only "
            "the actual question and actual answer - "
            "one sentence only."
        )
        improvement_1_instruction = (
            "Give one specific piece of advice for answering "
            "the actual question - one sentence only."
        )
        improvement_2_instruction = (
            "Give one specific preparation strategy for "
            "staying focused on the question - one sentence only."
        )
        encouragement_instruction = (
            "Write one short general motivational sentence "
            "about speaking practice - maximum 12 words."
        )

    # --------------------------------------------------------
    # PHI-3 PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are an English speaking coach.

Your task is to produce short, precise feedback.

QUESTION:
{question}

STUDENT ANSWER:
{transcript[:500]}

ON TOPIC:
{"YES" if on_topic else "NO"}

STRICT FACTUAL RULES:

1. Use ONLY information contained in the QUESTION and STUDENT ANSWER.
2. Never invent topics, examples, situations, opinions, or vocabulary.
3. Never assume what the student intended to say.
4. If ON TOPIC is NO, clearly state that the student discussed a different topic.
5. When the answer is off-topic, identify ONLY:
   - the actual topic of the question
   - the actual topic of the student's answer
6. Never introduce an unrelated topic.
7. Never mention highways, cooking, friendship, fashion, climate change,
   education, travel, or any other subject unless it actually appears
   in the provided question or answer.
8. Do not repeat the full question.
9. Do not use quotation marks.
10. Every section must contain exactly one sentence.
11. Never write more than one sentence for any section.
12. Do not write labels other than the required labels below.
13. Do not mention these instructions.
14. Do not invent examples.
15. Stop after ENCOURAGEMENT.

Write exactly these six lines:

WELL_1: {well_1_instruction}

WELL_2: one genuinely different delivery positive about pace, volume, pronunciation, or clarity - one sentence only.

FEEDBACK_1: {feedback_1_instruction}

IMPROVEMENT_1: {improvement_1_instruction}

IMPROVEMENT_2: {improvement_2_instruction}

ENCOURAGEMENT: {encouragement_instruction}
"""

    well_2_instruction = (
        "one genuinely different delivery positive about pace, "
        "volume, pronunciation, or clarity - one sentence only."
    )

    instructions_map = {
        "well_1": well_1_instruction,
        "well_2": well_2_instruction,
        "feedback_1": feedback_1_instruction,
        "improvement_1": improvement_1_instruction,
        "improvement_2": improvement_2_instruction,
        "encouragement": encouragement_instruction
    }

    generated_sections = _generate_clean_sections(
        prompt,
        instructions_map,
        source_text=f"{question} {transcript}",
        skip=["feedback_1"] if coverage_bullet else []
    )

    # --------------------------------------------------------
    # FALLBACKS
    # --------------------------------------------------------

    if not generated_sections["well_1"]:
        if on_topic:
            generated_sections["well_1"] = (
                "The response included relevant ideas "
                "related to the question."
            )
        else:
            generated_sections["well_1"] = (
                "The student maintained a clear and "
                "audible speaking delivery."
            )
    if not generated_sections["well_2"]:
        generated_sections["well_2"] = (
            "The student maintained a comfortable "
            "speaking pace throughout."
        )
    if not generated_sections["feedback_1"]:
        if on_topic:
            generated_sections["feedback_1"] = (
                coverage_bullet
                or
                "The response could be developed further "
                "with more specific detail."
            )
        else:
            generated_sections["feedback_1"] = (
                "The response did not answer the question "
                "that was asked."
            )
    if not generated_sections["improvement_1"]:
        if on_topic:
            generated_sections["improvement_1"] = (
                "Add specific examples or details to "
                "make the response more developed."
            )
        else:
            generated_sections["improvement_1"] = (
                "Focus on the main subject of the question "
                "before beginning your response."
            )
    if not generated_sections["improvement_2"]:
        if on_topic:
            generated_sections["improvement_2"] = (
                "Organise your ideas with a clear beginning, "
                "development, and conclusion."
            )
        else:
            generated_sections["improvement_2"] = (
                "Take a few seconds to identify the key "
                "words in the question before speaking."
            )
    if not generated_sections["encouragement"]:
        generated_sections["encouragement"] = (
            "Keep practising and your speaking skills "
            "will continue to improve."
        )
    return {
        "well_1": generated_sections["well_1"],
        "well_2": generated_sections["well_2"],
        "well_eye": eye_contact_well,
        "feedback_1": generated_sections["feedback_1"],
        "feedback_filler": filler_bullet,
        "feedback_eye": eye_contact_feedback,
        "improvement_1": generated_sections["improvement_1"],
        "improvement_2": generated_sections["improvement_2"],
        "improvement_eye": eye_contact_improvement,
        "encouragement": generated_sections["encouragement"]
    }

# ============================================================
# PLACEHOLDER CHECK
# ============================================================

def has_placeholders(text: str) -> bool:
    """
    Check whether Phi-3 left template placeholders such as
    [TOPIC] or [ANSWER].
    """
    return bool(re.search(r'\[[^\]]+\]', text))

# ============================================================
# CLEAN FEEDBACK
# ============================================================

def clean_feedback(text: str) -> str:
    """
    Clean common garbled outputs from Phi-3.
    """
    cleaned_text = text

    # Remove digits at the start of words
    cleaned_text = re.sub(r'\b\d+([a-zA-Z]{3,})', r'\1', cleaned_text)

    # Remove digits at the end of words
    cleaned_text = re.sub(r'([a-zA-Z]{3,})\d+\b', r'\1', cleaned_text)

    # Remove digits embedded inside words
    cleaned_text = re.sub(r'([a-zA-Z]+)\d+([a-zA-Z]+)', r'\1\2', cleaned_text)

    # Remove symbols embedded inside words
    cleaned_text = re.sub(r'([a-zA-Z])[0-9@#$%^&*+=|\\<>{}[\]~`]([a-zA-Z])', r'\1\2', cleaned_text)

    # Remove garbled symbols at start of words - explicitly excludes
    # legitimate punctuation (. , ! ? ' " - ; :) so real sentence
    # punctuation is never stripped, only stray characters like
    # @#$%^&* that indicate a garbled model output.

    _GARBLE_CHARS = r"@#$%^&*+=|\\<>{}\[\]~`_"

    cleaned_text = re.sub(rf'(?<!\w)[{_GARBLE_CHARS}]([a-zA-Z]{{3,}})', r'\1', cleaned_text)

    # Remove garbled symbols at end of words (same exclusion)
    cleaned_text = re.sub(rf'([a-zA-Z]{{3,}})[{_GARBLE_CHARS}](?!\w)', r'\1',cleaned_text)

    # Remove repeated consecutive words
    cleaned_text = re.sub(r'\b(\w+)\s+\1\b', r'\1', cleaned_text, flags=re.IGNORECASE)

    # Remove placeholders
    cleaned_text = re.sub(r'\[[^\]]+\]', '', cleaned_text)

    # Remove excessive spaces
    cleaned_text = re.sub(r'  +', ' ', cleaned_text)

    # Remove empty lines
    non_empty_lines = [
        line
        for line in cleaned_text.split("\n")
        if line.strip()
    ]

    cleaned_text = "\n".join(non_empty_lines)
    return cleaned_text.strip()

# ============================================================
# FULL REPORT GENERATION
# ============================================================

def generate_report(
    question: str,
    transcript: str,
    filler_data: dict,
    yamnet_summary: dict,
    gaze_data: dict = None,
    grammar_data: dict = None,
    pause_data: dict = None,   
    pace_data: dict = None,
    pitch_data: dict = None
) -> str:
    """
    Full analysis pipeline.

    Every LLM-dependent step below is individually wrapped in its
    own try/except with a safe fallback value. This means a single
    slow or broken step (topic check, coverage, quality, CEFR, or
    feedback prose) degrades gracefully instead of destroying the
    entire report - the student always gets scores, transcript,
    filler/grammar analysis, and SOME coaching text, even in the
    worst case where every LLM call fails.
    """

    filler_count = filler_data.get("count", 0)
    filler_instances = filler_data.get("instances", [])
    word_count = len(transcript.split())

    if word_count < MIN_RESPONSE_WORDS:
        return f"""OVERALL SCORE: N/A
CONTENT SCORE: N/A
DELIVERY SCORE: N/A
FILLER WORDS DETECTED: N/A

FEEDBACK COULD NOT BE GENERATED

Reason: Insufficient speech detected.
Please record yourself speaking for at least 30 seconds and try again.
"""

    # --------------------------------------------------------
    # FILLER SEVERITY / LIST
    # --------------------------------------------------------

    if filler_count <= 5:
        filler_severity = "LOW - good control of filler words"
    elif filler_count <= 10:
        filler_severity = "MODERATE - some fillers present"
    else:
        filler_severity = "HIGH - more than 10% of words were fillers"
    if filler_instances:
        filler_list = ", ".join(
            f"'{filler['word']}' at {filler['time']:.1f}s"
            for filler in filler_instances[:8]
        )
    else:
        filler_list = "none"

    # ========================================================
    # STEP 1 - TOPIC (fallback: assume on-topic)
    # ========================================================

    try:
        topic_result = check_topic(question, transcript)
        on_topic = topic_result["on_topic"]
        topic_mismatch = topic_result["topic_mismatch"]
        print(f"Final topic result: {'ON TOPIC' if on_topic else 'OFF TOPIC'}")
    except Exception as e:
        print(f"Topic check failed, assuming on-topic: {e}")
        on_topic = True
        topic_mismatch = "None"

    # ========================================================
    # STEP 2 - COVERAGE (fallback: treat as fully covered)
    # ========================================================

    missing_coverage = None
    if on_topic:
        try:
            missing_coverage = check_coverage(question, transcript)
        except Exception as e:
            print(f"Coverage check failed, skipping: {e}")
            missing_coverage = None

    # ========================================================
    # STEP 3 - CONTENT QUALITY (fallback: rating 2/4)
    # ========================================================

    if on_topic:
        try:
            quality_rating = assess_content_quality(question, transcript)
        except Exception as e:
            print(f"Quality assessment failed, defaulting to 2: {e}")
            quality_rating = 2
    else:
        quality_rating = 1

    # ========================================================
    # STEP 4 - SCORES (pure Python, cannot fail from Ollama)
    # ========================================================

    scores = calculate_scores(
        on_topic, filler_count, word_count,
        quality_rating, gaze_data, yamnet_summary
    )

    # ========================================================
    # STEP 5 - CEFR (fallback: generic B1)
    # ========================================================

    try:
        cefr_result = assess_cefr_level(transcript)
        print(f"CEFR level: {cefr_result['level']}")
    except Exception as e:
        print(f"CEFR assessment failed, using fallback: {e}")
        cefr_result = {
            "level": "B1",
            "reason": "CEFR could not be assessed for this attempt."
        }

    # ========================================================
    # STEP 6 - FEEDBACK (fallback: generic coaching prose)
    # ========================================================

    if filler_count == 0:
        _filler_fallback = "No filler words were detected - excellent control of spoken language."
    else:
        _filler_fallback = f"{filler_count} filler word(s) detected - practice pausing instead of filling silences."
    try:
        feedback = generate_feedback(
            question, transcript, on_topic, filler_count,
            filler_severity, filler_list, missing_coverage, gaze_data
        )
    except Exception as e:
        print(f"Feedback generation failed, using generic fallback: {e}")
        feedback = {
            "well_1": "The response was received and analysed.",
            "well_2": "The student completed the speaking task.",
            "well_eye": None,
            "feedback_1": "Detailed feedback could not be generated for this attempt.",
            "feedback_filler": _filler_fallback,
            "feedback_eye": None,
            "improvement_1": "Try recording another response to get full feedback.",
            "improvement_2": "Review your transcript above for areas to improve.",
            "improvement_eye": None,
            "encouragement": "Keep practising - every attempt helps."
        }

    # ========================================================
    # STEP 7 - CLEAN LLM OUTPUT
    # ========================================================

    for key in ["well_1", "well_2", "feedback_1", "improvement_1", "improvement_2", "encouragement"]:
        feedback[key] = clean_feedback(feedback[key])

    # ========================================================
    # STEP 8 - GAZE (pure Python/MediaPipe, already try/excepted in app.py)
    # ========================================================

    if gaze_data and "error" not in gaze_data:
        eye_contact_pct = round(gaze_data.get("eye_contact_ratio", 0) * 100)
        gaze_summary = gaze_data.get("gaze_summary", "Eye contact was analysed.")
        gaze_line = f"EYE CONTACT: {eye_contact_pct}% - {gaze_summary}"
        head_orientation = gaze_data.get("head_orientation")
        if head_orientation and head_orientation.get("orientation") != "insufficient data":
            gaze_line += (
                f"\nHEAD ORIENTATION: {head_orientation['forward_pct']}% forward - "
                f"{head_orientation['summary']}"
            )
    else:
        gaze_line = "EYE CONTACT: not analysed"

    # ========================================================
    # STEP 9 - YAMNET
    # ========================================================

    if yamnet_summary:
        silence_pct = round(yamnet_summary.get("silence_ratio", 0) * 100)
        notable_events = yamnet_summary.get("notable_audio_events", 0)
        yamnet_line = f"SILENCE RATIO: {silence_pct}% - NOTABLE AUDIO EVENTS: {notable_events}"
    else:
        yamnet_line = "AUDIO EVENTS: not analysed"

    # ========================================================
    # STEP 9b - PAUSES / PACE / PITCH
    # ========================================================

    if pause_data:
        pause_line = (
            f"PAUSES: {pause_data.get('long_pause_count', 0)} long pause(s), "
            f"avg {pause_data.get('avg_pause', 0)}s - {pause_data.get('summary', '')}"
        )
    else:
        pause_line = "PAUSES: not analysed"
    if pace_data:
        pace_line = f"PACE: {pace_data.get('avg_wpm', 0)} wpm - {pace_data.get('summary', '')}"
    else:
        pace_line = "PACE: not analysed"
    if pitch_data and pitch_data.get("monotone") is not None:
        pitch_line = f"PITCH: stddev {pitch_data.get('pitch_stddev', 0)}Hz - {pitch_data.get('summary', '')}"
    else:
        pitch_line = "PITCH: not analysed"

    # ========================================================
    # STEP 10 - GRAMMAR 
    # grammar_data is pre-computed by app.py, which handles
    # exceptions from check_grammar() 
    # ========================================================

    if grammar_data and grammar_data.get("error_count", 0) > 0:
        grammar_line = f"GRAMMAR: {grammar_data['summary']}"
        grammar_errors_section = "GRAMMAR ERRORS DETECTED:\n"
        for error_number, error in enumerate(grammar_data.get("errors", [])[:5], 1):
            wrong_text = error.get("wrong", "")
            suggested_text = error.get("suggestion", "")
            message = error.get("message", "")
            if suggested_text:
                grammar_errors_section += f"{error_number}. '{wrong_text}' -> '{suggested_text}' ({message})\n"
            else:
                grammar_errors_section += f"{error_number}. '{wrong_text}' - {message}\n"
    elif grammar_data:
        grammar_line = f"GRAMMAR: {grammar_data['summary']}"
        grammar_errors_section = ""
    else:
        grammar_line = "GRAMMAR: not analysed"
        grammar_errors_section = ""

    # ========================================================
    # ASSEMBLE SECTIONS
    # ========================================================

    well_section = "WHAT YOU DID WELL:\n"
    well_section += f"- {feedback['well_1']}\n"
    well_section += f"- {feedback['well_2']}\n"
    if feedback.get("well_eye"):
        well_section += f"- {feedback['well_eye']}\n"

    feedback_section = "KEY FEEDBACK:\n"
    feedback_section += f"- {feedback['feedback_1']}\n"
    feedback_section += f"- {feedback['feedback_filler']}\n"
    if feedback.get("feedback_eye"):
        feedback_section += f"- {feedback['feedback_eye']}\n"

    if filler_count == 0:
        improvement_3 = "Continue maintaining excellent control of filler words to support confident delivery."
    else:
        improvement_3 = "Before recording, practise your answer aloud once without stopping to organise your thoughts."

    improvements_section = "TOP IMPROVEMENTS:\n"
    improvements_section += f"1. {feedback['improvement_1']}\n"
    improvements_section += f"2. {feedback['improvement_2']}\n"
    improvements_section += f"3. {improvement_3}\n"
    if feedback.get("improvement_eye"):
        improvements_section += f"4. {feedback['improvement_eye']}\n"

    encouragement_section = f"ENCOURAGEMENT:\n{feedback['encouragement']}"
    coverage_line = f"QUESTION COVERAGE: {missing_coverage}\n" if missing_coverage else ""

    report = f"""OVERALL SCORE: {scores['overall_score']}/100
CONTENT SCORE: {scores['content_score']}/100{"  - answer was off topic" if not on_topic else ""}
DELIVERY SCORE: {scores['delivery_score']}/100
VISUAL SCORE: {scores['visual_score']}/100
FILLER WORDS DETECTED: {filler_count} ({filler_severity})
{gaze_line}
{yamnet_line}
{pause_line}
{pace_line}
{pitch_line}
{grammar_line}

ON TOPIC: {"Yes" if on_topic else "No"}
TOPIC MISMATCH: {topic_mismatch}
{coverage_line}CONTENT QUALITY RATING: {quality_rating}/4
CEFR LEVEL: {cefr_result['level']} - {cefr_result['reason']}

{well_section}
{feedback_section}
{grammar_errors_section}{improvements_section}
{encouragement_section}"""

    return report.strip()

# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":

    print("=== TESTING QUESTION GENERATION ===")
    question = generate_question("climate change","intermediate")
    print(f"Question: {question}")

    print("\n=== TESTING REPORT GENERATION ===")
    test_transcript = (
        "Climate change is caused by greenhouse gases "
        "like carbon dioxide trapping heat in the atmosphere. "
        "Human activities such as burning fossil fuels and "
        "deforestation are the main contributors. Governments "
        "need to invest in renewable energy and individuals "
        "can help by reducing waste and using public transport."
    )

    test_fillers = {
        "count": 2,
        "instances": [
            {
                "word": "um",
                "time": 5.1
            },
            {
                "word": "uh",
                "time": 12.4
            }
        ]
    }

    test_yamnet = {
        "speech_ratio": 0.94,
        "silence_ratio": 0.06,
        "notable_audio_events": 0
    }

    report = generate_report(
        question,
        test_transcript,
        test_fillers,
        test_yamnet
    )

    print("\n=== REPORT ===")
    print(report)