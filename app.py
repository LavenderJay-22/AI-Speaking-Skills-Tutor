import gradio as gr
import os
import sys
import re
import html
import tempfile
from fpdf import FPDF
from fpdf.enums import XPos, YPos

sys.path.append('.')

from models.grammar_model import check_grammar
from utils.video_processor import extract_audio
from models.whisper_model import transcribe_audio, detect_fillers, analyse_pauses, analyse_pace
from models.yamnet_model import analyse_audio
from models.mediapipe_model import analyse_gaze
from models.prosody_model import analyse_pitch
from models.llm_fusion import (
    generate_question,
    generate_report,
    generate_followup_question,
    MIN_RESPONSE_WORDS,
    is_ollama_available
)

# ============================================================
# APP STATE
# app.launch runs on the user's own computer. 
# ============================================================

current_question = {"text": ""}

# Tracks the previous attempt so a follow-up question can target
# whatever weak area it revealed (off-topic / missed coverage /
# shallow answer). "followup" is None when there's nothing to
# target. That's what hides the follow-up button.
last_attempt = {
    "question": "", "followup": None, "report_text": "",
    "filler_data": None, "grammar_data": None, "analysis_details": None
}

video_reset_mode = {"value": None}

# True only when an analysis finished successfully. 
# Decides whether we switch from the setup page to the results page.
analysis_state = {"ok": False}

# ============================================================
# PAGE SWITCHING (setup page <-> results page)
# ============================================================

def go_to_results():
    """Show the results page only if the analysis actually succeeded."""
    if analysis_state["ok"]:
        return gr.update(visible=False), gr.update(visible=True)
    return gr.update(visible=True), gr.update(visible=False)

def go_to_setup():
    return gr.update(visible=True), gr.update(visible=False)

# ============================================================
# GENERATE QUESTION
# ============================================================

def start_question_loading():
    """Runs instantly on click - greys out the button with an
    in-button spinner and shows a cycling status message."""
    loading_html = """
    <div class="analysis-status loading">
        <div class="spinner"></div>
        <div>
            <div class="status-title">Generating your question...</div>
            <div class="status-text" id="question-stage-text">
                Thinking of a good question...
            </div>
            <div class="status-text">
                Just a moment, this runs on your own computer.
            </div>
        </div>
    </div>

    <script>
    (function() {
        const stages = [
            "Thinking of a good question...",
            "Matching it to your topic...",
            "Adjusting for difficulty level...",
            "Finalising the wording..."
        ];

        let i = 0;
        const el = document.getElementById("question-stage-text");
        if (!el) return;
        const interval = setInterval(function() {
            i = (i + 1) % stages.length;
            if (!document.body.contains(el)) {
                clearInterval(interval);
                return;
            }
            el.textContent = stages[i];
        }, 1600);
    })();
    </script>
    """

    return (
        gr.update(
            interactive=False,
            elem_classes=["primary-btn", "btn-loading"]
        ),
        gr.update(interactive=False),
        gr.update(visible=True, value=loading_html),
        gr.update(interactive=False),
        gr.update(interactive=False)
    )

def topic_changed(topic):
    """Enable question generation only when a topic has been entered."""
    if topic and topic.strip():
        return gr.update(
            interactive=True,
            elem_classes=["primary-btn"]
        )
    return gr.update(
        interactive=False,
        elem_classes=["primary-btn"]
    )

def video_changed(video):
    """
    Enable analysis only when a question exists and a video is available.
    Also lock topic/difficulty editing the moment a video is attached, so
    the active question can never drift out of sync with what the student
    actually recorded an answer to.
    """
    if video:
        return (
            gr.update(
                interactive=bool(current_question["text"]),
                elem_classes=["primary-btn"]
            ),
            gr.update(interactive=False),  # Topic
            gr.update(interactive=False),  # Difficulty
            gr.update(interactive=False)   # Generate
        )

    # Try Again or Follow-up cleared the video.
    # Keep question-generation controls locked.
    if video_reset_mode["value"] in ("try_again", "followup"):
        video_reset_mode["value"] = None
        return (
            gr.update(interactive=False),  # Analyse
            gr.update(interactive=False),  # Topic
            gr.update(interactive=False),  # Difficulty
            gr.update(interactive=False)   # Generate
        )

    # Video was cleared normally / by New Question.
    return (
        gr.update(interactive=False),  # Analyse
        gr.update(interactive=True),   # Topic
        gr.update(interactive=True),   # Difficulty
        gr.update(interactive=False)   # Generate
    )

def get_question(topic, difficulty, video):
    """Generate a speaking question."""
    if not topic.strip():
        return (
            "Please enter a topic first.",
            gr.update(interactive=False, elem_classes=["primary-btn"]),
            gr.update(interactive=False),
            gr.update(visible=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False)
        )

    question = generate_question(topic, difficulty)
    current_question["text"] = question

    # If a video already exists, freeze the question controls.
    # Otherwise, allow the user to generate another question.
    controls_locked = bool(video)

    return (
        question,
        gr.update(interactive=not controls_locked, elem_classes=["primary-btn"]),
        gr.update(interactive=bool(video), elem_classes=["primary-btn"]),
        gr.update(visible=False),
        gr.update(interactive=not controls_locked),
        gr.update(interactive=not controls_locked),
        gr.update(interactive=not controls_locked)
    )

# ============================================================
# EXTRACT SCORES FROM EXISTING REPORT
# ============================================================

def extract_scores(report):
    """
    Extract the four scores from the generated report.

    Example:
    OVERALL SCORE: 43/100
    CONTENT SCORE: 8/100
    DELIVERY SCORE: 80/100
    VISUAL SCORE: 74/100
    """

    scores = {
        "overall": "-",
        "content": "-",
        "delivery": "-",
        "visual": "-"
    }

    if not report:
        return scores
    score_patterns = {
        "overall": r"OVERALL SCORE:\s*(\d+)\s*/\s*100",
        "content": r"CONTENT SCORE:\s*(\d+)\s*/\s*100",
        "delivery": r"DELIVERY SCORE:\s*(\d+)\s*/\s*100",
        "visual": r"VISUAL SCORE:\s*(\d+)\s*/\s*100"
    }

    for score_key, pattern in score_patterns.items():
        match = re.search(pattern, report, re.IGNORECASE)
        if match:
            scores[score_key] = match.group(1)
    return scores

# ============================================================
# EXTRACT CEFR LEVEL FROM EXISTING REPORT
# ============================================================

def extract_cefr(report):
    """
    Extract the CEFR level and reason from the generated report.

    Example line in report:
    CEFR LEVEL: B1 - Generally clear communication with a
    developing range of vocabulary.
    """

    cefr = {
        "level": "-",
        "reason": "Your CEFR level will appear here after analysis."
    }
    if not report:
        return cefr
    match = re.search(
        r"CEFR LEVEL:\s*([A-C][12])\s*(?:-|-)\s*(.+)",
        report
    )
    if match:
        cefr["level"] = match.group(1).strip()
        cefr["reason"] = match.group(2).strip().split("\n")[0].strip()

    return cefr

# ============================================================
# EXTRACT ANALYSIS DETAILS (ON TOPIC / EYE CONTACT / AUDIO)
# ============================================================

def extract_analysis_details(report):
    """
    Extract the on-topic, eye contact, audio, pace, pause and pitch
    lines from the generated report.
    """
    details = {
        "on_topic": None,
        "topic_mismatch": None,
        "coverage_missing": None,
        "eye_contact_pct": None,
        "eye_contact_summary": None,
        "silence_pct": None,
        "notable_events": None,
        "head_orientation_pct": None,
        "head_orientation_summary": None,
        "pause_summary": None,
        "pace_summary": None,
        "pitch_summary": None,
        "long_pause_count": None,     
        "pace_wpm": None,            
        "pitch_stddev": None,
    }

    if not report:
        return details
    match = re.search(r"ON TOPIC:\s*(Yes|No)", report, re.IGNORECASE)
    if match:
        details["on_topic"] = match.group(1).strip().title()

    match = re.search(r"TOPIC MISMATCH:\s*(.+)", report)
    if match:
        mismatch_text = match.group(1).strip()
        if mismatch_text and mismatch_text.lower() != "none":
            details["topic_mismatch"] = mismatch_text

    match = re.search(r"QUESTION COVERAGE:\s*(.+)", report)
    if match:
        details["coverage_missing"] = match.group(1).strip()

    match = re.search(r"EYE CONTACT:\s*(\d+)%\s*-\s*(.+)", report)
    if match:
        details["eye_contact_pct"] = match.group(1).strip()
        details["eye_contact_summary"] = match.group(2).strip().split("\n")[0].strip()

    match = re.search(r"HEAD ORIENTATION:\s*(\d+)%\s*forward\s*-\s*(.+)", report)
    if match:
        details["head_orientation_pct"] = match.group(1).strip()
        details["head_orientation_summary"] = match.group(2).strip().split("\n")[0].strip()

    match = re.search(r"PAUSES:\s*.+?-\s*(.+)", report)
    if match:
        details["pause_summary"] = match.group(1).strip().split("\n")[0].strip()

    match = re.search(r"PACE:\s*.+?-\s*(.+)", report)
    if match:
        details["pace_summary"] = match.group(1).strip().split("\n")[0].strip()

    match = re.search(r"PITCH:\s*.+?-\s*(.+)", report)
    if match:
        details["pitch_summary"] = match.group(1).strip().split("\n")[0].strip()

    match = re.search(
        r"SILENCE RATIO:\s*(\d+)%\s*-\s*NOTABLE AUDIO EVENTS:\s*(\d+)",
        report
    )
    if match:
        details["silence_pct"] = match.group(1).strip()
        details["notable_events"] = match.group(2).strip()

    match = re.search(r"PAUSES:\s*(\d+)\s*long pause", report)
    if match:
        details["long_pause_count"] = int(match.group(1))

    match = re.search(r"PACE:\s*(\d+(?:\.\d+)?)\s*wpm", report)
    if match:
        details["pace_wpm"] = float(match.group(1))

    match = re.search(r"PITCH:\s*stddev\s*(\d+(?:\.\d+)?)\s*Hz", report)
    if match:
        details["pitch_stddev"] = float(match.group(1))
    return details

# ============================================================
# EXTRACT CONTENT QUALITY RATING
# ============================================================

def extract_quality_rating(report):
    """Parse the '/4' quality rating line added in llm_fusion.py."""
    if not report:
        return None
    match = re.search(r"CONTENT QUALITY RATING:\s*(\d)\s*/\s*4", report)
    if match:
        return int(match.group(1))
    return None

# ============================================================
# DECIDE THE FOLLOW-UP FOCUS
# ============================================================

def determine_followup_focus(details, quality_rating):
    """
    Decide what the next follow-up question should target, based
    on the previous attempt. Returns None when nothing is worth
    targeting (fully on-topic, fully covered, good quality) -
    the caller uses that to hide the follow-up button entirely
    rather than manufacturing a fake weakness.
    """
    if details.get("on_topic") == "No":
        return {
            "focus": (
                "gently returning to the same subject as the "
                "previous question, using simpler wording"
            ),
            "detail": "",
            "label": "Topic mismatch"
        }

    coverage_missing = details.get("coverage_missing")
    if coverage_missing:
        return {
            "focus": "covering the part of the previous question that was missed",
            "detail": coverage_missing,
            "label": "part of the question was missed"
        }
    if quality_rating is not None and quality_rating <= 2:
        return {
            "focus": "encouraging more depth, detail, or a specific example",
            "detail": "",
            "label": "answer could go deeper"
        }
    return None

# ============================================================
# CEFR LOOK-UPS
# ============================================================

CEFR_LEVEL_DESCRIPTIONS = {
    "A1": "Beginner - uses very basic vocabulary and short, simple phrases.",
    "A2": "Elementary - communicates familiar ideas using mostly simple sentences.",
    "B1": "Intermediate range - clear sentence structures with some complex phrases.",
    "B2": "Upper-Intermediate - expresses ideas in detail with a good range of vocabulary.",
    "C1": "Advanced - communicates fluently and flexibly with precise vocabulary.",
    "C2": "Proficient - communicates with near-native fluency and sophistication.",
}

CEFR_SHORT_NAMES = {
    "A1": "Beginner",
    "A2": "Elementary",
    "B1": "Intermediate",
    "B2": "Upper-Intermediate",
    "C1": "Advanced",
    "C2": "Proficient",
}

# Keywords that, if present in the LLM's CEFR reason, can contradict the
# separately-computed grammar score - so we fall back to a general
# level description instead of showing a specific grammar claim here.
CEFR_GRAMMAR_KEYWORDS = ["grammar", "grammatical", "error", "mistake", "incorrect"]

def sanitize_cefr_reason(reason, level):
    """
    Avoid showing a CEFR reason that makes a specific claim about grammar
    mistakes, since that can contradict the separate Grammar Errors card
    (which is based on an actual grammar checker, not the LLM's impression).
    Falls back to a general, level-based description instead.
    """
    if reason and not any(
        keyword in reason.lower() for keyword in CEFR_GRAMMAR_KEYWORDS
    ):
        return reason
    return CEFR_LEVEL_DESCRIPTIONS.get(
        level,
        "Your CEFR level will appear here after analysis."
    )

# ============================================================
# SCORES OVERVIEW CARD (donut + CEFR badge + bars)
# ============================================================

def _score_color(score_pct):
    if score_pct >= 70:
        return "#4b9128"
    elif score_pct >= 40:
        return "#e8b446"
    return "#d0524d"

def _strip_redundant_level_prefix(reason, short_name):
    """
    The range (e.g. 'B1 - Intermediate') is already shown above this
    text, so strip a leading '<short_name>' or '<short_name> range -'
    from the reason if present, so it doesn't repeat itself.
    """
    prefix_pattern = rf"^{re.escape(short_name)}(?:\s+range)?\s*-\s*"
    return re.sub(prefix_pattern, "", reason, flags=re.IGNORECASE).strip() 

def make_scores_overview(scores, cefr):
    try:
        overall_pct = max(0, min(100, int(scores.get("overall", "-"))))
        has_overall = True
    except (ValueError, TypeError):
        overall_pct = 0
        has_overall = False
    donut_color = _score_color(overall_pct) if has_overall else "#e4e4e0"
    overall_display = scores.get("overall", "-")

    def bar_row(label, value):
        try:
            bar_pct = max(0, min(100, int(value)))
            bar_color = _score_color(bar_pct)
        except (ValueError, TypeError):
            bar_pct = 0
            bar_color = "#e4e4e0"

        return f"""
        <div class="score-bar-row">
            <div class="score-bar-head">
                <span class="score-bar-label">{html.escape(label)}</span>
                <span class="score-bar-value">{html.escape(str(value))}</span>
            </div>
            <div class="score-bar-track">
                <div class="score-bar-fill" style="width:{bar_pct}%; background:{bar_color};"></div>
            </div>
        </div>
        """

    level = str(cefr.get("level", "-"))
    short_name = CEFR_SHORT_NAMES.get(level, "-")
    reason = sanitize_cefr_reason(str(cefr.get("reason", "")), level)
    reason = _strip_redundant_level_prefix(reason, short_name)

    return f"""
    <div class="scores-overview-row">

        <div class="score-donut-wrap">
            <div class="score-donut" style="background: conic-gradient({donut_color} {overall_pct}%, #e7e7e3 {overall_pct}%);">
                <div class="score-donut-inner">
                    <div class="score-donut-number">{html.escape(str(overall_display))}</div>
                </div>
            </div>
            <div class="score-caption">Overall Score</div>
        </div>

        <div class="cefr-mini">
            <div class="cefr-mini-badge">{html.escape(level)}</div>
            <div class="score-caption">CEFR Score</div>
        </div>

        <div class="score-bars">
            {bar_row("Content Score", scores.get("content", "-"))}
            {bar_row("Delivery Score", scores.get("delivery", "-"))}
            {bar_row("Visual Score", scores.get("visual", "-"))}
        </div>

    </div>

    <div class="cefr-reason-row">
        <strong>{html.escape(level)} &mdash; {html.escape(short_name)}</strong>
        <div class="cefr-reason-text">{html.escape(reason)}</div>
    </div>
    """

def empty_scores_overview():
    return make_scores_overview(
        {"overall": "-", "content": "-", "delivery": "-", "visual": "-"},
        {"level": "-"}
    )

# ============================================================
# OVERVIEW TAB (session details)
# ============================================================

# Each helper returns "good" (green), "moderate" (yellow),
# "attention" (red) or None (no data -> neutral grey).

def _rate_topic(on_topic):
    if on_topic is None:
        return None
    return "good" if on_topic.lower() == "yes" else "attention"


def _rate_audio(silence_pct, notable_events):
    if silence_pct is None:
        return None
    silence_value = int(silence_pct)
    notable_event_count = int(notable_events or 0)
    if silence_value > 50:
        return "attention"
    if silence_value > 30 or notable_event_count > 3:
        return "moderate"
    return "good"

def _rate_pace(wpm):
    if wpm is None:
        return None
    if 110 <= wpm <= 170:
        return "good"
    if 90 <= wpm < 110 or 170 < wpm <= 190:
        return "moderate"
    return "attention"

def _rate_pitch(stddev):
    if stddev is None:
        return None
    if stddev >= 25:
        return "good"
    if stddev >= 15:
        return "moderate"
    return "attention"

def _rate_pauses(long_pause_count):
    if long_pause_count is None:
        return None
    if long_pause_count <= 2:
        return "good"
    if long_pause_count <= 5:
        return "moderate"
    return "attention"

def _rate_eye_contact(pct):
    if pct is None:
        return None
    pct = int(pct)
    if pct >= 60:
        return "good"
    if pct >= 40:
        return "moderate"
    return "attention"

def _rate_head(pct):
    if pct is None:
        return None
    pct = int(pct)
    if pct >= 80:
        return "good"
    if pct >= 60:
        return "moderate"
    return "attention"

def format_analysis_details(details):
    """
    Render session stats as grouped, flat grey tiles - Content /
    Voice and delivery / Body language.
    """

    def tile(label, value, note=None, level=None):
        level_class = f" {level}" if level else ""
        note_html = f'<div class="ov-tile-note">{html.escape(note)}</div>' if note else ""
        return f"""
        <div class="ov-tile{level_class}">
            <div class="ov-tile-label">{html.escape(label)}</div>
            <div class="ov-tile-value">{html.escape(str(value))}</div>
            {note_html}
        </div>
        """

    # CONTENT
    on_topic = details.get("on_topic")
    topic_mismatch = details.get("topic_mismatch")
    coverage_missing = details.get("coverage_missing")
    content_tile = tile(
        "Topic relevance",
        on_topic if on_topic is not None else "-",
        topic_mismatch or coverage_missing,
        _rate_topic(on_topic)
    )

    # VOICE AND DELIVERY
    silence_pct = details.get("silence_pct")
    notable_events = details.get("notable_events")

    if silence_pct is not None:
        audio_tile = tile(
            "Audio quality",
            f"{silence_pct}% silence",
            f"{notable_events} notable event{'s' if notable_events != '1' else ''}",
            _rate_audio(silence_pct, notable_events)
        )
    else:
        audio_tile = tile("Audio quality", "-")

    pace_tile = tile(
        "Pace", details.get("pace_summary") or "-",
        level=_rate_pace(details.get("pace_wpm"))
    )
    pitch_tile = tile(
        "Pitch", details.get("pitch_summary") or "-",
        level=_rate_pitch(details.get("pitch_stddev"))
    )
    pause_tile = tile(
        "Pauses", details.get("pause_summary") or "-",
        level=_rate_pauses(details.get("long_pause_count"))
    )

    # BODY LANGUAGE
    eye_pct = details.get("eye_contact_pct")
    eye_value = f"{eye_pct}%" if eye_pct is not None else "-"
    eye_tile = tile(
        "Eye contact", eye_value, details.get("eye_contact_summary"),
        _rate_eye_contact(eye_pct)
    )

    head_pct = details.get("head_orientation_pct")
    head_value = f"{head_pct}% forward" if head_pct is not None else "-"
    head_tile = tile(
        "Head", head_value, details.get("head_orientation_summary"),
        _rate_head(head_pct)
    )

    return f"""
    <div class="overview-card">

        <div class="overview-title">Performance Overview</div>

        <div class="overview-group">
            <div class="overview-group-label content-label">🎯 CONTENT</div>
            <div class="overview-grid single">{content_tile}</div>
        </div>

        <div class="overview-group">
            <div class="overview-group-label voice-label">🎙️VOICE &amp; DELIVERY</div>
            <div class="overview-grid double">
                {audio_tile}
                {pace_tile}
                {pitch_tile}
                {pause_tile}
            </div>
        </div>

        <div class="overview-group">
            <div class="overview-group-label body-label">🧍🏾BODY LANGUAGE</div>
            <div class="overview-grid double">
                {eye_tile}
                {head_tile}
            </div>
        </div>

    </div>
    """

def empty_session_details():
    return format_analysis_details({})

# ============================================================
# GROUP FILLER INSTANCES BY WORD
# ============================================================

def group_fillers(instances):
    """Group filler instances by word, preserving first-seen order."""
    word_counts = {}
    word_order = []
    for filler in instances:
        word = str(filler.get("word", "")).lower()
        if word not in word_counts:
            word_counts[word] = 0
            word_order.append(word)
        word_counts[word] += 1
    return [(word, word_counts[word]) for word in word_order]

# ============================================================
# QUESTION CARDS
# ============================================================

def make_question_card(question):
    """The 'Your question' block at the top of the Results page."""
    if not question:
        return """
        <div class="question-top-card">
            <div class="question-top-label">Your question</div>
            <div class="question-top-text placeholder">Your question will appear here after analysis.</div>
        </div>
        """
    return f"""
    <div class="question-top-card">
        <div class="question-top-label">Your question</div>
        <div class="question-top-text">{html.escape(question)}</div>
    </div>
    """

def empty_question_card():
    return make_question_card("")

def make_question_recap_box(question):
    """The light purple 'Your question' box shown on the Setup page."""
    if not question:
        return """
        <div class="question-recap-card">
            <div class="question-recap-text placeholder">Your generated question will appear here...</div>
        </div>
        """
    return f"""
    <div class="question-recap-card">
        <div class="question-recap-text">{html.escape(question)}</div>
    </div>
    """

def empty_question_recap_box():
    return make_question_recap_box("")

def make_encouragement_card(report):
    sections = parse_report_sections(report)
    if not sections["encouragement"]:
        return ""
    return f'<div class="encouragement-standalone">&ldquo;{html.escape(sections["encouragement"])}&rdquo;</div>'

def empty_encouragement_card():
    return ""

def make_weakspot_text(followup_info):
    if not followup_info:
        return ""
    label = followup_info.get("label") or "an area to practice"
    return f'<div class="weakspot-text">Weak spot: {html.escape(label)}</div>'

def empty_weakspot_text():
    return ""

# ============================================================
# HIGHLIGHT FILLER WORDS + GRAMMAR ERRORS IN TRANSCRIPT
# ============================================================

def highlight_transcript(text, fillers=None, grammar_data=None):
    """
    Highlight both filler words and grammar errors in the transcript.

    Filler words are matched by word anywhere they occur. Grammar
    errors use the exact character offsets LanguageTool reported
    against this same transcript string.

    Spans are resolved so overlaps don't produce broken/nested
    <mark> tags - if a filler word and a grammar error overlap,
    the grammar highlight takes priority.
    """

    highlight_spans = []
    if grammar_data and grammar_data.get("errors"):
        for error in grammar_data["errors"]:
            error_offset = error.get("offset")
            wrong_text = error.get("wrong", "")
            if error_offset is None or not wrong_text:
                continue
            highlight_spans.append((error_offset, error_offset + len(wrong_text), "grammar-highlight"))
    if fillers and fillers.get("instances"):
        filler_words = {
            str(filler.get("word", ""))
            for filler in fillers["instances"]
            if filler.get("word")
        }
        for word in filler_words:
            if not word:
                continue
            word_pattern = re.compile(rf"(?<!\w){re.escape(word)}(?!\w)", re.IGNORECASE)
            for word_match in word_pattern.finditer(text):
                highlight_spans.append((word_match.start(), word_match.end(), "filler-highlight"))

    if not highlight_spans:
        return html.escape(text)

    # Grammar spans take priority over filler spans when they overlap.
    span_priority = {"grammar-highlight": 0, "filler-highlight": 1}
    highlight_spans.sort(key=lambda span: (span[0], span_priority[span[2]]))
    resolved_spans = []
    last_end = -1

    for start, end, css_class in highlight_spans:
        if start >= last_end:
            resolved_spans.append((start, end, css_class))
            last_end = end

    text_pieces = []
    cursor_position = 0

    for start, end, css_class in resolved_spans:
        text_pieces.append(html.escape(text[cursor_position:start]))
        text_pieces.append(
            f'<mark class="{css_class}">{html.escape(text[start:end])}</mark>'
        )
        cursor_position = end
    text_pieces.append(html.escape(text[cursor_position:]))
    return "".join(text_pieces)

# ============================================================
# FORMAT TRANSCRIPT
# ============================================================

def format_transcript(transcript, fillers=None, grammar_data=None):
    if not transcript:
        return """
        <div class="transcript-card">
            <div class="card-header">
                <div>
                    <div class="card-header-title">Spoken Response Transcript</div>
                    <div class="card-header-subtitle">Your spoken response</div>
                </div>
                <div class="transcript-badge">SPEECH</div>
            </div>
            <div class="transcript-empty">Your transcript will appear here after analysis.</div>
        </div>
        """

    safe_transcript = highlight_transcript(transcript, fillers, grammar_data)
    has_fillers = fillers and fillers.get("count", 0) > 0
    has_grammar_errors = grammar_data and grammar_data.get("error_count", 0) > 0

    legend_items = ""
    if has_fillers:
        legend_items += """
        <span class="transcript-legend-item">
            <span class="filler-highlight-swatch"></span> filler word detected
        </span>
        """
    if has_grammar_errors:
        legend_items += """
        <span class="transcript-legend-item">
            <span class="grammar-highlight-swatch"></span> grammar issue detected
        </span>
        """

    legend = f'<div class="transcript-legend">{legend_items}</div>' if legend_items else ""
    return f"""
    <div class="transcript-card">
        <div class="card-header">
            <div>
                <div class="card-header-title">Spoken Response Transcript</div>
                <div class="card-header-subtitle">Your spoken response</div>
            </div>
            <div class="transcript-badge">SPEECH</div>
        </div>
        <div class="transcript-text-body" style="white-space: pre-wrap;">{safe_transcript}</div>
        {legend}
    </div>
    """

# ============================================================
# FORMAT FILLER WORDS
# ============================================================

def format_fillers(fillers):
    if not fillers or fillers["count"] == 0:
        return """
        <div class="filler-card">
            <div class="card-header">
                <div>
                    <div class="card-header-title">Filler Words Detected</div>
                    <div class="card-header-subtitle">Speech fluency</div>
                </div>
            </div>
            <div class="filler-big-count">0</div>
            <div class="filler-status good">No filler words detected</div>
            <div class="filler-message">Excellent control of filler words.</div>
        </div>
        """

    filler_count = fillers["count"]
    filler_instances = fillers["instances"]
    filler_groups = group_fillers(filler_instances)

    if filler_count <= 5:
        status_class = "good"
        status_message = "Good control of filler words."
    elif filler_count <= 10:
        status_class = "medium"
        status_message = "Try replacing some filler words with short pauses."
    else:
        status_class = "high"
        status_message = "Try practising pauses to reduce filler words."
    breakdown_summary = ", ".join(f"{word} x{count}" for word, count in filler_groups)

    instances_html = ""
    for filler in filler_instances:
        instances_html += f"""
        <div class="filler-instance">
            <span class="filler-word">'{html.escape(str(filler['word']))}'</span>
            <span class="filler-time">{filler['time']:.1f}s</span>
        </div>
        """
    return f"""
    <div class="filler-card">
        <div class="card-header">
            <div>
                <div class="card-header-title">Filler Words Detected</div>
                <div class="card-header-subtitle">Speech fluency</div>
            </div>
        </div>
        <div class="filler-big-count">{filler_count}</div>
        <div class="filler-breakdown-summary">{html.escape(breakdown_summary)}</div>
        <div class="filler-status {status_class}">{status_message}</div>

        <details class="filler-details">
            <summary>Show word-level timestamps</summary>
            <div class="filler-list">{instances_html}</div>
        </details>
    </div>
    """

# ============================================================
# FORMAT GRAMMAR ERRORS
# ============================================================

def format_grammar(grammar_data):
    if not grammar_data or grammar_data.get("error_count", 0) == 0:
        summary = (
            grammar_data.get("summary")
            if grammar_data
            else "No grammar errors detected - excellent accuracy."
        )
        return f"""
        <div class="grammar-card">
            <div class="grammar-head">
                <div>
                    <div class="grammar-title">Grammar Errors Detected</div>
                    <div class="grammar-subtitle">Sentence accuracy</div>
                </div>
            </div>
            <div class="grammar-empty">{html.escape(str(summary))}</div>
        </div>
        """

    errors = grammar_data.get("errors", [])
    error_count = grammar_data.get("error_count", 0)
    summary = grammar_data.get("summary", "")

    errors_html = ""
    for error_number, error in enumerate(errors, 1):
        wrong_text = str(error.get("wrong", ""))
        suggested_text = error.get("suggestion")
        message = str(error.get("message", ""))
        category = str(error.get("category") or "Grammar").replace("_", " ").title()
        correction_html = html.escape(str(suggested_text)) if suggested_text else '<em>No suggestion available</em>'
        errors_html += f"""
        <div class="grammar-error">
            <div class="grammar-error-head">
                <span class="grammar-error-index">Error {error_number}</span>
                <span class="grammar-error-badge">{html.escape(category)}</span>
            </div>
            <div class="grammar-error-cols">
                <div>
                    <div class="grammar-col-label">Incorrect</div>
                    <div class="grammar-wrong">{html.escape(wrong_text)}</div>
                </div>
                <div>
                    <div class="grammar-col-label">Correction</div>
                    <div class="grammar-right">{correction_html}</div>
                </div>
            </div>
            <div class="grammar-explanation">{html.escape(message)}</div>
        </div>
        """

    return f"""
    <div class="grammar-card">

        <div class="grammar-head">
            <div>
                <div class="grammar-title">Grammar Errors Detected</div>
                <div class="grammar-subtitle">Sentence accuracy</div>
            </div>
            <div class="grammar-count-badge">{error_count} {'issue' if error_count == 1 else 'issues'}</div>
        </div>

        <div class="grammar-overall-banner"><strong>Overall:</strong> {html.escape(str(summary))}</div>

        {errors_html}

        <div class="grammar-footer-note">
            Capitalisation, punctuation, spacing and typography rules have
            been excluded, as they reflect written conventions not
            applicable to transcribed speech.
        </div>

    </div>
    """

def empty_grammar_card():
    return format_grammar(None)

# ============================================================
# PARSE REPORT INTO SECTIONS
# ============================================================

def parse_report_sections(report):
    sections = {
        "well": [],
        "feedback": [],
        "improvements": [],
        "encouragement": ""
    }
    if not report:
        return sections

    def extract_bullets(header, text):
        header_pattern = rf"{re.escape(header)}:\s*\n((?:(?:[-\u2022]|\d+\.).*\n?)*)"
        header_match = re.search(header_pattern, text)

        if not header_match:
            return []

        bullet_block = header_match.group(1)
        bullet_items = re.findall(r"^(?:[-\u2022]|\d+\.)\s*(.+)$", bullet_block, re.MULTILINE)
        return [item.strip() for item in bullet_items if item.strip()]

    sections["well"] = extract_bullets("WHAT YOU DID WELL", report)
    sections["feedback"] = extract_bullets("KEY FEEDBACK", report)
    sections["improvements"] = extract_bullets("TOP IMPROVEMENTS", report)

    encouragement_match = re.search(r"ENCOURAGEMENT:\s*\n?(.+)", report, re.DOTALL)

    if encouragement_match:
        raw_encouragement = encouragement_match.group(1).strip()

        # Ensure every line ends with terminal punctuation BEFORE
        # joining, so lines merged from separate model outputs never
        # run together without a full stop between them.
        encouragement_lines = [line.strip() for line in raw_encouragement.split("\n") if line.strip()]
        punctuated_lines = []
        for line in encouragement_lines:
            if line and not line.endswith((".", "!", "?")):
                line += "."
            punctuated_lines.append(line)
        sections["encouragement"] = " ".join(punctuated_lines)
    return sections

# ============================================================
# FORMAT COACHING REPORT
# ============================================================

def format_report(report):
    if not report:
        return """
        <div class="coaching-empty">Your coaching report will appear here after analysis.</div>
        """
    sections = parse_report_sections(report)

    def bullets_html(items, css_class):
        if not items:
            return '<div class="report-empty-row">Nothing to show.</div>'
        return "".join(
            f'<div class="report-bullet {css_class}">{html.escape(item)}</div>'
            for item in items
        )

    return f"""
    <div class="coaching-columns">

        <div class="report-section">
            <div class="report-section-title well">✅ What you did well</div>
            {bullets_html(sections["well"], "well")}
        </div>

        <div class="report-section">
            <div class="report-section-title feedback">💡 Key feedback</div>
            {bullets_html(sections["feedback"], "feedback")}
        </div>

        <div class="report-section">
            <div class="report-section-title improvement">🎯 Top improvements</div>
            {bullets_html(sections["improvements"], "improvement")}
        </div>

    </div>
    """

# ============================================================
# PDF EXPORT (plain, neatly formatted)
# ============================================================

def _clean(text):
    """Strip characters the core PDF font can't render, collapse whitespace."""
    latin1_safe_text = text.encode("latin-1", "ignore").decode("latin-1")
    return " ".join(latin1_safe_text.replace("\r", " ").replace("\n", " ").split())


def _wrap_lines(pdf, text, max_width):
    """
    Manually wrap text into lines guaranteed to fit max_width, measuring
    real rendered width so fpdf2 never has to guess where to break.
    """
    wrapped_lines = []
    current_line = ""
    for word in text.split(" "):
        if not word:
            continue
        candidate_line = (current_line + " " + word).strip()
        if pdf.get_string_width(candidate_line) <= max_width:
            current_line = candidate_line
            continue
        if current_line:
            wrapped_lines.append(current_line)
            current_line = ""
        if pdf.get_string_width(word) <= max_width:
            current_line = word
            continue
        current_piece = ""
        for character in word:
            if pdf.get_string_width(current_piece + character) <= max_width:
                current_piece += character
            else:
                wrapped_lines.append(current_piece)
                current_piece = character
        current_line = current_piece
    if current_line:
        wrapped_lines.append(current_line)
    return wrapped_lines

def export_report_to_pdf():
    """
    Build a neatly formatted PDF from the coaching report - headings,
    spacing and bullet points, no colour, reusing the same
    section-parsing logic as the on-screen report card.
    """
    question = current_question["text"] or "No question recorded."
    report_text = last_attempt.get("report_text", "")
    if not report_text:
        return None
    sections = parse_report_sections(report_text)
    scores = extract_scores(report_text)
    pdf = FPDF()
    pdf.set_margins(18, 18, 18)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    usable_width = pdf.w - pdf.l_margin - pdf.r_margin

    def heading(text, size=13):
        pdf.set_font("Helvetica", "B", size)
        pdf.cell(0, 9, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)
    
    def body(text, size=11):
        pdf.set_font("Helvetica", "", size)
        for line in _wrap_lines(pdf, _clean(text), usable_width):
            pdf.cell(usable_width, 6, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def bullets(items, size=11):
        pdf.set_font("Helvetica", "", size)
        for item in items:
            is_first_line = True
            for line in _wrap_lines(pdf, _clean(item), usable_width - 6):
                prefix = "-  " if is_first_line else "   "
                pdf.cell(usable_width, 6, prefix + line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                is_first_line = False
            pdf.ln(1)

    # Title
    pdf.set_font("Helvetica", "B", 17)
    pdf.cell(0, 11, "AI Speaking Skills - Coaching Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    # Question
    heading("Speaking Question", 12)
    body(question)
    pdf.ln(4)

    # Scores
    heading("Scores", 12)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 6, f"Overall: {scores.get('overall', '-')}/100    "
                    f"Content: {scores.get('content', '-')}/100    "
                    f"Delivery: {scores.get('delivery', '-')}/100    "
                    f"Visual: {scores.get('visual', '-')}/100", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)

    # Performance overview
    details = last_attempt.get("analysis_details") or {}
    overview_lines = []

    if details.get("on_topic") is not None:
        overview_lines.append(f"Topic Relevance: {details['on_topic']}")
    if details.get("pace_summary"):
        overview_lines.append(f"Pace: {details['pace_summary']}")
    if details.get("pitch_summary"):
        overview_lines.append(f"Pitch: {details['pitch_summary']}")
    if details.get("pause_summary"):
        overview_lines.append(f"Pauses: {details['pause_summary']}")
    if details.get("eye_contact_pct"):
        overview_lines.append(f"Eye Contact: {details['eye_contact_pct']}%")
    if details.get("head_orientation_pct"):
        overview_lines.append(f"Head Orientation: {details['head_orientation_pct']}% forward")

    if overview_lines:
        heading("Performance Overview")
        bullets(overview_lines)
        pdf.ln(3)

    # Filler word summary
    filler_data = last_attempt.get("filler_data")
    if filler_data is not None:
        heading("Filler Words")
        filler_count = filler_data.get("count", 0)
        body("No filler words detected." if filler_count == 0 else f"{filler_count} filler word(s) detected.")
        pdf.ln(3)

    # Grammar errors
    grammar_data = last_attempt.get("grammar_data")
    if grammar_data and grammar_data.get("error_count", 0) > 0:
        heading("Grammar Errors")
        for error_number, error in enumerate(grammar_data.get("errors", []), 1):
            wrong_text = error.get("wrong", "")
            suggested_text = error.get("suggestion") or "no suggestion"
            message = error.get("message", "")
            bullets([f"{wrong_text} -> {suggested_text}: {message}"])
        pdf.ln(3)
    elif grammar_data:
        heading("Grammar Errors")
        body(grammar_data.get("summary", "No grammar errors detected."))
        pdf.ln(3)

    # What you did well
    if sections["well"]:
        heading("What You Did Well")
        bullets(sections["well"])
        pdf.ln(3)

    # Key feedback
    if sections["feedback"]:
        heading("Key Feedback")
        bullets(sections["feedback"])
        pdf.ln(3)

    # Top improvements
    if sections["improvements"]:
        heading("Top Improvements")
        bullets(sections["improvements"])
        pdf.ln(3)

    # Encouragement
    if sections["encouragement"]:
        heading("Encouragement")
        pdf.set_font("Helvetica", "I", 11)
        for line in _wrap_lines(pdf, _clean(sections["encouragement"]), usable_width):
            pdf.cell(usable_width, 6, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    output_path = os.path.join(tempfile.gettempdir(), "speaking_coaching_report.pdf")
    pdf.output(output_path)
    return output_path

def prepare_pdf():
    """Build the PDF as soon as the analysis finishes, so the download
    button already holds the file when the results page appears."""
    if not analysis_state["ok"]:
        return gr.update()
    return export_report_to_pdf()

# ============================================================
# STATUS MESSAGES
# ============================================================

def make_loading_status(stage_text, note_text=""):
    """
    Inline loading card (spinner + text). Used for the question and
    follow-up messages, which sit in the page rather than covering it.
    """
    note_html = f'<div class="status-text">{html.escape(note_text)}</div>' if note_text else ""
    return f"""
    <div class="analysis-status loading">
        <div class="spinner"></div>
        <div>
            <div class="status-title">{html.escape(stage_text)}</div>
            {note_html}
        </div>
    </div>
    """

def make_static_status(title_text, note_text=""):
    """
    Like make_loading_status, but without the spinner - used for
    short-lived confirmation or error messages.
    """
    note_html = f'<div class="status-text">{html.escape(note_text)}</div>' if note_text else ""
    return f"""
    <div class="analysis-status">
        <div>
            <div class="status-title">{html.escape(title_text)}</div>
            {note_html}
        </div>
    </div>
    """

def make_overlay_status(stage_text, note_text=""):
    """Full-screen translucent overlay with a live status message."""
    note_html = f'<div class="status-text">{html.escape(note_text)}</div>' if note_text else ""
    return f"""
    <div class="analysis-overlay">
        <div class="analysis-overlay-card">
            <div class="spinner"></div>
            <div class="status-title">{html.escape(stage_text)}</div>
            {note_html}
            <div class="status-text overlay-footnote">
                This runs on your own computer, so the full analysis can take
                a while. Please keep this tab open.
            </div>
        </div>
    </div>
    """

# ============================================================
# ANALYSE VIDEO
# ============================================================

def start_loading():
    """
    Runs instantly on click. Greys out the button and shows the overlay
    with the first status. From here on, every further status update
    comes from analyse_video itself as each pipeline step actually
    starts, so the text on screen reflects real progress.
    """
    return (
        gr.update(interactive=False, elem_classes=["primary-btn", "btn-loading"]),
        gr.update(visible=True, value=make_overlay_status("Starting analysis...")),
        gr.update(interactive=False),
        gr.update(interactive=False)
    )

def analyse_video(video_path):
    """
    Full analysis pipeline.

    This is a generator: it yields a full update tuple before each real
    step and once more at the end with the final results. Every yield
    must supply values for ALL outputs bound to this event; for
    stage-progress yields we leave everything except status_output
    and analyse_btn untouched with gr.update().
    """
    analysis_state["ok"] = False
    NOOP = gr.update()

    def stage(text, note=""):
        """One progress yield: only status_output (the overlay) changes."""
        return (
            NOOP,                                       # question_recap_output2
            NOOP,                                       # scores_overview_output
            NOOP, NOOP,                                 # report_output, encouragement_output
            NOOP, NOOP, NOOP, NOOP,                     # transcript / filler / session / grammar
            NOOP, NOOP, NOOP,                           # try_again_btn, new_question_btn, followup_btn
            NOOP,                                       # weakspot_output
            gr.update(visible=True, value=make_overlay_status(text, note)),  # status_output
            gr.update(interactive=False, elem_classes=["primary-btn", "btn-loading"]),  # analyse_btn
            NOOP,                                       # topic_input
            NOOP                                        # difficulty_input
        )

    def blank_results(message, analyse_interactive, lock_inputs):
        """Reset every result card and show a plain (non-overlay) message."""
        return (
            empty_question_card(),
            empty_scores_overview(),
            format_report(""),
            empty_encouragement_card(),
            format_transcript(""),
            format_fillers({"count": 0, "instances": []}),
            empty_session_details(),
            empty_grammar_card(),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            empty_weakspot_text(),
            gr.update(visible=True, value=message),
            gr.update(interactive=analyse_interactive, elem_classes=["primary-btn"]),
            gr.update(interactive=not lock_inputs),
            gr.update(interactive=not lock_inputs)
        )

    # --------------------------------------------------------
    # Initial validation
    # --------------------------------------------------------

    if not video_path:
        yield blank_results(
            make_static_status("Please upload a video first."),
            analyse_interactive=True,
            lock_inputs=False
        )
        return
    if not current_question["text"]:
        yield blank_results(
            make_static_status("Please generate a question first."),
            analyse_interactive=True,
            lock_inputs=True
        )
        return
    if not is_ollama_available():
        yield blank_results(
            make_static_status(
                "Ollama isn't running",
                "Please start Ollama (run 'ollama serve' in a terminal), then try again."
            ),
            analyse_interactive=True,
            lock_inputs=bool(video_path)
        )
        return
    audio_path = None

    try:
        # ----------------------------------------------------
        # STEP 1 - extract + transcribe
        # ----------------------------------------------------

        yield stage("Extracting audio from your video...")
        audio_path = extract_audio(video_path)
        yield stage("Transcribing your speech...", "This can take a while for longer clips.")
        transcript = transcribe_audio(audio_path)
        word_count = len(transcript["words"])

        # Uses the same MIN_RESPONSE_WORDS constant that
        # generate_report() checks internally, so the UI-level
        # rejection and the report-level rejection can never disagree.

        if word_count < MIN_RESPONSE_WORDS:
            yield blank_results(
                make_static_status("Your recording was too short."),
                analyse_interactive=True,
                lock_inputs=True
            )
            return

        # ----------------------------------------------------
        # STEP 2 - filler detection
        # ----------------------------------------------------

        yield stage("Checking for filler words...")
        fillers = detect_fillers(transcript)

        # ----------------------------------------------------
        # STEP 2b - pause and pace analysis
        # ----------------------------------------------------

        yield stage("Analysing pacing and pauses...")
        pause_data = analyse_pauses(transcript)
        pace_data = analyse_pace(transcript)

        # ----------------------------------------------------
        # STEP 3 - gaze analysis 
        # ----------------------------------------------------

        yield stage("Analysing eye contact and gaze...")
        try:
            gaze_data = analyse_gaze(video_path)
        except Exception as gaze_error:
            print(f"Gaze analysis skipped: {gaze_error}")
            gaze_data = None

        # ----------------------------------------------------
        # STEP 4 - audio event analysis 
        # ----------------------------------------------------

        yield stage("Analysing audio quality...")
        try:
            yamnet_result = analyse_audio(audio_path)
        except Exception as audio_error:
            print(f"Audio analysis skipped: {audio_error}")
            yamnet_result = {
                "summary": "Audio analysis unavailable."
            }

        # ----------------------------------------------------
        # STEP 4b - pitch analysis 
        # ----------------------------------------------------

        yield stage("Analysing pitch and tone...")
        try:
            pitch_data = analyse_pitch(audio_path)
        except Exception as pitch_error:
            print(f"Pitch analysis skipped: {pitch_error}")
            pitch_data = None

        # ----------------------------------------------------
        # STEP 5 - grammar analysis 
        # ----------------------------------------------------

        yield stage("Checking grammar...")
        try:
            grammar_data = check_grammar(transcript["text"])
        except Exception as grammar_error:
            print(f"Grammar check skipped: {grammar_error}")
            grammar_data = None

        # ----------------------------------------------------
        # STEP 6 - generate full report
        # ----------------------------------------------------

        yield stage(
            "Generating your coaching report...",
            "This can take a few minutes on this computer."
        )
        report = generate_report(
            question=current_question["text"],
            transcript=transcript["text"],
            filler_data=fillers,
            yamnet_summary=yamnet_result["summary"],
            gaze_data=gaze_data,
            grammar_data=grammar_data,
            pause_data=pause_data,
            pace_data=pace_data,
            pitch_data=pitch_data
        )

        # ----------------------------------------------------
        # EXTRACT SCORES / CEFR / DETAILS
        # ----------------------------------------------------

        scores = extract_scores(report)
        cefr = extract_cefr(report)
        analysis_details = extract_analysis_details(report)
        quality_rating = extract_quality_rating(report)
        followup_info = determine_followup_focus(analysis_details, quality_rating)

        last_attempt["question"] = current_question["text"]
        last_attempt["followup"] = followup_info
        last_attempt["report_text"] = report
        last_attempt["filler_data"] = fillers
        last_attempt["grammar_data"] = grammar_data
        last_attempt["analysis_details"] = analysis_details

        # ----------------------------------------------------
        # FORMAT RESULTS
        # ----------------------------------------------------

        transcript_html = format_transcript(transcript["text"], fillers, grammar_data)
        filler_html = format_fillers(fillers)
        report_html = format_report(report)
        encouragement_html = make_encouragement_card(report)
        session_html = format_analysis_details(analysis_details)
        grammar_html = format_grammar(grammar_data)
        scores_html = make_scores_overview(scores, cefr)

        # ----------------------------------------------------
        # CLEAN UP AUDIO
        # ----------------------------------------------------

        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
            audio_path = None

        # ----------------------------------------------------
        # RETURN FINAL RESULTS
        # ----------------------------------------------------

        analysis_state["ok"] = True
        yield (
            make_question_card(current_question["text"]),
            scores_html,
            report_html,
            encouragement_html,
            transcript_html,
            filler_html,
            session_html,
            grammar_html,

            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=followup_info is not None),
            make_weakspot_text(followup_info),

            gr.update(visible=False, value=""),
            gr.update(interactive=False, elem_classes=["primary-btn"]),
            gr.update(interactive=False),
            gr.update(interactive=False)
        )

    except Exception as analysis_error:
        print(f"Analysis failed: {analysis_error}")
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception:
                pass
        error_text = str(analysis_error).lower()

        if "timed out" in error_text or "timeout" in error_text:
            friendly_title = "The AI is taking too long to respond"
            friendly_note = "This can happen if the model is under heavy load. Please try again."
        elif "connectionerror" in error_text or "connection" in error_text:
            friendly_title = "Could not reach the AI model"
            friendly_note = "Please make sure Ollama is running, then try again."
        else:
            friendly_title = "Something went wrong during analysis"
            friendly_note = "Please try again."

        yield blank_results(
            make_static_status(friendly_title, friendly_note),
            analyse_interactive=True,
            lock_inputs=bool(video_path)
        )

# ============================================================
# TRY AGAIN
# ============================================================

def try_again():
    """Keep same question and clear previous results."""
    video_reset_mode["value"] = "try_again"
    return (
        None,
        empty_question_card(),
        empty_scores_overview(),
        format_report(""),
        empty_encouragement_card(),
        format_transcript(""),
        format_fillers({"count": 0, "instances": []}),
        empty_session_details(),
        empty_grammar_card(),

        gr.update(visible=False),  # try_again_btn
        gr.update(visible=False),  # new_question_btn
        gr.update(visible=False),  # followup_btn
        empty_weakspot_text(),
        gr.update(visible=False, value=""),  # status_output
        gr.update(visible=False),  # question_status

        # Same question stays on screen for the next attempt.
        make_question_recap_box(current_question["text"]),
        gr.update(interactive=False, elem_classes=["primary-btn"]),  # analyse_btn
        gr.update(interactive=False),  # topic_input
        gr.update(interactive=False),  # difficulty_input
        gr.update(interactive=False)   # generate_btn
    )


# ============================================================
# NEW QUESTION
# ============================================================
def new_question():
    """Reset everything for a completely new question."""
    video_reset_mode["value"] = "new_question"
    current_question["text"] = ""
    last_attempt["question"] = ""
    last_attempt["followup"] = None
    return (
        gr.update(value="", interactive=True),  # Topic
        gr.update(interactive=True),            # Difficulty
        "",                                     # Question
        gr.update(value=None, interactive=False),  # Video

        empty_question_card(),
        empty_scores_overview(),
        format_report(""),
        empty_encouragement_card(),
        format_transcript(""),
        format_fillers({"count": 0, "instances": []}),
        empty_session_details(),
        empty_grammar_card(),

        gr.update(interactive=False, elem_classes=["primary-btn"]),  # Generate
        gr.update(interactive=False, elem_classes=["primary-btn"]),    # Analyse
        gr.update(visible=False),  # Try Again
        gr.update(visible=False),  # New Question
        gr.update(visible=False),  # Follow-up
        empty_weakspot_text(),

        gr.update(visible=False),  # status_output
        gr.update(visible=False, value=""),  
        empty_question_recap_box()
    )


# ============================================================
# FOLLOW-UP QUESTION HANDLER
# ============================================================
def generate_followup():
    """
    Generate a follow-up question targeting last_attempt's weak
    area, then reset the results like Try Again does, so the
    student gets a clean slate to answer it.
    """
    video_reset_mode["value"] = "followup"
    followup_info = last_attempt.get("followup")
    if not followup_info:
        # The button shouldn't be visible without a real focus,
        # but stay safe if this ever fires anyway.
        yield (
            gr.update(),               # question_output
            gr.update(),               # video_input
            empty_question_card(),
            empty_scores_overview(),
            format_report(""),
            empty_encouragement_card(),
            format_transcript(""),
            format_fillers({"count": 0, "instances": []}),
            empty_session_details(),
            empty_grammar_card(),
            gr.update(visible=False),  # try_again_btn
            gr.update(visible=False),  # new_question_btn
            gr.update(visible=False),  # followup_btn
            empty_weakspot_text(),

            gr.update(visible=False),  # question_status
            gr.update(),               # question_recap_output (setup box)
            gr.update(visible=False),  # status_output
            gr.update(interactive=True, elem_classes=["primary-btn"]),  # analyse_btn
            gr.update(interactive=False),  # topic_input
            gr.update(interactive=False),  # difficulty_input
            gr.update(interactive=False)   # generate_btn
        )
        return

    # Immediate loading feedback in the live message under Step 01.
    yield (
        gr.update(value=""),                       # question_output - clear immediately
        gr.update(value=None, interactive=False),   # video_input - clear immediately
        empty_question_card(),                      # question_recap_output2
        empty_scores_overview(),                    # scores_overview_output

        gr.update(),  # report_output
        gr.update(),  # encouragement_output
        gr.update(),  # transcript_output
        gr.update(),  # filler_output
        gr.update(),  # session_output
        gr.update(),  # grammar_output
        gr.update(),  # try_again_btn
        gr.update(),  # new_question_btn
        gr.update(),  # followup_btn
        gr.update(),  # weakspot_output
        gr.update(
            visible=True,
            value=make_loading_status(
                "Preparing your follow-up question...",
                "Just a moment, this runs on your own computer."
            )
        ),  # question_status
        empty_question_recap_box(),  # question_recap_output (setup box) - clear immediately

        gr.update(visible=False, value=""),  # status_output
        gr.update(interactive=False),  # analyse_btn
        gr.update(interactive=False),  # topic_input
        gr.update(interactive=False),  # difficulty_input
        gr.update(interactive=False)   # generate_btn
    )

    focus_text = followup_info["focus"]
    question = generate_followup_question(
        last_attempt["question"],
        focus_text,
        followup_info.get("detail", "")
    )
    current_question["text"] = question
    last_attempt["followup"] = None

    yield (
        question,
        gr.update(value=None, interactive=True),
        empty_question_card(),
        empty_scores_overview(),
        format_report(""),
        empty_encouragement_card(),
        format_transcript(""),
        format_fillers({"count": 0, "instances": []}),
        empty_session_details(),
        empty_grammar_card(),
        gr.update(visible=False),  # try_again_btn
        gr.update(visible=False),  # new_question_btn
        gr.update(visible=False),  # followup_btn
        empty_weakspot_text(),

        gr.update(
            visible=True,
            value=make_static_status(
                "Follow-up question ready",
                f"Focus: {focus_text}"
            )
        ),  # question_status
        make_question_recap_box(question),

        gr.update(visible=False),  # status_output
        gr.update(interactive=False, elem_classes=["primary-btn"]),  # analyse_btn
        gr.update(interactive=False),  # topic_input
        gr.update(interactive=False),  # difficulty_input
        gr.update(interactive=False)   # generate_btn
    )

# ============================================================
# CUSTOM CSS
# ============================================================

CUSTOM_CSS = """

/* ============================================================
   PAGE
   ============================================================ */

:root { --body-background-fill: #fdfcff; }

body, gradio-app { background: #fdfcff !important; }

.gradio-container { background: transparent !important; }

.gr-form, .block, .gradio-html, .prose {
    --block-border-color: transparent !important;
}

.generating, .pending {
    border: none !important;
    box-shadow: none !important;
    animation: none !important;
}

.gradio-container {
    max-width: 980px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    padding: 18px 24px 60px;
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    overflow-x: hidden;
    box-sizing: border-box;
}

.gradio-html {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

.report-bullet-line,
.transcript-text-body,
.grammar-row-wrong,
.grammar-row-correct,
.grammar-row-explanation,
.ov-tile-note,
.status-text,
.status-title {
    overflow-wrap: break-word;
    word-break: break-word;
}

/* ============================================================
   HEADER + SECTION TEXT 
   ============================================================ */

.app-header {
    background: linear-gradient(135deg, #17172b 0%, #29275c 100%);
    border-radius: 20px;
    padding: 24px 30px;
    margin-bottom: 22px;
    color: white;
    box-shadow: 0 12px 35px rgba(30, 30, 70, 0.15);
}

.app-header h1 {
    font-size: 26px;
    color: white;
    font-weight: 750;
    margin: 0 0 8px 0;
    letter-spacing: -1px;
}

.app-header p {
    margin: 0;
    font-size: 16px;
    color: #c9c8e8;
}

.section-title {
    font-size: 18px;
    font-weight: 700;
    color: #1d1d32;
    margin: 20px 0 6px 4px;
}

.section-subtitle {
    color: #77778c;
    font-size: 14px;
    margin: 0 0 10px 4px;
    line-height: 1.5;
}

.step-badge {
    display: inline-block;
    background: #ecebff;
    color: #6258d6;
    font-size: 11px;
    font-weight: 800;
    padding: 7px 13px;
    border-radius: 999px;
    margin-top: 10px;
    letter-spacing: 0.6px;
}

.webcam-help {
    background: #f7f6ff;
    border: 1px solid #dedcff;
    border-radius: 14px;
    padding: 14px 18px;
    font-size: 13px;
    line-height: 1.6;
    color: #55556b;
    margin-bottom: 6px;
}

.webcam-help strong { color: #292844; }

.results-header {
    background: linear-gradient(135deg, #17172b 0%, #29275c 100%);
    border-radius: 22px;
    padding: 25px 30px;
    color: white;
    margin-top: 12px;
    margin-bottom: 20px;
}

.results-header h2 {
    margin: 0;
    color: white;
    font-size: 25px;
}

.results-header p {
    margin: 6px 0 0 0;
    color: #c9c8e8;
}

.footer-text {
    text-align: center;
    color: #9999aa;
    font-size: 12px;
    margin-top: 35px;
}

/* ============================================================
   CARDS (white rounded containers)
   ============================================================ */

.card-col {
    background: white;
    border: 1px solid #e7e7ef;
    border-radius: 20px;
    padding: 22px;
    box-shadow: 0 5px 18px rgba(25, 25, 55, 0.05);
    box-sizing: border-box;
    margin-bottom: 18px;
}

.card-col .block,
.card-col .form {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* labels above inputs */
.card-col [data-testid="block-info"] {
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #66667a !important;
    background: transparent !important;
}

/* ============================================================
   SETUP PAGE
   ============================================================ */

.card-col .form {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: wrap !important;
    align-items: flex-start !important;
    gap: 16px !important;
}

#topic-input { flex: 3 1 280px !important; min-width: 0 !important; }
#difficulty-input { flex: 2 1 300px !important; min-width: 0 !important; }

#topic-input label,
#difficulty-input > label {
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #66667a !important;
    margin: 0 0 6px 0 !important;
    padding: 0 !important;
    line-height: 1.3 !important;
    display: block !important;
}

#topic-input input,
#topic-input textarea {
    background: white !important;
    border: 1px solid #d9d9e0 !important;
    border-radius: 12px !important;
    padding: 12px 14px !important;
    font-size: 15px !important;
    box-shadow: none !important;
    color: #1d1d32 !important;
}

#topic-input input:focus,
#topic-input textarea:focus {
    border-color: #6c63e8 !important;
    outline: none !important;
}

#difficulty-input .wrap {
    display: flex !important;
    flex-wrap: nowrap !important;
    gap: 0 !important;
    padding: 0 !important;
    margin-top: 0 !important;
    background: white !important;
    border: 1px solid #d9d9e0 !important;
    border-radius: 12px !important;
    overflow: hidden;
}

#difficulty-input label {
    flex: 1 1 0 !important;
    display: flex !important;
    align-items: center;
    justify-content: center;
    margin: 0 !important;
    padding: 11px 8px !important;
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    color: #66667a !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    text-transform: capitalize;
    cursor: pointer;
}

#difficulty-input label span { color: inherit !important; }

#difficulty-input label.selected,
#difficulty-input label:has(input:checked) {
    background: #ecebfa !important;
    color: #4b43cf !important;
}

#difficulty-input input[type="radio"] {
    position: absolute;
    opacity: 0;
    width: 0;
    height: 0;
    margin: 0;
    pointer-events: none;
}

/* Buttons */

.primary-btn {
    background: #5a4bb0 !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    min-height: 48px;
    box-shadow: none !important;
    position: relative;
}

.primary-btn:hover { background: #4d3fa0 !important; }

.secondary-btn {
    background: white !important;
    color: #1d1d32 !important;
    border: 1px solid #d9d9e0 !important;
    border-radius: 12px !important;
    font-weight: 500 !important;
    min-height: 46px;
    box-shadow: none !important;
    position: relative;
}

.secondary-btn:hover { background: #f7f7fa !important; }

.primary-btn:disabled,
.primary-btn[disabled],
.secondary-btn:disabled,
.secondary-btn[disabled] {
    opacity: 0.45 !important;
    cursor: not-allowed !important;
}

.primary-btn.btn-loading,
.secondary-btn.btn-loading {
    color: transparent !important;
    pointer-events: none;
}

.primary-btn.btn-loading::after,
.secondary-btn.btn-loading::after {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 18px;
    height: 18px;
    margin: -9px 0 0 -9px;
    border-radius: 50%;
    border: 2.5px solid rgba(255, 255, 255, 0.35);
    border-top-color: #ffffff;
    animation: spin 0.8s linear infinite;
}

.secondary-btn.btn-loading::after {
    border: 2.5px solid rgba(98, 88, 214, 0.25);
    border-top-color: #6258d6;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

.caption-text {
    text-align: center;
    font-size: 13px;
    color: #9999aa;
    margin-top: 2px;
}

/* Question box on the Setup page */

.question-recap-card {
    background: #ecebfa;
    border-radius: 16px;
    padding: 18px 22px;
    margin-bottom: 8px;
}

.question-recap-label {
    font-size: 14px;
    font-weight: 500;
    color: #5f57b8;
    margin-bottom: 6px;
}

.question-recap-text {
    font-size: 19px;
    font-weight: 500;
    color: #3a3390;
    line-height: 1.45;
}

.question-recap-text.placeholder {
    font-size: 15px;
    font-weight: 400;
    color: #8b87c4;
}

#setup-card { margin-bottom: 0 !important; }

.section-title.tight-top { margin-top: 0; margin-bottom: 4px; }

.section-title.tight-top + .section-subtitle {
    margin-bottom: 8px;
}

/* Video */

#upload-box {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

#upload-box .block {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    overflow: visible !important;
}

/* the dashed box is only the drop zone */
#upload-box .upload-container {
    background: white !important;
    border: 1.5px dashed #d0d0dc !important;
    border-radius: 16px !important;
    overflow: hidden;
}

#upload-box video:not(.hide) {
    width: 100% !important;
    max-width: 100% !important;
    height: 380px !important;
    max-height: 380px !important;
    object-fit: contain !important;
    margin: 0 auto !important;
}

#upload-box video.hide { display: none !important; }

/* Upload / Webcam source buttons under the drop zone */

#upload-box .source-selection {
    display: flex !important;
    gap: 12px !important;
    height: auto !important;
    padding: 12px 0 0 0 !important;
    background: transparent !important;
    border: none !important;
}

#upload-box .source-selection button {
    flex: 1 1 0;
    display: flex !important;
    align-items: center;
    justify-content: center;
    gap: 8px;
    width: auto !important;
    height: auto !important;
    margin: 0 !important;
    padding: 10px 12px !important;
    background: white !important;
    border: 1px solid #d9d9e0 !important;
    border-radius: 12px !important;
    color: #1d1d32 !important;
    font-size: 14px !important;
    font-weight: 500 !important;
}

#upload-box .source-selection button.selected {
    background: #ecebfa !important;
    color: #4b43cf !important;
}

#upload-box .source-selection button svg {
    width: 18px !important;
    height: 18px !important;
    flex-shrink: 0;
}

#upload-box .source-selection button:first-of-type::after { content: "Upload"; }
#upload-box .source-selection button:last-of-type::after  { content: "Webcam"; }

/* ============================================================
   LIVE STATUS MESSAGES 
   ============================================================ */

.analysis-status {
    display: flex;
    align-items: center;
    gap: 16px;
    background: #f7f6ff;
    border: 1px solid #dedcff;
    border-radius: 18px;
    padding: 18px 22px;
    margin: 15px 0;
}

.analysis-status.loading {
    background: #ecebff;
    border-color: #d8d6ff;
}

.spinner {
    width: 22px;
    height: 22px;
    flex-shrink: 0;
    border-radius: 50%;
    border: 3px solid #d8d6ff;
    border-top-color: #6258d6;
    animation: spin 0.8s linear infinite;
}

.status-title {
    font-size: 16px;
    font-weight: 700;
    color: #292844;
}

.status-text {
    font-size: 13px;
    color: #77778c;
    margin-top: 3px;
}

.analysis-overlay {
    position: fixed;
    top: 0;
    right: 0;
    bottom: 0;
    left: 0;
    z-index: 9999;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(23, 23, 43, 0.55);
    backdrop-filter: blur(4px);
}

.analysis-overlay-card {
    background: white;
    border-radius: 20px;
    padding: 28px 34px;
    width: calc(100% - 40px);
    max-width: 420px;
    text-align: center;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.25);
}

.analysis-overlay-card .spinner {
    width: 34px;
    height: 34px;
    margin: 0 auto 16px;
}

.overlay-footnote {
    margin-top: 16px;
    padding-top: 12px;
    border-top: 1px solid #ececf1;
    font-size: 12.5px;
    line-height: 1.5;
}

/* ============================================================
   RESULTS PAGE - top card (question + Save PDF + scores)
   ============================================================ */

.result-top-card { gap: 0 !important; }

.result-top-card > .row,
.result-top-card .row {
    flex-wrap: nowrap !important;
    align-items: flex-start !important;
    gap: 16px !important;
}

.question-top-label {
    font-size: 15px;
    color: #6b6b78;
    margin-bottom: 4px;
}

.question-top-text {
    font-size: 21px;
    font-weight: 500;
    color: #1d1d32;
    line-height: 1.35;
}

.question-top-text.placeholder {
    font-size: 15px;
    font-weight: 400;
    color: #9999aa;
}

.pdf-btn-top {
    width: 100%;
    max-width: 170px;
    margin-left: auto;
}

/* donut + CEFR + bars */

.scores-overview-row {
    display: grid;
    grid-template-columns: auto auto 1fr;
    align-items: center;
    column-gap: 44px;
    row-gap: 22px;
    margin-top: 20px;
    padding-top: 22px;
    border-top: 1px solid #ececf1;
}

.score-donut-wrap,
.cefr-mini {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
}

.score-donut {
    width: 124px;
    height: 124px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
}

.score-donut-inner {
    width: 96px;
    height: 96px;
    background: white;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
}

.score-donut-number {
    font-size: 34px;
    font-weight: 500;
    color: #1d1d32;
}

.score-caption {
    font-size: 15px;
    color: #55556b;
    text-align: center;
}

.cefr-mini-badge {
    width: 72px;
    height: 56px;
    border-radius: 14px;
    background: #ecebfa;
    color: #4b43cf;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    font-weight: 500;
}

.cefr-reason-row {
    margin-top: 16px;
    padding-top: 16px;
    border-top: 1px solid #ececf1;
    font-size: 14px;
    line-height: 1.5;
    color: #55556b;
}

.cefr-reason-row strong {
    color: #1d1d32;
}

.cefr-reason-text {
    margin-top: 4px;
    font-weight: 400;
    color: #55556b;
}

.score-bars {
    display: flex;
    flex-direction: column;
    gap: 16px;
    min-width: 0;
}

.score-bar-head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 6px;
}

.score-bar-label {
    font-size: 15px;
    color: #55556b;
}

.score-bar-value {
    font-size: 15px;
    color: #1d1d32;
}

.score-bar-track {
    width: 100%;
    height: 8px;
    border-radius: 999px;
    background: #f1f1ee;
    overflow: hidden;
}

.score-bar-fill {
    height: 100%;
    border-radius: 999px;
}

/* ============================================================
   RESULTS PAGE - tabs (Overview / Speech / Grammar / Coaching)
   ============================================================ */

#results-subtabs [role="tablist"],
#results-subtabs .tab-nav {
    background: transparent !important;
    border: none !important;
    border-bottom: 1px solid #e3e3ea !important;
    gap: 26px !important;
    padding: 0 4px !important;
    margin-bottom: 0 !important;
}

#results-subtabs button[role="tab"] {
    border: none !important;
    background: transparent !important;
    color: #8a8a99 !important;
    font-weight: 500 !important;
    font-size: 15px !important;
    padding: 10px 2px 12px 2px !important;
    margin: 0 !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    box-shadow: none !important;
}

#results-subtabs button[role="tab"]::after { display: none !important; }

#results-subtabs button[role="tab"].selected {
    color: #4b43cf !important;
    border-bottom: 2px solid #6c63e8 !important;
}

#results-subtabs [role="tabpanel"],
#results-subtabs .tabitem {
    background: white !important;
    border: 1px solid #e7e7ef !important;
    border-radius: 18px !important;
    padding: 18px !important;
    margin-top: 14px !important;
    box-shadow: 0 5px 18px rgba(25, 25, 55, 0.05);
    box-sizing: border-box;
}

/* Overview tab */

.overview-card { padding: 0; }

.overview-group { margin-bottom: 18px; }
.overview-group:last-child { margin-bottom: 0; }

.overview-title {
    font-size: 17px;
    font-weight: 750;
    color: #292844;
    margin-bottom: 18px;
}

.overview-group-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.8px;
    margin-bottom: 10px;
    padding-left: 2px;
}

.content-label { color: #6258d6; }
.voice-label   { color: #1a9c8f; }
.body-label    { color: #d67a2c; }

.overview-grid { display: grid; gap: 12px; }

.overview-grid.single { grid-template-columns: 1fr; }
.overview-grid.double { grid-template-columns: repeat(2, minmax(0, 1fr)); }

.ov-tile {
    background: #f7f7fa;
    border-radius: 12px;
    padding: 14px 16px;
    min-width: 0;
}

.ov-tile-label {
    font-size: 14px;
    color: #8b8b96;
    margin-bottom: 2px;
}

.ov-tile-value {
    font-size: 18px;
    font-weight: 500;
    color: #1d1d32;
}

.ov-tile-note {
    font-size: 12.5px;
    color: #77778c;
    margin-top: 4px;
    line-height: 1.4;
}

/* Performance colours (pastel) */

.ov-tile.good      { background: #eef8f0; }
.ov-tile.moderate  { background: #fdf6e3; }
.ov-tile.attention { background: #fbeceb; }

.ov-tile.good .ov-tile-value      { color: #3f8a55; }
.ov-tile.moderate .ov-tile-value  { color: #a67c12; }
.ov-tile.attention .ov-tile-value { color: #c4605c; }

/* Speech tab */

.filler-card,
.transcript-card {
    background: white;
    border: 1px solid #e7e7ef;
    border-radius: 22px;
    padding: 24px 26px;
    box-shadow: 0 5px 18px rgba(25, 25, 55, 0.05);
    height: 100%;
    box-sizing: border-box;
}

.card-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 18px;
}

.card-header-title {
    font-size: 18px;
    font-weight: 750;
    color: #211d3a;
    line-height: 1.3;
}

.card-header-subtitle {
    font-size: 14px;
    color: #9a9aa8;
    margin-top: 3px;
}

.transcript-badge {
    background: #ecebfa;
    color: #4b43cf;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.4px;
    padding: 5px 11px;
    border-radius: 999px;
    flex-shrink: 0;
}

.grammar-count-badge {
    background: #f9e0de;
    color: #a43d3d;
    font-size: 11px;
    font-weight: 750;
    padding: 5px 11px;
    border-radius: 999px;
    flex-shrink: 0;
    white-space: nowrap;
}

.filler-big-count {
    font-size: 36px;
    font-weight: 500;
    color: #1d1d32;
    line-height: 1.15;
}

.filler-breakdown-summary {
    font-size: 14px;
    color: #66667a;
    margin-top: 4px;
    line-height: 1.5;
}

.filler-status {
    display: inline-block;
    margin-top: 12px;
    padding: 5px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
}

.filler-status.good   { background: #e9f8ef; color: #27804c; }
.filler-status.medium { background: #fff5dd; color: #9b6c10; }
.filler-status.high   { background: #ffe8e8; color: #a43d3d; }

.filler-message {
    color: #66667a;
    font-size: 13px;
    line-height: 1.5;
    margin: 10px 0 0 0;
}

.filler-details summary {
    cursor: pointer;
    font-size: 12px;
    font-weight: 650;
    color: #6258d6;
    margin-top: 12px;
}

.filler-list { margin-top: 10px; }

.filler-instance {
    display: flex;
    justify-content: space-between;
    border-top: 1px solid #e7e7ef;
    padding: 8px 0;
    font-size: 12.5px;
}

.filler-word { font-weight: 650; color: #4a4960; }
.filler-time { color: #9999aa; font-size: 12px; }

.transcript-text-body {
    color: #1d1d32;
    font-size: 17px;
    line-height: 1.7;
    max-height: 320px;
    overflow-y: auto;
    padding-right: 5px;
}

.transcript-empty {
    color: #9999aa;
    font-size: 14px;
    line-height: 1.6;
}

.filler-highlight {
    background: #f1dfb0;
    color: #1d1d32;
    padding: 1px 4px;
    border-radius: 5px;
}

.grammar-highlight {
    background: #f3d5d2;
    color: #7a2b2b;
    padding: 1px 4px;
    border-radius: 5px;
}

.transcript-legend {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 14px;
    margin-top: 14px;
    font-size: 12px;
    color: #9999aa;
}

.transcript-legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
}

.filler-highlight-swatch,
.grammar-highlight-swatch {
    width: 11px;
    height: 11px;
    border-radius: 3px;
    display: inline-block;
}

.filler-highlight-swatch  { background: #f1dfb0; }
.grammar-highlight-swatch { background: #f3d5d2; }

/* Grammar card */

.grammar-card {
    background: white;
    border: 1px solid #e7e7ef;
    border-radius: 22px;
    padding: 26px 28px;
    box-shadow: 0 5px 18px rgba(25, 25, 55, 0.05);
    box-sizing: border-box;
}

.grammar-head {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 20px;
}

.grammar-title {
    font-size: 18px;
    font-weight: 750;
    color: #211d3a;
    line-height: 1.3;
}

.grammar-subtitle {
    font-size: 14px;
    color: #9a9aa8;
    margin-top: 3px;
}

.grammar-count-badge {
    background: #f7e4e2;
    color: #8b3a35;
    font-size: 13px;
    font-weight: 750;
    padding: 7px 16px;
    border-radius: 999px;
    flex-shrink: 0;
    white-space: nowrap;
}

.grammar-overall-banner {
    background: #fbf4dc;
    color: #6f5b16;
    border-radius: 14px;
    padding: 14px 18px;
    font-size: 14.5px;
    margin-bottom: 14px;
}

.grammar-overall-banner strong { color: #1d1d1d; }

.grammar-empty {
    color: #77778c;
    font-size: 14px;
    line-height: 1.6;
}

.grammar-error {
    background: #fdf8f8;
    border: 1px solid #edd6d9;
    border-radius: 18px;
    padding: 18px 22px;
    margin-bottom: 14px;
}

.grammar-error-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 14px;
}

.grammar-error-index {
    font-size: 14.5px;
    font-weight: 750;
    color: #8b3a44;
}

.grammar-error-badge {
    background: #f7e4e2;
    color: #8b3a44;
    font-size: 12px;
    font-weight: 750;
    letter-spacing: 0.3px;
    padding: 5px 13px;
    border-radius: 999px;
}

.grammar-error-cols {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 24px;
}

.grammar-col-label {
    font-size: 12px;
    color: #9a9aa8;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    margin-bottom: 5px;
}

.grammar-wrong {
    font-size: 17px;
    color: #a3413c;
    text-decoration: line-through;
}

.grammar-right {
    font-size: 17px;
    font-weight: 650;
    color: #4b7c4b;
}

.grammar-explanation {
    margin-top: 14px;
    font-size: 14px;
    line-height: 1.6;
    color: #5c5c70;
}

.grammar-footer-note {
    margin-top: 20px;
    padding-top: 16px;
    border-top: 1px solid #ececf1;
    font-size: 12px;
    line-height: 1.6;
    color: #a0a0b2;
}

.grammar-wrong,
.grammar-right,
.grammar-explanation {
    overflow-wrap: break-word;
    word-break: break-word;
}

/* Coaching tab */

.coaching-columns {
    display: grid;
    grid-template-columns: 1fr;
    gap: 18px;
}

.report-section-title {
    font-size: 13px;
    font-weight: 750;
    margin-bottom: 10px;
}

.report-section-title.well        { color: #27804c; }
.report-section-title.feedback    { color: #9b6c10; }
.report-section-title.improvement { color: #6258d6; }

.report-bullet {
    font-size: 13.5px;
    line-height: 1.6;
    color: #45455b;
    padding: 9px 12px;
    border-radius: 10px;
    margin-bottom: 6px;
    overflow-wrap: break-word;
    word-break: break-word;
}

.report-bullet.well        { background: #f2faf5; }
.report-bullet.feedback    { background: #fffaf0; }
.report-bullet.improvement { background: #f7f6ff; }

.report-empty-row,
.coaching-empty {
    font-size: 14px;
    color: #9999aa;
}

.encouragement-standalone {
    text-align: center;
    font-size: 18px;
    font-style: italic;
    color: #292844;
    padding: 18px 16px 2px;
}

/* How-to-read-results guide */

.nav-guide {
    background: white;
    border: 1px solid #e7e7ef;
    border-radius: 18px;
    padding: 18px 22px;
    margin-bottom: 14px;
    box-shadow: 0 5px 18px rgba(25, 25, 55, 0.05);
}

.nav-guide-title { font-size: 16px; font-weight: 750; color: #211d3a; }
.nav-guide-sub   { font-size: 13.5px; color: #77778c; margin: 3px 0 14px 0; }

.nav-guide-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
}

.nav-guide-item { background: #f7f6ff; border-radius: 12px; padding: 12px 14px; }
.nav-guide-tab  { font-size: 13px; font-weight: 700; color: #4b43cf; margin-bottom: 4px; }
.nav-guide-text { font-size: 13px; line-height: 1.5; color: #55556b; }

.nav-guide-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    margin-top: 14px;
    padding-top: 12px;
    border-top: 1px solid #ececf1;
    font-size: 12.5px;
    color: #77778c;
}

.legend-item { display: flex; align-items: center; gap: 6px; }
.legend-dot  { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.legend-dot.good      { background: #8cc9a0; }
.legend-dot.moderate  { background: #e8c766; }
.legend-dot.attention { background: #e59a96; }

.nav-guide-grid.two { grid-template-columns: repeat(2, minmax(0, 1fr)); }

.nav-guide-tip {
    margin-top: 12px;
    padding-top: 10px;
    border-top: 1px solid #ececf1;
    font-size: 12.5px;
    line-height: 1.5;
    color: #77778c;
}

/* Bottom action bar */

.action-bar {
    align-items: center !important;
    gap: 12px !important;
    margin-top: 18px;
    padding: 16px 20px;
    background: white;
    border: 1px solid #e7e7ef;
    border-radius: 18px;
    box-sizing: border-box;
}

.weakspot-text {
    font-size: 14.5px;
    color: #66667a;
}

/* ============================================================
   RESPONSIVE DESIGN
   ============================================================ */

@media (max-width: 900px) {
    .gradio-container {
        max-width: 100% !important;
        width: 100% !important;
        padding: 16px 16px 36px !important;
    }

    .app-header { padding: 26px 26px; }
    .app-header h1 { font-size: 24px; line-height: 1.2; }
}

@media (max-width: 760px) {
    .coaching-columns { grid-template-columns: 1fr; }
    .nav-guide-grid { grid-template-columns: 1fr; }
    .nav-guide-grid.two { grid-template-columns: 1fr; }

    .scores-overview-row {
        grid-template-columns: 1fr 1fr;
        column-gap: 20px;
    }

    .score-bars { grid-column: 1 / -1; }
}

@media (max-width: 600px) {
    html, body { max-width: 100% !important; overflow-x: hidden !important; }

    .gradio-container { padding: 12px 12px 28px !important; }

    .app-header { padding: 22px 18px; border-radius: 18px; }
    .app-header h1 { font-size: 21px; }
    .app-header p { font-size: 14px; line-height: 1.5; }

    .section-title { font-size: 17px; }
    .section-subtitle { font-size: 13px; }

    .card-col { padding: 15px; border-radius: 16px; }

    .results-header { padding: 18px; border-radius: 16px; }
    .results-header h2 { font-size: 20px; }
    .results-header p { font-size: 13px; }

    #upload-box video:not(.hide) {
        height: 260px !important;
        max-height: 260px !important;
    }

    .result-top-card .row { flex-wrap: wrap !important; }
    .pdf-btn-top { max-width: 100%; }

    .question-top-text { font-size: 18px; }

    .overview-grid.double { grid-template-columns: 1fr; }

    .grammar-error-cols { grid-template-columns: 1fr; gap: 12px; }
    .grammar-card { padding: 20px 18px; }

    #results-subtabs [role="tablist"],
    #results-subtabs .tab-nav { gap: 16px !important; }

    #results-subtabs [role="tabpanel"],
    #results-subtabs .tabitem { padding: 14px !important; }

    .action-bar { flex-direction: column !important; align-items: stretch !important; padding: 14px; }
    .action-bar > * { width: 100% !important; max-width: 100% !important; min-width: 0 !important; }

    .analysis-status { padding: 14px 16px; }
    .status-title { font-size: 14px; }
    .status-text { font-size: 12px; line-height: 1.5; }

    img, video, canvas, svg { max-width: 100% !important; }
}

@media (max-width: 420px) {
    .score-donut { width: 108px; height: 108px; }
    .score-donut-inner { width: 82px; height: 82px; }
    .score-donut-number { font-size: 30px; }
}
"""

# ============================================================
# SCROLL HELPERS (JavaScript)
# scrollWhenReady() waits until a marker element on the
# new page is actually visible, then scrolls after a short settle time
# (so the reset that happens at the same time doesn't shift the layout
# under the scroll).
# ============================================================

APP_JS = """
function scrollWhenReady(waitId, targetId, block, settleMs) {
    const started = Date.now();
    const timer = setInterval(function () {
        const waitEl = document.getElementById(waitId);
        const ready = waitEl && waitEl.offsetParent !== null;
        if (ready || Date.now() - started > 4000) {
            clearInterval(timer);
            setTimeout(function () {
                const target = document.getElementById(targetId);
                if (target) {
                    target.scrollIntoView({
                        behavior: "smooth",
                        block: block
                    });
                }
            }, settleMs);
        }
    }, 50);
}

function scrollToQuestionStatus() {
    const target = document.getElementById("question-status-anchor");
    if (target) {
        target.scrollIntoView({ behavior: "smooth", block: "start" });
    }
}

function scrollToResultsTop()          { scrollWhenReady("results-top", "results-top", "start", 150); }
function scrollToTopLater()            { scrollWhenReady("practice-setup-anchor", "app-top", "start", 200); }
function scrollToUploadBox()           { scrollWhenReady("practice-setup-anchor", "upload-box", "center", 350); }
function scrollToQuestionStatusLater() { scrollWhenReady("practice-setup-anchor", "question-status-anchor", "start", 100); }
"""

# ============================================================
# BUILD GRADIO APP
#   setup_page   - Step 01, question, Step 02, Analyse
#   results_page - question + scores, tabs, action bar
# ============================================================

with gr.Blocks(
    title="AI Speaking Skills Tutor"
) as app:

    # ========================================================
    # PAGE 1 - SETUP
    # ========================================================

    with gr.Column(visible=True) as setup_page:
        gr.HTML("""
            <div class="app-header" id="app-top">
                <h1> AI Speaking Skills Tutor</h1>
                <p>
                    Practice your spoken English, analyse your performance,
                    and receive personalised AI coaching.
                </p>
            </div>
        """)

        # ----------------------------------------------------
        # STEP 1
        # ----------------------------------------------------

        gr.HTML("""
            <div id="practice-setup-anchor">
                <div class="step-badge">STEP 01</div>
                <div class="section-title">
                    Practice Setup
                </div>
                <div class="section-subtitle">
                    Type a topic of your choice and select a difficulty level to generate
                    your speaking question.
                </div>
            </div>
        """)

        with gr.Column(elem_classes="card-col", elem_id="setup-card"):
            with gr.Row():
                topic_input = gr.Textbox(
                    label="Topic",
                    placeholder="e.g. fashion, technology, education...",
                    elem_id="topic-input",
                    scale=3
                )
                difficulty_input = gr.Radio(
                    choices=[
                        "beginner",
                        "intermediate",
                        "advanced"
                    ],
                    value="intermediate",
                    label="Difficulty",
                    elem_id="difficulty-input",
                    scale=2
                )
            generate_btn = gr.Button(
                "Generate question",
                variant="primary",
                interactive=False,
                elem_classes=["primary-btn"]
            )

        # Live message for questions, follow-ups and "new question".
        gr.HTML('<div id="question-status-anchor"></div>')
        question_status = gr.HTML(
            visible=False
        )

        # ----------------------------------------------------
        # QUESTION
        # ----------------------------------------------------

        gr.HTML("""
            <div class="section-title tight-top">
                Your Speaking Question
            </div>
            <div class="section-subtitle">
                Take a moment to think about your answer before recording.
            </div>
        """)

        question_recap_output = gr.HTML(empty_question_recap_box())

        # Holds the current question text (used as state by the events).
        question_output = gr.Textbox(visible=False)

        # ----------------------------------------------------
        # STEP 2
        # ----------------------------------------------------

        gr.HTML("""
            <div class="step-badge">STEP 02</div>
            <div class="section-title">
                Submit Your Response
            </div>
            <div class="section-subtitle">
                Upload a video, or record yourself live using your webcam.
                Speak naturally and try not to read from a script.
            </div>
        """)

        gr.HTML("""
            <div class="nav-guide">
                <div class="nav-guide-title">How to submit your response</div>
                <div class="nav-guide-sub">Choose Upload or Webcam below the video box, then click inside the box.</div>
                <div class="nav-guide-grid two">
                    <div class="nav-guide-item">
                        <div class="nav-guide-tab">Upload</div>
                        <div class="nav-guide-text">
                            Click inside the box and choose a video from your computer.
                        </div>
                    </div>
                    <div class="nav-guide-item">
                        <div class="nav-guide-tab">Webcam</div>
                        <div class="nav-guide-text">
                            Click inside the box to turn on your camera and allow access
                            if your browser asks. Press the red button to start recording,
                            and press it again to stop.
                        </div>
                    </div>
                </div>

                <div class="nav-guide-tip">
                    Camera not starting? Click the lock icon in the address bar and
                    allow camera and microphone.
                </div>
            </div>
        """)

        with gr.Column(elem_classes="card-col"):
            with gr.Column(elem_id="upload-box"):
                video_input = gr.Video(
                    label="Upload your video answer",
                    show_label=False,
                    sources=["upload", "webcam"],
                    interactive=False
                )

            analyse_btn = gr.Button(
                "Analyse my answer",
                variant="primary",
                interactive=False,
                elem_classes=["primary-btn"]
            )
            gr.HTML('<div class="caption-text">This runs on your own computer, so it can take a while</div>')

        # ----------------------------------------------------
        # STATUS (shown as the full-screen overlay while analysing,
        # and as a plain message for errors / "try again")
        # ----------------------------------------------------

        status_output = gr.HTML(
            visible=False
        )

    # ========================================================
    # PAGE 2 - RESULTS
    # ========================================================

    with gr.Column(visible=False) as results_page:
        gr.HTML("""
            <div class="results-header" id="results-top">
                <h2> Speaking Performance Results </h2>
                <p>
                    Here's your AI-powered speaking performance analysis.
                </p>
            </div>
        """)

        with gr.Column(elem_classes=["card-col", "result-top-card"]):
            with gr.Row():
                with gr.Column(scale=4, min_width=220):
                    question_recap_output2 = gr.HTML(empty_question_card())
                with gr.Column(scale=1, min_width=150):
                    download_pdf_btn = gr.DownloadButton(
                        "Save PDF",
                        variant="secondary",
                        elem_classes=["secondary-btn", "pdf-btn-top"]
                    )
            scores_overview_output = gr.HTML(empty_scores_overview())
        gr.HTML("""
            <div class="nav-guide">
                <div class="nav-guide-title">How to read your results</div>
                <div class="nav-guide-sub">Use the tabs below to move between the sections.</div>

                <div class="nav-guide-grid">
                    <div class="nav-guide-item">
                        <div class="nav-guide-tab">Overview</div>
                        <div class="nav-guide-text">
                            Topic relevance, voice and delivery (audio, pace, pitch, pauses)
                            and body language (eye contact, head orientation).
                        </div>
                    </div>
                    <div class="nav-guide-item">
                        <div class="nav-guide-tab">Speech Analysis</div>
                        <div class="nav-guide-text">
                            Your filler words, a highlighted transcript of what you said,
                            and any grammar errors with corrections.
                        </div>
                    </div>
                    <div class="nav-guide-item">
                        <div class="nav-guide-tab">Coaching Report</div>
                        <div class="nav-guide-text">
                            Personalised feedback: what you did well, key feedback
                            and your top improvements.
                        </div>
                    </div>
                </div>

                <div class="nav-guide-legend">
                    <span class="legend-item"><span class="legend-dot good"></span> Going well</span>
                    <span class="legend-item"><span class="legend-dot moderate"></span> Could improve</span>
                    <span class="legend-item"><span class="legend-dot attention"></span> Needs attention</span>
                </div>
            </div>
        """)

        with gr.Tabs(elem_id="results-subtabs"):
            with gr.Tab("Overview"):
                session_output = gr.HTML(empty_session_details())
            with gr.Tab("Speech Analysis"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=220):
                        filler_output = gr.HTML(
                            format_fillers({"count": 0, "instances": []})
                        )
                    with gr.Column(scale=2, min_width=260):
                        transcript_output = gr.HTML(format_transcript(""))
                grammar_output = gr.HTML(empty_grammar_card())
            with gr.Tab("Coaching Report"):
                report_output = gr.HTML(format_report(""))
                encouragement_output = gr.HTML(empty_encouragement_card())

        # ----------------------------------------------------
        # ACTIONS
        # ----------------------------------------------------

        with gr.Row(elem_classes="action-bar"):
            with gr.Column(scale=1, min_width=200):
                weakspot_output = gr.HTML(empty_weakspot_text())
            try_again_btn = gr.Button(
                "Try Again - Same Question",
                variant="secondary",
                visible=False,
                scale=0,
                min_width=230,
                elem_classes="secondary-btn"
            )
            new_question_btn = gr.Button(
                "New Question",
                variant="secondary",
                visible=False,
                scale=0,
                min_width=140,
                elem_classes="secondary-btn"
            )
            followup_btn = gr.Button(
                "Practice Weak Spot",
                variant="primary",
                visible=False,
                scale=0,
                min_width=170,
                elem_classes=["primary-btn"]
            )

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    gr.HTML("""
        <div class="footer-text">
            AI Speaking Skills Tutor · Practice · Analyse · Improve
        </div>
    """)

    # ========================================================
    # BUTTON EVENTS
    # ========================================================

    # The same list is used by Try Again (as its outputs).
    TRY_AGAIN_OUTPUTS = [
        video_input,
        question_recap_output2,
        scores_overview_output,
        report_output,
        encouragement_output,
        transcript_output,
        filler_output,
        session_output,
        grammar_output,
        try_again_btn,
        new_question_btn,
        followup_btn,
        weakspot_output,
        status_output,
        question_status,
        question_recap_output,
        analyse_btn,
        topic_input,
        difficulty_input,
        generate_btn
    ]

    topic_input.change(
        fn=topic_changed,
        inputs=topic_input,
        outputs=generate_btn
    )

    video_input.change(
        fn=video_changed,
        inputs=video_input,
        outputs=[analyse_btn, topic_input, difficulty_input, generate_btn]
    )

    # ---- GENERATE QUESTION ---------------------------------

    generate_btn.click(
        fn=start_question_loading,
        inputs=None,
        outputs=[
            generate_btn,
            analyse_btn,
            question_status,
            topic_input,
            difficulty_input
        ],
        show_progress="hidden",
        js="scrollToQuestionStatus"
    ).then(
        fn=get_question,
        inputs=[
            topic_input,
            difficulty_input,
            video_input
        ],
        outputs=[
            question_output,
            generate_btn,
            analyse_btn,
            question_status,
            video_input,
            topic_input,
            difficulty_input
        ],
        show_progress="hidden"
    ).then(
        fn=make_question_recap_box,
        inputs=question_output,
        outputs=question_recap_output,
        show_progress="hidden"
    )

    # ---- ANALYSE ------------------------------------------
    # overlay + live status  ->  results page  ->  pan to top

    analyse_btn.click(
        fn=start_loading,
        inputs=None,
        outputs=[
            analyse_btn,
            status_output,
            topic_input,
            difficulty_input
        ],
        show_progress="hidden"
    ).then(
        fn=analyse_video,
        inputs=video_input,
        outputs=[
            question_recap_output2,
            scores_overview_output,
            report_output,
            encouragement_output,
            transcript_output,
            filler_output,
            session_output,
            grammar_output,
            try_again_btn,
            new_question_btn,
            followup_btn,
            weakspot_output,
            status_output,
            analyse_btn,
            topic_input,
            difficulty_input
        ],
        show_progress="hidden"
    ).then(
        fn=prepare_pdf,                    
        inputs=None,
        outputs=download_pdf_btn,
        show_progress="hidden"
    ).then(
        fn=go_to_results,
        inputs=None,
        outputs=[setup_page, results_page],
        js="scrollToResultsTop"
    )

    # ---- TRY AGAIN -----------------------------------------
    # setup page  ->  reset  ->  pan to the upload box

    try_again_btn.click(
        fn=go_to_setup,
        inputs=None,
        outputs=[setup_page, results_page],
        js="scrollToUploadBox"
    ).then(
        fn=try_again,
        inputs=None,
        outputs=TRY_AGAIN_OUTPUTS
    )

    # ---- FOLLOW-UP -----------------------------------------
    # setup page  ->  pan to the live message  ->  generate

    followup_btn.click(
        fn=go_to_setup,
        inputs=None,
        outputs=[setup_page, results_page],
        js="scrollToQuestionStatusLater"
    ).then(
        fn=generate_followup,
        inputs=None,
        outputs=[
            question_output,
            video_input,
            question_recap_output2,
            scores_overview_output,
            report_output,
            encouragement_output,
            transcript_output,
            filler_output,
            session_output,
            grammar_output,
            try_again_btn,
            new_question_btn,
            followup_btn,
            weakspot_output,
            question_status,
            question_recap_output,
            status_output,
            analyse_btn,
            topic_input,
            difficulty_input,
            generate_btn
        ],
        show_progress="hidden"
    )

    # ---- NEW QUESTION --------------------------------------
    # setup page  ->  reset everything  ->  scroll to very top

    new_question_btn.click(
        fn=go_to_setup,
        inputs=None,
        outputs=[setup_page, results_page],
        js="scrollToTopLater"
    ).then(
        fn=new_question,
        inputs=None,
        outputs=[
            topic_input,
            difficulty_input,
            question_output,
            video_input,
            question_recap_output2,
            scores_overview_output,
            report_output,
            encouragement_output,
            transcript_output,
            filler_output,
            session_output,
            grammar_output,
            generate_btn,
            analyse_btn,
            try_again_btn,
            new_question_btn,
            followup_btn,
            weakspot_output,
            status_output,
            question_status,
            question_recap_output
        ]
    )

# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":
    app.launch(inbrowser=True, css=CUSTOM_CSS, js=APP_JS)