# gemini_integration.py
# Drop-in replacement for claude_integration.py — uses Gemini 1.5 Flash (free tier).
# Same function signatures, same return types. No changes needed in notebook 04.
#
# Setup (one-time):
#   1. Get free API key at https://aistudio.google.com
#   2. pip install google-generativeai
#   3. Set GEMINI_API_KEY in your environment or paste directly below
#
# Free tier limits: 15 requests/min, 1M tokens/day — sufficient for full dataset runs.

import os
import json
import re
import time
from dataclasses import dataclass
from typing import Optional

import google.generativeai as genai

# ── API setup ─────────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")  # or paste key as string here
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")


# ─────────────────────────────────────────────
# DATA CLASSES (identical to claude_integration.py)
# ─────────────────────────────────────────────

@dataclass
class CulturalContext:
    distress_frame: str           # academic | somatic | relational | direct | mixed
    cultural_markers: list        # e.g. ["exam failure framing", "filial guilt"]
    severity_indicators: list
    language_mix: str             # zh | en | mixed
    raw: dict                     # full parsed JSON from Gemini

@dataclass
class CounsellorReport:
    risk_score: int
    risk_tier: str                # low | medium | high
    primary_drivers: list
    cultural_summary: str
    recommended_action: str
    urgency_window: str
    escalation_flag: bool
    full_report: str


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _clean_json(raw_text: str) -> str:
    """Strip markdown fences Gemini sometimes adds around JSON."""
    raw_text = re.sub(r"^```json\s*", "", raw_text.strip(), flags=re.MULTILINE)
    raw_text = re.sub(r"```$", "", raw_text.strip(), flags=re.MULTILINE)
    return raw_text.strip()


def _call_gemini(prompt: str, max_retries: int = 3) -> str:
    """
    Call Gemini with simple retry logic for rate-limit errors.
    Free tier allows 15 requests/min — adds a short delay between calls.
    """
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            time.sleep(1.5)  # stay comfortably within 15 req/min
            return response.text
        except Exception as e:
            if attempt < max_retries - 1:
                wait = (attempt + 1) * 10
                print(f"[gemini_integration] Retry {attempt + 1} after {wait}s — {e}")
                time.sleep(wait)
            else:
                raise


# ─────────────────────────────────────────────
# TIER 1→2 BRIDGE: CULTURAL CONTEXT ENRICHMENT
# ─────────────────────────────────────────────

CULTURAL_CONTEXT_PROMPT = """You are a clinical NLP assistant specialising in Asian adolescent mental health.
Analyse the social media post or journal entry below for distress expression patterns.

Return ONLY valid JSON — no preamble, no markdown fences — with this exact schema:
{{
  "distress_frame": "<academic|somatic|relational|direct|mixed>",
  "cultural_markers": ["<marker1>", "<marker2>"],
  "severity_indicators": ["<indicator1>", "<indicator2>"],
  "language_mix": "<zh|en|mixed>",
  "code_switch_patterns": ["<example if mixed, else empty list>"],
  "indirect_expression_score": <0.0 to 1.0, where 1.0 = fully indirect>
}}

Definitions:
- academic: distress framed through exam failure, grades, academic pressure
- somatic: headaches, fatigue, sleep issues, physical complaints
- relational: family shame, disappointing parents, peer rejection
- direct: explicit emotional disclosure ("I feel sad/hopeless")
- mixed: two or more frames present

Cultural markers to watch for (Chinese examples → frame):
- 考试 / 成绩 / 失败 (exam/grades/failure) → academic
- 头痛 / 睡不着 / 好累 (headache/can't sleep/exhausted) → somatic
- 让爸妈失望 / 丢脸 (disappointing parents/losing face) → relational
- 不想说话 / 消失 / 没意思 (don't want to talk/disappear/pointless) → severity indicators

Post to analyse:
{post_text}"""


def get_cultural_context(post_text: str) -> CulturalContext:
    """
    Enrich a MentalBERT-flagged post with cultural distress framing.
    Call this AFTER Tier 1 flags a post as distress-positive.

    Args:
        post_text: raw social media post or journal entry (zh / en / mixed)
    Returns:
        CulturalContext dataclass
    """
    prompt = CULTURAL_CONTEXT_PROMPT.format(post_text=post_text)
    raw_text = _call_gemini(prompt)
    parsed = json.loads(_clean_json(raw_text))

    return CulturalContext(
        distress_frame=parsed.get("distress_frame", "mixed"),
        cultural_markers=parsed.get("cultural_markers", []),
        severity_indicators=parsed.get("severity_indicators", []),
        language_mix=parsed.get("language_mix", "mixed"),
        raw=parsed
    )


def build_xgboost_features(context: CulturalContext) -> dict:
    """
    Convert CulturalContext into numeric/binary features for XGBoost.
    Merge the returned dict with your existing temporal + linguistic features.

    Returns:
        Feature dict ready to merge into your XGBoost input DataFrame row.
    """
    frames = ["academic", "somatic", "relational", "direct", "mixed"]
    frame_ohe = {f"frame_{f}": int(context.distress_frame == f) for f in frames}

    return {
        **frame_ohe,
        "cultural_marker_count": len(context.cultural_markers),
        "severity_indicator_count": len(context.severity_indicators),
        "indirect_expression_score": context.raw.get("indirect_expression_score", 0.5),
        "is_code_switching": int(context.language_mix == "mixed"),
        "is_zh_only": int(context.language_mix == "zh"),
        "has_somatic_markers": int(
            any(m in context.cultural_markers for m in
                ["somatic complaint", "fatigue", "sleep complaint", "headache"])
        ),
        "has_relational_markers": int(
            any(m in context.cultural_markers for m in
                ["filial guilt", "face-saving", "family shame", "disappointing parents"])
        ),
    }


# ─────────────────────────────────────────────
# TIER 2 → COUNSELLOR REPORT
# ─────────────────────────────────────────────

COUNSELLOR_REPORT_PROMPT = """You are a clinical communication specialist generating risk summaries
for school counsellors in Singapore and APAC.

Your audience has NO data science background. They need to know:
1. How worried to be
2. Why (plain language — no ML jargon, no feature names, no scores)
3. What to do next and when

Rules:
- Never use terms like SHAP, XGBoost, features, model, probability, token
- Never diagnose — use "indicators suggest" or "pattern consistent with"
- Always include a concrete next action with a time window
- If escalation_flag is true, lead with it in the first sentence
- Translate Chinese examples into English in parentheses
- Maximum 200 words for full_report
- Return ONLY valid JSON — no preamble, no markdown fences:
{{
  "risk_tier": "<low|medium|high>",
  "primary_drivers": ["<plain English driver 1>", "<driver 2>", "<driver 3>"],
  "cultural_summary": "<1-2 sentences on how distress is being expressed>",
  "recommended_action": "<concrete next step>",
  "urgency_window": "<within 24 hours | within 48 hours | within 1 week>",
  "escalation_flag": <true|false>,
  "full_report": "<complete formatted report for the counsellor>"
}}

Student data:
Student ID: {user_id}
Risk score: {risk_score}/100
Temporal trend: {temporal_trend}

Top risk drivers (internal — do not expose to student):
{shap_summary}

Cultural expression pattern:
- Distress frame: {distress_frame}
- Cultural markers observed: {cultural_markers}
- Severity indicators: {severity_indicators}
- Language: {language_mix}
- Indirect expression score: {indirect_score}

Generate the counsellor report now."""


def generate_counsellor_report(
    risk_score: int,
    shap_values: dict,
    cultural_context: CulturalContext,
    temporal_trend: Optional[str] = None,
    user_id: str = "anonymous"
) -> CounsellorReport:
    """
    Generate a plain-language counsellor report from XGBoost output + SHAP + context.

    Args:
        risk_score:       XGBoost risk score 0-100
        shap_values:      dict of {feature_name: shap_value} from your SHAP explainer
        cultural_context: CulturalContext from get_cultural_context()
        temporal_trend:   optional string from LSTM e.g. "escalating over 7 days"
        user_id:          anonymised student ID for the report header
    Returns:
        CounsellorReport dataclass
    """
    top_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    shap_summary = "\n".join(
        f"  - {feat}: {'↑ risk' if val > 0 else '↓ risk'} (magnitude {abs(val):.3f})"
        for feat, val in top_features
    )

    prompt = COUNSELLOR_REPORT_PROMPT.format(
        user_id=user_id,
        risk_score=risk_score,
        temporal_trend=temporal_trend or "single observation",
        shap_summary=shap_summary,
        distress_frame=cultural_context.distress_frame,
        cultural_markers=", ".join(cultural_context.cultural_markers) or "none",
        severity_indicators=", ".join(cultural_context.severity_indicators) or "none",
        language_mix=cultural_context.language_mix,
        indirect_score=cultural_context.raw.get("indirect_expression_score", "N/A")
    )

    raw_text = _call_gemini(prompt)
    parsed = json.loads(_clean_json(raw_text))

    return CounsellorReport(
        risk_score=risk_score,
        risk_tier=parsed["risk_tier"],
        primary_drivers=parsed["primary_drivers"],
        cultural_summary=parsed["cultural_summary"],
        recommended_action=parsed["recommended_action"],
        urgency_window=parsed["urgency_window"],
        escalation_flag=parsed["escalation_flag"],
        full_report=parsed["full_report"]
    )


# ─────────────────────────────────────────────
# BATCH RUNNER (bonus — processes full dataset efficiently)
# ─────────────────────────────────────────────

def batch_process_flagged_posts(
    flagged_df,
    text_column: str = "post_text",
    id_column: str = "user_id",
    risk_score_column: str = "xgb_risk_score",
    shap_columns: Optional[list] = None
) -> list:
    """
    Process a full DataFrame of MentalBERT-flagged posts through the pipeline.
    Respects Gemini free tier rate limits automatically.

    Args:
        flagged_df:        pandas DataFrame of posts flagged by MentalBERT
        text_column:       column name containing post text
        id_column:         column name containing anonymised student ID
        risk_score_column: column name containing XGBoost risk score
        shap_columns:      list of column names that are SHAP values (optional)
    Returns:
        List of CounsellorReport objects
    """
    reports = []

    for i, row in flagged_df.iterrows():
        print(f"Processing {i+1}/{len(flagged_df)}: {row[id_column]}")

        try:
            ctx = get_cultural_context(row[text_column])

            shap_vals = {}
            if shap_columns:
                shap_vals = {col: row[col] for col in shap_columns if col in row}

            report = generate_counsellor_report(
                risk_score=int(row[risk_score_column]),
                shap_values=shap_vals,
                cultural_context=ctx,
                user_id=str(row[id_column])
            )
            reports.append(report)

        except Exception as e:
            print(f"  [!] Failed for {row[id_column]}: {e}")
            reports.append(None)

    return reports
