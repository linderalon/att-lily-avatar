#!/usr/bin/env python3
"""
Local proxy server — replaces AWS Lambda for local development.

Accepts the same POST body the HR/Code Interview demos send, builds prompts,
calls LiteLLM (OpenAI-compatible), and returns the same { success, summary, usage }
response shape the demos expect.

Usage:
    LITELLM_API_KEY=your-key python3 local_proxy.py

Environment variables:
    LITELLM_API_KEY   Required. API key for LiteLLM.
    LITELLM_URL       LiteLLM base URL (default: http://localhost:10006/v1/chat/completions)
    MODEL             Model name (default: claude-4-6-sonnet)
    MAX_TOKENS        Max output tokens for full/HR mode (default: 2048)
    TEMPERATURE       Sampling temperature (default: 0.3)
    PORT              Port to listen on (default: 8090)
"""

import json
import os
import sys
import requests
from http.server import BaseHTTPRequestHandler, HTTPServer

# =============================================================================
# CONFIGURATION
# =============================================================================

LITELLM_URL  = os.environ.get('LITELLM_URL', 'http://localhost:10006/v1/chat/completions')
LITELLM_KEY  = os.environ.get('LITELLM_API_KEY', '')
MODEL        = os.environ.get('MODEL', 'claude-4-6-sonnet')
MAX_TOKENS   = int(os.environ.get('MAX_TOKENS', '2048'))
TEMPERATURE  = float(os.environ.get('TEMPERATURE', '0.3'))
PORT         = int(os.environ.get('PORT', '8090'))

CORS_HEADERS = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    'Content-Type': 'application/json',
}

# =============================================================================
# PROMPTS (identical to lambda_function.py)
# =============================================================================

PER_PROBLEM_SYSTEM_PROMPT = """Analyze ONE coding problem from an interview transcript. Focus ONLY on the specified problem.
Output ONLY valid JSON. No markdown. Be concise — all strings 1 sentence max.

OUTPUT JSON SHAPE:
{"problem_id":"str","problem_title":"str","difficulty":"easy|medium|hard","outcome":"solved|partial|stuck|skipped","tests_passed":int,"tests_total":int,"approach":"brute_force|hash_map|two_pointer|sorting|dynamic_programming|recursion|greedy|other|incomplete","approach_used":"str","time_complexity":"str","space_complexity":"str","optimal":bool,"time_spent_minutes":int,"hints_used":int,"scores":{"creativity":1-5,"logic":1-5,"code_quality":1-5,"explainability":1-5,"complexity":1-5,"scale":1-5},"eval_notes":"1-2 sentences"}"""

SYNTHESIS_SYSTEM_PROMPT = """Synthesize coding interview results into overall assessment. Output ONLY valid JSON. Be VERY concise — max 5 words per string field. fit.score_0_100=skill(60%)+potential(40%).

JSON:
{"overview":"20-30 words","skill_assessment":{"problem_solving":1-5,"problem_solving_e":"str","code_fluency":1-5,"code_fluency_e":"str","communication":1-5,"communication_e":"str","efficiency_awareness":1-5,"efficiency_awareness_e":"str"},"potential_assessment":{"creativity_score":1-5,"creativity_a":"str","tenacity_score":1-5,"tenacity_a":"str","aptitude_score":1-5,"aptitude_a":"str","propensity_score":1-5,"propensity_a":"str","talent_indicators":["max3"],"potential_vs_performance":"potential_exceeds|matches|performance_exceeds|insufficient","growth_trajectory":"high|moderate|limited|unknown"},"fit":{"score_0_100":num,"rec":"strong_yes|yes|lean_yes|lean_no|no","conf":"high|medium|low","rationale":"str"},"strengths":["max3"],"areas_for_improvement":["max3"],"cq":{"emo":"calm|confident|neutral|frustrated|stressed|positive|unknown","tone":"collaborative|independent|receptive|defensive|unknown","eng":"high|medium|low|unknown","think_aloud":bool},"risk":{"flags":["none"],"escalated":false,"reason":""},"next_steps":["max2"]}"""

HR_SYSTEM_PROMPT = """You are an expert HR analyst. Analyze HR call transcripts and produce structured JSON summaries.

CRITICAL RULES:
1. Output ONLY valid JSON - no markdown, no explanations, no code blocks
2. Follow the schema EXACTLY
3. Be evidence-based and concise
4. Never reference internal terms like "DPP" in output text

REQUIRED OUTPUT STRUCTURE:
{
  "v": "4.1",
  "mode": "<interview|post_interview|separation>",
  "ctx": {"org":"str","role":"str","role_id":"str","loc":"str","person":"str","subj_id":"str"},
  "dpp_digest": {"mins":int,"focus":["str"],"must":["str"],"nice":["str"],"cv_provided":bool,"role_id":"str","subj_id":"str"},
  "turns": int,
  "overview": "80-200 word summary",
  "key_answers": [{"id":"str","q":"str","a":"str","status":"answered|partially_answered|not_answered","strength":"strong|ok|weak|unknown"}],
  "fit": {"score_0_100":num,"rec":"strong_yes|yes|lean_yes|lean_no|no","conf":"high|medium|low","dims":[{"id":"str","score_1_5":1-5,"e":"str"}]},
  "star_analysis": null,
  "believability": {"score_0_100":num,"cv_consistency":"consistent|mixed|inconsistent|no_cv|unknown","mismatches":[],"signals":["str"],"notes":"str"},
  "gaps": [{"missing":"str","why_matters":"str","next_q":"str"}],
  "cq": {"emo":"str","tone":"str","eng":"str"},
  "risk": {"flags":["none"],"escalated":false,"reason":""},
  "next_steps": ["str"]
}

Return ONLY the JSON object."""

# =============================================================================
# LiteLLM CALL
# =============================================================================

def call_litellm(user_prompt, system_prompt, max_tokens=None):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {LITELLM_KEY}',
    }
    payload = {
        'model': MODEL,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user',   'content': user_prompt},
        ],
        'max_tokens': max_tokens or MAX_TOKENS,
        'temperature': TEMPERATURE,
    }

    resp = requests.post(LITELLM_URL, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    content = data['choices'][0]['message']['content'].strip()

    # Strip markdown code fences if model wrapped the JSON
    if content.startswith('```'):
        lines = content.split('\n')
        content = '\n'.join(lines[1:-1] if lines[-1].strip() == '```' else lines[1:])

    summary = json.loads(content)

    usage = {
        'input_tokens':  data.get('usage', {}).get('prompt_tokens', 0),
        'output_tokens': data.get('usage', {}).get('completion_tokens', 0),
    }
    return summary, usage

# =============================================================================
# REPORT ANALYSIS PROMPTS
# =============================================================================

KNOWLEDGE_CHECK_SYSTEM_PROMPT = """You are an AT&T sales training evaluator. Analyze a knowledge check session transcript and output ONLY a valid JSON object. No markdown, no explanation.

OUTPUT SCHEMA:
{
  "product": "<AT&T product assessed>",
  "overall_score": <0-100 integer>,
  "grade": "<A+|A|A-|B+|B|B-|C+|C|C-|D|F>",
  "summary": "<2-3 sentence overview of the session>",
  "strong_spots": ["<up to 4 specific strengths observed>"],
  "weak_spots": ["<up to 4 specific gaps or mistakes>"],
  "areas_to_improve": ["<up to 4 concrete, actionable improvement items>"],
  "study_suggestions": [
    {"topic": "<topic name>", "why": "<why this matters for selling>", "priority": "<high|medium|low>"}
  ],
  "question_breakdown": [
    {"question_summary": "<short label for the question>", "score": <1-5>, "quality": "<strong|adequate|weak>", "feedback": "<1 sentence specific feedback>"}
  ],
  "readiness": "<ready_to_sell|needs_review|not_ready>"
}"""

GENERAL_ANALYSIS_SYSTEM_PROMPT = """You are an AT&T sales training evaluator. Analyze a sales training conversation and output ONLY a valid JSON object. No markdown, no explanation.

OUTPUT SCHEMA:
{
  "session_type": "<short label for the type of session based on the transcript>",
  "overall_score": <0-100 integer>,
  "grade": "<A+|A|A-|B+|B|B-|C+|C|C-|D|F>",
  "summary": "<2-3 sentence overview of the session>",
  "strong_spots": ["<up to 4 specific strengths observed>"],
  "weak_spots": ["<up to 4 specific gaps or weaknesses>"],
  "areas_to_improve": ["<up to 4 concrete, actionable items>"],
  "study_suggestions": [
    {"topic": "<topic>", "why": "<why it matters>", "priority": "<high|medium|low>"}
  ],
  "engagement": "<high|medium|low>",
  "confidence": "<high|medium|low>"
}"""

# =============================================================================
# HANDLERS (same logic as lambda_function.py)
# =============================================================================

def format_transcript(transcript):
    lines = []
    for i, turn in enumerate(transcript, 1):
        role    = turn.get('role', 'unknown')
        content = turn.get('content', '')
        speaker = 'AI' if role == 'assistant' else 'Candidate'
        lines.append(f"[{i}] {speaker}: {content}")
    return '\n'.join(lines)


def handle_per_problem(body):
    transcript = body.get('transcript', [])
    problem    = body.get('problem_focus', {})
    dpp        = body.get('dpp', {})

    if not transcript:
        return error_body('Missing: transcript', 'VALIDATION_ERROR')
    if not problem.get('id'):
        return error_body('Missing: problem_focus.id', 'VALIDATION_ERROR')

    lang = 'python'
    live_code = dpp.get('live_code')
    if isinstance(live_code, dict):
        lang = live_code.get('language', 'python')

    user_prompt = (
        f"Analyze ONLY the problem \"{problem.get('title', problem['id'])}\" "
        f"(id: {problem['id']}, difficulty: {problem.get('difficulty', '?')}).\n\n"
        f"## Session Context\n"
        f"Language: {lang}\n"
        f"Session problems: {json.dumps(dpp.get('all_problems_in_session', []), separators=(',', ':'))}\n\n"
        f"## Transcript\n{format_transcript(transcript)}\n\n"
        f"Output the JSON for this ONE problem only."
    )

    result, usage = call_litellm(user_prompt, PER_PROBLEM_SYSTEM_PROMPT, max_tokens=512)
    return success_body(result, usage)


def handle_synthesis(body):
    problem_results = body.get('problem_results', [])
    dpp             = body.get('dpp', {})

    if not problem_results:
        return error_body('Missing: problem_results', 'VALIDATION_ERROR')

    candidate = dpp.get('candidate', {})
    name      = candidate.get('full_name', candidate.get('first_name', 'Candidate'))
    elapsed   = dpp.get('session', {}).get('elapsed_minutes', '?')
    total     = dpp.get('session', {}).get('total_problems', len(problem_results))

    user_prompt = (
        f"Candidate: {name}\n"
        f"Session: {elapsed} minutes, {len(problem_results)} of {total} problems attempted.\n"
        f"Hints given: {dpp.get('session', {}).get('hints_given', 0)}\n\n"
        f"## Per-Problem Results\n"
        f"```json\n{json.dumps(problem_results, separators=(',', ':'))}\n```\n\n"
        f"Synthesize these results into one overall assessment JSON."
    )

    result, usage = call_litellm(user_prompt, SYNTHESIS_SYSTEM_PROMPT, max_tokens=512)
    return success_body(result, usage)


def handle_full(body):
    transcript    = body.get('transcript', [])
    dpp           = body.get('dpp', {})
    schema        = body.get('schema')
    custom_prompt = body.get('summary_prompt')

    if not transcript:
        return error_body('Missing: transcript', 'VALIDATION_ERROR')
    if not dpp:
        return error_body('Missing: dpp', 'VALIDATION_ERROR')

    transcript_text = format_transcript(transcript)
    turn_count      = len([t for t in transcript if t.get('role') == 'user'])
    dpp_clean       = {k: v for k, v in dpp.items() if k != 'summary_prompt'}

    parts = [
        "Analyze this session and produce a JSON summary.\n",
        f"## Session Mode\n{dpp_clean.get('mode', 'interview')}\n",
        f"## Turn Count\n{turn_count} user turns\n",
        f"## DPP\n```json\n{json.dumps(dpp_clean, separators=(',', ':'))}\n```\n",
        f"## Transcript\n{transcript_text}\n",
    ]
    if schema:
        parts.append(f"## Schema\n```json\n{json.dumps(schema, separators=(',', ':'))}\n```\n")
    parts.append(
        "\n## Instructions\n"
        "Follow the system prompt schema exactly.\n"
        "Output ONLY the JSON object, no other text."
    )

    user_prompt = '\n'.join(parts)
    system      = custom_prompt or HR_SYSTEM_PROMPT
    summary, usage = call_litellm(user_prompt, system)

    final_code = dpp.get('final_code') or (
        dpp.get('live_code', {}).get('current_code', '') if isinstance(dpp.get('live_code'), dict) else ''
    )
    if final_code and 'final_code' not in summary:
        summary['final_code'] = final_code

    return success_body(summary, usage)


def handle_knowledge_check(body):
    """Analyze a knowledge check session and produce a graded report."""
    transcript = body.get('transcript', [])
    product    = body.get('product', 'AT&T Product')
    questions  = body.get('questions', [])

    if not transcript:
        return error_body('Missing: transcript', 'VALIDATION_ERROR')

    q_block = '\n'.join(f'{i+1}. {q}' for i, q in enumerate(questions)) if questions else 'Not provided'
    user_prompt = (
        f"Product assessed: {product}\n\n"
        f"Questions asked during the session:\n{q_block}\n\n"
        f"## Transcript\n{format_transcript(transcript)}\n\n"
        f"Analyze this knowledge check and output the JSON report."
    )

    result, usage = call_litellm(user_prompt, KNOWLEDGE_CHECK_SYSTEM_PROMPT, max_tokens=1500)
    return success_body(result, usage)


def handle_general(body):
    """Analyze a general sales training conversation and produce a report."""
    transcript = body.get('transcript', [])
    context    = body.get('context', '')

    if not transcript:
        return error_body('Missing: transcript', 'VALIDATION_ERROR')

    user_prompt = (
        (f"Session context: {context}\n\n" if context else '') +
        f"## Transcript\n{format_transcript(transcript)}\n\n"
        f"Analyze this sales training session and output the JSON report."
    )

    result, usage = call_litellm(user_prompt, GENERAL_ANALYSIS_SYSTEM_PROMPT, max_tokens=1200)
    return success_body(result, usage)


# =============================================================================
# RESPONSE HELPERS
# =============================================================================

def success_body(data, usage):
    return (200, json.dumps({'success': True, 'summary': data, 'usage': usage}))

def error_body(message, code, status=400):
    return (status, json.dumps({'success': False, 'error': message, 'code': code}))

# =============================================================================
# HTTP SERVER
# =============================================================================

class ProxyHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f'[proxy] {fmt % args}', file=sys.stderr)

    def _send(self, status, body):
        encoded = body.encode()
        self.send_response(status)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.send_header('Content-Length', str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_OPTIONS(self):
        self._send(200, '')

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        raw    = self.rfile.read(length)

        try:
            body = json.loads(raw)
        except json.JSONDecodeError as e:
            self._send(*error_body(f'Invalid JSON: {e}', 'VALIDATION_ERROR'))
            return

        mode = body.get('analysis_mode')
        try:
            if mode == 'knowledge_check':
                status, resp = handle_knowledge_check(body)
            elif mode == 'general':
                status, resp = handle_general(body)
            elif mode == 'per_problem':
                status, resp = handle_per_problem(body)
            elif mode == 'synthesis':
                status, resp = handle_synthesis(body)
            else:
                status, resp = handle_full(body)
        except requests.HTTPError as e:
            self._send(*error_body(f'LiteLLM error: {e}', 'LITELLM_ERROR', 502))
            return
        except json.JSONDecodeError as e:
            self._send(*error_body(f'LLM returned invalid JSON: {e}', 'PARSE_ERROR', 502))
            return
        except Exception as e:
            print(f'[proxy] error: {e}', file=sys.stderr)
            self._send(*error_body(f'Proxy error: {e}', 'PROXY_ERROR', 500))
            return

        self._send(status, resp)


if __name__ == '__main__':
    if not LITELLM_KEY:
        print('Warning: LITELLM_API_KEY is not set', file=sys.stderr)

    server = HTTPServer(('localhost', PORT), ProxyHandler)
    print(f'Local proxy running on http://localhost:{PORT}')
    print(f'  → LiteLLM: {LITELLM_URL}')
    print(f'  → Model:   {MODEL}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')
