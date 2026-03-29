"""
Avatar Analysis Lambda Function

Shared Lambda serving both the HR Avatar and Code Interview demos.
Analyzes call/session transcripts using AWS Bedrock (Claude) and returns
structured JSON summaries.

Analysis modes (selected by the `analysis_mode` request field):
  - "per_problem": Analyze a single coding problem from a transcript (~5s, max_tokens=512)
  - "synthesis":   Synthesize per-problem results into an overall assessment (~8s, max_tokens=512)
  - (default):     Full single-call HR analysis using HR_SYSTEM_PROMPT (v4.1 schema, max_tokens from env)

The Code Interview client fires N parallel per_problem calls then one synthesis call.
The HR Avatar client sends a single request with no analysis_mode (hits the default path).

Environment Variables:
    MODEL_ID:    Bedrock model ID (default: claude-3-haiku)
    MAX_TOKENS:  Max output tokens for default/full mode (default: 2048)
    TEMPERATURE: Model temperature (default: 0.3)
"""

import json
import os
import boto3
from botocore.config import Config

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_ID = os.environ.get('MODEL_ID', 'anthropic.claude-3-haiku-20240307-v1:0')
MAX_TOKENS = int(os.environ.get('MAX_TOKENS', '2048'))
TEMPERATURE = float(os.environ.get('TEMPERATURE', '0.3'))

bedrock_config = Config(
    retries={'max_attempts': 3, 'mode': 'adaptive'},
    read_timeout=60,
    connect_timeout=10
)

bedrock = boto3.client('bedrock-runtime', config=bedrock_config)

# =============================================================================
# CORS HEADERS
# =============================================================================

CORS_HEADERS = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Content-Type': 'application/json'
}

# =============================================================================
# CODE INTERVIEW PROMPTS (per-problem + synthesis)
# =============================================================================

PER_PROBLEM_SYSTEM_PROMPT = """Analyze ONE coding problem from an interview transcript. Focus ONLY on the specified problem.
Output ONLY valid JSON. No markdown. Be concise — all strings 1 sentence max.

OUTPUT JSON SHAPE:
{"problem_id":"str","problem_title":"str","difficulty":"easy|medium|hard","outcome":"solved|partial|stuck|skipped","tests_passed":int,"tests_total":int,"approach":"brute_force|hash_map|two_pointer|sorting|dynamic_programming|recursion|greedy|other|incomplete","approach_used":"str","time_complexity":"str","space_complexity":"str","optimal":bool,"time_spent_minutes":int,"hints_used":int,"scores":{"creativity":1-5,"logic":1-5,"code_quality":1-5,"explainability":1-5,"complexity":1-5,"scale":1-5},"eval_notes":"1-2 sentences"}"""

SYNTHESIS_SYSTEM_PROMPT = """Synthesize coding interview results into overall assessment. Output ONLY valid JSON. Be VERY concise — max 5 words per string field. fit.score_0_100=skill(60%)+potential(40%).

JSON:
{"overview":"20-30 words","skill_assessment":{"problem_solving":1-5,"problem_solving_e":"str","code_fluency":1-5,"code_fluency_e":"str","communication":1-5,"communication_e":"str","efficiency_awareness":1-5,"efficiency_awareness_e":"str"},"potential_assessment":{"creativity_score":1-5,"creativity_a":"str","tenacity_score":1-5,"tenacity_a":"str","aptitude_score":1-5,"aptitude_a":"str","propensity_score":1-5,"propensity_a":"str","talent_indicators":["max3"],"potential_vs_performance":"potential_exceeds|matches|performance_exceeds|insufficient","growth_trajectory":"high|moderate|limited|unknown"},"fit":{"score_0_100":num,"rec":"strong_yes|yes|lean_yes|lean_no|no","conf":"high|medium|low","rationale":"str"},"strengths":["max3"],"areas_for_improvement":["max3"],"cq":{"emo":"calm|confident|neutral|frustrated|stressed|positive|unknown","tone":"collaborative|independent|receptive|defensive|unknown","eng":"high|medium|low|unknown","think_aloud":bool},"risk":{"flags":["none"],"escalated":false,"reason":""},"next_steps":["max2"]}"""

# =============================================================================
# HR DEMO SYSTEM PROMPT (default / full mode)
# =============================================================================

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
# LAMBDA HANDLER
# =============================================================================

def lambda_handler(event, context):
    # Handle CORS preflight
    if event.get('requestContext', {}).get('http', {}).get('method') == 'OPTIONS':
        return {'statusCode': 200, 'headers': CORS_HEADERS, 'body': ''}

    try:
        body = event.get('body', '{}')
        if isinstance(body, str):
            body = json.loads(body)

        mode = body.get('analysis_mode')

        if mode == 'per_problem':
            return handle_per_problem(body)
        elif mode == 'synthesis':
            return handle_synthesis(body)
        else:
            return handle_full(body)

    except json.JSONDecodeError as e:
        return error_response(f'Invalid JSON: {str(e)}', 'VALIDATION_ERROR')
    except bedrock.exceptions.ThrottlingException:
        return error_response('Service busy, please retry', 'THROTTLING', 429)
    except bedrock.exceptions.ModelTimeoutException:
        return error_response('Analysis took too long, please retry', 'TIMEOUT', 504)
    except Exception as e:
        print(f'Error: {str(e)}')
        return error_response(f'Analysis failed: {str(e)}', 'BEDROCK_ERROR', 500)


def handle_per_problem(body):
    """Analyze a single problem from the transcript."""
    transcript = body.get('transcript', [])
    problem = body.get('problem_focus', {})
    dpp = body.get('dpp', {})

    if not transcript:
        return error_response('Missing: transcript', 'VALIDATION_ERROR')
    if not problem.get('id'):
        return error_response('Missing: problem_focus.id', 'VALIDATION_ERROR')

    transcript_text = format_transcript(transcript)

    user_prompt = (
        f"Analyze ONLY the problem \"{problem.get('title', problem['id'])}\" "
        f"(id: {problem['id']}, difficulty: {problem.get('difficulty', '?')}).\n\n"
        f"## Session Context\n"
        f"Language: {dpp.get('live_code', dpp.get('session', {})).get('language', 'python') if isinstance(dpp.get('live_code'), dict) else 'python'}\n"
        f"Session problems: {json.dumps(dpp.get('all_problems_in_session', []), separators=(',', ':'))}\n\n"
        f"## Transcript\n{transcript_text}\n\n"
        f"Output the JSON for this ONE problem only."
    )

    result, usage = call_bedrock(user_prompt, PER_PROBLEM_SYSTEM_PROMPT, max_tokens=512)

    return success_response(result, usage)


def handle_synthesis(body):
    """Synthesize per-problem results into an overall assessment."""
    problem_results = body.get('problem_results', [])
    dpp = body.get('dpp', {})

    if not problem_results:
        return error_response('Missing: problem_results', 'VALIDATION_ERROR')

    candidate = dpp.get('candidate', {})
    name = candidate.get('full_name', candidate.get('first_name', 'Candidate'))
    elapsed = dpp.get('session', {}).get('elapsed_minutes', '?')
    total_problems = dpp.get('session', {}).get('total_problems', len(problem_results))

    user_prompt = (
        f"Candidate: {name}\n"
        f"Session: {elapsed} minutes, {len(problem_results)} of {total_problems} problems attempted.\n"
        f"Hints given: {dpp.get('session', {}).get('hints_given', 0)}\n\n"
        f"## Per-Problem Results\n"
        f"```json\n{json.dumps(problem_results, separators=(',', ':'))}\n```\n\n"
        f"Synthesize these results into one overall assessment JSON."
    )

    result, usage = call_bedrock(user_prompt, SYNTHESIS_SYSTEM_PROMPT, max_tokens=512)

    return success_response(result, usage)


def handle_full(body):
    """Full single-call analysis (HR demo or legacy code interview)."""
    transcript = body.get('transcript', [])
    dpp = body.get('dpp', {})
    schema = body.get('schema')
    custom_prompt = body.get('summary_prompt')

    if not transcript:
        return error_response('Missing: transcript', 'VALIDATION_ERROR')
    if not dpp:
        return error_response('Missing: dpp', 'VALIDATION_ERROR')

    user_prompt = build_full_prompt(transcript, dpp, schema, custom_prompt)
    summary, usage = call_bedrock(user_prompt, custom_prompt or HR_SYSTEM_PROMPT)

    # Inject final_code from DPP
    final_code = dpp.get('final_code') or dpp.get('live_code', {}).get('current_code', '')
    if final_code and 'final_code' not in summary:
        summary['final_code'] = final_code

    return success_response(summary, usage)


# =============================================================================
# HELPERS
# =============================================================================

def build_full_prompt(transcript, dpp, schema=None, custom_prompt=None):
    transcript_text = format_transcript(transcript)
    turn_count = len([t for t in transcript if t.get('role') == 'user'])
    dpp_clean = {k: v for k, v in dpp.items() if k != 'summary_prompt'}

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

    return '\n'.join(parts)


def format_transcript(transcript):
    lines = []
    for i, turn in enumerate(transcript, 1):
        role = turn.get('role', 'unknown')
        content = turn.get('content', '')
        speaker = 'AI' if role == 'assistant' else 'Candidate'
        lines.append(f"[{i}] {speaker}: {content}")
    return '\n'.join(lines)


def call_bedrock(user_prompt, system_prompt, max_tokens=None):
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens or MAX_TOKENS,
        "temperature": TEMPERATURE,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}]
    }

    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(request_body),
        contentType='application/json',
        accept='application/json'
    )

    response_body = json.loads(response['body'].read())
    content = response_body.get('content', [{}])[0].get('text', '{}').strip()

    if content.startswith('```'):
        lines = content.split('\n')
        content = '\n'.join(lines[1:-1] if lines[-1] == '```' else lines[1:])

    try:
        summary = json.loads(content)
    except json.JSONDecodeError as e:
        print(f'Failed to parse LLM response: {content[:500]}')
        raise ValueError(f'LLM returned invalid JSON: {str(e)}')

    usage = {
        'input_tokens': response_body.get('usage', {}).get('input_tokens', 0),
        'output_tokens': response_body.get('usage', {}).get('output_tokens', 0)
    }

    return summary, usage


def success_response(data, usage):
    return {
        'statusCode': 200,
        'headers': CORS_HEADERS,
        'body': json.dumps({'success': True, 'summary': data, 'usage': usage})
    }


def error_response(message, code, status_code=400):
    return {
        'statusCode': status_code,
        'headers': CORS_HEADERS,
        'body': json.dumps({'success': False, 'error': message, 'code': code})
    }
