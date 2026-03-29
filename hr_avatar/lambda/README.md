# Avatar Analysis - Lambda Backend

A serverless API that analyzes Avatar call/session transcripts using AWS Bedrock (Claude 3.5 Haiku) and returns structured JSON summaries. Shared by both the **HR Avatar** and **Code Interview** demos.

## Architecture

```
┌─────────────┐      HTTPS       ┌──────────────┐     ┌─────────────┐
│   Browser   │ ────────────────▶│  API Gateway │────▶│   Lambda    │
│ (Demo Page) │ ◀──────────────── │   (HTTP v2)  │◀────│  (Python)   │
└─────────────┘      JSON        └──────────────┘     └──────┬──────┘
                                                             │
                                                     ┌───────▼───────┐
                                                     │    Bedrock    │
                                                     │Claude 3.5 Haiku│
                                                     └───────────────┘
```

**Security:** The Lambda is NOT directly accessible from the internet. It is exposed only via API Gateway, enforced by the Lambda resource policy (`sourceArn` scoped to the specific API Gateway ID).

## Analysis Modes

The Lambda supports three analysis modes, selected by the `analysis_mode` field in the request body:

| Mode | Used By | Description | max_tokens |
|------|---------|-------------|------------|
| `per_problem` | Code Interview | Analyze one coding problem from a transcript | 512 |
| `synthesis` | Code Interview | Synthesize per-problem results into an overall assessment | 512 |
| *(default)* | HR Avatar | Full single-call analysis using `HR_SYSTEM_PROMPT` (v4.1) | 2048 (env var) |

The Code Interview client calls `per_problem` in parallel for each problem (fast, ~5s each), then one `synthesis` call (~8s). The HR Avatar client sends a single request with no `analysis_mode` (falls through to the default full-analysis path).

## Features

- **Serverless** — Pay only for what you use
- **Multi-mode** — Three analysis paths (per-problem, synthesis, full) from one function
- **Custom prompts** — Full mode supports optional `summary_prompt` field to override the default system prompt
- **Secure** — No API keys in browser; Lambda accessible only via API Gateway
- **CORS-enabled** — Works from any origin
- **Resilient** — 90-second Lambda timeout, automatic Bedrock retries, structured errors

## Prerequisites

- AWS CLI configured (`aws sts get-caller-identity` should work)
- Bedrock access enabled in your AWS account
- Claude 3.5 Haiku enabled in Bedrock console (us-west-2 region)

## Quick Deploy

```bash
cd lambda/
./deploy.sh
```

The script will:
1. Create an IAM role (or use existing)
2. Package and deploy the Lambda function
3. Create an HTTP API Gateway with CORS
4. Output the API endpoint URL

**Note:** If you already have a manually-created API Gateway (e.g., `30vsmo8j0l`), you can skip the deploy script and just update the Lambda code:

```bash
zip -j function.zip lambda_function.py
aws lambda update-function-code \
  --function-name hr-avatar-analysis \
  --zip-file fileb://function.zip
```

## API Usage

### Mode: Default (HR Avatar)

Full single-call analysis. Returns a v4.1 summary with fit score, gaps, next steps, etc.

```bash
POST https://YOUR_API_ENDPOINT/
Content-Type: application/json

{
  "transcript": [
    {"role": "assistant", "content": "Hello, I'm calling about..."},
    {"role": "user", "content": "Hi, yes this is..."}
  ],
  "dpp": {
    "mode": "interview",
    "org": {"n": "Company Name"},
    "role": {"t": "Job Title", "id": "ROLE-001"},
    "subj": {"name": "Candidate Name", "id": "CAND-001"},
    "mtg": {"mins": 5, "focus": ["topic1", "topic2"]}
  },
  "summary_prompt": "(optional) Custom system prompt to override HR_SYSTEM_PROMPT"
}
```

### Mode: `per_problem` (Code Interview)

Analyze a single coding problem from the session transcript.

```bash
POST https://YOUR_API_ENDPOINT/
Content-Type: application/json

{
  "analysis_mode": "per_problem",
  "transcript": [{"role": "assistant", "content": "..."}, ...],
  "problem_focus": {"id": "two-sum", "title": "Two Sum", "difficulty": "easy"},
  "dpp": {
    "mode": "coding_challenge",
    "session": {"elapsed_minutes": 6, "total_problems": 4},
    "live_code": {"language": "python"},
    "candidate": {"full_name": "Jane Doe"},
    "all_problems_in_session": [...]
  }
}
```

### Mode: `synthesis` (Code Interview)

Combine per-problem results into one overall assessment.

```bash
POST https://YOUR_API_ENDPOINT/
Content-Type: application/json

{
  "analysis_mode": "synthesis",
  "problem_results": [
    {"problem_id": "two-sum", "scores": {...}, ...},
    {"problem_id": "valid-palindrome", "scores": {...}, ...}
  ],
  "dpp": {
    "session": {"elapsed_minutes": 6, "total_problems": 4, "hints_given": 0},
    "candidate": {"full_name": "Jane Doe"}
  }
}
```

### Response (Success)

All modes return the same envelope:

```json
{
  "success": true,
  "summary": { ... },
  "usage": {
    "input_tokens": 1234,
    "output_tokens": 567
  }
}
```

The `summary` shape varies by mode — see the embedded prompts in `lambda_function.py` for each mode's output schema.

### Response (Error)

```json
{
  "success": false,
  "error": "Error message",
  "code": "VALIDATION_ERROR|BEDROCK_ERROR|TIMEOUT|THROTTLING"
}
```

## Configuration

Environment variables (set in Lambda console or via CLI):

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `anthropic.claude-3-5-haiku-20241022-v1:0` | Bedrock model ID |
| `MAX_TOKENS` | `2048` | Max output tokens (used by full mode; per-problem and synthesis override to 512) |
| `TEMPERATURE` | `0.3` | Model temperature (lower = more deterministic) |

### Change Model

```bash
aws lambda update-function-configuration \
  --function-name hr-avatar-analysis \
  --environment "Variables={MODEL_ID=anthropic.claude-3-5-haiku-20241022-v1:0,MAX_TOKENS=2048,TEMPERATURE=0.3}"
```

## Benchmark

An automated benchmark (`benchmark.py`) stress-tests the iterative pipeline (4 parallel per-problem calls + 1 synthesis call) against the live Lambda.

```bash
# Run 10 iterations, print report, save raw JSON
python3 benchmark.py --runs 10 --json

# Custom threshold and URL
python3 benchmark.py --runs 5 --threshold 19 --url https://YOUR_API_ENDPOINT/
```

The benchmark:
- Fires 4 parallel per-problem calls + 1 synthesis call per iteration (matching the real client flow)
- Retries on 503/429 with 3s backoff (matching the client's retry logic)
- Reports per-run timing, aggregate stats (min/avg/p95/max), bottleneck analysis, and a PASS/FAIL verdict
- Exits with code 0 (pass) or 1 (fail), suitable for CI

No dependencies beyond Python 3 stdlib.

## Updating the Function

After editing `lambda_function.py`:

```bash
zip -j function.zip lambda_function.py
aws lambda update-function-code \
  --function-name hr-avatar-analysis \
  --zip-file fileb://function.zip
```

Then update `ANALYSIS_API_URL` in both client files if the endpoint changes:
- `hr_avatar/hr-demo.js`
- `code_interview/code-interview.js`

## Monitoring

### View Logs

```bash
aws logs tail /aws/lambda/hr-avatar-analysis --follow
```

### Check Function Status

```bash
aws lambda get-function --function-name hr-avatar-analysis \
  --query 'Configuration.{State:State,LastModified:LastModified}'
```

## Cleanup

```bash
./cleanup.sh

# Or manually:
aws apigatewayv2 delete-api --api-id YOUR_API_ID
aws lambda delete-function --function-name hr-avatar-analysis
aws iam delete-role-policy --role-name hr-avatar-analysis-lambda-role --policy-name bedrock-invoke
aws iam detach-role-policy --role-name hr-avatar-analysis-lambda-role --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
aws iam delete-role --role-name hr-avatar-analysis-lambda-role
```

## Troubleshooting

### 403 Forbidden
- Check Lambda resource policy allows API Gateway
- Verify API Gateway has correct Lambda integration

### 503 Service Unavailable
- HTTP API Gateway has a hard 30-second timeout — this is not configurable
- The Code Interview client uses iterative parallel calls (5-8s each) to stay well under this limit
- The HR Avatar client makes a single call; ensure the transcript isn't excessively long

### Timeout (504)
- Increase Lambda timeout (max 900s)
- Check if transcript is unusually large

### Invalid JSON Response
- Check CloudWatch logs for parsing errors
- Increase `MAX_TOKENS` if output is being truncated

### Bedrock Throttling
- Both clients implement retry with backoff for 429 responses
- Request quota increase from AWS if persistent

## Files

| File | Description |
|------|-------------|
| `lambda_function.py` | Lambda handler — three analysis modes, Bedrock integration, embedded prompts |
| `benchmark.py` | Performance benchmark for iterative pipeline (stdlib only, no dependencies) |
| `deploy.sh` | Automated deployment script (IAM + Lambda + API Gateway) |
| `cleanup.sh` | Resource cleanup script |
| `trust-policy.json` | IAM trust policy for Lambda execution role |
| `bedrock-policy.json` | IAM policy granting Bedrock invoke access |
| `.api-url` | Stores deployed API URL (generated by deploy.sh) |
