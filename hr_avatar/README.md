# HR Avatar Demo

A complete HR use-case demonstration built with the Kaltura Avatar SDK. Features interview simulations, post-interview calls, and separation meetings with AI-powered call analysis.

## Features

- **Interview Scenarios** - Phone screen simulations for various roles (drivers, engineers, analysts)
- **Post-Interview Calls** - Job offer and rejection feedback conversations
- **Separation Meetings** - Layoff, performance, and misconduct terminations
- **CV Upload** - Upload candidate resumes (PDF) for personalized interviews
- **Editable Fields** - Customize candidate name, role, company, location before starting
- **Live Transcript** - Real-time conversation recording with TXT/MD download
- **AI Call Analysis** - Automatic call summary generation using AWS Bedrock

## Quick Start

1. **Serve the files** (from the `hr_avatar` directory):
   ```bash
   # Using Python
   python3 -m http.server 8080

   # Using Node.js
   npx serve .
   ```

2. **Open in browser**: http://localhost:8080

3. **Select a scenario** from the sidebar and start talking!

## Project Structure

```
hr_avatar/
├── index.html                    # Main demo page
├── hr-demo.js                    # Application logic (scenarios, SDK integration)
├── hr-demo.css                   # Styles (warm professional theme)
├── README.md                     # This file
│
├── dynamic_page_prompt.schema.json   # DPP v2 JSON Schema
├── call_summary.schema.json          # Call summary output schema (v4.1)
├── base_prompt.txt                   # Base system prompt for Nora HR avatar
│
├── dynamic_page_prompt_samples/      # Sample scenario files
│   ├── interview_amazon_*.json       # Interview scenarios
│   ├── post-interview_*.json         # Post-interview scenarios
│   └── separation_*.json             # Separation scenarios
│
└── lambda/                           # Call analysis backend (shared with code_interview)
    ├── README.md                     # Lambda deployment guide + API reference
    ├── lambda_function.py            # Analysis function — 3 modes (per-problem, synthesis, full)
    ├── benchmark.py                  # Performance benchmark (iterative pipeline)
    ├── deploy.sh                     # Deployment script
    ├── cleanup.sh                    # Cleanup script
    └── *.json                        # IAM policy files
```

## How It Works

### 1. Scenario Selection

The demo loads scenario configurations from JSON files in `dynamic_page_prompt_samples/`. Each scenario defines:

- **mode**: `interview`, `post_interview`, or `separation`
- **org**: Company details (name, values, tone)
- **role**: Job details (title, location, requirements)
- **subj**: Candidate/employee info
- **mtg**: Meeting config (duration, focus areas)
- **case**: For separations - type, effective date, talking points

### 2. SDK Integration

```javascript
// Initialize the SDK
const sdk = new KalturaAvatarSDK({
    clientId: 'YOUR_CLIENT_ID',
    flowId: 'YOUR_FLOW_ID',
    container: '#avatar-container'
});

// Best practice: Inject DPP when avatar becomes visible
sdk.on(KalturaAvatarSDK.Events.SHOWING_AGENT, () => {
    // Add small delay for safety margin
    setTimeout(() => {
        sdk.injectPrompt(JSON.stringify(scenarioData));
    }, 500);
});

// Listen for events
sdk.on(KalturaAvatarSDK.Events.AGENT_TALKED, (data) => {
    // Avatar spoke
});

sdk.on(KalturaAvatarSDK.Events.USER_TRANSCRIPTION, (data) => {
    // User spoke
});

sdk.on(KalturaAvatarSDK.Events.CONVERSATION_ENDED, () => {
    // Conversation ended - analyze and reset
});

// Start avatar
await sdk.start();
```

### 3. Call Analysis

When a conversation ends, the demo:
1. Sends the transcript + DPP to the Lambda analysis API
2. Receives a structured JSON summary (fit score, gaps, next steps, etc.)
3. Displays the summary in a modal with download option

## Adding New Scenarios

### Step 1: Create Scenario JSON

Create a new file in `dynamic_page_prompt_samples/`:

```json
{
  "v": "2",
  "mode": "interview",
  "org": {
    "n": "Your Company",
    "val": "Company values statement",
    "tone": "warm, professional"
  },
  "role": {
    "id": "ROLE-001",
    "t": "Job Title",
    "loc": "Location",
    "must": ["requirement 1", "requirement 2"],
    "nice": ["nice to have 1"]
  },
  "subj": {
    "id": "CAND-001",
    "name": "Candidate Name"
  },
  "mtg": {
    "mins": 5,
    "focus": ["topic 1", "topic 2"]
  }
}
```

### Step 2: Register in hr-demo.js

Add to the `SCENARIOS` object:

```javascript
const SCENARIOS = {
    interview: [
        // ... existing scenarios
        {
            id: 'interview_yourcompany_role',
            name: 'Your Company Role',
            description: 'Brief description',
            file: 'dynamic_page_prompt_samples/your-scenario.json',
            company: 'Your Company',
            role: 'Job Title',
            location: 'Location',
            duration: '5 min'
        }
    ]
};
```

### Step 3: Test

Refresh the page - your scenario appears in the sidebar automatically.

## Configuration

All configuration is centralized in the `CONFIG` object at the top of `hr-demo.js`:

```javascript
const CONFIG = Object.freeze({
    VERSION: '1.0.16',                    // Bump to bust browser cache
    CLIENT_ID: '115767973963657880005',   // Kaltura Avatar client
    FLOW_ID: 'agent-15',                   // Nora HR agent
    ANALYSIS_API_URL: 'https://...',       // Lambda API endpoint
    PROMPT_INJECTION_DELAY_MS: 2000,
    AVATAR_NAME: 'Nora (HR)',
    PDFJS_WORKER_URL: '...'
});
```

To customize for your deployment, update the values in `CONFIG`.

## Schemas

### Dynamic Page Prompt (DPP) v2

The DPP schema defines the structure for scenario configurations. Key sections:

| Section | Purpose |
|---------|---------|
| `org` | Company name, values, tone |
| `role` | Job title, location, requirements (must/nice) |
| `subj` | Candidate/employee name and profile |
| `mtg` | Meeting duration, focus areas |
| `eval` | Evaluation dimensions (for interviews) |
| `case` | Case details (for separations) |
| `limits` | Guardrails (banned topics, escalation triggers) |
| `inst` | Additional instructions for the avatar |

See `dynamic_page_prompt.schema.json` for the complete specification.

### Call Summary v4.1

The analysis output follows a structured schema with:

| Field | Description |
|-------|-------------|
| `ctx` | Context (company, role, person) |
| `overview` | 80-200 word summary |
| `key_answers` | Responses to critical questions |
| `fit` | Role fit assessment (score, dimensions) |
| `star_analysis` | STAR methodology analysis |
| `believability` | Credibility assessment |
| `gaps` | Missing information needing follow-up |
| `cq` | Call quality (emotion, tone, engagement) |
| `risk` | Risk flags and escalation status |
| `next_steps` | Recommended actions |

See `call_summary.schema.json` for the complete specification.

## Lambda Backend

The call analysis feature requires a serverless backend. See `lambda/README.md` for deployment instructions.

**Quick deploy:**
```bash
cd lambda
./deploy.sh
```

**Cost estimate:** ~$0.002 per call analysis

## Customization

### Styling

Edit `hr-demo.css`. The design uses CSS custom properties:

```css
:root {
    --accent-interview: #6b9080;      /* Sage green */
    --accent-post-interview: #7c6ca8; /* Purple */
    --accent-separation: #b56576;     /* Dusty rose */
    --bg-primary: #faf8f5;            /* Warm off-white */
}
```

### Avatar Behavior

Modify `base_prompt.txt` to change Nora's base personality and instructions.

### Analysis Output

Edit `HR_SYSTEM_PROMPT` in `lambda/lambda_function.py` to customize how HR call summaries are generated. The Lambda also contains separate prompts for Code Interview analysis (`PER_PROBLEM_SYSTEM_PROMPT`, `SYNTHESIS_SYSTEM_PROMPT`) — see `lambda/README.md` for details.

## Troubleshooting

### Avatar won't load
- Check browser console for errors
- Verify microphone/camera permissions
- Ensure valid clientId and flowId

### Call analysis fails
- Check if Lambda is deployed: `curl -X POST $API_URL -d '{"test":true}'`
- Check CloudWatch logs: `aws logs tail /aws/lambda/hr-avatar-analysis --follow`

### Transcript not recording
- Verify `sdk.setTranscriptEnabled(true)` (enabled by default)
- Check for JavaScript errors in console

## Browser Support

Chrome 60+ · Firefox 55+ · Safari 11+ · Edge 79+

## Version

Current: v1.0.21

### Changelog

- **v1.0.24**: Refactored DPP injection to use `SHOWING_AGENT` event — removed arbitrary timeout, DPP now injected when avatar is actually ready, added configurable `DPP_INJECTION_DELAY_MS` (500ms default)
- **v1.0.23**: Added call end trigger detection — avatar saying "Ending call now" triggers end handler
- **v1.0.22**: Added debug panel to show full DPP injection
- **v1.0.21**: Fixed avatar context confusion — increased DPP injection delay (200ms → 800ms), added DPP validation before injection, fixed state leakage on scenario switch
- **v1.0.20**: Performance improvements, refined call summary schema v4.1

## License

MIT
