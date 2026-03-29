# Kaltura Avatar SDK

> **Note:** This is a non-official, research-level SDK. For official Kaltura SDKs, see [github.com/kaltura](https://github.com/kaltura/).

Embed AI avatar conversations in any website with just a few lines of code.

**Zero dependencies** · **~4KB minified** · **Works everywhere**

## Quick Start

```html
<div id="avatar" style="width: 800px; height: 600px;"></div>
<script src="kaltura-avatar-sdk.min.js"></script>
<script>
  new KalturaAvatarSDK({
    clientId: 'YOUR_CLIENT_ID',
    flowId: 'YOUR_FLOW_ID',
    container: '#avatar'
  }).start();
</script>
```

That's it! The avatar loads and users can start talking.

## Installation

**Option 1: Direct download**
```html
<script src="kaltura-avatar-sdk.min.js"></script>
```

**Option 2: ES Module**
```javascript
import KalturaAvatarSDK from './kaltura-avatar-sdk.js';
```

## Basic Usage

```javascript
// Create and start
const sdk = new KalturaAvatarSDK({
  clientId: 'YOUR_CLIENT_ID',
  flowId: 'YOUR_FLOW_ID',
  container: '#avatar'
});

await sdk.start();

// Listen to avatar speech
sdk.on('agent-talked', (data) => {
  console.log('Avatar said:', data.agentContent || data);
});

// Listen to user speech
sdk.on('user-transcription', (data) => {
  console.log('User said:', data.userTranscription || data);
});

// Send a prompt programmatically
sdk.injectPrompt('Tell me a joke');

// End conversation
sdk.end();
```

## API Reference

### Constructor

```javascript
new KalturaAvatarSDK({
  clientId: string,      // Required: Your client ID
  flowId: string,        // Required: Your flow ID
  container: string,     // Optional: CSS selector or HTMLElement
  config: {              // Optional: Configuration
    debug: boolean,      // Enable console logging
    apiBaseUrl: string,  // Override API URL
    meetBaseUrl: string  // Override meet URL
  }
})
```

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `start(options?)` | `Promise<HTMLIFrameElement>` | Start the avatar conversation |
| `end()` | `void` | End the conversation |
| `destroy()` | `void` | Cleanup SDK resources |
| `injectPrompt(text)` | `boolean` | Send a prompt to the avatar |
| `sendMessage(message)` | `boolean` | Send raw message to iframe |
| `on(event, callback)` | `() => void` | Subscribe to events (returns unsubscribe fn) |
| `off(event, callback)` | `void` | Unsubscribe from event |
| `once(event, callback)` | `() => void` | Subscribe once |
| `getState()` | `string` | Get current SDK state |
| `getAssets()` | `object\|null` | Get loaded avatar assets |
| `getAvatarInfo()` | `object\|null` | Get avatar details |
| `getIframe()` | `HTMLIFrameElement\|null` | Get iframe element |
| `getTalkUrl()` | `string\|null` | Get talk URL |
| `getClientId()` | `string` | Get client ID |
| `getFlowId()` | `string` | Get flow ID |
| `getTranscript()` | `Array` | Get conversation transcript |
| `getTranscriptText(options?)` | `string` | Get transcript as formatted text |
| `downloadTranscript(options?)` | `void` | Download transcript as file |
| `clearTranscript()` | `void` | Clear the transcript |
| `setTranscriptEnabled(enabled)` | `void` | Enable/disable transcript recording |

### Events

Subscribe using `sdk.on(event, callback)`:

```javascript
// Avatar events
sdk.on('showing-join-meeting', () => {});  // Join screen displayed
sdk.on('join-meeting-clicked', () => {});  // User clicked join
sdk.on('showing-agent', () => {});         // Avatar is visible
sdk.on('agent-talked', (data) => {});      // Avatar spoke
sdk.on('user-transcription', (data) => {}); // User speech transcribed
sdk.on('pronunciation-score', (data) => {}); // Pronunciation score
sdk.on('permissions-denied', () => {});    // Mic/camera denied
sdk.on('conversation-ended', () => {});    // Conversation ended
sdk.on('load-agent-error', () => {});      // Failed to load avatar

// SDK events
sdk.on('ready', ({ assets }) => {});       // SDK initialized
sdk.on('started', ({ iframe }) => {});     // Conversation started
sdk.on('ended', () => {});                 // Conversation ended
sdk.on('error', ({ message }) => {});      // Error occurred
sdk.on('stateChange', ({ from, to }) => {}); // State changed

// Wildcard - receive all events
sdk.on('*', ({ event, data }) => {});
```

You can also use constants:
```javascript
sdk.on(KalturaAvatarSDK.Events.AGENT_TALKED, (data) => {});
sdk.on(KalturaAvatarSDK.Events.USER_TRANSCRIPTION, (data) => {});
```

### States

```javascript
sdk.getState(); // Returns one of:
// 'uninitialized' - SDK created but not initialized
// 'initializing'  - Loading avatar assets
// 'ready'         - Ready to start conversation
// 'in-conversation' - Conversation active
// 'ended'         - Conversation ended
// 'error'         - Error occurred
```

## Examples

### Auto-start on Page Load

```javascript
const sdk = new KalturaAvatarSDK({
  clientId: 'xxx',
  flowId: 'yyy',
  container: '#avatar'
});

sdk.start(); // Starts immediately
```

### Custom Styling

```javascript
await sdk.start({
  styles: {
    borderRadius: '20px',
    boxShadow: '0 10px 40px rgba(0,0,0,0.3)'
  }
});
```

### Programmatic Prompts

```javascript
// Send a prompt to the avatar
sdk.injectPrompt('What is the weather today?');

// The avatar will respond as if the user said this
```

### React Component

```jsx
import { useEffect, useRef } from 'react';
import KalturaAvatarSDK from './kaltura-avatar-sdk';

function Avatar({ clientId, flowId }) {
  const ref = useRef();

  useEffect(() => {
    const sdk = new KalturaAvatarSDK({
      clientId,
      flowId,
      container: ref.current
    });
    sdk.start();
    return () => sdk.destroy();
  }, [clientId, flowId]);

  return <div ref={ref} style={{ height: 500 }} />;
}
```

### Vue Component

```vue
<template>
  <div ref="container" style="height: 500px"></div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue';
import KalturaAvatarSDK from './kaltura-avatar-sdk';

const props = defineProps(['clientId', 'flowId']);
const container = ref();
let sdk;

onMounted(() => {
  sdk = new KalturaAvatarSDK({
    clientId: props.clientId,
    flowId: props.flowId,
    container: container.value
  });
  sdk.start();
});

onUnmounted(() => sdk?.destroy());
</script>
```

### Event Logging

```javascript
// Log all events
sdk.on('*', ({ event, data }) => {
  console.log(`[${event}]`, data);
});
```

### Transcript Recording

The SDK automatically records conversation transcripts.

```javascript
// Get transcript as array
const transcript = sdk.getTranscript();
// Returns: [{ role: 'Avatar'|'User', text: string, timestamp: Date }, ...]

// Get transcript as text
const text = sdk.getTranscriptText();
const markdown = sdk.getTranscriptText({ format: 'markdown' });
const json = sdk.getTranscriptText({ format: 'json' });

// Download transcript
sdk.downloadTranscript(); // Downloads as .txt
sdk.downloadTranscript({ format: 'markdown' }); // Downloads as .md
sdk.downloadTranscript({
  format: 'text',
  filename: 'my-transcript.txt',
  includeTimestamps: true
});

// Disable/enable transcript recording
sdk.setTranscriptEnabled(false);

// Clear transcript
sdk.clearTranscript();
```

## Demo Applications

Both demos share a single Lambda backend for AI-powered analysis. See [`hr_avatar/lambda/README.md`](hr_avatar/lambda/README.md) for deployment.

### HR Avatar Demo

The `hr_avatar/` folder contains a complete HR use case demo with interview, post-interview, and separation scenarios. See [`hr_avatar/README.md`](hr_avatar/README.md) for detailed documentation.

**Features:** Interview simulations, post-interview calls, separation meetings, CV upload, editable fields, live transcript, AI call analysis.

```bash
cd hr_avatar
python3 -m http.server 8080
# Open http://localhost:8080
```

### Code Interview Demo

The `code_interview/` folder contains an AI pair programming interview. See [`code_interview/README.md`](code_interview/README.md) for detailed documentation.

**Features:** Real-time code context injection (Monaco editor), multi-problem sessions, avatar-controlled flow, iterative parallel analysis with per-problem scoring and synthesis.

```bash
cd code_interview
python3 -m http.server 8081
# Open http://localhost:8081
```

**Recent Improvements (v1.5.7):**
- DPP injection now uses `SHOWING_AGENT` event for proper timing
- Removed arbitrary timeout — DPP injected when avatar is actually ready
- Added configurable `DPP_INJECTION_DELAY_MS` (500ms default) for safety margin

**Previous (v1.5.6):**
- Fixed state leakage between problems (test results no longer persist to next problem)
- Added race condition prevention during problem transitions
- DPP now tracks problem ID to handle identical starter code

### Dynamic Page Prompt (DPP)

Both demos use the Dynamic Page Prompt system to customize avatar behavior at runtime.

#### Best Practice: Inject DPP on SHOWING_AGENT

The `SHOWING_AGENT` event fires when the avatar is fully loaded and visible. This is the **correct time** to inject the DPP — the avatar is ready to receive and process context.

```javascript
const sdk = new KalturaAvatarSDK({
  clientId: 'YOUR_CLIENT_ID',
  flowId: 'YOUR_FLOW_ID',
  container: '#avatar'
});

// Load scenario/config JSON
const response = await fetch('scenario.json');
const scenarioData = await response.json();

// Listen for SHOWING_AGENT to inject DPP at the right time
sdk.on(KalturaAvatarSDK.Events.SHOWING_AGENT, () => {
  // Optional: add a small delay for safety margin
  setTimeout(() => {
    sdk.injectPrompt(JSON.stringify(scenarioData));
  }, 500); // 500ms is a good default
});

// Start avatar
await sdk.start();
```

**Why this pattern?**
- `sdk.start()` creates the iframe and begins loading, but returns immediately
- The iframe loads asynchronously (user may see join screen, grant permissions)
- `SHOWING_AGENT` fires when the avatar is actually visible and ready
- Injecting before this event may fail (iframe not ready to receive postMessage)

**Avoid this anti-pattern:**
```javascript
// ❌ BAD: Arbitrary timeout after start() — may fire too early or too late
await sdk.start();
setTimeout(() => sdk.injectPrompt(json), 800);

// ✅ GOOD: Wait for SHOWING_AGENT event
sdk.on('showing-agent', () => {
  setTimeout(() => sdk.injectPrompt(json), 500);
});
await sdk.start();
```

### Analysis Backend

Both demos share a single Lambda function (`hr_avatar/lambda/`) that routes requests based on the `analysis_mode` field:

| Mode | Used By | Description |
|------|---------|-------------|
| `per_problem` | Code Interview | Analyze one coding problem (~5s) |
| `synthesis` | Code Interview | Synthesize results into overall assessment (~8s) |
| *(default)* | HR Avatar | Full single-call analysis (v4.1 schema) |

```bash
cd hr_avatar/lambda
./deploy.sh  # Deploys to AWS Lambda + API Gateway
```

See [`hr_avatar/lambda/README.md`](hr_avatar/lambda/README.md) for deployment details, API usage, and benchmarking.

## Files

### SDK Core

| File | Size | Description |
|------|------|-------------|
| `kaltura-avatar-sdk.min.js` | ~4KB | Production (minified) |
| `kaltura-avatar-sdk.js` | ~12KB | Development (readable, documented) |
| `kaltura-avatar-sdk.d.ts` | ~4KB | TypeScript definitions |

### Basic Demo

| File | Description |
|------|-------------|
| `index.html` | Simple demo page |
| `demo.js` | Basic SDK usage example |
| `demo.css` | Demo styles |

### HR Avatar Demo

| File | Description |
|------|-------------|
| `hr_avatar/README.md` | Comprehensive HR demo documentation |
| `hr_avatar/index.html` | HR demo page |
| `hr_avatar/hr-demo.js` | Application logic (scenarios, SDK integration, analysis) |
| `hr_avatar/hr-demo.css` | Styles (warm professional theme) |
| `hr_avatar/base_prompt.txt` | Base system prompt for Nora HR avatar |
| `hr_avatar/dynamic_page_prompt.schema.json` | DPP v2 JSON Schema |
| `hr_avatar/call_summary.schema.json` | Call analysis output schema (v4.1) |
| `hr_avatar/dynamic_page_prompt_samples/` | Sample scenario JSON files |

### Code Interview Demo

| File | Description |
|------|-------------|
| `code_interview/README.md` | Code interview documentation |
| `code_interview/index.html` | Interview page (Monaco editor + avatar) |
| `code_interview/code-interview.js` | Application logic (problems, SDK, iterative analysis) |
| `code_interview/code-interview.css` | Dark theme styles |
| `code_interview/base_prompt.txt` | Avatar persona and behavior instructions |
| `code_interview/goals.txt` | Session goals for the avatar |
| `code_interview/dynamic_page_prompt.schema.json` | DPP JSON Schema for code interview |

### Lambda Backend (shared)

| File | Description |
|------|-------------|
| `hr_avatar/lambda/README.md` | Lambda deployment guide + API reference |
| `hr_avatar/lambda/lambda_function.py` | Bedrock Claude analysis — 3 modes (per-problem, synthesis, full) |
| `hr_avatar/lambda/benchmark.py` | Performance benchmark for iterative pipeline |
| `hr_avatar/lambda/deploy.sh` | Automated deployment script |
| `hr_avatar/lambda/cleanup.sh` | Resource cleanup script |
| `hr_avatar/lambda/*.json` | IAM policy files |

## Browser Support

Chrome 60+ · Firefox 55+ · Safari 11+ · Edge 79+

## TypeScript

TypeScript definitions are included. Import types:

```typescript
import KalturaAvatarSDK from './kaltura-avatar-sdk';

const sdk = new KalturaAvatarSDK({
  clientId: 'xxx',
  flowId: 'yyy'
});

sdk.on('agent-talked', (data) => {
  // data is typed as string | { agentContent: string }
});
```

## License

MIT
