# Code Interview POC - AI Pair Programming

An interactive coding interview demo using the Kaltura Avatar SDK. Features a real-time AI pair programming assistant that watches code as it's typed and provides contextual guidance.

## Quick Start

1. **Serve the files** via any HTTP server (e.g., Apache, nginx, or `python -m http.server`)
2. **Open** `index.html` in a browser
3. **Register** with your name and email on the opening screen
4. **Click "Start Peer-Programming Interview"** to begin
5. **Solve problems** - the avatar sees your code in real-time and guides you
6. **View your report** - detailed analysis shown at the end

## Application Flow

The application has three distinct screens:

```
+------------------+     +------------------+     +------------------+
|  OPENING SCREEN  | --> | INTERVIEW SCREEN | --> |   END SCREEN     |
|  - Registration  |     |  - Avatar + Code |     |  - Full Report   |
|  - Name/Email    |     |  - Live Coding   |     |  - Download JSON |
|  - Validation    |     |  - Multi-Problem |     |  - Restart       |
+------------------+     +------------------+     +------------------+
```

1. **Opening Screen**: User enters their first name, last name, and email. Validation ensures all fields are properly filled.

2. **Interview Screen**: The main coding interface with the avatar on the left and Monaco editor on the right. The avatar addresses the user by name.

3. **End Screen**: Displays the comprehensive session analysis report with candidate info, scores, and detailed feedback. Option to download as JSON or start a new interview.

## Files

| File | Purpose |
|------|---------|
| `index.html` | Main HTML page with Monaco editor and avatar container |
| `code-interview.js` | Application logic, SDK integration, DPP injection, iterative analysis |
| `code-interview.css` | Dark theme styles for editor and UI |
| `base_prompt.txt` | Avatar persona, behavior instructions, and DPP schema |
| `goals.txt` | Primary goals the avatar should achieve during conversation |
| `summary_prompt.txt` | Analysis schema (legacy — used only by the full-mode fallback) |
| `dynamic_page_prompt.schema.json` | JSON Schema for DPP validation and reference |

## Architecture

```
+------------------+     +-------------------+
|  Monaco Editor   |---->|  DPP Builder      |
|  (user's code)   |     |  (buildDPP())     |
+------------------+     +---------+---------+
                                   |
                                   v
                         +-------------------+
                         |  Kaltura Avatar   |
                         |  SDK              |
                         |  (injectPrompt)   |
                         +---------+---------+
                                   |
                                   v
                         +-------------------+
                         |  Avatar Speech    |
                         |  Trigger Detection|
                         +-------------------+
                                   |
              +--------------------+--------------------+
              |                                         |
              v                                         v
    "Switching to the next             "Ending the session now."
     challenge now."                              |
              |                                   v
              v                    +------------------------------+
    +-------------------+          | Iterative Parallel Analysis  |
    | Load Next Problem |          | (Lambda via API Gateway)     |
    +-------------------+          +------------------------------+
                                     |
                         +-----------+-----------+
                         |                       |
                         v                       v
                   Phase 1: N parallel     Phase 2: 1 synthesis
                   per-problem calls       call (~8s)
                   (~5s wall-clock)              |
                         |                       v
                         +--------> Phase 3: Assemble
                                    final v1.5 summary
                                         |
                                         v
                                  +-------------+
                                  | End Screen  |
                                  | (Report UI) |
                                  +-------------+
```

## How It Works

1. **Real-time Code Injection**: As the user types, their code is packaged into a DPP (Dynamic Page Prompt) and injected to the avatar every 200ms (debounced).

2. **Phase Detection**: The system detects what phase the user is in (START, CODING, DEBUG, PARTIAL, COMPLETE) and includes this in the DPP.

3. **Avatar-Controlled Flow**: The avatar controls problem switching and session ending by saying specific trigger phrases that the UI listens for.

4. **Iterative Parallel Analysis**: When the session ends, the client orchestrates analysis in three phases:
   - **Phase 1** — Fire N parallel `per_problem` calls to the Lambda (one per problem attempted). Each call analyzes a single problem from the full transcript (~5s each, running concurrently).
   - **Phase 2** — Send one `synthesis` call with all per-problem results. The Lambda produces an overall assessment (~8s).
   - **Phase 3** — The client assembles the final v1.5 summary object from per-problem analyses + synthesis + session metadata, then displays the report.

## Session Phases

| Phase | Trigger | Avatar Behavior |
|-------|---------|-----------------|
| `START` | Just joined, hasn't typed | Brief greeting, then silent |
| `STUCK` | No changes for 60+ seconds | Check-in, offer hints |
| `CODING` | Actively writing code | Stay silent, observe |
| `DEBUG` | Code threw an error | Help diagnose |
| `PARTIAL` | Some tests pass | Encourage, prompt edge cases |
| `WRONG_OUTPUT` | Code runs, 0 tests pass | Suggest tracing |
| `COMPLETE` | All tests pass | Verify understanding, then switch/end |

## Avatar Trigger Phrases

The avatar controls the session flow using exact phrases:

- **"Switching to the next challenge now."** - Loads the next problem automatically
- **"Ending the session now."** - Ends session and triggers analysis

## Problems Included

1. **Two Sum** (Easy) - Hash map solution, O(n)
2. **Valid Palindrome** (Easy) - Two pointer solution, O(n)
3. **Reverse Linked List** (Medium) - Iterative pointer manipulation, O(n)
4. **Fizz Buzz** (Easy) - Modulo operations, O(n)

## Adding New Problems

Add to the `PROBLEMS` object in `code-interview.js`:

```javascript
'problem-id': {
    id: 'problem-id',
    title: 'Problem Name',
    difficulty: 'easy', // easy | medium | hard
    description: `Problem description...`,
    examples: [{ input: '...', output: '...', explanation: '...' }],
    starterCode: {
        python: `def solution():\n    pass`,
        javascript: `function solution() {\n\n}`
    },
    testCases: [{ input: [...], expected: ... }],
    hints: ['Hint 1', 'Hint 2'],
    optimalComplexity: 'O(n) time, O(1) space'
}
```

Then add the ID to the `PROBLEM_ORDER` array. Problem details are passed dynamically to the avatar via DPP, so no prompt file updates are needed. The new problem will automatically be included in the iterative analysis pipeline.

## Configuration

Edit `CONFIG` in `code-interview.js`:

```javascript
const CONFIG = {
    CLIENT_ID: '...',              // Kaltura SDK client ID
    FLOW_ID: 'agent-16',           // Avatar agent ID
    ANALYSIS_API_URL: '...',       // Lambda API endpoint (shared with HR demo)
    SUMMARY_PROMPT_PATH: '...',    // Legacy summary prompt file (not used by iterative flow)
    DEBOUNCE_MS: 200,              // Code change debounce (ms)
    MAX_INTERVAL_MS: 15000,        // Max interval between DPP updates
    AVATAR_NAME: 'Alex'            // Avatar display name
};
```

## Session Analysis

When the session ends, the client runs an iterative parallel analysis pipeline:

1. **Per-problem calls** (parallel) — One Lambda call per problem attempted, each receiving the full transcript plus a `problem_focus` parameter. Each call returns scores (creativity, logic, code quality, explainability, complexity, scale), approach details, complexity analysis, and evaluation notes. These run concurrently for ~5s wall-clock time.

2. **Synthesis call** — One Lambda call that receives all per-problem results and produces an overall assessment: overview, skill assessment, potential assessment, fit score (0-100), strengths, areas for improvement, and next steps. ~8s.

3. **Assembly** — The client combines per-problem analyses, synthesis output, session metadata, and the candidate's final code into a v1.5 summary object displayed in the end-screen report.

The report includes:
- **Skills**: Problem solving, code fluency, communication, efficiency awareness (1-5 each)
- **Potential**: Creativity, tenacity, aptitude, propensity (1-5 each)
- **Per-Problem Evaluation**: Detailed scores, approach, complexity for each problem
- **Fit Score**: 0-100 with hiring recommendation

If any per-problem call fails, a fallback entry with neutral scores (3/5) is used so the report always renders.

## Technical Notes

- **User Registration**: Validates first name, last name (letters, hyphens, apostrophes only, min 2 chars), and email (standard format). User info is passed to both DPP and analysis.

- **Personalized Avatar**: The `user.first_name` field in DPP allows the avatar to address the candidate by name.

- **Simulated Execution**: `simulateExecution()` uses pattern matching, not real code execution. For production, integrate a sandboxed code runner.

- **Multi-Problem Sessions**: Avatar tracks progress across problems. The DPP includes pre-calculated fields (`is_last_problem`, `action_when_done`) to simplify avatar decision-making.

- **Retry Logic**: `callAnalysisAPI()` retries on 503/429 with 3s backoff (up to 2 attempts), matching API Gateway timeout behavior.

- **Graceful Degradation**: If a per-problem call fails, a neutral fallback entry is inserted so the synthesis and report still work.

- **Session Restart**: Full state reset clears all user data, problem progress, and re-initializes the SDK for a fresh session.

## Version

Current: v1.5.6

### Changelog

- **v1.5.6**: Fixed avatar context confusion — state leakage between problems (test results now properly cleared), race condition prevention during problem transitions (code tracking suspended), DPP tracks problem ID (handles identical starter code), DPP validation catches state inconsistencies, reduced transition delays (1500ms → 800ms)
- **v1.5.5**: Iterative parallel analysis pipeline (per-problem + synthesis), v1.5 summary schema with nested scores, retry logic with 3s backoff, performance benchmark
- **v1.5.4**: Fixed session-end race condition (transcript captured before SDK stop), SDK end-call button triggers end screen, 10-minute default session, problems passed dynamically via DPP (removed hardcoded solutions from base prompt), avatar patience improvements (no rephrasing/summarizing)
- **v1.5.0**: Three-screen flow (Opening > Interview > End), user registration with validation, personalized avatar interaction using candidate name, inline report display (replaces modal), restart functionality
- **v1.4.0**: Custom summary prompt support, comprehensive analysis modal, simplified DPP fields
- **v1.3.0**: Avatar-controlled problem switching and session ending via trigger phrases
- **v1.2.0**: Problem-specific code simulation, multi-problem session tracking
- **v1.1.0**: Added Fizz Buzz problem, session analysis with potential assessment
- **v1.0.0**: Initial release with Two Sum, Valid Palindrome, Reverse Linked List
