/**
 * Code Interview POC - AI Pair Programming
 *
 * Demonstrates real-time code context injection to an AI avatar,
 * enabling intelligent pair programming assistance.
 *
 * Session analysis uses an iterative parallel pipeline:
 *   Phase 1: N parallel per-problem Lambda calls (one per problem attempted)
 *   Phase 2: 1 synthesis Lambda call (combines per-problem results)
 *   Phase 3: Client-side assembly of final v1.5 summary
 *
 * @see analyzeSessionWithData — orchestrates the 3-phase analysis
 * @see callAnalysisAPI — HTTP helper with retry logic (503/429 → 3s backoff)
 */

// =============================================================================
// CONFIGURATION
// =============================================================================

const CONFIG = Object.freeze({
    VERSION: '1.5.7',

    // Kaltura Avatar SDK
    CLIENT_ID: '115767973963657880005',
    FLOW_ID: 'agent-16',

    // Call analysis API endpoint (local proxy → LiteLLM; run: python3 local_proxy.py)
    ANALYSIS_API_URL: 'http://localhost:8090',

    // Summary prompt file path (legacy — not used by iterative analysis flow)
    SUMMARY_PROMPT_PATH: 'summary_prompt.txt',

    // DPP injection timing
    DPP_INJECTION_DELAY_MS: 500,  // Delay after SHOWING_AGENT before initial DPP injection
    DEBOUNCE_MS: 200,             // Wait 200ms after typing stops for code updates
    MAX_INTERVAL_MS: 15000,       // Force update every 15s during active coding

    // Avatar name
    AVATAR_NAME: 'Alex'
});

// =============================================================================
// PROBLEM DEFINITIONS
// =============================================================================

const PROBLEMS = {
    'two-sum': {
        id: 'two-sum',
        title: 'Two Sum',
        difficulty: 'easy',
        description: `Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.`,
        examples: [
            {
                input: 'nums = [2,7,11,15], target = 9',
                output: '[0,1]',
                explanation: 'nums[0] + nums[1] = 2 + 7 = 9'
            }
        ],
        starterCode: {
            python: `def two_sum(nums, target):
    # Your code here
    pass

# Test
print(two_sum([2, 7, 11, 15], 9))`,
            javascript: `function twoSum(nums, target) {
    // Your code here

}

// Test
console.log(twoSum([2, 7, 11, 15], 9));`
        },
        testCases: [
            { input: [[2, 7, 11, 15], 9], expected: [0, 1] },
            { input: [[3, 2, 4], 6], expected: [1, 2] },
            { input: [[3, 3], 6], expected: [0, 1] }
        ],
        hints: [
            'Consider using a hash map to store values you\'ve seen',
            'For each number, check if (target - number) exists in your map',
            'You can solve this in O(n) time with O(n) space'
        ],
        optimalComplexity: 'O(n) time, O(n) space'
    },

    'valid-palindrome': {
        id: 'valid-palindrome',
        title: 'Valid Palindrome',
        difficulty: 'easy',
        description: `A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward.

Given a string s, return true if it is a palindrome, or false otherwise.`,
        examples: [
            {
                input: 's = "A man, a plan, a canal: Panama"',
                output: 'true',
                explanation: '"amanaplanacanalpanama" is a palindrome.'
            }
        ],
        starterCode: {
            python: `def is_palindrome(s):
    # Your code here
    pass

# Test
print(is_palindrome("A man, a plan, a canal: Panama"))`,
            javascript: `function isPalindrome(s) {
    // Your code here

}

// Test
console.log(isPalindrome("A man, a plan, a canal: Panama"));`
        },
        testCases: [
            { input: ['A man, a plan, a canal: Panama'], expected: true },
            { input: ['race a car'], expected: false },
            { input: [' '], expected: true }
        ],
        hints: [
            'First, filter out non-alphanumeric characters and convert to lowercase',
            'Compare the string with its reverse',
            'You can also use two pointers from both ends'
        ],
        optimalComplexity: 'O(n) time, O(1) space with two pointers'
    },

    'reverse-linked-list': {
        id: 'reverse-linked-list',
        title: 'Reverse Linked List',
        difficulty: 'medium',
        description: `Given the head of a singly linked list, reverse the list, and return the reversed list.

Each node has a value and a next pointer to the following node (or null/None for the last node).`,
        examples: [
            {
                input: 'head = [1,2,3,4,5]',
                output: '[5,4,3,2,1]',
                explanation: 'The list is reversed.'
            }
        ],
        starterCode: {
            python: `class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_list(head):
    # Your code here
    pass

# Test with: 1 -> 2 -> 3 -> 4 -> 5
head = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))
result = reverse_list(head)
# Print result
vals = []
while result:
    vals.append(result.val)
    result = result.next
print(vals)`,
            javascript: `class ListNode {
    constructor(val = 0, next = null) {
        this.val = val;
        this.next = next;
    }
}

function reverseList(head) {
    // Your code here

}

// Test with: 1 -> 2 -> 3 -> 4 -> 5
const head = new ListNode(1, new ListNode(2, new ListNode(3, new ListNode(4, new ListNode(5)))));
const result = reverseList(head);
// Print result
const vals = [];
let curr = result;
while (curr) {
    vals.push(curr.val);
    curr = curr.next;
}
console.log(vals);`
        },
        testCases: [
            { input: [[1,2,3,4,5]], expected: [5,4,3,2,1] },
            { input: [[1,2]], expected: [2,1] },
            { input: [[]], expected: [] }
        ],
        hints: [
            'You need to change where each node points to',
            'Keep track of the previous node as you iterate',
            'Think about what happens to the head and tail'
        ],
        optimalComplexity: 'O(n) time, O(1) space'
    },

    'fizz-buzz': {
        id: 'fizz-buzz',
        title: 'Fizz Buzz',
        difficulty: 'easy',
        description: `Given an integer n, return a string array answer (1-indexed) where:

- answer[i] == "FizzBuzz" if i is divisible by 3 and 5.
- answer[i] == "Fizz" if i is divisible by 3.
- answer[i] == "Buzz" if i is divisible by 5.
- answer[i] == i (as a string) if none of the above conditions are true.`,
        examples: [
            {
                input: 'n = 15',
                output: '["1","2","Fizz","4","Buzz","Fizz","7","8","Fizz","Buzz","11","Fizz","13","14","FizzBuzz"]',
                explanation: 'Numbers divisible by 3 become "Fizz", by 5 become "Buzz", by both become "FizzBuzz".'
            }
        ],
        starterCode: {
            python: `def fizz_buzz(n):
    # Your code here
    pass

# Test
print(fizz_buzz(15))`,
            javascript: `function fizzBuzz(n) {
    // Your code here

}

// Test
console.log(fizzBuzz(15));`
        },
        testCases: [
            { input: [3], expected: ["1", "2", "Fizz"] },
            { input: [5], expected: ["1", "2", "Fizz", "4", "Buzz"] },
            { input: [15], expected: ["1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8", "Fizz", "Buzz", "11", "Fizz", "13", "14", "FizzBuzz"] }
        ],
        hints: [
            'Loop through numbers 1 to n',
            'Check divisibility by 15 first (both 3 and 5), then by 3, then by 5',
            'Use the modulo operator (%) to check divisibility'
        ],
        optimalComplexity: 'O(n) time, O(n) space'
    }
};

// Problem sequence
const PROBLEM_ORDER = ['two-sum', 'valid-palindrome', 'reverse-linked-list', 'fizz-buzz'];

// =============================================================================
// APPLICATION STATE
// =============================================================================

const state = {
    // Current screen: 'opening' | 'interview' | 'end'
    currentScreen: 'opening',

    // User info (collected at registration)
    user: {
        firstName: '',
        lastName: '',
        email: ''
    },

    // SDK instance
    sdk: null,

    // Monaco editor instance
    editor: null,

    // Current problem tracking
    currentProblemIndex: 0,
    currentProblem: PROBLEMS['two-sum'],
    completedProblems: [],  // IDs of completed problems

    // Selected language
    language: 'python',

    // Session timing
    sessionStartTime: null,

    // Code injection tracking
    lastInjectedCode: '',
    lastInjectedProblemId: null, // Track which problem was last injected
    lastInjectionTime: 0,
    debounceTimer: null,
    intervalTimer: null,

    // Run results
    lastRunResult: null,
    runCount: 0,

    // Hints given (per problem)
    hintsGiven: 0,

    // Problem completion tracking
    problemCompleted: false,

    // Analysis results
    lastSessionSummary: null,

    // Custom summary prompt (loaded from file)
    summaryPrompt: null,

    // Session end guard (prevents duplicate end handling)
    isEndingSession: false,

    /**
     * Build the complete DPP object for injection.
     */
    buildDPP() {
        const code = this.editor?.getValue() || '';
        const starterCode = this.currentProblem.starterCode[this.language];
        const isStarterCode = code.trim() === starterCode.trim();
        const elapsedMins = this.sessionStartTime
            ? Math.floor((Date.now() - this.sessionStartTime) / 60000)
            : 0;
        const elapsedSecs = this.sessionStartTime
            ? Math.floor((Date.now() - this.sessionStartTime) / 1000)
            : 0;

        const phase = this.getCurrentPhase(code, isStarterCode, elapsedSecs);

        // Track if problem was just completed
        if (phase === 'COMPLETE' && !this.problemCompleted) {
            this.problemCompleted = true;
            // Add to completed list if not already there
            if (!this.completedProblems.includes(this.currentProblem.id)) {
                this.completedProblems.push(this.currentProblem.id);
            }
        }

        // Determine next problem
        const nextProblemIndex = this.currentProblemIndex + 1;
        const hasNextProblem = nextProblemIndex < PROBLEM_ORDER.length;
        const nextProblem = hasNextProblem ? PROBLEMS[PROBLEM_ORDER[nextProblemIndex]] : null;

        return {
            v: '2',
            mode: 'coding_challenge',

            // User info (for personalized interaction)
            user: {
                first_name: this.user.firstName,
                full_name: `${this.user.firstName} ${this.user.lastName}`.trim(),
                email: this.user.email
            },

            // Session config
            mtg: {
                mins: 10
            },

            // Problem being solved
            problem: {
                id: this.currentProblem.id,
                title: this.currentProblem.title,
                difficulty: this.currentProblem.difficulty,
                description: this.currentProblem.description,
                example: this.currentProblem.examples[0]?.input + ' → ' + this.currentProblem.examples[0]?.output,
                optimal_complexity: this.currentProblem.optimalComplexity
            },

            // Live code state (updates in real-time)
            live_code: {
                language: this.language,
                current_code: code,
                is_starter_code: isStarterCode,
                line_count: code.split('\n').length,
                code_observations: this.analyzeCode(code)
            },

            // Execution results (null until they click Run)
            last_execution: this.lastRunResult ? {
                tests_passed: this.lastRunResult.passed,
                tests_failed: this.lastRunResult.failed,
                total_tests: this.lastRunResult.total,
                error_message: this.lastRunResult.error,
                output: this.lastRunResult.output
            } : null,

            // Session timing and progress
            session: {
                elapsed_seconds: elapsedSecs,
                elapsed_minutes: elapsedMins,
                times_code_was_run: this.runCount,
                hints_given: this.hintsGiven,
                phase: phase,
                problem_completed: this.problemCompleted,
                // Progress: "Problem X of Y"
                current_problem: this.currentProblemIndex + 1,
                total_problems: PROBLEM_ORDER.length,
                // Pre-calculated - just read these directly
                is_last_problem: !hasNextProblem,
                action_when_done: hasNextProblem ? 'SWITCH' : 'END'
            },

            // Next problem info (for avatar to offer transition)
            next_problem: hasNextProblem ? {
                id: nextProblem.id,
                title: nextProblem.title,
                difficulty: nextProblem.difficulty
            } : null,

            // Clear instruction for the avatar - ONLY included when problem is COMPLETE
            instruction_on_complete: phase === 'COMPLETE'
                ? (hasNextProblem
                    ? `SWITCH: Problem completed! After verifying understanding, say "Switching to the next challenge now." to proceed to ${nextProblem.title}.`
                    : `END: Final problem completed! After verifying understanding, say "Ending the session now."`)
                : null
        };
    },

    /**
     * Determine current phase based on state.
     */
    getCurrentPhase(code, isStarterCode, elapsedSecs) {
        if (this.lastRunResult?.passed === this.lastRunResult?.total && this.lastRunResult?.total > 0) {
            return 'COMPLETE';
        }
        if (this.lastRunResult?.error) {
            return 'DEBUG';
        }
        if (this.lastRunResult?.passed > 0) {
            return 'PARTIAL';
        }
        if (this.lastRunResult && this.lastRunResult.passed === 0) {
            return 'WRONG_OUTPUT';
        }
        if (!isStarterCode) {
            return 'CODING';
        }
        if (isStarterCode && elapsedSecs < 60) {
            return 'START';
        }
        return 'STUCK';
    },

    /**
     * Analyze the code and provide observations for the avatar.
     */
    analyzeCode(code) {
        const observations = [];

        if (!code || code.trim() === '' || code === this.currentProblem.starterCode[this.language]) {
            observations.push('Candidate has not started coding yet');
            return observations;
        }

        // Check for common patterns
        if (this.language === 'python') {
            if (code.includes('for') && code.match(/for.*:[\s\S]*for.*:/)) {
                observations.push('Using nested loops (O(n²) approach)');
            }
            if (code.includes('dict(') || code.includes('= {}') || code.includes(': {')) {
                observations.push('Using a dictionary/hash map (good for O(n) solution)');
            }
            if (code.includes('enumerate')) {
                observations.push('Using enumerate for index tracking');
            }
            if (code.includes('return') && !code.includes('pass')) {
                observations.push('Has a return statement');
            }
            if (code.includes('pass') && !code.includes('return [')) {
                observations.push('Still has placeholder pass statement');
            }
        } else if (this.language === 'javascript') {
            if (code.includes('for') && code.match(/for.*\{[\s\S]*for.*\{/)) {
                observations.push('Using nested loops (O(n²) approach)');
            }
            if (code.includes('new Map') || code.includes('= {}') || code.includes('Object.')) {
                observations.push('Using an object/Map (good for O(n) solution)');
            }
            if (code.includes('return') && !code.includes('return;')) {
                observations.push('Has a return statement with value');
            }
        }

        if (observations.length === 0) {
            observations.push('Code in progress');
        }

        return observations;
    }
};

// =============================================================================
// UI ELEMENTS
// =============================================================================

let ui = {};

function initUI() {
    ui = {
        // Screens
        openingScreen: document.getElementById('opening-screen'),
        interviewScreen: document.getElementById('interview-screen'),
        endScreen: document.getElementById('end-screen'),

        // Registration form
        registrationForm: document.getElementById('registration-form'),
        firstNameInput: document.getElementById('first-name'),
        lastNameInput: document.getElementById('last-name'),
        emailInput: document.getElementById('email'),
        firstNameError: document.getElementById('first-name-error'),
        lastNameError: document.getElementById('last-name-error'),
        emailError: document.getElementById('email-error'),
        startBtn: document.getElementById('start-btn'),

        // Interview screen
        avatarContainer: document.getElementById('avatar-container'),
        loadingState: document.getElementById('loading-state'),
        statusDot: document.getElementById('status-dot'),
        statusText: document.getElementById('status-text'),
        userBadge: document.getElementById('user-badge'),
        editorContainer: document.getElementById('editor-container'),
        languageSelect: document.getElementById('language-select'),
        runBtn: document.getElementById('run-btn'),
        resetBtn: document.getElementById('reset-btn'),
        outputContent: document.getElementById('output-content'),
        testResults: document.getElementById('test-results'),
        debugDpp: document.getElementById('debug-dpp'),
        problemTitle: document.getElementById('problem-title'),
        problemDifficulty: document.getElementById('problem-difficulty'),
        problemDescription: document.getElementById('problem-description'),

        // End screen
        endUserName: document.getElementById('end-user-name'),
        reportContent: document.getElementById('report-content'),
        downloadReportBtn: document.getElementById('download-report-btn'),
        restartBtn: document.getElementById('restart-btn')
    };
}

// =============================================================================
// MONACO EDITOR
// =============================================================================

function initMonaco() {
    return new Promise((resolve) => {
        require.config({
            paths: {
                vs: 'https://cdn.jsdelivr.net/npm/monaco-editor@0.45.0/min/vs'
            }
        });

        require(['vs/editor/editor.main'], function() {
            state.editor = monaco.editor.create(ui.editorContainer, {
                value: state.currentProblem.starterCode[state.language],
                language: state.language,
                theme: 'vs-dark',
                fontSize: 14,
                fontFamily: "'Fira Code', monospace",
                minimap: { enabled: false },
                lineNumbers: 'on',
                scrollBeyondLastLine: false,
                automaticLayout: true,
                tabSize: 4,
                insertSpaces: true,
                wordWrap: 'on'
            });

            state.editor.onDidChangeModelContent(() => {
                onCodeChange();
            });

            resolve();
        });
    });
}

// =============================================================================
// AVATAR SDK
// =============================================================================

function initSDK() {
    state.sdk = new KalturaAvatarSDK({
        clientId: CONFIG.CLIENT_ID,
        flowId: CONFIG.FLOW_ID,
        container: '#avatar-container'
    });

    state.sdk.on('stateChange', ({ from, to }) => {
        updateStatus(to);
    });

    // Inject initial DPP when avatar becomes visible (SHOWING_AGENT)
    // This is the proper time to inject - the avatar is ready to receive context
    state.sdk.on(KalturaAvatarSDK.Events.SHOWING_AGENT, () => {
        console.log('[SDK] Avatar visible - scheduling initial DPP injection');
        setTimeout(() => {
            console.log('[SDK] Injecting initial DPP after SHOWING_AGENT delay');
            injectDPP('initial');
            startCodeTracking();
        }, CONFIG.DPP_INJECTION_DELAY_MS);
    });

    state.sdk.on(KalturaAvatarSDK.Events.AGENT_TALKED, (data) => {
        const text = data?.agentContent || data;
        console.log(`${CONFIG.AVATAR_NAME}:`, text);

        // Check if avatar is suggesting to move to next problem
        if (typeof text === 'string') {
            checkForProblemSwitchTrigger(text);
        }
    });

    state.sdk.on(KalturaAvatarSDK.Events.USER_TRANSCRIPTION, (data) => {
        const text = data?.userTranscription || data;
        console.log('User:', text);
    });

    state.sdk.on(KalturaAvatarSDK.Events.CONVERSATION_ENDED, () => {
        console.log('SDK CONVERSATION_ENDED event received');
        handleSessionEnd();
    });

    state.sdk.on('error', ({ message }) => {
        console.error('SDK Error:', message);
        updateStatus('error');
    });
}

async function startAvatar() {
    try {
        updateStatus('connecting');
        // Start the SDK - initial DPP injection happens on SHOWING_AGENT event
        await state.sdk.start();

        ui.loadingState.style.display = 'none';
        state.sessionStartTime = Date.now();
    } catch (error) {
        console.error('Failed to start avatar:', error);
        updateStatus('error');
    }
}

// =============================================================================
// DPP INJECTION
// =============================================================================

/**
 * Validate DPP consistency before injection.
 * Catches corrupted state early and logs warnings.
 */
function validateDPP(dpp) {
    const warnings = [];

    // Validate problem consistency
    if (dpp.session.current_problem !== state.currentProblemIndex + 1) {
        warnings.push(`Problem mismatch: DPP=${dpp.session.current_problem}, state=${state.currentProblemIndex + 1}`);
    }

    // Validate phase/completion consistency
    if (dpp.session.phase === 'COMPLETE' && !dpp.session.problem_completed) {
        warnings.push('Phase=COMPLETE but problem_completed=false');
    }

    // Validate last_execution alignment with state
    if (dpp.last_execution && !state.lastRunResult) {
        warnings.push('DPP has last_execution but state.lastRunResult is null');
    }

    // Log warnings but don't block injection
    if (warnings.length > 0) {
        console.warn('[DPP Validation]', warnings);
    }

    return warnings.length === 0;
}

function injectDPP(reason = 'update') {
    if (!state.sdk) return;

    const dpp = state.buildDPP();

    // Validate before injection (warn but don't block)
    validateDPP(dpp);

    const json = JSON.stringify(dpp);

    state.sdk.injectPrompt(json);
    state.lastInjectedCode = dpp.live_code.current_code;
    state.lastInjectedProblemId = state.currentProblem?.id; // Track problem for change detection
    state.lastInjectionTime = Date.now();

    if (ui.debugDpp) {
        ui.debugDpp.textContent = `// Injection reason: ${reason}\n// Time: ${new Date().toLocaleTimeString()}\n\n${JSON.stringify(dpp, null, 2)}`;
    }

    console.log(`[DPP] Injected (${reason}):`, {
        problem: dpp.session.problem_id,
        phase: dpp.session.phase,
        code_lines: dpp.live_code.line_count,
        elapsed_secs: dpp.session.elapsed_seconds
    });
}

// =============================================================================
// CODE CHANGE TRACKING
// =============================================================================

function onCodeChange() {
    if (state.debounceTimer) {
        clearTimeout(state.debounceTimer);
    }

    state.debounceTimer = setTimeout(() => {
        const currentCode = state.editor?.getValue() || '';
        const currentProblemId = state.currentProblem?.id;
        // Inject if code changed OR problem changed (handles identical starter code)
        const codeChanged = currentCode !== state.lastInjectedCode;
        const problemChanged = currentProblemId !== state.lastInjectedProblemId;

        if (codeChanged || problemChanged) {
            injectDPP('code_change');
        }
    }, CONFIG.DEBOUNCE_MS);
}

function startCodeTracking() {
    stopCodeTracking();

    state.intervalTimer = setInterval(() => {
        const currentCode = state.editor?.getValue() || '';
        const currentProblemId = state.currentProblem?.id;
        const timeSinceLastInjection = Date.now() - state.lastInjectionTime;

        // Inject if code or problem changed and enough time has passed
        const codeChanged = currentCode !== state.lastInjectedCode;
        const problemChanged = currentProblemId !== state.lastInjectedProblemId;

        if ((codeChanged || problemChanged) && timeSinceLastInjection >= CONFIG.MAX_INTERVAL_MS) {
            injectDPP('interval');
        }
    }, CONFIG.MAX_INTERVAL_MS);
}

function stopCodeTracking() {
    if (state.debounceTimer) {
        clearTimeout(state.debounceTimer);
        state.debounceTimer = null;
    }
    if (state.intervalTimer) {
        clearInterval(state.intervalTimer);
        state.intervalTimer = null;
    }
}

// =============================================================================
// CODE EXECUTION (SIMULATED)
// =============================================================================

function runCode() {
    const code = state.editor.getValue();
    state.runCount++;

    ui.outputContent.textContent = 'Running...';
    ui.outputContent.className = 'output-content';
    ui.testResults.textContent = '';

    setTimeout(() => {
        try {
            const result = simulateExecution(code, state.language);
            state.lastRunResult = result;

            if (result.error) {
                ui.outputContent.textContent = `Error: ${result.error}`;
                ui.outputContent.className = 'output-content error';
                ui.testResults.textContent = 'Error';
                ui.testResults.className = 'test-results fail';
            } else {
                ui.outputContent.textContent = result.output;
                ui.outputContent.className = result.passed === result.total
                    ? 'output-content success'
                    : 'output-content';
                ui.testResults.textContent = `${result.passed}/${result.total} tests passed`;
                ui.testResults.className = result.passed === result.total
                    ? 'test-results pass'
                    : 'test-results fail';

            }

            injectDPP('code_run');

        } catch (e) {
            state.lastRunResult = {
                timestamp: new Date().toISOString(),
                error: e.message,
                passed: 0,
                failed: 0,
                total: 0,
                output: null
            };
            ui.outputContent.textContent = `Execution error: ${e.message}`;
            ui.outputContent.className = 'output-content error';
            injectDPP('code_error');
        }
    }, 500);
}

/**
 * DEMO MODE: Simulates code execution using pattern matching.
 *
 * This is NOT real code execution - it detects common patterns in the code
 * to simulate pass/fail results based on the current problem.
 * For production, replace with actual code execution via a sandboxed backend.
 */
function simulateExecution(code, language) {
    const testCases = state.currentProblem.testCases;
    const problemId = state.currentProblem.id;

    // Check for empty/placeholder code
    if (language === 'python' && code.includes('pass') && !code.includes('return')) {
        return makeResult(0, testCases.length, 'Output: None');
    }
    if (language === 'javascript' && !code.includes('return')) {
        return makeResult(0, testCases.length, 'Output: undefined');
    }

    // Problem-specific pattern matching
    switch (problemId) {
        case 'two-sum':
            return simulateTwoSum(code, language, testCases);
        case 'valid-palindrome':
            return simulatePalindrome(code, language, testCases);
        case 'reverse-linked-list':
            return simulateReverseList(code, language, testCases);
        case 'fizz-buzz':
            return simulateFizzBuzz(code, language, testCases);
        default:
            return makeResult(0, testCases.length, 'Unknown problem type.');
    }
}

function makeResult(passed, total, output, error = null) {
    return {
        timestamp: new Date().toISOString(),
        error: error,
        passed: passed,
        failed: total - passed,
        total: total,
        output: output
    };
}

function simulateTwoSum(code, language, testCases) {
    const hasHashMap = language === 'python'
        ? (code.includes('dict') || code.includes('{}') || code.includes('enumerate'))
        : (code.includes('Map') || code.includes('{}') || code.includes('Object'));
    const hasNestedLoops = code.match(/for.*for/s);
    const hasLoop = code.includes('for') || code.includes('while');

    if (hasHashMap || hasNestedLoops) {
        return makeResult(testCases.length, testCases.length,
            'Output: [0, 1]\n\nAll test cases passed!');
    }
    if (hasLoop) {
        return makeResult(1, testCases.length,
            'Output: [0, 1]\n\nPartial success - 1 test case passed.');
    }
    return makeResult(0, testCases.length, 'No output produced.');
}

function simulatePalindrome(code, language, testCases) {
    const hasStringClean = language === 'python'
        ? (code.includes('isalnum') || code.includes('lower'))
        : (code.includes('replace') || code.includes('toLowerCase') || code.includes('match'));
    const hasReverse = language === 'python'
        ? (code.includes('[::-1]') || code.includes('reversed'))
        : (code.includes('.reverse()') || code.includes('split') && code.includes('join'));
    const hasTwoPointer = code.includes('left') && code.includes('right') || code.includes('i]') && code.includes('j]');

    if ((hasStringClean && hasReverse) || (hasStringClean && hasTwoPointer)) {
        return makeResult(testCases.length, testCases.length,
            'Output: True\n\nAll test cases passed!');
    }
    if (hasStringClean || hasReverse) {
        return makeResult(2, testCases.length,
            'Output: True\n\nPartial success - 2 test cases passed.');
    }
    if (code.includes('==') || code.includes('return')) {
        return makeResult(1, testCases.length,
            'Output: False\n\nPartial success - 1 test case passed.');
    }
    return makeResult(0, testCases.length, 'No output produced.');
}

function simulateReverseList(code, language, testCases) {
    const hasThreePointers = (code.includes('prev') && code.includes('curr')) ||
                            (code.includes('previous') && code.includes('current'));
    const hasNextSave = code.includes('next') || code.includes('temp');
    const hasRecursion = code.includes('reverse') && (code.includes('head.next') || code.includes('node.next'));

    if ((hasThreePointers && hasNextSave) || hasRecursion) {
        return makeResult(testCases.length, testCases.length,
            'Output: [5, 4, 3, 2, 1]\n\nAll test cases passed!');
    }
    if (hasThreePointers || hasNextSave) {
        return makeResult(1, testCases.length,
            'Output: [5, 4, 3, 2, 1]\n\nPartial success - 1 test case passed.');
    }
    if (code.includes('while') || code.includes('for')) {
        return makeResult(1, testCases.length,
            'Output: None\n\nPartial success - 1 test case passed.');
    }
    return makeResult(0, testCases.length, 'No output produced.');
}

function simulateFizzBuzz(code, language, testCases) {
    const hasModulo = code.includes('%');
    const hasThreeCheck = code.includes('3');
    const hasFiveCheck = code.includes('5');
    const hasFifteenCheck = code.includes('15') || (code.includes('3') && code.includes('5') && code.match(/%\s*3.*%\s*5|%\s*5.*%\s*3/s));
    const hasFizzBuzz = code.includes('FizzBuzz') || code.includes('Fizz') && code.includes('Buzz');

    if (hasModulo && hasFizzBuzz && hasThreeCheck && hasFiveCheck) {
        if (hasFifteenCheck || code.includes('elif') || code.includes('else if')) {
            return makeResult(testCases.length, testCases.length,
                'Output: ["1","2","Fizz","4","Buzz",...,"FizzBuzz"]\n\nAll test cases passed!');
        }
        return makeResult(2, testCases.length,
            'Output: ["1","2","Fizz","4","Buzz",...]\n\nPartial success - 2 test cases passed.');
    }
    if (hasModulo && (hasThreeCheck || hasFiveCheck)) {
        return makeResult(1, testCases.length,
            'Output: ["1","2","Fizz",...]\n\nPartial success - 1 test case passed.');
    }
    return makeResult(0, testCases.length, 'No output produced.');
}

// =============================================================================
// SESSION ANALYSIS
// =============================================================================

/**
 * Handle session end - called by SDK event or trigger phrase detection.
 * Prevents duplicate processing if already ending.
 */
async function handleSessionEnd() {
    // Prevent duplicate calls
    if (state.isEndingSession || state.currentScreen === 'end') {
        console.log('Session end already in progress, ignoring');
        return;
    }
    state.isEndingSession = true;

    console.log('Handling session end...');

    if (!state.sdk) {
        state.isEndingSession = false;
        return;
    }

    stopCodeTracking();
    updateStatus('ending...');

    // IMPORTANT: Get transcript BEFORE stopping SDK (stop may clear it)
    const transcript = state.sdk.getTranscript();
    const dpp = state.buildDPP();

    console.log('Captured transcript with', transcript?.length || 0, 'entries');

    // Switch to end screen first (so user sees immediate feedback)
    switchScreen('end');

    // Update end screen header with user name
    if (ui.endUserName) {
        ui.endUserName.textContent = `Thank you, ${state.user.firstName}!`;
    }

    // Show analyzing state in report area
    if (ui.reportContent) {
        ui.reportContent.innerHTML = `
            <div class="analyzing-state">
                <div class="spinner"></div>
                <p>Analyzing your session...</p>
            </div>
        `;
    }

    // Hide download button until report is ready
    if (ui.downloadReportBtn) {
        ui.downloadReportBtn.style.display = 'none';
    }

    // Stop the avatar: end() removes the iframe and stops all audio
    try {
        state.sdk.end();
        console.log('SDK ended successfully');
    } catch (e) {
        console.log('SDK end error:', e.message);
    }

    // Analyze with the transcript we captured earlier
    await analyzeSessionWithData(transcript, dpp);

    state.isEndingSession = false;
}

/**
 * Orchestrate iterative parallel analysis of the coding session.
 *
 * Phase 1: Fire N parallel per-problem Lambda calls (via callAnalysisAPI).
 * Phase 2: Send one synthesis call with all per-problem results.
 * Phase 3: Assemble the final v1.5 summary and display the report.
 *
 * Failed per-problem calls produce a neutral fallback entry (scores=3)
 * so the synthesis and report still render.
 *
 * @param {Array} transcript - SDK transcript entries ({role, text, timestamp})
 * @param {Object} dpp - Dynamic Page Prompt snapshot from session end
 */
async function analyzeSessionWithData(transcript, dpp) {
    if (!transcript?.length) {
        console.log('Empty transcript, skipping analysis');
        showSessionSummary(null);
        return;
    }

    console.log('Analyzing session with', transcript.length, 'transcript entries');

    try {
        const formattedTranscript = transcript.map(entry => ({
            role: entry.role === 'Avatar' ? 'assistant' : 'user',
            content: entry.text
        }));

        // Build list of problems attempted
        const problemsAttemptedList = state.completedProblems.map(id => ({
            id, title: PROBLEMS[id]?.title || id, difficulty: PROBLEMS[id]?.difficulty || 'unknown'
        }));
        if (!state.completedProblems.includes(state.currentProblem.id)) {
            problemsAttemptedList.push({
                id: state.currentProblem.id,
                title: state.currentProblem.title,
                difficulty: state.currentProblem.difficulty
            });
        }

        // Build lean DPP for API calls
        const leanDpp = {
            mode: dpp.mode,
            session: dpp.session,
            live_code: dpp.live_code ? { language: dpp.live_code.language } : undefined,
            candidate: {
                first_name: state.user.firstName,
                last_name: state.user.lastName,
                full_name: `${state.user.firstName} ${state.user.lastName}`.trim()
            },
            all_problems_in_session: PROBLEM_ORDER.map(id => ({
                id, title: PROBLEMS[id].title, difficulty: PROBLEMS[id].difficulty
            }))
        };

        // --- Phase 1: Analyze each problem in parallel ---
        updateAnalysisProgress('Analyzing problems...', 0, problemsAttemptedList.length);

        const perProblemPromises = problemsAttemptedList.map(problem =>
            callAnalysisAPI({
                analysis_mode: 'per_problem',
                transcript: formattedTranscript,
                problem_focus: problem,
                dpp: leanDpp
            })
        );

        const perProblemResults = await Promise.allSettled(perProblemPromises);

        // Collect successful results, log failures
        const problemAnalyses = [];
        perProblemResults.forEach((result, idx) => {
            const problem = problemsAttemptedList[idx];
            if (result.status === 'fulfilled' && result.value.success) {
                problemAnalyses.push(result.value.summary);
                console.log(`Problem "${problem.title}" analyzed:`, result.value.usage);
            } else {
                const error = result.status === 'rejected' ? result.reason : result.value?.error;
                console.warn(`Problem "${problem.title}" analysis failed:`, error);
                // Insert a minimal fallback entry
                problemAnalyses.push({
                    problem_id: problem.id,
                    problem_title: problem.title,
                    difficulty: problem.difficulty,
                    outcome: 'solved',
                    tests_passed: 0, tests_total: 0,
                    approach: 'other', approach_used: 'Analysis unavailable',
                    time_complexity: '?', space_complexity: '?',
                    optimal: false, time_spent_minutes: 0, hints_used: 0,
                    scores: { creativity: 3, logic: 3, code_quality: 3, explainability: 3, complexity: 3, scale: 3 },
                    eval_notes: 'Per-problem analysis failed; default scores applied.'
                });
            }
            updateAnalysisProgress('Analyzing problems...', idx + 1, problemsAttemptedList.length);
        });

        // --- Phase 2: Synthesize into overall assessment ---
        updateAnalysisProgress('Generating overall assessment...', null, null);

        const synthesisResult = await callAnalysisAPI({
            analysis_mode: 'synthesis',
            problem_results: problemAnalyses,
            dpp: leanDpp
        });

        if (!synthesisResult.success) {
            console.error('Synthesis failed:', synthesisResult.error);
            showSessionSummary(null, `Synthesis failed: ${synthesisResult.error}`);
            return;
        }

        console.log('Synthesis complete:', synthesisResult.usage);

        // --- Phase 3: Assemble final summary ---
        const synthesis = synthesisResult.summary;
        const finalSummary = {
            v: '1.5',
            mode: dpp.mode || 'coding_challenge',
            ctx: {
                org: '', problem_id: state.currentProblem.id,
                problem_title: state.currentProblem.title,
                difficulty: state.currentProblem.difficulty || '',
                language: dpp.live_code?.language || 'python',
                person: leanDpp.candidate.full_name, subj_id: ''
            },
            session_stats: {
                elapsed_minutes: dpp.session?.elapsed_minutes || 0,
                target_minutes: dpp.mtg?.mins || 0,
                times_code_run: dpp.session?.times_code_was_run || 0,
                hints_given: dpp.session?.hints_given || 0,
                tests_passed: dpp.last_execution?.tests_passed || 0,
                tests_total: dpp.last_execution?.total_tests || 0,
                problems_completed: state.completedProblems.length,
                problems_total: dpp.session?.total_problems || PROBLEM_ORDER.length
            },
            problems_attempted: problemAnalyses,
            debugging_history: [],
            turns: formattedTranscript.filter(t => t.role === 'user').length,
            overview: synthesis.overview || '',
            skill_assessment: synthesis.skill_assessment || {},
            potential_assessment: synthesis.potential_assessment || {},
            fit: synthesis.fit || {},
            strengths: synthesis.strengths || [],
            areas_for_improvement: synthesis.areas_for_improvement || [],
            gaps: synthesis.gaps || [],
            remaining_problems: [],
            cq: synthesis.cq || {},
            risk: synthesis.risk || { flags: ['none'], escalated: false, reason: '' },
            next_steps: synthesis.next_steps || [],
            final_code: state.editor?.getValue() || dpp.final_code || ''
        };

        state.lastSessionSummary = finalSummary;
        console.log('Session analysis complete (iterative):', finalSummary);
        showSessionSummary(finalSummary);

    } catch (error) {
        console.error('Failed to analyze session:', error);
        showSessionSummary(null, error.message);
    }
}

/**
 * Call the analysis API with retry logic.
 * Returns { success, summary, usage } or { success: false, error }.
 */
async function callAnalysisAPI(payload) {
    const MAX_RETRIES = 2;
    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
        try {
            if (attempt > 1) {
                await new Promise(r => setTimeout(r, 3000));
            }
            const response = await fetch(CONFIG.ANALYSIS_API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if ((response.status === 503 || response.status === 429) && attempt < MAX_RETRIES) {
                console.warn(`API returned ${response.status}, retrying...`);
                continue;
            }
            if (!response.ok) {
                return { success: false, error: `HTTP ${response.status}` };
            }
            return await response.json();
        } catch (err) {
            if (attempt === MAX_RETRIES) return { success: false, error: err.message };
        }
    }
    return { success: false, error: 'Max retries exceeded' };
}

/**
 * Update the analysis progress indicator in the report area.
 */
function updateAnalysisProgress(message, done, total) {
    if (!ui.reportContent) return;
    const progress = (done != null && total) ? ` (${done}/${total})` : '';
    ui.reportContent.innerHTML = `
        <div class="analyzing-state">
            <div class="spinner"></div>
            <p>${message}${progress}</p>
        </div>
    `;
}

function showSessionSummary(summary, errorDetail = null) {
    if (ui.downloadReportBtn) {
        ui.downloadReportBtn.style.display = summary ? 'inline-flex' : 'none';
    }

    if (!summary) {
        if (ui.reportContent) {
            ui.reportContent.innerHTML = `
                <div class="report-empty">
                    <p>Session ended. Analysis not available.</p>
                    ${errorDetail ? `<p style="font-size: 12px; color: var(--text-muted); margin-top: 8px;">${escapeHtml(errorDetail)}</p>` : ''}
                </div>`;
        }
        return;
    }

    summary.candidate = {
        first_name: state.user.firstName,
        last_name: state.user.lastName,
        full_name: `${state.user.firstName} ${state.user.lastName}`.trim(),
        email: state.user.email
    };

    if (!ui.reportContent) return;

    const fitScore = summary.fit?.score_0_100;
    const fitClass = fitScore >= 70 ? 'good' : fitScore >= 50 ? 'ok' : 'poor';
    const rec = summary.fit?.rec || 'N/A';
    const initials = `${state.user.firstName.charAt(0)}${state.user.lastName.charAt(0)}`.toUpperCase();
    const lastProblem = summary.problems_attempted?.length ? summary.problems_attempted[summary.problems_attempted.length - 1] : null;

    ui.reportContent.innerHTML = `
        <div class="report-sections">

            <!-- Candidate Header -->
            <div class="candidate-header">
                <div class="candidate-avatar">${initials}</div>
                <div class="candidate-info">
                    <h3>${escapeHtml(state.user.firstName)} ${escapeHtml(state.user.lastName)}</h3>
                    <p>${escapeHtml(state.user.email)}</p>
                </div>
            </div>

            <!-- Overview -->
            <div class="summary-section">
                <h4>Overview</h4>
                <p>${escapeHtml(summary.overview || 'No overview available')}</p>
            </div>

            <!-- Session Statistics -->
            ${summary.session_stats ? `
            <div class="summary-section">
                <h4>Session Statistics</h4>
                <div class="stats-grid">
                    <div class="stat-item"><span class="stat-label">Duration:</span> <span class="stat-value">${summary.session_stats.elapsed_minutes ?? '?'}/${summary.session_stats.target_minutes ?? '?'} min</span></div>
                    <div class="stat-item"><span class="stat-label">Code Runs:</span> <span class="stat-value">${summary.session_stats.times_code_run ?? '?'}</span></div>
                    <div class="stat-item"><span class="stat-label">Hints Given:</span> <span class="stat-value">${summary.session_stats.hints_given ?? '?'}</span></div>
                    <div class="stat-item"><span class="stat-label">Tests Passed:</span> <span class="stat-value">${summary.session_stats.tests_passed ?? '?'}/${summary.session_stats.tests_total ?? '?'}</span></div>
                    <div class="stat-item"><span class="stat-label">Problems Completed:</span> <span class="stat-value">${summary.session_stats.problems_completed ?? '?'}/${summary.session_stats.problems_total ?? '?'}</span></div>
                </div>
            </div>
            ` : ''}

            <!-- Problems Attempted -->
            ${summary.problems_attempted?.length ? `
            <div class="summary-section">
                <h4>Problems Attempted (${summary.problems_attempted.length})</h4>
                <div class="problems-attempted">
                    ${summary.problems_attempted.map((p, idx) => `
                    <div class="problem-card ${p.outcome || ''}">
                        <div class="problem-card-header">
                            <span class="problem-number">#${idx + 1}</span>
                            <span class="problem-name">${escapeHtml(p.problem_title || 'Unknown')}</span>
                            <span class="difficulty-tag ${p.difficulty || ''}">${escapeHtml(p.difficulty || '?')}</span>
                            <span class="outcome-tag ${p.outcome || ''}">${escapeHtml(p.outcome || '?')}</span>
                        </div>
                        <div class="problem-card-body">
                            <div class="problem-stats">
                                <span>Tests: ${p.tests_passed ?? '?'}/${p.tests_total ?? '?'}</span>
                                <span>Time: ${p.time_spent_minutes ?? '?'}min</span>
                                <span>Hints: ${p.hints_used ?? 0}</span>
                            </div>
                            <div class="problem-complexity">
                                <span><strong>Time:</strong> ${escapeHtml(p.time_complexity || '?')}</span>
                                <span><strong>Space:</strong> ${escapeHtml(p.space_complexity || '?')}</span>
                                ${p.optimal ? '<span class="solution-badge optimal">Optimal</span>' : ''}
                            </div>
                            ${p.approach_used ? `<div class="problem-approach"><strong>Approach:</strong> ${escapeHtml(p.approach_used)}</div>` : ''}

                            ${p.scores ? `
                            <div class="problem-eval-section">
                                <div class="problem-eval-grid">
                                    <div class="eval-item"><span class="eval-label">Creativity</span><span class="eval-score">${p.scores.creativity}/5</span></div>
                                    <div class="eval-item"><span class="eval-label">Logic</span><span class="eval-score">${p.scores.logic}/5</span></div>
                                    <div class="eval-item"><span class="eval-label">Code Quality</span><span class="eval-score">${p.scores.code_quality}/5</span></div>
                                    <div class="eval-item"><span class="eval-label">Explainability</span><span class="eval-score">${p.scores.explainability}/5</span></div>
                                    <div class="eval-item"><span class="eval-label">Complexity</span><span class="eval-score">${p.scores.complexity}/5</span></div>
                                    <div class="eval-item"><span class="eval-label">Scale</span><span class="eval-score">${p.scores.scale}/5</span></div>
                                </div>
                                ${p.eval_notes ? `<div class="problem-eval-notes"><div class="eval-note">${escapeHtml(p.eval_notes)}</div></div>` : ''}
                            </div>
                            ` : ''}
                        </div>
                    </div>
                    `).join('')}
                </div>
            </div>
            ` : ''}

            <!-- Solution Analysis (from last problem) -->
            ${lastProblem ? `
            <div class="summary-section">
                <h4>Solution Analysis</h4>
                <div class="solution-overview">
                    <span class="solution-badge ${lastProblem.outcome === 'solved' ? 'solved' : 'unsolved'}">${lastProblem.outcome === 'solved' ? 'Solved' : 'Not Solved'}</span>
                    ${lastProblem.optimal ? '<span class="solution-badge optimal">Optimal</span>' : ''}
                    <span class="solution-badge approach">${escapeHtml((lastProblem.approach || '').replace(/_/g, ' '))}</span>
                </div>
                ${lastProblem.approach_used ? `<p class="approach-desc">${escapeHtml(lastProblem.approach_used)}</p>` : ''}
                <div class="complexity-summary">
                    <span><strong>Time:</strong> ${escapeHtml(lastProblem.time_complexity || '?')}</span>
                    <span><strong>Space:</strong> ${escapeHtml(lastProblem.space_complexity || '?')}</span>
                </div>
            </div>
            ` : ''}

            <!-- Debugging History -->
            ${summary.debugging_history?.length ? `
            <div class="summary-section">
                <h4>Debugging History</h4>
                <div class="debug-history">
                    ${summary.debugging_history.map(d => `
                    <div class="debug-entry ${d.resolved ? 'resolved' : 'unresolved'}">
                        <span class="debug-type">${escapeHtml(d.error_type || 'Error')}</span>
                        <span class="debug-desc">${escapeHtml(d.description || '')}</span>
                        <span class="${d.resolved ? 'check-pass' : 'check-fail'}">${d.resolved ? 'Resolved' : 'Unresolved'}</span>
                        ${d.hint_needed ? '<span class="hint-badge">Hint Needed</span>' : ''}
                    </div>
                    `).join('')}
                </div>
            </div>
            ` : ''}

            <!-- Technical Fit -->
            ${fitScore != null ? `
            <div class="summary-section">
                <h4>Technical Fit</h4>
                <div class="fit-score ${fitClass}">
                    <span class="score-value">${fitScore}</span>
                    <span class="score-label">/ 100</span>
                </div>
                <div class="recommendation">Recommendation: <strong>${rec.replace(/_/g, ' ')}</strong></div>
                ${summary.fit.conf ? `<div class="confidence-badge">Confidence: ${escapeHtml(summary.fit.conf)}</div>` : ''}
                ${summary.fit.rationale ? `<p class="fit-rationale">${escapeHtml(summary.fit.rationale)}</p>` : ''}
            </div>
            ` : ''}

            <!-- Skill Assessment -->
            ${summary.skill_assessment ? `
            <div class="summary-section">
                <h4>Skill Assessment</h4>
                <div class="skill-assessment-detail">
                    ${[
                        {name: 'Problem Solving', score: summary.skill_assessment.problem_solving, evidence: summary.skill_assessment.problem_solving_e},
                        {name: 'Code Fluency', score: summary.skill_assessment.code_fluency, evidence: summary.skill_assessment.code_fluency_e},
                        {name: 'Communication', score: summary.skill_assessment.communication, evidence: summary.skill_assessment.communication_e},
                        {name: 'Efficiency Awareness', score: summary.skill_assessment.efficiency_awareness, evidence: summary.skill_assessment.efficiency_awareness_e}
                    ].map(d => `
                    <div class="skill-item">
                        <div class="skill-header">
                            <span class="skill-name">${d.name}</span>
                            <span class="skill-score">${d.score ?? '?'}/5</span>
                        </div>
                        ${d.evidence ? `<p class="skill-evidence">${escapeHtml(d.evidence)}</p>` : ''}
                    </div>`).join('')}
                </div>
            </div>
            ` : ''}

            <!-- Potential Assessment -->
            ${summary.potential_assessment ? `
            <div class="summary-section">
                <h4>Potential Assessment</h4>
                <p class="potential-note">Evaluated independently of problem outcome</p>
                <div class="potential-dims">
                    ${[
                        {name: 'Creativity', score: summary.potential_assessment.creativity_score, analysis: summary.potential_assessment.creativity_a},
                        {name: 'Tenacity', score: summary.potential_assessment.tenacity_score, analysis: summary.potential_assessment.tenacity_a},
                        {name: 'Aptitude', score: summary.potential_assessment.aptitude_score, analysis: summary.potential_assessment.aptitude_a},
                        {name: 'Propensity', score: summary.potential_assessment.propensity_score, analysis: summary.potential_assessment.propensity_a}
                    ].map(d => `
                    <div class="potential-dim">
                        <div class="potential-header">
                            <span class="potential-name">${d.name}</span>
                            <span class="potential-score">${d.score || '?'}/5</span>
                        </div>
                        ${d.analysis ? `<p class="potential-analysis">${escapeHtml(d.analysis)}</p>` : ''}
                    </div>`).join('')}
                </div>
                ${summary.potential_assessment.talent_indicators?.length ? `
                <div class="talent-indicators">
                    <strong>Talent Indicators:</strong>
                    <ul>${summary.potential_assessment.talent_indicators.map(t => `<li>${escapeHtml(t)}</li>`).join('')}</ul>
                </div>` : ''}
                <div class="potential-meta">
                    <span class="badge potential-badge">${escapeHtml((summary.potential_assessment.potential_vs_performance || '').replace(/_/g, ' '))}</span>
                    <span class="badge growth-badge">Growth: ${escapeHtml((summary.potential_assessment.growth_trajectory || '').replace(/_/g, ' '))}</span>
                </div>
            </div>
            ` : ''}

            <!-- Strengths -->
            ${summary.strengths?.length ? `
            <div class="summary-section">
                <h4>Strengths</h4>
                <ul class="strengths-list">${summary.strengths.map(s => `<li>${escapeHtml(s)}</li>`).join('')}</ul>
            </div>` : ''}

            <!-- Areas for Improvement -->
            ${summary.areas_for_improvement?.length ? `
            <div class="summary-section">
                <h4>Areas for Improvement</h4>
                <ul class="improvement-list">${summary.areas_for_improvement.map(s => `<li>${escapeHtml(s)}</li>`).join('')}</ul>
            </div>` : ''}

            <!-- Gaps -->
            ${summary.gaps?.length ? `
            <div class="summary-section">
                <h4>Gaps & Follow-ups</h4>
                <div class="gaps-detail">
                    ${summary.gaps.map(g => `
                    <div class="gap-item">
                        <div class="gap-missing"><strong>Gap:</strong> ${escapeHtml(g.missing || '')}</div>
                        ${g.why_matters ? `<div class="gap-why"><strong>Why it matters:</strong> ${escapeHtml(g.why_matters)}</div>` : ''}
                        ${g.next_q ? `<div class="gap-followup"><strong>Follow-up:</strong> ${escapeHtml(g.next_q)}</div>` : ''}
                    </div>`).join('')}
                </div>
            </div>` : ''}

            <!-- Remaining Problems -->
            ${summary.remaining_problems?.length ? `
            <div class="summary-section">
                <h4>Remaining Problems</h4>
                <p class="section-note">Not attempted in this session</p>
                <ul class="remaining-list">
                    ${summary.remaining_problems.map(p => `
                    <li>
                        <span class="problem-name">${escapeHtml(p.problem_title || '')}</span>
                        <span class="difficulty-tag ${p.difficulty || ''}">${escapeHtml(p.difficulty || '')}</span>
                        ${p.reason_not_attempted ? `<span class="reason-tag">${escapeHtml(p.reason_not_attempted.replace(/_/g, ' '))}</span>` : ''}
                    </li>`).join('')}
                </ul>
            </div>` : ''}

            <!-- Next Steps -->
            ${summary.next_steps?.length ? `
            <div class="summary-section">
                <h4>Next Steps</h4>
                <ul class="next-steps-list">${summary.next_steps.map(s => `<li>${escapeHtml(s)}</li>`).join('')}</ul>
            </div>` : ''}

            <!-- Risk Flags -->
            ${summary.risk && (summary.risk.escalated || (summary.risk.flags?.length && !summary.risk.flags.includes('none'))) ? `
            <div class="summary-section">
                <h4>Risk Flags</h4>
                <div class="risk-section ${summary.risk.escalated ? 'escalated' : ''}">
                    ${summary.risk.flags?.filter(f => f !== 'none').map(f => `<span class="risk-flag">${escapeHtml(f.replace(/_/g, ' '))}</span>`).join('') || ''}
                    ${summary.risk.escalated ? '<span class="risk-escalated">Escalation Required</span>' : ''}
                    ${summary.risk.reason ? `<p class="risk-reason">${escapeHtml(summary.risk.reason)}</p>` : ''}
                </div>
            </div>` : ''}

            <!-- Session Quality -->
            <div class="summary-section">
                <h4>Session Quality</h4>
                <div class="cq-badges">
                    <span class="badge">Emotion: ${escapeHtml(summary.cq?.emo || 'unknown')}</span>
                    <span class="badge">Tone: ${escapeHtml(summary.cq?.tone || 'unknown')}</span>
                    <span class="badge">Engagement: ${escapeHtml(summary.cq?.eng || 'unknown')}</span>
                    ${summary.cq?.think_aloud !== undefined ? `<span class="badge">Think Aloud: ${summary.cq.think_aloud ? 'Yes' : 'No'}</span>` : ''}
                </div>
            </div>

            <!-- Debug -->
            <details class="summary-debug">
                <summary>Debug: Raw JSON Response</summary>
                <pre>${escapeHtml(JSON.stringify(summary, null, 2))}</pre>
            </details>
        </div>
    `;
}

function downloadSessionSummary() {
    if (!state.lastSessionSummary) return;

    // Include user info in the download
    const reportData = {
        ...state.lastSessionSummary,
        candidate: {
            first_name: state.user.firstName,
            last_name: state.user.lastName,
            full_name: `${state.user.firstName} ${state.user.lastName}`.trim(),
            email: state.user.email
        },
        generated_at: new Date().toISOString()
    };

    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;

    // Create filename with user name
    const safeName = `${state.user.firstName}-${state.user.lastName}`.toLowerCase().replace(/[^a-z0-9]/g, '-');
    a.download = `coding-interview-${safeName}-${Date.now()}.json`;

    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// =============================================================================
// UI UPDATES
// =============================================================================

function updateStatus(status) {
    ui.statusDot.className = 'status-dot';

    switch (status) {
        case 'connecting':
            ui.statusText.textContent = 'Connecting...';
            break;
        case 'ready':
            ui.statusDot.classList.add('connected');
            ui.statusText.textContent = 'Ready';
            break;
        case 'in-conversation':
            ui.statusDot.classList.add('connected');
            ui.statusText.textContent = 'In conversation';
            break;
        case 'ended':
            ui.statusText.textContent = 'Session ended';
            break;
        case 'analyzing...':
            ui.statusText.textContent = 'Analyzing...';
            break;
        case 'error':
            ui.statusDot.classList.add('error');
            ui.statusText.textContent = 'Connection error';
            break;
        default:
            ui.statusText.textContent = status;
    }
}

/**
 * Reset output panel and execution state to initial values.
 */
function resetOutputState() {
    state.lastRunResult = null;
    state.problemCompleted = false;
    ui.outputContent.textContent = 'Click "Run Code" to execute your solution...';
    ui.outputContent.className = 'output-content';
    ui.testResults.textContent = '';
}

function resetCode() {
    if (state.editor) {
        state.editor.setValue(state.currentProblem.starterCode[state.language]);
    }
    state.runCount = 0;
    resetOutputState();
    injectDPP('reset');
}

function onLanguageChange(newLanguage) {
    state.language = newLanguage;

    if (state.editor) {
        monaco.editor.setModelLanguage(state.editor.getModel(), newLanguage);
        state.editor.setValue(state.currentProblem.starterCode[newLanguage]);
    }

    resetOutputState();
    injectDPP('language_change');
}

/**
 * Switch to the next problem in the sequence.
 */
function switchToNextProblem() {
    const nextIndex = state.currentProblemIndex + 1;

    if (nextIndex >= PROBLEM_ORDER.length) {
        console.log('No more problems available');
        return false;
    }

    // Update problem state
    state.currentProblemIndex = nextIndex;
    state.currentProblem = PROBLEMS[PROBLEM_ORDER[nextIndex]];
    state.runCount = 0;
    state.hintsGiven = 0;

    // Update editor
    if (state.editor) {
        state.editor.setValue(state.currentProblem.starterCode[state.language]);
    }

    // Update UI
    updateProblemUI();
    resetOutputState();

    // Inject updated DPP
    injectDPP('problem_switch');

    console.log(`Switched to problem: ${state.currentProblem.title}`);
    return true;
}

/**
 * Update the problem display in the UI.
 */
function updateProblemUI() {
    const problem = state.currentProblem;

    if (ui.problemTitle) {
        ui.problemTitle.textContent = problem.title;
    }

    if (ui.problemDifficulty) {
        ui.problemDifficulty.textContent = problem.difficulty.charAt(0).toUpperCase() + problem.difficulty.slice(1);
        ui.problemDifficulty.className = `difficulty ${problem.difficulty}`;
    }

    if (ui.problemDescription) {
        const example = problem.examples[0];
        ui.problemDescription.innerHTML = `
            <p>${escapeHtml(problem.description).replace(/\n/g, '</p><p>')}</p>
            <div class="example">
                <strong>Example:</strong>
                <pre>Input: ${escapeHtml(example.input)}
Output: ${escapeHtml(example.output)}
Explanation: ${escapeHtml(example.explanation)}</pre>
            </div>
        `;
    }
}

/**
 * Check if the avatar's speech contains action triggers.
 * - "switching to the next challenge now" → switch to next problem
 * - "ending the session now" → end session and show analysis
 *
 * IMPORTANT: Stops code tracking during transitions to prevent race conditions
 * where stale DPP injections occur during state changes.
 */
function checkForProblemSwitchTrigger(text) {
    const lowerText = text.toLowerCase();

    // Trigger: switch to next problem
    if (lowerText.includes('switching to the next challenge now')) {
        const hasNext = state.currentProblemIndex + 1 < PROBLEM_ORDER.length;
        if (hasNext) {
            console.log('[Avatar] Switch trigger detected - suspending code tracking');
            stopCodeTracking(); // Prevent race conditions during transition
            setTimeout(() => {
                switchToNextProblem();
                startCodeTracking(); // Resume after state is fully updated
            }, 800); // Reduced from 1500ms to minimize race window
        }
        return;
    }

    // Trigger: end session
    if (lowerText.includes('ending the session now')) {
        console.log('[Avatar] End session trigger detected - suspending code tracking');
        stopCodeTracking(); // Prevent race conditions during transition
        setTimeout(() => {
            handleSessionEnd();
            // No need to restart tracking - session is ending
        }, 1500); // Reduced from 2000ms
    }
}

// Make switchToNextProblem available globally for avatar integration
window.switchToNextProblem = switchToNextProblem;

// =============================================================================
// EVENT LISTENERS
// =============================================================================

function attachEventListeners() {
    // Registration form
    ui.registrationForm?.addEventListener('submit', handleRegistrationSubmit);

    // Input validation on blur
    ui.firstNameInput?.addEventListener('blur', () => validateField('firstName'));
    ui.lastNameInput?.addEventListener('blur', () => validateField('lastName'));
    ui.emailInput?.addEventListener('blur', () => validateField('email'));

    // Clear errors on input
    ui.firstNameInput?.addEventListener('input', () => clearFieldError('firstName'));
    ui.lastNameInput?.addEventListener('input', () => clearFieldError('lastName'));
    ui.emailInput?.addEventListener('input', () => clearFieldError('email'));

    // Interview controls
    ui.runBtn?.addEventListener('click', runCode);
    ui.resetBtn?.addEventListener('click', resetCode);
    ui.languageSelect?.addEventListener('change', (e) => {
        onLanguageChange(e.target.value);
    });

    // End screen
    ui.downloadReportBtn?.addEventListener('click', downloadSessionSummary);
    ui.restartBtn?.addEventListener('click', restartInterview);
}

// =============================================================================
// SCREEN TRANSITIONS
// =============================================================================

/**
 * Switch to a different screen (opening, interview, end).
 */
function switchScreen(screenName) {
    state.currentScreen = screenName;

    // Hide all screens
    ui.openingScreen.style.display = 'none';
    ui.interviewScreen.style.display = 'none';
    ui.endScreen.style.display = 'none';

    // Show the target screen
    switch (screenName) {
        case 'opening':
            ui.openingScreen.style.display = 'flex';
            break;
        case 'interview':
            ui.interviewScreen.style.display = 'flex';
            break;
        case 'end':
            ui.endScreen.style.display = 'flex';
            break;
    }

    console.log(`[Screen] Switched to: ${screenName}`);
}

// =============================================================================
// FORM VALIDATION
// =============================================================================

/**
 * Validate a single field.
 */
function validateField(fieldName) {
    let value, errorEl, inputEl, isValid = true, errorMessage = '';

    switch (fieldName) {
        case 'firstName':
            value = ui.firstNameInput.value.trim();
            errorEl = ui.firstNameError;
            inputEl = ui.firstNameInput;
            if (!value) {
                isValid = false;
                errorMessage = 'First name is required';
            } else if (value.length < 2) {
                isValid = false;
                errorMessage = 'First name must be at least 2 characters';
            } else if (!/^[a-zA-Z\s\-']+$/.test(value)) {
                isValid = false;
                errorMessage = 'First name can only contain letters, spaces, hyphens, and apostrophes';
            }
            break;

        case 'lastName':
            value = ui.lastNameInput.value.trim();
            errorEl = ui.lastNameError;
            inputEl = ui.lastNameInput;
            if (!value) {
                isValid = false;
                errorMessage = 'Last name is required';
            } else if (value.length < 2) {
                isValid = false;
                errorMessage = 'Last name must be at least 2 characters';
            } else if (!/^[a-zA-Z\s\-']+$/.test(value)) {
                isValid = false;
                errorMessage = 'Last name can only contain letters, spaces, hyphens, and apostrophes';
            }
            break;

        case 'email':
            value = ui.emailInput.value.trim();
            errorEl = ui.emailError;
            inputEl = ui.emailInput;
            if (!value) {
                isValid = false;
                errorMessage = 'Email is required';
            } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value)) {
                isValid = false;
                errorMessage = 'Please enter a valid email address';
            }
            break;
    }

    // Update UI
    if (errorEl) {
        errorEl.textContent = errorMessage;
    }
    if (inputEl) {
        inputEl.classList.toggle('invalid', !isValid);
    }

    return isValid;
}

/**
 * Clear error for a single field.
 */
function clearFieldError(fieldName) {
    let errorEl, inputEl;

    switch (fieldName) {
        case 'firstName':
            errorEl = ui.firstNameError;
            inputEl = ui.firstNameInput;
            break;
        case 'lastName':
            errorEl = ui.lastNameError;
            inputEl = ui.lastNameInput;
            break;
        case 'email':
            errorEl = ui.emailError;
            inputEl = ui.emailInput;
            break;
    }

    if (errorEl) {
        errorEl.textContent = '';
    }
    if (inputEl) {
        inputEl.classList.remove('invalid');
    }
}

/**
 * Validate all fields and return overall validity.
 */
function validateAllFields() {
    const isFirstNameValid = validateField('firstName');
    const isLastNameValid = validateField('lastName');
    const isEmailValid = validateField('email');

    return isFirstNameValid && isLastNameValid && isEmailValid;
}

/**
 * Handle registration form submission.
 */
function handleRegistrationSubmit(e) {
    e.preventDefault();

    if (!validateAllFields()) {
        return;
    }

    // Store user info
    state.user.firstName = ui.firstNameInput.value.trim();
    state.user.lastName = ui.lastNameInput.value.trim();
    state.user.email = ui.emailInput.value.trim();

    console.log('[Registration] User registered:', state.user);

    // Start the interview
    startInterview();
}

/**
 * Start the interview session.
 */
async function startInterview() {
    // Switch to interview screen
    switchScreen('interview');

    // Update user badge
    if (ui.userBadge) {
        ui.userBadge.textContent = `${state.user.firstName} ${state.user.lastName}`;
    }

    // Initialize Monaco editor if not already done
    if (!state.editor) {
        await initMonaco();
    }

    // Start the avatar
    await startAvatar();
}

/**
 * Restart the interview (from end screen).
 */
function restartInterview() {
    // Reset all state
    resetFullState();

    // Switch back to opening screen
    switchScreen('opening');

    // Clear form fields
    if (ui.firstNameInput) ui.firstNameInput.value = '';
    if (ui.lastNameInput) ui.lastNameInput.value = '';
    if (ui.emailInput) ui.emailInput.value = '';

    // Clear any validation errors
    clearFieldError('firstName');
    clearFieldError('lastName');
    clearFieldError('email');
}

/**
 * Reset all application state for a new session.
 */
function resetFullState() {
    // Stop any running timers
    stopCodeTracking();

    // Reset user
    state.user = { firstName: '', lastName: '', email: '' };

    // Destroy previous SDK instance (cleanup listeners, remove iframe)
    if (state.sdk) {
        try { state.sdk.destroy(); } catch (e) { /* ignore */ }
    }
    state.sdk = null;

    // Reset problem tracking
    state.currentProblemIndex = 0;
    state.currentProblem = PROBLEMS['two-sum'];
    state.completedProblems = [];

    // Reset language
    state.language = 'python';

    // Reset timing
    state.sessionStartTime = null;

    // Reset code injection
    state.lastInjectedCode = '';
    state.lastInjectionTime = 0;
    state.debounceTimer = null;
    state.intervalTimer = null;

    // Reset run results
    state.lastRunResult = null;
    state.runCount = 0;

    // Reset hints
    state.hintsGiven = 0;

    // Reset completion
    state.problemCompleted = false;

    // Reset analysis
    state.lastSessionSummary = null;

    // Reset session end guard
    state.isEndingSession = false;

    // Reset editor content if exists
    if (state.editor) {
        state.editor.setValue(state.currentProblem.starterCode[state.language]);
    }

    // Reset UI elements
    if (ui.outputContent) {
        ui.outputContent.textContent = 'Click "Run Code" to execute your solution...';
        ui.outputContent.className = 'output-content';
    }
    if (ui.testResults) {
        ui.testResults.textContent = '';
    }
    // Reset problem UI
    updateProblemUI();

    // Re-initialize SDK
    initSDK();

    console.log('[State] Full state reset');
}

// =============================================================================
// INITIALIZATION
// =============================================================================

async function init() {
    initUI();
    attachEventListeners();

    // Load custom summary prompt for detailed analysis
    await loadSummaryPrompt();

    // Show opening screen
    switchScreen('opening');

    // Pre-initialize SDK (but don't start yet)
    initSDK();
}

/**
 * Load the custom summary prompt from file.
 * This enables detailed code interview analysis with per-problem evaluations.
 */
async function loadSummaryPrompt() {
    try {
        const response = await fetch(CONFIG.SUMMARY_PROMPT_PATH);
        if (response.ok) {
            state.summaryPrompt = await response.text();
            console.log('Summary prompt loaded successfully');
        } else {
            console.warn('Could not load summary prompt, using API default');
        }
    } catch (error) {
        console.warn('Failed to load summary prompt:', error);
    }
}

document.addEventListener('DOMContentLoaded', init);
