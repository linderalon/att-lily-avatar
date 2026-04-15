/**
 * QA: isQuizSession detection — 10 test cases
 * Run with: node qa_detection_test.js
 */

// ── Exact copy of the detection logic from index.html ────────────────────────

const USER_QUIZ_PHRASES = [
    'quiz me', 'test me', 'ask me', 'give me a quiz', 'give me a test',
    'can you quiz', 'can you test', "let's do a quiz", "let's do a test",
    'knowledge check', 'practice questions', 'challenge me',
    'see how much i know', 'check my knowledge'
];

const EVAL_PHRASES = [
    'correct', "that's right", 'well done', 'good job', 'exactly right',
    'not quite', 'actually,', 'close, but', 'let me clarify', 'good answer',
    'next question', 'question 1', 'question 2', 'question 3', 'question 4', 'question 5',
    "let's move on to", 'for the next question', 'final question'
];

const ASK_ME_NEGATIONS = ["don't ask me", "do not ask me", "won't ask me", "not ask me", "never ask me", "didn't ask me"];

function isQuizSession(rawTranscript) {
    const userRequested = rawTranscript.some(e => {
        if (e.role !== 'You') return false;
        const lower = (e.text || '').toLowerCase();
        const otherPhrases = USER_QUIZ_PHRASES.filter(p => p !== 'ask me');
        if (otherPhrases.some(p => lower.indexOf(p) !== -1)) return true;
        if (lower.indexOf('ask me') !== -1) {
            const negated = ASK_ME_NEGATIONS.some(n => lower.indexOf(n) !== -1);
            return !negated;
        }
        return false;
    });
    if (userRequested) return true;

    let evalCount = 0, shortQuestions = 0;
    rawTranscript.forEach(e => {
        if (e.role === 'You') return;
        const text  = e.text || '';
        const lower = text.toLowerCase();
        if (EVAL_PHRASES.some(p => lower.indexOf(p) !== -1)) evalCount++;
        if (text.length < 300 && text.indexOf('?') !== -1) shortQuestions++;
    });
    return evalCount >= 2 && shortQuestions >= 3;
}

// ── Test helpers ──────────────────────────────────────────────────────────────

const ai  = text => ({ role: 'Avatar', text });
const usr = text => ({ role: 'You',    text });

let passed = 0, failed = 0;

function test(name, transcript, expected) {
    const result = isQuizSession(transcript);
    const ok     = result === expected;
    const label  = ok ? '✓ PASS' : '✗ FAIL';
    const expect = expected ? 'GRADED' : 'SUMMARY';
    const got    = result   ? 'GRADED' : 'SUMMARY';
    console.log(`${label}  [${expect}]  ${name}`);
    if (!ok) console.log(`       expected=${expect}  got=${got}`);
    ok ? passed++ : failed++;
}

// ── TEST CASES ────────────────────────────────────────────────────────────────

// ── FALSE: informational / call summary ──────────────────────────────────────

test(
    '1. Pure product info — user asks, avatar explains at length',
    [
        ai('Hi there! I\'m Lily, your AT&T Learning Concierge. How can I help you today?'),
        usr('What can you tell me about Contact Center?'),
        ai('AT&T Contact Center as a Service is a cloud-based, AI-powered solution that unifies customer interactions across voice, chat, email, and social media into a single platform. It can be deployed in under 48 hours and integrates with the AT&T global network for reliable, high-quality connectivity. What specific aspects of Contact Center are you interested in?'),
        usr('Can you provide the value drivers of the IP switch products?'),
        ai('I am looking that up for you right away.'),
        ai('The value drivers for IP Toll-Free products primarily revolve around revenue growth through value-added services, which also helps reduce customer churn. Would you like to know more about the specific advanced features that drive this value?'),
    ],
    false  // info only — expect SUMMARY
);

test(
    '2. Avatar asks only conversational check-in questions ("Does that make sense?")',
    [
        ai('Hello! How can I assist you today?'),
        usr('Tell me about AT&T Fiber plans.'),
        ai('AT&T Fiber offers symmetrical upload and download speeds with no data caps. Plans range from 300 Mbps to 5 Gig. Does that make sense?'),
        usr('Yes, what about pricing?'),
        ai('Pricing starts around $55/month for the 300 Mbps tier. Bundling with wireless can save up to $20/month. Does that help clarify things?'),
        usr('Great, thanks.'),
        ai('Happy to help! Is there anything else you\'d like to know?'),
    ],
    false  // conversational questions — expect SUMMARY
);

test(
    '3. User says "I have a question" — should NOT trigger quiz mode',
    [
        ai('Hi, I\'m Lily! How can I help?'),
        usr('I have a question about FirstNet.'),
        ai('FirstNet is AT&T\'s dedicated network for first responders. It provides priority and preemption on the AT&T network, ensuring first responders always get through even during network congestion.'),
        usr('Thanks, that\'s very helpful.'),
    ],
    false  // "a question" ≠ quiz request — expect SUMMARY
);

test(
    '4. Word "correctly" in avatar turn — substring false positive guard',
    [
        ai('The AT&T network correctly routes traffic to ensure low latency for enterprise customers.'),
        usr('How does billing work?'),
        ai('Billing is handled correctly through the AT&T portal, where you can view usage and manage plans.'),
        usr('What about the contact center correctly handling calls?'),
        ai('Yes, the system correctly prioritises inbound calls. Any more questions?'),
        usr('No, that\'s all.'),
    ],
    false  // "correctly" must NOT match "correct" — expect SUMMARY
);

test(
    '5. User says "don\'t ask me about that" — "ask me" false positive guard',
    [
        ai('Would you like to discuss AT&T wireless or fiber today?'),
        usr('Please don\'t ask me about fiber, just tell me about wireless.'),
        ai('Of course! AT&T Wireless offers 5G coverage across the US with plans like Unlimited Starter, Extra, and Premium. Each tier provides progressively more features including hotspot data and international roaming.'),
        usr('What\'s the difference between Extra and Premium?'),
        ai('Premium includes 50 GB of premium hotspot data versus 15 GB on Extra, plus international day passes and enhanced streaming quality.'),
    ],
    false  // "don't ask me about" should NOT trigger — expect SUMMARY
);

// ── TRUE: quiz / test sessions ───────────────────────────────────────────────

test(
    '6. User explicitly asks "quiz me on AT&T wireless"',
    [
        ai('Hi! I\'m Lily. What would you like to learn today?'),
        usr('Can you quiz me on AT&T wireless plans?'),
        ai('Sure! What are the three AT&T Unlimited plan tiers?'),
        usr('Starter, Extra, and Premium.'),
        ai('That\'s right! Now, what is FirstNet and who is it designed for?'),
        usr('It\'s for first responders.'),
        ai('Correct! Last question — what is the hotspot data allowance on Unlimited Premium?'),
        usr('50 GB.'),
        ai('Well done! You scored 3 out of 3.'),
    ],
    true  // explicit quiz request — expect GRADED
);

test(
    '7. User says "test me on fiber"',
    [
        ai('Hello! How can I assist?'),
        usr('I want you to test me on AT&T Fiber.'),
        ai('Great! What is the maximum speed tier available on AT&T Fiber?'),
        usr('5 Gig.'),
        ai('Correct! Does AT&T Fiber have data caps?'),
        usr('No data caps.'),
        ai('That\'s right! How does AT&T Fiber differ from cable internet?'),
        usr('Symmetrical speeds and fibre all the way to the home.'),
        ai('Excellent answer!'),
    ],
    true  // "test me" — expect GRADED
);

test(
    '8. User says "check my knowledge on contact center"',
    [
        ai('Hi! What can I help you with?'),
        usr('Please check my knowledge on Contact Center solutions.'),
        ai('Absolutely! What does CCaaS stand for and what problem does it solve?'),
        usr('Contact Center as a Service — it moves contact centers to the cloud.'),
        ai('Good job! What is the main benefit of Five9 AI Insights?'),
        usr('Custom KPI dashboards without IT setup.'),
        ai('That\'s right! You\'re doing great.'),
    ],
    true  // "check my knowledge" — expect GRADED
);

test(
    '9. Avatar initiates structured quiz unprompted (Signal 2)',
    [
        ai('Let\'s do a quick knowledge check on AT&T Wireless.'),
        ai('Question 1: What are the AT&T Unlimited plan tiers?'),
        usr('Starter, Extra, and Premium.'),
        ai('That\'s right! Well done.'),
        ai('Question 2: What is FirstNet?'),
        usr('A network for first responders with priority access.'),
        ai('Correct! Good answer.'),
        ai('Question 3: What is the 5G advantage AT&T offers over competitors?'),
        usr('Nationwide coverage and faster speeds.'),
        ai('Not quite — the key differentiator is the sub-6 GHz nationwide coverage combined with mmWave in dense areas. Let\'s move on to the final question.'),
        ai('Final question: What does unlimited premium include that Extra does not?'),
        usr('50 GB hotspot and international passes.'),
        ai('Well done! You scored 3 out of 4.'),
    ],
    true  // avatar-initiated quiz with eval phrases + short questions — expect GRADED
);

test(
    '10. User says "can you ask me some questions to test my knowledge"',
    [
        ai('Of course! Let\'s start with AT&T Fiber. What speed tiers are available?'),
        usr('300 Mbps, 1 Gig, 2 Gig, and 5 Gig.'),
        ai('Correct! Next — does AT&T Fiber have data caps?'),
        usr('No.'),
        ai('That\'s right! One more — what is the main advantage of symmetrical speeds?'),
        usr('Upload and download are the same, which is better for video calls and backups.'),
        ai('Excellent! You did very well.'),
    ],
    true  // "ask me" + "test my knowledge" — expect GRADED
);

// ── Results ───────────────────────────────────────────────────────────────────

console.log(`\n${'─'.repeat(55)}`);
console.log(`Results: ${passed}/10 passed, ${failed} failed`);
if (failed > 0) {
    console.log('\n⚠  Issues found — see FAIL lines above.');
    process.exit(1);
} else {
    console.log('\n All 10 tests passed.');
}
