/**
 * Kaltura HR Avatar Demo
 * Interview, Post-Interview, and Separation Scenario Player
 *
 * This demo showcases the Kaltura Avatar SDK for HR use cases:
 * - Interview: Initial candidate phone screens
 * - Post-Interview: Offer calls and rejection feedback
 * - Separation: Layoff, performance, and misconduct meetings
 *
 * @version 1.0.16
 * @see dynamic_page_prompt.schema.json - DPP v2 schema
 * @see call_summary.schema.json - Analysis output schema v4.1
 */

// =============================================================================
// CONFIGURATION
// =============================================================================

/**
 * Application configuration constants.
 * Update these values for different deployments.
 */
const CONFIG = Object.freeze({
    // Version - bump when making changes to bust browser cache
    VERSION: '1.0.24',

    // Kaltura Avatar SDK credentials
    CLIENT_ID: '115767973963657880005',
    FLOW_ID: 'agent-15',

    // Call analysis API endpoint (local proxy → LiteLLM; run: python3 local_proxy.py)
    ANALYSIS_API_URL: 'http://localhost:8090',

    // Delay (ms) after SHOWING_AGENT before injecting DPP
    // This ensures the avatar is fully ready to receive context
    DPP_INJECTION_DELAY_MS: 500,

    // Avatar display name
    AVATAR_NAME: 'Nora (HR)',

    // PDF.js worker URL
    PDFJS_WORKER_URL: 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js'
});

// =============================================================================
// PDF.js INITIALIZATION
// =============================================================================

if (typeof pdfjsLib !== 'undefined') {
    pdfjsLib.GlobalWorkerOptions.workerSrc = CONFIG.PDFJS_WORKER_URL;
}

// =============================================================================
// SCENARIO DEFINITIONS
// =============================================================================

/**
 * Scenario configuration registry.
 * Each scenario defines metadata for display and the path to its DPP JSON file.
 *
 * To add a new scenario:
 * 1. Create the DPP JSON file in dynamic_page_prompt_samples/
 * 2. Add an entry here with: id, name, description, file, company, and type-specific fields
 * 3. The UI will automatically render the new scenario card
 *
 * @type {Object.<string, Array<Object>>}
 */
const SCENARIOS = Object.freeze({
    /**
     * Interview scenarios - Initial candidate phone screens
     * DPP mode: "interview"
     */
    interview: [
        {
            id: 'interview_amazon_driver',
            name: 'Amazon Delivery Driver',
            description: 'Experienced delivery driver interview',
            file: 'dynamic_page_prompt_samples/interview_amazon_experienced-delivery-driver.json',
            company: 'Amazon',
            role: 'Delivery Driver',
            location: 'Lisbon, PT',
            duration: '5 min'
        },
        {
            id: 'interview_acme_dispatch',
            name: 'Acme Dispatch Coordinator',
            description: 'Dispatch coordinator phone screen',
            file: 'dynamic_page_prompt_samples/interview_acme_logistics-dispenser.json',
            company: 'Acme Logistics',
            role: 'Dispatch Coordinator',
            location: 'Porto, PT',
            duration: '5 min'
        },
        {
            id: 'interview_mcdonalds_crew',
            name: 'McDonald\'s Crew Member',
            description: 'Entry-level crew member interview',
            file: 'dynamic_page_prompt_samples/interview_mcdonalds_crew-worker-entry-level.json',
            company: 'McDonald\'s',
            role: 'Crew Member',
            location: 'Orlando, FL',
            duration: '5 min'
        },
        {
            id: 'interview_aws_sde',
            name: 'AWS Software Engineer',
            description: 'Technical interview with code walkthrough',
            file: 'dynamic_page_prompt_samples/interview_aws_engineering-software-developer.json',
            company: 'AWS',
            role: 'Software Development Engineer',
            location: 'EMEA (Hybrid)',
            duration: '5 min'
        },
        {
            id: 'interview_amazon_ads_analyst',
            name: 'Amazon Ads Data Analyst',
            description: 'AI/Data analyst technical interview',
            file: 'dynamic_page_prompt_samples/interview_amazon_ads_engineering-ai-data-analyst.json',
            company: 'Amazon Ads',
            role: 'AI Specialist / Data Analyst',
            location: 'EMEA (Hybrid)',
            duration: '5 min'
        }
    ],

    /**
     * Post-Interview scenarios - Follow-up outcome calls
     * DPP mode: "post_interview"
     */
    postInterview: [
        {
            id: 'post_mcdonalds_offer',
            name: 'McDonald\'s Offer Call',
            description: 'Job offer and next steps',
            file: 'dynamic_page_prompt_samples/post-interview_mcdonalds_offer-call.json',
            company: 'McDonald\'s',
            role: 'Crew Member',
            type: 'Offer'
        },
        {
            id: 'post_mcdonalds_not_selected',
            name: 'McDonald\'s Not Selected',
            description: 'Candidate not selected feedback call',
            file: 'dynamic_page_prompt_samples/post-interview_mcdonalds_candidate_not_selected.json',
            company: 'McDonald\'s',
            role: 'Crew Member',
            type: 'Rejection'
        }
    ],

    /**
     * Separation scenarios - Employee termination meetings
     * DPP mode: "separation"
     */
    separation: [
        {
            id: 'separation_amazon_layoff',
            name: 'Amazon Layoff',
            description: 'Business restructure layoff',
            file: 'dynamic_page_prompt_samples/separation_redundancy_amazon_delivery-driver.json',
            company: 'Amazon',
            role: 'Delivery Driver',
            type: 'Layoff'
        },
        {
            id: 'separation_acme_misconduct',
            name: 'Acme Policy Violation',
            description: 'Termination for policy violation',
            file: 'dynamic_page_prompt_samples/separation_misconduct_acme-logistics_warehouse-associate.json',
            company: 'Acme Logistics',
            role: 'Warehouse Associate',
            type: 'Misconduct'
        },
        {
            id: 'separation_mcdonalds_attendance',
            name: 'McDonald\'s Attendance',
            description: 'Termination for attendance issues',
            file: 'dynamic_page_prompt_samples/separation_mcdonalds_performance-attendance.json',
            company: 'McDonald\'s',
            role: 'Crew Member',
            type: 'Performance'
        },
        {
            id: 'separation_mcdonalds_misconduct',
            name: 'McDonald\'s Policy Violation',
            description: 'Termination for policy violation',
            file: 'dynamic_page_prompt_samples/separation_mcdonalds_misconduct.json',
            company: 'McDonald\'s',
            role: 'Cashier',
            type: 'Misconduct'
        }
    ]
});

// =============================================================================
// APPLICATION STATE
// =============================================================================

/**
 * Application state container.
 * Centralizes all mutable state for easier debugging and maintenance.
 */
const state = {
    /** @type {KalturaAvatarSDK|null} SDK instance */
    sdk: null,

    /** @type {Object|null} Currently selected scenario metadata from SCENARIOS */
    currentScenario: null,

    /** @type {Object|null} Loaded DPP JSON data */
    scenarioData: null,

    /** @type {string|null} Extracted CV text content */
    cvText: null,

    /** @type {boolean} Whether a conversation is currently active */
    isConversationActive: false,

    /** @type {string|null} Scenario name preserved for download after reset */
    lastScenarioName: null,

    /** @type {Object|null} Last call summary from analysis API */
    lastCallSummary: null,

    /** @type {boolean} Flag to prevent duplicate end-of-call handling */
    isEndingCall: false,

    /**
     * User-edited field overrides.
     * These override the defaults from scenario JSON when building the prompt.
     * null means "use default from scenario"
     */
    editedFields: {
        candidate: null,   // subj.name (all modes)
        role: null,        // role.t (all modes)
        company: null,     // org.n (all modes)
        location: null,    // role.loc (interview only)
        focus: null,       // mtg.focus (interview only)
        effective: null    // case.eff (separation only)
    }
};

/**
 * Reset edited fields to defaults.
 */
function resetEditedFields() {
    state.editedFields = {
        candidate: null,
        role: null,
        company: null,
        location: null,
        focus: null,
        effective: null
    };
}

// =============================================================================
// DOM ELEMENT CACHE
// =============================================================================

/**
 * Cached DOM element references.
 * Populated on DOMContentLoaded for performance.
 * @type {Object.<string, HTMLElement>}
 */
let ui = {};

/**
 * Initialize DOM element cache.
 * Call once after DOM is ready.
 */
function initUI() {
    ui = {
        // Main containers
        avatarContainer: document.getElementById('avatar-container'),
        emptyState: document.getElementById('empty-state'),
        scenarioDetails: document.getElementById('scenario-details'),

        // CV Upload
        cvUploadPanel: document.getElementById('cv-upload-panel'),
        cvUploadArea: document.getElementById('cv-upload-area'),
        cvFileInput: document.getElementById('cv-file-input'),
        cvStatus: document.getElementById('cv-status'),
        cvFilename: document.getElementById('cv-filename'),
        cvRemoveBtn: document.getElementById('cv-remove-btn'),
        startInterviewBtn: document.getElementById('start-interview-btn'),

        // Transcript
        transcriptContent: document.getElementById('transcript-content'),
        transcriptEmpty: document.getElementById('transcript-empty'),

        // Status display
        statusValue: document.getElementById('status-value'),
        scenarioValue: document.getElementById('scenario-value'),

        // Download buttons
        downloadBtn: document.getElementById('download-btn'),
        downloadMdBtn: document.getElementById('download-md-btn')
    };
}

// =============================================================================
// UTILITIES
// =============================================================================

/**
 * Escape HTML entities to prevent XSS.
 * @param {*} text - Input to escape (converted to string)
 * @returns {string} HTML-safe string
 */
function escapeHtml(text) {
    if (text == null) return '';
    const div = document.createElement('div');
    div.textContent = String(text);
    return div.innerHTML;
}

/**
 * Safely get nested property from object.
 * @param {Object} obj - Source object
 * @param {string} path - Dot-separated property path
 * @param {*} defaultValue - Value to return if path not found
 * @returns {*} Property value or default
 */
function getNestedValue(obj, path, defaultValue = '') {
    return path.split('.').reduce((acc, key) => acc?.[key], obj) ?? defaultValue;
}

/**
 * Deep clone an object (JSON-safe only).
 * @param {Object} obj - Object to clone
 * @returns {Object} Cloned object
 */
function deepClone(obj) {
    return JSON.parse(JSON.stringify(obj));
}

/**
 * Generate a filename-safe slug from text.
 * @param {string} text - Input text
 * @returns {string} Lowercase, hyphenated slug
 */
function slugify(text) {
    return (text || 'untitled').toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9-]/g, '');
}

/**
 * Get today's date as YYYY-MM-DD.
 * @returns {string} ISO date string
 */
function getTodayISO() {
    return new Date().toISOString().slice(0, 10);
}

// =============================================================================
// SDK INITIALIZATION
// =============================================================================

/**
 * Initialize the Kaltura Avatar SDK with event handlers.
 */
function initSDK() {
    state.sdk = new KalturaAvatarSDK({
        clientId: CONFIG.CLIENT_ID,
        flowId: CONFIG.FLOW_ID,
        container: '#avatar-container'
    });

    // State change tracking
    state.sdk.on('stateChange', ({ from, to }) => {
        updateStatus(to);
        updateDownloadButtons();
    });

    // Inject DPP when avatar becomes visible (SHOWING_AGENT)
    // This is the proper time to inject - the avatar is ready to receive context
    state.sdk.on(KalturaAvatarSDK.Events.SHOWING_AGENT, () => {
        console.log('[SDK] Avatar visible - scheduling DPP injection');
        setTimeout(() => {
            const promptJson = buildDynamicPrompt();
            if (promptJson) {
                console.log('[SDK] Injecting DPP after SHOWING_AGENT delay');
                state.sdk.injectPrompt(promptJson);
            }
        }, CONFIG.DPP_INJECTION_DELAY_MS);
    });

    // Avatar speech -> transcript
    state.sdk.on(KalturaAvatarSDK.Events.AGENT_TALKED, (data) => {
        const text = data?.agentContent || (typeof data === 'string' ? data : null);
        if (text) {
            addTranscriptEntry('avatar', CONFIG.AVATAR_NAME, text);

            // Check for call-ending trigger phrases
            checkForCallEndTrigger(text);
        }
    });

    // User speech -> transcript
    state.sdk.on(KalturaAvatarSDK.Events.USER_TRANSCRIPTION, (data) => {
        const text = data?.userTranscription || (typeof data === 'string' ? data : null);
        if (text) {
            addTranscriptEntry('user', 'You', text);
        }
    });

    // Conversation ended -> use centralized handler
    state.sdk.on(KalturaAvatarSDK.Events.CONVERSATION_ENDED, async () => {
        console.log('[SDK] CONVERSATION_ENDED event received');
        await handleCallEnd();
    });

    // Error handling
    state.sdk.on('error', ({ message }) => {
        console.error('SDK Error:', message);
        updateStatus('error');
    });
}

// =============================================================================
// AVATAR SPEECH TRIGGERS
// =============================================================================

/**
 * Check if avatar speech contains call-ending trigger phrases.
 * When detected, triggers the end-of-call handler after a short delay.
 *
 * @param {string} text - Avatar speech text to check
 */
function checkForCallEndTrigger(text) {
    if (!text || typeof text !== 'string') return;

    const lowerText = text.toLowerCase();

    // Trigger phrases that indicate the avatar is ending the call
    const endCallPhrases = [
        'ending call now',
        'ending the call now',
        'end the call now',
        'ending this call',
        'ending our call',
        'conclude our call',
        'concluding the call',
        'wrapping up the call',
        'wrapping up now',
        'goodbye and take care',
        'thank you for your time today'
    ];

    const shouldEndCall = endCallPhrases.some(phrase => lowerText.includes(phrase));

    if (shouldEndCall && state.isConversationActive && !state.isEndingCall) {
        console.log('[Avatar] Call end trigger detected:', text.substring(0, 50) + '...');
        state.isEndingCall = true; // Prevent duplicate handling

        // Delay to allow avatar to finish speaking before triggering end handler
        setTimeout(async () => {
            console.log('[Avatar] Triggering end-of-call handler');
            await handleCallEnd();
        }, 2000);
    }
}

/**
 * Handle end of call - analyze and reset.
 * Centralized handler to prevent duplicate processing.
 */
async function handleCallEnd() {
    // Prevent duplicate calls
    if (!state.isConversationActive && !state.isEndingCall) {
        console.log('[handleCallEnd] Already ended, skipping');
        return;
    }

    console.log('[handleCallEnd] Processing end of call');
    showAnalyzingState();
    await analyzeCall();
    highlightDownloadButton();
    resetToInitialState();
    state.isEndingCall = false; // Reset flag after handling
}

// =============================================================================
// SCENARIO RENDERING
// =============================================================================

/**
 * Render all scenario cards in the sidebar.
 */
function renderScenarios() {
    const lists = {
        interview: document.getElementById('interview-list'),
        postInterview: document.getElementById('post-interview-list'),
        separation: document.getElementById('separation-list')
    };

    // Render each category
    Object.entries(SCENARIOS).forEach(([type, scenarios]) => {
        const listEl = lists[type];
        if (listEl) {
            listEl.innerHTML = scenarios.map(s => createScenarioCardHTML(s, type)).join('');
        }
    });

    // Attach click handlers using event delegation
    document.querySelectorAll('.scenario-list').forEach(list => {
        list.addEventListener('click', (e) => {
            const card = e.target.closest('.scenario-card');
            if (card) {
                selectScenario(card.dataset.id, card.dataset.type);
            }
        });
    });
}

/**
 * Create HTML for a scenario card.
 * @param {Object} scenario - Scenario configuration
 * @param {string} type - Category: 'interview', 'postInterview', 'separation'
 * @returns {string} HTML string
 */
function createScenarioCardHTML(scenario, type) {
    const meta = type === 'interview'
        ? `<span>${escapeHtml(scenario.company)}</span><span>${escapeHtml(scenario.duration)}</span>`
        : `<span>${escapeHtml(scenario.company)}</span><span>${escapeHtml(scenario.type)}</span>`;

    // CSS class needs hyphen for post-interview
    const cssClass = type === 'postInterview' ? 'post-interview' : type;

    return `
        <div class="scenario-card ${cssClass}" data-id="${escapeHtml(scenario.id)}" data-type="${type}">
            <h3>${escapeHtml(scenario.name)}</h3>
            <p>${escapeHtml(scenario.description)}</p>
            <div class="meta">${meta}</div>
        </div>
    `;
}

// =============================================================================
// SCENARIO SELECTION & DETAILS
// =============================================================================

/**
 * Handle scenario selection.
 * @param {string} id - Scenario ID
 * @param {string} type - Category type
 */
async function selectScenario(id, type) {
    // Confirm if switching from active conversation
    if (state.sdk?.getState() === 'in-conversation') {
        if (!confirm('End the current conversation and start a new scenario?')) {
            return;
        }
        state.sdk.end();
    }

    // Reset state - fully clean up to prevent context leakage between scenarios
    state.isConversationActive = false;
    state.lastCallSummary = null;      // Clear analysis from previous call
    state.lastScenarioName = null;     // Clear scenario name
    resetEditedFields();
    clearCV();

    // Update UI selection
    document.querySelectorAll('.scenario-card').forEach(card => {
        card.classList.toggle('active', card.dataset.id === id);
    });

    // Find and load scenario
    const scenarioList = SCENARIOS[type];
    state.currentScenario = scenarioList?.find(s => s.id === id);

    if (!state.currentScenario) {
        console.error(`Scenario not found: ${id}`);
        return;
    }

    try {
        const url = `${state.currentScenario.file}?v=${CONFIG.VERSION}`;
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        state.scenarioData = await response.json();

        showScenarioDetails();
        ui.scenarioValue.textContent = state.currentScenario.name;
    } catch (error) {
        console.error('Failed to load scenario:', error);
        ui.scenarioValue.textContent = 'Error loading scenario';
    }
}

/**
 * Display scenario details panel with editable fields.
 */
function showScenarioDetails() {
    if (!state.scenarioData) return;

    const mode = state.scenarioData.mode;
    const { subj, role, org, mtg } = state.scenarioData;

    let html = `<h4>Scenario: ${escapeHtml(state.currentScenario?.name)}</h4>`;
    html += buildScenarioFieldsHTML(mode, { subj, role, org, mtg, case: state.scenarioData.case });

    ui.scenarioDetails.innerHTML = html;
    ui.scenarioDetails.style.display = 'block';

    // Set accent color based on mode
    const accentColors = {
        interview: 'var(--accent-interview)',
        post_interview: 'var(--accent-post-interview)',
        separation: 'var(--accent-separation)'
    };
    ui.scenarioDetails.style.borderLeftColor = accentColors[mode] || 'var(--accent-warm)';

    // Setup editable field listeners
    attachEditableFieldListeners();

    // Show appropriate start button/CV panel
    updateStartPanel(mode);
}

/**
 * Build HTML for scenario detail fields based on mode.
 * @param {string} mode - DPP mode
 * @param {Object} data - Scenario data fields
 * @returns {string} HTML string
 */
function buildScenarioFieldsHTML(mode, data) {
    const { subj, role, org, mtg } = data;

    // Common editable fields for all modes
    const personLabel = mode === 'separation' ? 'Employee' : 'Candidate';
    let html = `
        <div class="detail-row editable">
            <label class="label" for="edit-candidate">${personLabel}:</label>
            <input type="text" id="edit-candidate" class="edit-input"
                   value="${escapeHtml(subj?.name)}" placeholder="${personLabel} name">
        </div>
        <div class="detail-row editable">
            <label class="label" for="edit-role">Role:</label>
            <input type="text" id="edit-role" class="edit-input"
                   value="${escapeHtml(role?.t)}" placeholder="Job title">
        </div>
        <div class="detail-row editable">
            <label class="label" for="edit-company">Company:</label>
            <input type="text" id="edit-company" class="edit-input"
                   value="${escapeHtml(org?.n)}" placeholder="Company name">
        </div>
    `;

    // Mode-specific fields
    if (mode === 'interview') {
        html += `
            <div class="detail-row editable">
                <label class="label" for="edit-location">Location:</label>
                <input type="text" id="edit-location" class="edit-input"
                       value="${escapeHtml(role?.loc)}" placeholder="Location">
            </div>
            <div class="detail-row">
                <span class="label">Duration:</span>
                <span class="value">${mtg?.mins || 5} minutes</span>
            </div>
            <div class="detail-row editable focus-row">
                <label class="label" for="edit-focus">Focus:</label>
                <input type="text" id="edit-focus" class="edit-input"
                       value="${escapeHtml(mtg?.focus?.join(', '))}"
                       placeholder="Focus areas (comma-separated)">
            </div>
        `;
    } else if (mode === 'post_interview') {
        const caseType = inferPostInterviewType(data.case);
        html += `
            <div class="detail-row">
                <span class="label">Call Type:</span>
                <span class="value">${escapeHtml(caseType)}</span>
            </div>
            <div class="detail-row">
                <span class="label">Duration:</span>
                <span class="value">${mtg?.mins || 5} minutes</span>
            </div>
        `;
    } else if (mode === 'separation') {
        html += `
            <div class="detail-row">
                <span class="label">Type:</span>
                <span class="value">${escapeHtml(data.case?.type || 'N/A')}</span>
            </div>
            <div class="detail-row editable">
                <label class="label" for="edit-effective">Effective:</label>
                <input type="text" id="edit-effective" class="edit-input"
                       value="${escapeHtml(data.case?.eff)}" placeholder="Effective date">
            </div>
            <div class="detail-row">
                <span class="label">Immediate:</span>
                <span class="value">${data.case?.imm ? 'Yes' : 'No'}</span>
            </div>
        `;
    }

    return html;
}

/**
 * Infer the post-interview call type from case data.
 * @param {Object} caseData - DPP case object
 * @returns {string} Human-readable call type
 */
function inferPostInterviewType(caseData) {
    if (caseData?.type && caseData.type !== 'Other') {
        return caseData.type;
    }
    const firstTalk = caseData?.talk?.[0]?.toLowerCase() || '';
    return firstTalk.includes('offer') ? 'Offer Call' : 'Feedback Call';
}

/**
 * Attach event listeners to editable input fields.
 * Removes existing listeners first to prevent duplicates.
 */
function attachEditableFieldListeners() {
    const fieldIds = ['candidate', 'role', 'company', 'location', 'focus', 'effective'];

    fieldIds.forEach(fieldId => {
        const input = document.getElementById(`edit-${fieldId}`);
        if (!input) return;

        // Create handler with closure over fieldId
        const handler = () => {
            state.editedFields[fieldId] = input.value.trim() || null;
        };

        // Store handler reference for potential cleanup
        input._editHandler = handler;
        input.addEventListener('input', handler);
    });
}

/**
 * Set disabled state on all editable input fields.
 * @param {boolean} disabled
 */
function setEditableFieldsDisabled(disabled) {
    document.querySelectorAll('.scenario-details .edit-input').forEach(input => {
        input.disabled = disabled;
        input.classList.toggle('disabled', disabled);
    });
}

// =============================================================================
// START PANELS (Interview CV Upload / Other Start Button)
// =============================================================================

/**
 * Update which start panel is visible based on mode.
 * Re-enables controls that may have been disabled from a previous conversation.
 * @param {string} mode - DPP mode
 */
function updateStartPanel(mode) {
    if (mode === 'interview') {
        ui.cvUploadPanel.style.display = 'block';
        setCVUploadDisabled(false);
        hideStartCallPanel();
    } else {
        ui.cvUploadPanel.style.display = 'none';
        showStartCallPanel(mode);
        setStartCallPanelDisabled(false);
    }
}

/**
 * Show the start call button panel for non-interview scenarios.
 * @param {string} mode - DPP mode
 */
function showStartCallPanel(mode) {
    let panel = document.getElementById('start-call-panel');

    if (!panel) {
        panel = document.createElement('div');
        panel.id = 'start-call-panel';
        panel.className = 'start-call-panel';
        panel.innerHTML = `
            <button class="btn btn-start-call" id="start-call-btn">
                <span>&#9654;</span> Start Call
            </button>
        `;
        ui.scenarioDetails.after(panel);

        document.getElementById('start-call-btn').addEventListener('click', async () => {
            setStartCallPanelDisabled(true);
            setEditableFieldsDisabled(true);
            await startConversation();
        });
    }

    // Set accent color
    const accentColors = {
        post_interview: 'var(--accent-post-interview)',
        separation: 'var(--accent-separation)'
    };
    panel.style.setProperty('--panel-accent', accentColors[mode] || 'var(--accent-warm)');
    panel.style.display = 'block';
}

/**
 * Hide the start call panel.
 */
function hideStartCallPanel() {
    const panel = document.getElementById('start-call-panel');
    if (panel) {
        panel.style.display = 'none';
    }
}

/**
 * Set disabled state on start call button.
 * @param {boolean} disabled
 */
function setStartCallPanelDisabled(disabled) {
    const btn = document.getElementById('start-call-btn');
    if (btn) {
        btn.disabled = disabled;
        btn.classList.toggle('disabled', disabled);
    }
}

// =============================================================================
// CV UPLOAD
// =============================================================================

/**
 * Handle CV file selection.
 * @param {File} file - Selected PDF file
 */
async function handleCVFile(file) {
    if (!file || file.type !== 'application/pdf') {
        alert('Please upload a PDF file.');
        return;
    }

    ui.cvUploadPanel.classList.add('processing');

    try {
        state.cvText = await extractTextFromPDF(file);

        ui.cvFilename.textContent = file.name;
        ui.cvStatus.style.display = 'flex';
        ui.cvUploadPanel.classList.add('has-cv');
        ui.cvUploadPanel.classList.remove('processing');

        console.log('CV extracted:', state.cvText.substring(0, 200) + '...');
    } catch (error) {
        console.error('Failed to extract CV text:', error);
        alert('Failed to read PDF. Please try another file.');
        ui.cvUploadPanel.classList.remove('processing');
        clearCV();
    }
}

/**
 * Extract text from a PDF file using PDF.js.
 * @param {File} file - PDF file
 * @returns {Promise<string>} Extracted text
 */
async function extractTextFromPDF(file) {
    const arrayBuffer = await file.arrayBuffer();
    const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;

    const pages = [];
    for (let i = 1; i <= pdf.numPages; i++) {
        const page = await pdf.getPage(i);
        const textContent = await page.getTextContent();
        pages.push(textContent.items.map(item => item.str).join(' '));
    }

    return pages.join('\n').trim();
}

/**
 * Clear uploaded CV.
 */
function clearCV() {
    state.cvText = null;
    if (ui.cvFileInput) ui.cvFileInput.value = '';
    if (ui.cvStatus) ui.cvStatus.style.display = 'none';
    if (ui.cvUploadPanel) ui.cvUploadPanel.classList.remove('has-cv');
}

/**
 * Set disabled state on CV upload controls.
 * @param {boolean} disabled
 */
function setCVUploadDisabled(disabled) {
    if (ui.cvFileInput) ui.cvFileInput.disabled = disabled;
    if (ui.startInterviewBtn) ui.startInterviewBtn.disabled = disabled;
    if (ui.cvRemoveBtn) ui.cvRemoveBtn.disabled = disabled;
    if (ui.cvUploadPanel) {
        ui.cvUploadPanel.classList.toggle('disabled', disabled);
    }
}

// =============================================================================
// CONVERSATION CONTROL
// =============================================================================

/**
 * Start the avatar conversation.
 */
async function startConversation() {
    if (!state.currentScenario || !state.scenarioData) return;

    state.isConversationActive = true;
    setEditableFieldsDisabled(true);
    setCVUploadDisabled(true);

    clearTranscriptUI();

    ui.emptyState.style.display = 'none';
    ui.avatarContainer.classList.remove('empty');

    try {
        // Start the SDK - DPP injection happens on SHOWING_AGENT event
        await state.sdk.start();
    } catch (error) {
        console.error('Failed to start conversation:', error);
        updateStatus('error');
        state.isConversationActive = false;
        setEditableFieldsDisabled(false);
        setCVUploadDisabled(false);
    }
}

/**
 * Validate DPP before injection.
 * Catches corrupted state and logs warnings.
 * @param {Object} dpp - DPP object to validate
 * @returns {boolean} True if valid, false if critical errors
 */
function validateDPP(dpp) {
    const errors = [];
    const warnings = [];

    // Required fields
    if (!dpp.v) errors.push('Missing: v (schema version)');
    if (!dpp.mode || !['interview', 'post_interview', 'separation'].includes(dpp.mode)) {
        errors.push(`Invalid mode: ${dpp.mode}`);
    }
    if (!dpp.org?.n) errors.push('Missing: org.n (company name)');
    if (!dpp.role?.t) errors.push('Missing: role.t (role title)');
    if (!dpp.subj?.name) errors.push('Missing: subj.name (person name)');

    // Mode-specific validation
    if (dpp.mode === 'interview') {
        if (!dpp.mtg?.mins) warnings.push('Interview missing mtg.mins');
        if (!dpp.eval) warnings.push('Interview missing eval block');
    } else if (dpp.mode === 'post_interview') {
        if (!dpp.case?.talk?.length) errors.push('post_interview requires case.talk array');
    } else if (dpp.mode === 'separation') {
        if (!dpp.case?.talk?.length) errors.push('separation requires case.talk array');
        if (!dpp.case?.type) errors.push('separation requires case.type');
    }

    // Log results
    if (errors.length > 0) {
        console.error('[DPP Validation] Errors:', errors);
        return false;
    }
    if (warnings.length > 0) {
        console.warn('[DPP Validation] Warnings:', warnings);
    }

    return true;
}

/**
 * Build the dynamic prompt JSON string.
 * Applies user edits and CV data to the scenario data.
 * @returns {string|null} JSON string for prompt injection
 */
function buildDynamicPrompt() {
    if (!state.scenarioData) return null;

    const promptData = applyUserEdits(deepClone(state.scenarioData));

    // Add CV information if available
    if (state.cvText) {
        promptData.subj = promptData.subj || {};
        promptData.subj.prof = promptData.subj.prof || {};
        promptData.subj.prof.notes = promptData.subj.prof.notes || [];

        promptData.subj.prof.cv_summary = state.cvText;
        promptData.subj.prof.notes.push(
            'IMPORTANT: A CV/resume was provided. Review the cv_summary and ask relevant follow-up questions about their experience, skills, and background mentioned in the CV.'
        );

        promptData.inst = promptData.inst || [];
        promptData.inst.push(
            'Reference specific details from the candidate\'s CV when asking questions. Ask about gaps, interesting projects, or skills mentioned.'
        );
    }

    // Validate before returning
    if (!validateDPP(promptData)) {
        console.error('[DPP] Validation failed - DPP may be malformed');
        // Still return the prompt - let the avatar handle gracefully
    }

    // Debug logging - summary
    console.log('[DPP Build] Context:', {
        mode: promptData.mode,
        company: promptData.org?.n,
        role: promptData.role?.t,
        candidate: promptData.subj?.name,
        duration: promptData.mtg?.mins,
        hasCv: !!state.cvText,
        editedFields: Object.keys(state.editedFields).filter(k => state.editedFields[k])
    });

    const dppJson = JSON.stringify(promptData);

    // Debug logging - full DPP (for troubleshooting)
    console.log('[DPP Build] Full DPP being injected:', dppJson);

    // Update debug panel if it exists
    const debugPanel = document.getElementById('debug-dpp');
    if (debugPanel) {
        debugPanel.textContent = JSON.stringify(promptData, null, 2);
    }

    return dppJson;
}

/**
 * Apply user-edited field overrides to a DPP object.
 * @param {Object} data - DPP object to modify
 * @returns {Object} Modified DPP object
 */
function applyUserEdits(data) {
    const { editedFields } = state;

    if (editedFields.candidate && data.subj) {
        data.subj.name = editedFields.candidate;
    }
    if (editedFields.role && data.role) {
        data.role.t = editedFields.role;
    }
    if (editedFields.company && data.org) {
        data.org.n = editedFields.company;
    }
    if (editedFields.location && data.role) {
        data.role.loc = editedFields.location;
    }
    if (editedFields.focus) {
        data.mtg = data.mtg || {};
        data.mtg.focus = editedFields.focus.split(',').map(s => s.trim()).filter(Boolean);
    }
    if (editedFields.effective) {
        data.case = data.case || {};
        data.case.eff = editedFields.effective;
    }

    return data;
}

// =============================================================================
// UI STATE MANAGEMENT
// =============================================================================

/**
 * Update the status display.
 * @param {string} statusText - Status text to display
 */
function updateStatus(statusText) {
    ui.statusValue.textContent = statusText;
    ui.statusValue.className = 'value';

    if (statusText === 'in-conversation') {
        ui.statusValue.classList.add('status-active');
    } else if (statusText === 'ended' || statusText === 'error') {
        ui.statusValue.classList.add('status-ended');
    }
}

/**
 * Update download button enabled state based on transcript availability.
 */
function updateDownloadButtons() {
    const hasTranscript = state.sdk?.getTranscript().length > 0;
    if (ui.downloadBtn) ui.downloadBtn.disabled = !hasTranscript;
    if (ui.downloadMdBtn) ui.downloadMdBtn.disabled = !hasTranscript;
}

/**
 * Show analyzing state while call summary is being generated.
 * Clears the avatar iframe and shows a loading spinner.
 */
function showAnalyzingState() {
    // End SDK to clear the avatar iframe
    if (state.sdk) {
        state.sdk.end();
    }

    // Show loading spinner in avatar container
    ui.avatarContainer.classList.add('empty');
    ui.avatarContainer.innerHTML = `
        <div class="analyzing-state">
            <div class="spinner"></div>
            <h3>Analyzing Call...</h3>
            <p>Generating your call summary</p>
        </div>
    `;

    updateStatus('analyzing...');
}

/**
 * Highlight download button to draw attention.
 */
function highlightDownloadButton() {
    if (!state.sdk?.getTranscript().length) return;

    ui.downloadBtn.classList.add('pulse');
    setTimeout(() => ui.downloadBtn.classList.remove('pulse'), 2000);
}

/**
 * Reset UI to initial clean state.
 */
function resetToInitialState() {
    // End SDK session
    if (state.sdk) {
        state.sdk.end();
    }

    // Reset state
    state.currentScenario = null;
    state.scenarioData = null;
    state.isConversationActive = false;
    state.isEndingCall = false;
    resetEditedFields();
    clearCV();

    // Clear UI selections
    document.querySelectorAll('.scenario-card').forEach(card => {
        card.classList.remove('active');
    });

    // Hide panels
    ui.scenarioDetails.style.display = 'none';
    ui.scenarioDetails.innerHTML = '';
    ui.cvUploadPanel.style.display = 'none';
    hideStartCallPanel();

    // Show empty state (clear any existing content like analyzing spinner first)
    ui.avatarContainer.innerHTML = '';
    ui.avatarContainer.classList.add('empty');
    ui.avatarContainer.appendChild(ui.emptyState);
    ui.emptyState.style.display = '';

    // Reset status
    ui.statusValue.textContent = 'Ready';
    ui.statusValue.className = 'value';
    ui.scenarioValue.textContent = 'None selected';
}

// =============================================================================
// TRANSCRIPT MANAGEMENT
// =============================================================================

/**
 * Add an entry to the transcript display.
 * @param {string} type - 'avatar' or 'user'
 * @param {string} speaker - Display name
 * @param {string} text - Spoken text
 */
function addTranscriptEntry(type, speaker, text) {
    ui.transcriptEmpty.style.display = 'none';

    const entry = document.createElement('div');
    entry.className = `transcript-entry ${type}`;
    entry.innerHTML = `
        <div class="role">${escapeHtml(speaker)}</div>
        <div class="text">${escapeHtml(text)}</div>
        <div class="time">${new Date().toLocaleTimeString()}</div>
    `;

    ui.transcriptContent.appendChild(entry);
    ui.transcriptContent.scrollTop = ui.transcriptContent.scrollHeight;

    // Enable download buttons
    updateDownloadButtons();
}

/**
 * Clear the transcript UI.
 */
function clearTranscriptUI() {
    ui.transcriptContent.innerHTML = '';
    ui.transcriptEmpty.style.display = 'block';
    ui.transcriptContent.appendChild(ui.transcriptEmpty);
}

/**
 * Download transcript in specified format.
 * @param {string} format - 'text' or 'markdown'
 */
function downloadTranscript(format) {
    if (!state.sdk) return;

    const name = slugify(state.currentScenario?.name || 'session');
    const ext = format === 'markdown' ? 'md' : 'txt';

    state.sdk.downloadTranscript({
        format,
        filename: `hr-transcript-${name}-${getTodayISO()}.${ext}`,
        includeTimestamps: true
    });
}

// =============================================================================
// CALL ANALYSIS
// =============================================================================

/**
 * Analyze the completed call using the Lambda API.
 */
async function analyzeCall() {
    if (!state.sdk || !state.scenarioData) {
        console.log('No call to analyze');
        return;
    }

    const transcript = state.sdk.getTranscript();
    if (!transcript?.length) {
        console.log('Empty transcript, skipping analysis');
        return;
    }

    // Preserve scenario name for download
    state.lastScenarioName = state.currentScenario?.name || 'call';

    updateStatus('analyzing...');

    try {
        const dpp = buildDPPForAnalysis();

        // Format transcript for API (SDK uses 'role', need 'assistant'/'user')
        const formattedTranscript = transcript.map(entry => ({
            role: entry.role === 'Avatar' ? 'assistant' : 'user',
            content: entry.text
        }));

        const response = await fetch(CONFIG.ANALYSIS_API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ transcript: formattedTranscript, dpp })
        });

        const result = await response.json();

        if (result.success) {
            state.lastCallSummary = result.summary;
            console.log('Call analysis complete:', state.lastCallSummary);
            showCallSummary(result.summary);
        } else {
            console.error('Analysis failed:', result.error);
        }
    } catch (error) {
        console.error('Failed to analyze call:', error);
    }
}

/**
 * Build DPP object for analysis API.
 * Ensures CV context is threaded through for consistent evaluation.
 * @returns {Object} DPP with user edits applied
 */
function buildDPPForAnalysis() {
    if (!state.scenarioData) return {};

    const dpp = applyUserEdits(deepClone(state.scenarioData));

    // Add CV information if provided - same as in buildDynamicPrompt for consistency
    if (state.cvText) {
        dpp.subj = dpp.subj || {};
        dpp.subj.prof = dpp.subj.prof || {};
        dpp.subj.prof.notes = dpp.subj.prof.notes || [];

        dpp.subj.prof.cv_summary = state.cvText;
        dpp.subj.prof.notes.push(
            'IMPORTANT: A CV/resume was provided. Review the cv_summary and ask relevant follow-up questions about their experience, skills, and background mentioned in the CV.'
        );

        dpp.inst = dpp.inst || [];
        dpp.inst.push(
            'Reference specific details from the candidate\'s CV when asking questions. Ask about gaps, interesting projects, or skills mentioned.'
        );
    }

    return dpp;
}

/**
 * Display call summary modal.
 * @param {Object} summary - Analysis result
 */
function showCallSummary(summary) {
    let modal = document.getElementById('summary-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'summary-modal';
        modal.className = 'summary-modal';
        document.body.appendChild(modal);
    }

    const fitScore = summary.fit?.score_0_100;
    const fitClass = fitScore >= 70 ? 'good' : fitScore >= 50 ? 'ok' : 'poor';

    modal.innerHTML = `
        <div class="summary-modal-content">
            <div class="summary-header">
                <h3>Call Summary</h3>
                <button class="summary-close-btn" onclick="closeSummaryModal()">&times;</button>
            </div>
            <div class="summary-body">
                <div class="summary-section">
                    <h4>Overview</h4>
                    <p>${escapeHtml(summary.overview || 'No overview available')}</p>
                </div>

                ${fitScore != null ? `
                <div class="summary-section">
                    <h4>Fit Assessment</h4>
                    <div class="fit-score ${fitClass}">
                        <span class="score-value">${fitScore}</span>
                        <span class="score-label">/ 100</span>
                    </div>
                    ${summary.fit.dims?.length ? `
                    <div class="fit-dims">
                        ${summary.fit.dims.map(d => `
                            <div class="dim">
                                <span class="dim-name">${escapeHtml(d.id || d.name || 'Unknown')}</span>
                                <span class="dim-score">${d.score_1_5}/5</span>
                            </div>
                        `).join('')}
                    </div>
                    ` : ''}
                </div>
                ` : ''}

                ${summary.gaps?.length ? `
                <div class="summary-section">
                    <h4>Gaps & Follow-ups</h4>
                    <ul class="gaps-list">
                        ${summary.gaps.map(g => `<li>${escapeHtml(typeof g === 'string' ? g : g.missing || '')}</li>`).join('')}
                    </ul>
                </div>
                ` : ''}

                ${summary.next_steps?.length ? `
                <div class="summary-section">
                    <h4>Next Steps</h4>
                    <ul class="next-steps-list">
                        ${summary.next_steps.map(s => `<li>${escapeHtml(s)}</li>`).join('')}
                    </ul>
                </div>
                ` : ''}

                <div class="summary-section">
                    <h4>Call Quality</h4>
                    <div class="cq-badges">
                        <span class="badge">Emotion: ${escapeHtml(summary.cq?.emo || 'unknown')}</span>
                        <span class="badge">Tone: ${escapeHtml(summary.cq?.tone || 'unknown')}</span>
                        <span class="badge">Engagement: ${escapeHtml(summary.cq?.eng || 'unknown')}</span>
                    </div>
                </div>
            </div>
            <div class="summary-footer">
                <button class="btn btn-secondary" onclick="downloadCallSummary()">Download JSON</button>
                <button class="btn btn-primary" onclick="closeSummaryModal()">Close</button>
            </div>
        </div>
    `;

    modal.style.display = 'flex';
}

/**
 * Close summary modal.
 */
function closeSummaryModal() {
    const modal = document.getElementById('summary-modal');
    if (modal) modal.style.display = 'none';
}

/**
 * Download call summary as JSON file.
 */
function downloadCallSummary() {
    if (!state.lastCallSummary) return;

    const name = slugify(state.lastScenarioName || 'call');
    const filename = `call-summary-${name}-${getTodayISO()}.json`;

    const blob = new Blob([JSON.stringify(state.lastCallSummary, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();

    URL.revokeObjectURL(url);
}

// =============================================================================
// EVENT LISTENERS SETUP
// =============================================================================

/**
 * Attach all event listeners.
 * Called once after DOM is ready.
 */
function attachEventListeners() {
    // CV file input
    ui.cvFileInput?.addEventListener('change', (e) => {
        if (e.target.files?.length) handleCVFile(e.target.files[0]);
    });

    // CV remove button
    ui.cvRemoveBtn?.addEventListener('click', clearCV);

    // Start interview button
    ui.startInterviewBtn?.addEventListener('click', async () => {
        setCVUploadDisabled(true);
        setEditableFieldsDisabled(true);
        await startConversation();
    });

    // Drag and drop for CV
    ui.cvUploadArea?.addEventListener('dragover', (e) => {
        e.preventDefault();
        ui.cvUploadArea.classList.add('dragover');
    });

    ui.cvUploadArea?.addEventListener('dragleave', () => {
        ui.cvUploadArea.classList.remove('dragover');
    });

    ui.cvUploadArea?.addEventListener('drop', (e) => {
        e.preventDefault();
        ui.cvUploadArea.classList.remove('dragover');
        if (e.dataTransfer.files?.length) handleCVFile(e.dataTransfer.files[0]);
    });

    // Download buttons
    ui.downloadBtn?.addEventListener('click', () => downloadTranscript('text'));
    ui.downloadMdBtn?.addEventListener('click', () => downloadTranscript('markdown'));
}

// =============================================================================
// INITIALIZATION
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    initUI();
    attachEventListeners();
    renderScenarios();
    initSDK();
});
