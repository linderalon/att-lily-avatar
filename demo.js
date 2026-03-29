/**
 * Kaltura Avatar SDK Demo
 * Demonstrates all SDK capabilities
 */

// Initialize SDK with your credentials
const sdk = new KalturaAvatarSDK({
    clientId: '115767973963657880005',
    flowId: 'agent-15',
    container: '#avatar-container'
});

// UI Elements
const ui = {
    status: document.getElementById('status'),
    avatarName: document.getElementById('avatar-name'),
    version: document.getElementById('sdk-version'),
    messages: document.getElementById('messages'),
    events: document.getElementById('events'),
    promptInput: document.getElementById('prompt-input'),
    sendBtn: document.getElementById('send-btn'),
    endBtn: document.getElementById('end-btn'),
    restartBtn: document.getElementById('restart-btn')
};

// Display SDK version
ui.version.textContent = KalturaAvatarSDK.VERSION;

// ============================================================================
// STATE TRACKING
// ============================================================================

sdk.on('stateChange', ({ from, to }) => {
    ui.status.textContent = to;
    ui.status.className = '';

    if (to === 'ready' || to === 'in-conversation') {
        ui.status.classList.add('status-active');
    } else if (to === 'error') {
        ui.status.classList.add('status-error');
    }

    // Update button states
    updateButtons(to);
    logEvent('stateChange', `${from} → ${to}`);
});

function updateButtons(state) {
    const inConversation = state === 'in-conversation';
    const canStart = state === 'ready' || state === 'ended';

    ui.sendBtn.disabled = !inConversation;
    ui.promptInput.disabled = !inConversation;
    ui.endBtn.disabled = !inConversation;
    ui.restartBtn.disabled = !canStart && state !== 'in-conversation';
}

// ============================================================================
// AVATAR EVENTS
// ============================================================================

// Ready - SDK initialized and assets loaded
sdk.on('ready', ({ assets }) => {
    ui.avatarName.textContent = assets.avatar.given_name;
    logEvent('ready', `Avatar: ${assets.avatar.given_name}`);
});

// Showing join meeting screen
sdk.on(KalturaAvatarSDK.Events.SHOWING_JOIN_MEETING, () => {
    logEvent('showing-join-meeting', 'Join screen displayed');
    addSystemMessage('Join screen displayed');
});

// User clicked join
sdk.on(KalturaAvatarSDK.Events.JOIN_MEETING_CLICKED, () => {
    logEvent('join-meeting-clicked', 'User joined');
    addSystemMessage('Joining conversation...');
});

// Avatar is visible
sdk.on(KalturaAvatarSDK.Events.SHOWING_AGENT, () => {
    logEvent('showing-agent', 'Avatar visible');
    addSystemMessage('Avatar is ready');
});

// Avatar spoke
sdk.on(KalturaAvatarSDK.Events.AGENT_TALKED, (data) => {
    const text = data.agentContent || data;
    if (text && typeof text === 'string') {
        addMessage('avatar', 'Avatar', text);
        logEvent('agent-talked', truncate(text, 50));
    }
});

// User speech transcribed
sdk.on(KalturaAvatarSDK.Events.USER_TRANSCRIPTION, (data) => {
    const text = data.userTranscription || data;
    if (text && typeof text === 'string') {
        addMessage('user', 'You', text);
        logEvent('user-transcription', truncate(text, 50));
    }
});

// Pronunciation score (for language learning)
sdk.on(KalturaAvatarSDK.Events.PRONUNCIATION_SCORE, (data) => {
    const score = data.pronunciationScore || data;
    logEvent('pronunciation-score', `Score: ${score}`);
});

// Permissions denied
sdk.on(KalturaAvatarSDK.Events.PERMISSIONS_DENIED, () => {
    logEvent('permissions-denied', 'Camera/mic denied');
    addSystemMessage('Camera or microphone access denied');
});

// Conversation ended
sdk.on(KalturaAvatarSDK.Events.CONVERSATION_ENDED, () => {
    logEvent('conversation-ended', 'Session ended');
    addSystemMessage('Conversation ended');
});

// Load error
sdk.on(KalturaAvatarSDK.Events.LOAD_AGENT_ERROR, () => {
    logEvent('load-agent-error', 'Failed to load');
    addSystemMessage('Failed to load avatar');
});

// Error handler
sdk.on('error', ({ message }) => {
    logEvent('error', message);
    addSystemMessage(`Error: ${message}`);
});

// ============================================================================
// MESSAGE DISPLAY
// ============================================================================

function addMessage(type, label, text) {
    clearEmpty();
    const div = document.createElement('div');
    div.className = `msg ${type}`;
    div.innerHTML = `<div class="msg-label">${label}</div>${escapeHtml(text)}`;
    ui.messages.appendChild(div);
    ui.messages.scrollTop = ui.messages.scrollHeight;
}

function addSystemMessage(text) {
    clearEmpty();
    const div = document.createElement('div');
    div.className = 'msg system';
    div.textContent = text;
    ui.messages.appendChild(div);
    ui.messages.scrollTop = ui.messages.scrollHeight;
}

function clearEmpty() {
    const empty = ui.messages.querySelector('.empty');
    if (empty) empty.remove();
}

// ============================================================================
// EVENT LOG
// ============================================================================

function logEvent(name, data) {
    const time = new Date().toLocaleTimeString('en-US', {
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });

    const div = document.createElement('div');
    div.className = 'event-item';
    div.innerHTML = `
        <span class="time">${time}</span>
        <span class="name">${name}</span>
        <span class="data">${escapeHtml(String(data))}</span>
    `;
    ui.events.appendChild(div);
    ui.events.scrollTop = ui.events.scrollHeight;

    // Keep only last 50 events
    while (ui.events.children.length > 50) {
        ui.events.removeChild(ui.events.firstChild);
    }
}

// ============================================================================
// USER ACTIONS
// ============================================================================

// Send prompt
ui.sendBtn.onclick = () => {
    const text = ui.promptInput.value.trim();
    if (text) {
        sdk.injectPrompt(text);
        ui.promptInput.value = '';
        logEvent('injectPrompt', truncate(text, 50));
    }
};

// Enter key sends
ui.promptInput.onkeypress = (e) => {
    if (e.key === 'Enter') ui.sendBtn.click();
};

// End conversation
ui.endBtn.onclick = () => {
    sdk.end();
    logEvent('end', 'User ended');
};

// Restart conversation
ui.restartBtn.onclick = () => {
    sdk.start();
    logEvent('start', 'User restarted');
};

// ============================================================================
// HELPERS
// ============================================================================

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function truncate(text, maxLength) {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

// ============================================================================
// AUTO-START
// ============================================================================

// Start the avatar conversation when page loads
sdk.start();
logEvent('start', 'Auto-started');
