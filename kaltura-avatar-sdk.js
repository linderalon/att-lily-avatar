/*!
 * Kaltura Avatar SDK v1.0.0
 * https://github.com/anthropics/claude-code
 *
 * Embed Kaltura AI Avatar conversations in any website.
 * Zero dependencies. Works with any framework.
 *
 * MIT License
 */

(function (global, factory) {
    typeof exports === 'object' && typeof module !== 'undefined' ? module.exports = factory() :
    typeof define === 'function' && define.amd ? define(factory) :
    (global = typeof globalThis !== 'undefined' ? globalThis : global || self, global.KalturaAvatarSDK = factory());
})(this, (function () {
    'use strict';

    /**
     * Avatar events emitted during conversation.
     * @readonly
     * @enum {string}
     */
    const Events = Object.freeze({
        /** Join screen displayed */
        SHOWING_JOIN_MEETING: 'showing-join-meeting',
        /** User clicked join */
        JOIN_MEETING_CLICKED: 'join-meeting-clicked',
        /** Avatar is visible */
        SHOWING_AGENT: 'showing-agent',
        /** Avatar spoke */
        AGENT_TALKED: 'agent-talked',
        /** User speech transcribed */
        USER_TRANSCRIPTION: 'user-transcription',
        /** Pronunciation score */
        PRONUNCIATION_SCORE: 'pronunciation-score',
        /** Permissions denied */
        PERMISSIONS_DENIED: 'permissions-denied',
        /** Conversation ended */
        CONVERSATION_ENDED: 'conversation-ended',
        /** Load error */
        LOAD_AGENT_ERROR: 'load-agent-error'
    });

    /**
     * SDK lifecycle states.
     * @readonly
     * @enum {string}
     */
    const State = Object.freeze({
        UNINITIALIZED: 'uninitialized',
        INITIALIZING: 'initializing',
        READY: 'ready',
        IN_CONVERSATION: 'in-conversation',
        ENDED: 'ended',
        ERROR: 'error'
    });

    /** Default configuration */
    const DEFAULT_CONFIG = {
        apiBaseUrl: 'https://api.avatar.us.kaltura.ai',
        meetBaseUrl: 'https://meet.avatar.us.kaltura.ai',
        debug: false,
        iframeClass: 'kaltura-avatar-iframe',
        iframeStyles: {
            width: '100%',
            height: '100%',
            border: 'none',
            borderRadius: '12px'
        }
    };

    /**
     * Kaltura Avatar SDK
     * @class
     * @example
     * const sdk = new KalturaAvatarSDK({
     *     clientId: 'your-client-id',
     *     flowId: 'your-flow-id',
     *     container: '#avatar'
     * });
     * await sdk.start();
     */
    class KalturaAvatarSDK {

        /**
         * Create SDK instance.
         * @param {Object} options
         * @param {string} options.clientId - Your Kaltura client ID
         * @param {string} options.flowId - Your Kaltura flow ID
         * @param {string|HTMLElement} [options.container] - Container element or selector
         * @param {Object} [options.config] - Configuration overrides
         */
        constructor(options = {}) {
            if (!options.clientId) throw new Error('clientId is required');
            if (!options.flowId) throw new Error('flowId is required');

            this._clientId = options.clientId;
            this._flowId = options.flowId;
            this._config = { ...DEFAULT_CONFIG, ...options.config };
            this._container = null;
            this._iframe = null;
            this._state = State.UNINITIALIZED;
            this._assets = null;
            this._listeners = new Map();
            this._onMessage = this._handleMessage.bind(this);
            this._transcript = [];
            this._transcriptEnabled = true;

            if (options.container) {
                this.setContainer(options.container);
            }
        }

        // =====================================================================
        // PUBLIC API
        // =====================================================================

        /**
         * Set container element.
         * @param {string|HTMLElement} container - CSS selector or element
         * @returns {this}
         */
        setContainer(container) {
            if (typeof container === 'string') {
                this._container = document.querySelector(container);
                if (!this._container) throw new Error(`Container not found: ${container}`);
            } else if (container instanceof HTMLElement) {
                this._container = container;
            } else {
                throw new Error('Container must be HTMLElement or selector string');
            }
            return this;
        }

        /**
         * Initialize SDK and load avatar assets.
         * @returns {Promise<Object>} Avatar assets
         */
        async init() {
            if (this._state !== State.UNINITIALIZED) {
                return this._assets;
            }

            this._setState(State.INITIALIZING);

            try {
                const url = `${this._config.apiBaseUrl}/clients/${this._clientId}/flow/${this._flowId}/sdk-assets`;
                const response = await fetch(url);

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }

                this._assets = await response.json();
                window.addEventListener('message', this._onMessage);

                this._setState(State.READY);
                this._emit('ready', { assets: this._assets });

                return this._assets;
            } catch (error) {
                this._setState(State.ERROR);
                this._emit('error', { message: error.message });
                throw error;
            }
        }

        /**
         * Start the avatar conversation.
         * @param {Object} [options]
         * @param {Object} [options.styles] - Custom iframe styles
         * @returns {Promise<HTMLIFrameElement>}
         */
        async start(options = {}) {
            if (!this._container) {
                throw new Error('Container not set. Call setContainer() first.');
            }

            if (this._state === State.UNINITIALIZED) {
                await this.init();
            }

            if (this._state !== State.READY && this._state !== State.ENDED) {
                return this._iframe;
            }

            // Clear transcript for new conversation
            this._transcript = [];

            this._iframe = this._createIframe(options.styles);
            this._container.innerHTML = '';
            this._container.appendChild(this._iframe);

            this._setState(State.IN_CONVERSATION);
            this._emit('started', { iframe: this._iframe });

            return this._iframe;
        }

        /**
         * End the conversation.
         */
        end() {
            if (this._iframe) {
                this._iframe.src = 'about:blank';
                this._iframe.remove();
                this._iframe = null;
            }
            this._setState(State.ENDED);
            this._emit('ended', {});
        }

        /**
         * Destroy SDK and cleanup resources.
         */
        destroy() {
            this.end();
            window.removeEventListener('message', this._onMessage);
            this._listeners.clear();
            this._assets = null;
            this._setState(State.UNINITIALIZED);
        }

        /**
         * Inject a prompt into the conversation.
         * @param {string} text - Prompt text
         * @returns {boolean} Success
         * @example
         * sdk.injectPrompt('Tell me a joke');
         */
        injectPrompt(text) {
            if (!text || typeof text !== 'string') return false;
            return this._postToIframe({
                type: 'eself-dynamic-prompt-message',
                content: text
            });
        }

        /**
         * Send raw message to iframe.
         * @param {Object} message
         * @returns {boolean} Success
         */
        sendMessage(message) {
            return this._postToIframe(message);
        }

        /**
         * Subscribe to event.
         * @param {string} event - Event name or '*' for all
         * @param {Function} callback
         * @returns {Function} Unsubscribe function
         * @example
         * sdk.on('agent-talked', (data) => console.log(data));
         * sdk.on('*', ({ event, data }) => console.log(event, data));
         */
        on(event, callback) {
            if (!this._listeners.has(event)) {
                this._listeners.set(event, new Set());
            }
            this._listeners.get(event).add(callback);
            return () => this.off(event, callback);
        }

        /**
         * Unsubscribe from event.
         * @param {string} event
         * @param {Function} callback
         */
        off(event, callback) {
            const listeners = this._listeners.get(event);
            if (listeners) listeners.delete(callback);
        }

        /**
         * Subscribe once.
         * @param {string} event
         * @param {Function} callback
         * @returns {Function} Unsubscribe function
         */
        once(event, callback) {
            const wrapper = (data) => {
                this.off(event, wrapper);
                callback(data);
            };
            return this.on(event, wrapper);
        }

        /** Get current state */
        getState() { return this._state; }

        /** Get loaded assets */
        getAssets() { return this._assets; }

        /** Get avatar info */
        getAvatarInfo() { return this._assets?.avatar || null; }

        /** Get iframe element */
        getIframe() { return this._iframe; }

        /** Get talk URL */
        getTalkUrl() { return this._assets?.talk_url || null; }

        /** Get client ID */
        getClientId() { return this._clientId; }

        /** Get flow ID */
        getFlowId() { return this._flowId; }

        // =====================================================================
        // TRANSCRIPT METHODS
        // =====================================================================

        /**
         * Enable or disable transcript recording.
         * @param {boolean} enabled
         */
        setTranscriptEnabled(enabled) {
            this._transcriptEnabled = !!enabled;
        }

        /**
         * Get the current transcript.
         * @returns {Array<{role: string, text: string, timestamp: Date}>}
         */
        getTranscript() {
            return [...this._transcript];
        }

        /**
         * Clear the transcript.
         */
        clearTranscript() {
            this._transcript = [];
        }

        /**
         * Get transcript as formatted text.
         * @param {Object} [options]
         * @param {boolean} [options.includeTimestamps=true] - Include timestamps
         * @param {string} [options.format='text'] - 'text', 'markdown', or 'json'
         * @returns {string}
         */
        getTranscriptText(options = {}) {
            const { includeTimestamps = true, format = 'text' } = options;

            if (format === 'json') {
                return JSON.stringify(this._transcript, null, 2);
            }

            const lines = this._transcript.map(entry => {
                const time = includeTimestamps
                    ? `[${entry.timestamp.toLocaleTimeString()}] `
                    : '';
                const role = format === 'markdown'
                    ? `**${entry.role}**`
                    : entry.role;
                return `${time}${role}: ${entry.text}`;
            });

            return lines.join(format === 'markdown' ? '\n\n' : '\n');
        }

        /**
         * Download transcript as a file.
         * @param {Object} [options]
         * @param {string} [options.filename] - Custom filename
         * @param {string} [options.format='text'] - 'text', 'markdown', or 'json'
         * @param {boolean} [options.includeTimestamps=true]
         */
        downloadTranscript(options = {}) {
            const {
                filename,
                format = 'text',
                includeTimestamps = true
            } = options;

            const content = this.getTranscriptText({ format, includeTimestamps });
            const ext = format === 'json' ? 'json' : format === 'markdown' ? 'md' : 'txt';
            const mimeType = format === 'json' ? 'application/json' : 'text/plain';
            const defaultName = `transcript-${new Date().toISOString().slice(0, 10)}.${ext}`;

            const blob = new Blob([content], { type: mimeType });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename || defaultName;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }

        // =====================================================================
        // PRIVATE
        // =====================================================================

        _createIframe(customStyles = {}) {
            const iframe = document.createElement('iframe');
            iframe.id = 'kaltura-avatar-iframe';
            iframe.className = this._config.iframeClass;
            iframe.src = this._assets.talk_url;
            iframe.frameBorder = '0';
            iframe.sandbox = 'allow-same-origin allow-scripts allow-forms';
            iframe.allow = [
                `camera ${this._config.meetBaseUrl}`,
                `microphone ${this._config.meetBaseUrl}`,
                `display-capture ${this._config.meetBaseUrl}`
            ].join('; ');

            Object.assign(iframe.style, this._config.iframeStyles, customStyles);
            return iframe;
        }

        _postToIframe(message) {
            if (!this._iframe?.contentWindow) return false;
            this._iframe.contentWindow.postMessage(message, '*');
            return true;
        }

        _handleMessage(event) {
            const { data } = event;
            if (!data || data.issuer !== 'eself-conversation-events') return;

            if (data.event) {
                this._emit(data.event, data.data || {});
            }

            // Record transcript entries
            if (this._transcriptEnabled) {
                if (data.event === Events.AGENT_TALKED) {
                    const text = data.data?.agentContent || data.data;
                    if (text && typeof text === 'string') {
                        this._transcript.push({
                            role: 'Avatar',
                            text: text,
                            timestamp: new Date()
                        });
                    }
                } else if (data.event === Events.USER_TRANSCRIPTION) {
                    const text = data.data?.userTranscription || data.data;
                    if (text && typeof text === 'string') {
                        this._transcript.push({
                            role: 'User',
                            text: text,
                            timestamp: new Date()
                        });
                    }
                }
            }

            if (data.event === Events.CONVERSATION_ENDED) {
                this._setState(State.ENDED);
            } else if (data.event === Events.LOAD_AGENT_ERROR) {
                this._setState(State.ERROR);
            }
        }

        _setState(newState) {
            const oldState = this._state;
            this._state = newState;
            if (oldState !== newState) {
                this._emit('stateChange', { from: oldState, to: newState });
            }
        }

        _emit(event, data) {
            const listeners = this._listeners.get(event);
            if (listeners) {
                listeners.forEach(cb => {
                    try { cb(data); } catch (e) { console.error(e); }
                });
            }

            const wildcardListeners = this._listeners.get('*');
            if (wildcardListeners) {
                wildcardListeners.forEach(cb => {
                    try { cb({ event, data }); } catch (e) { console.error(e); }
                });
            }
        }
    }

    // Static properties
    KalturaAvatarSDK.Events = Events;
    KalturaAvatarSDK.State = State;
    KalturaAvatarSDK.VERSION = '1.0.0';

    return KalturaAvatarSDK;
}));
