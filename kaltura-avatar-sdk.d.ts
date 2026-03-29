/*!
 * Kaltura Avatar SDK - TypeScript Declarations
 * @version 1.0.0
 */

export = KalturaAvatarSDK;
export as namespace KalturaAvatarSDK;

declare class KalturaAvatarSDK {
    constructor(options: KalturaAvatarSDK.Options);

    /** Set container element */
    setContainer(container: HTMLElement | string): this;

    /** Initialize SDK and load assets */
    init(): Promise<KalturaAvatarSDK.Assets>;

    /** Start the conversation */
    start(options?: { styles?: Partial<CSSStyleDeclaration> }): Promise<HTMLIFrameElement>;

    /** End the conversation */
    end(): void;

    /** Destroy SDK and cleanup */
    destroy(): void;

    /** Inject a prompt into the conversation */
    injectPrompt(text: string): boolean;

    /** Send raw message to iframe */
    sendMessage(message: Record<string, unknown>): boolean;

    /** Subscribe to event */
    on<K extends keyof KalturaAvatarSDK.EventMap>(
        event: K,
        callback: (data: KalturaAvatarSDK.EventMap[K]) => void
    ): () => void;
    on(event: '*', callback: (data: { event: string; data: unknown }) => void): () => void;

    /** Unsubscribe from event */
    off(event: string, callback: Function): void;

    /** Subscribe once */
    once<K extends keyof KalturaAvatarSDK.EventMap>(
        event: K,
        callback: (data: KalturaAvatarSDK.EventMap[K]) => void
    ): () => void;

    /** Get current state */
    getState(): KalturaAvatarSDK.StateValue;

    /** Get loaded assets */
    getAssets(): KalturaAvatarSDK.Assets | null;

    /** Get avatar info */
    getAvatarInfo(): KalturaAvatarSDK.AvatarInfo | null;

    /** Get iframe element */
    getIframe(): HTMLIFrameElement | null;

    /** Get talk URL */
    getTalkUrl(): string | null;

    /** Get client ID */
    getClientId(): string;

    /** Get flow ID */
    getFlowId(): string;

    /** Enable or disable transcript recording */
    setTranscriptEnabled(enabled: boolean): void;

    /** Get the current transcript */
    getTranscript(): KalturaAvatarSDK.TranscriptEntry[];

    /** Clear the transcript */
    clearTranscript(): void;

    /** Get transcript as formatted text */
    getTranscriptText(options?: {
        includeTimestamps?: boolean;
        format?: 'text' | 'markdown' | 'json';
    }): string;

    /** Download transcript as a file */
    downloadTranscript(options?: {
        filename?: string;
        format?: 'text' | 'markdown' | 'json';
        includeTimestamps?: boolean;
    }): void;

    static readonly Events: typeof KalturaAvatarSDK.Events;
    static readonly State: typeof KalturaAvatarSDK.State;
    static readonly VERSION: string;
}

declare namespace KalturaAvatarSDK {
    interface Options {
        clientId: string;
        flowId: string;
        container?: HTMLElement | string;
        config?: Partial<Config>;
    }

    interface Config {
        apiBaseUrl: string;
        meetBaseUrl: string;
        debug: boolean;
        iframeClass: string;
        iframeStyles: Partial<CSSStyleDeclaration>;
    }

    interface Assets {
        avatar: AvatarInfo;
        language: { languageCode: string };
        design: { logo: string; theme: string; color: string };
        talk_url: string;
    }

    interface AvatarInfo {
        given_name: string;
        images: string[];
        videos: string[];
    }

    interface TranscriptEntry {
        role: 'Avatar' | 'User';
        text: string;
        timestamp: Date;
    }

    const Events: {
        readonly SHOWING_JOIN_MEETING: 'showing-join-meeting';
        readonly JOIN_MEETING_CLICKED: 'join-meeting-clicked';
        readonly SHOWING_AGENT: 'showing-agent';
        readonly AGENT_TALKED: 'agent-talked';
        readonly USER_TRANSCRIPTION: 'user-transcription';
        readonly PRONUNCIATION_SCORE: 'pronunciation-score';
        readonly PERMISSIONS_DENIED: 'permissions-denied';
        readonly CONVERSATION_ENDED: 'conversation-ended';
        readonly LOAD_AGENT_ERROR: 'load-agent-error';
    };

    const State: {
        readonly UNINITIALIZED: 'uninitialized';
        readonly INITIALIZING: 'initializing';
        readonly READY: 'ready';
        readonly IN_CONVERSATION: 'in-conversation';
        readonly ENDED: 'ended';
        readonly ERROR: 'error';
    };

    type StateValue = typeof State[keyof typeof State];

    interface EventMap {
        'showing-join-meeting': unknown;
        'join-meeting-clicked': unknown;
        'showing-agent': unknown;
        'agent-talked': string | { agentContent: string };
        'user-transcription': string | { userTranscription: string };
        'pronunciation-score': number | { pronunciationScore: number };
        'permissions-denied': unknown;
        'conversation-ended': unknown;
        'load-agent-error': unknown;
        'ready': { assets: Assets };
        'started': { iframe: HTMLIFrameElement };
        'ended': {};
        'error': { message: string };
        'stateChange': { from: StateValue; to: StateValue };
    }
}
