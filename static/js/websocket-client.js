let ws = null;
let reconnectAttempts = 0;
const maxReconnectAttempts = 10;

function initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('WebSocket connected');
        addLog('info', 'Connected to training server');
        reconnectAttempts = 0;
    };

    ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        handleWebSocketMessage(message);
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        addLog('error', 'WebSocket connection error');
    };

    ws.onclose = () => {
        console.log('WebSocket disconnected');
        addLog('warning', 'Disconnected from training server');

        // Attempt to reconnect
        if (reconnectAttempts < maxReconnectAttempts) {
            reconnectAttempts++;
            const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000);
            addLog('info', `Reconnecting in ${delay / 1000}s... (attempt ${reconnectAttempts}/${maxReconnectAttempts})`);
            setTimeout(initWebSocket, delay);
        } else {
            addLog('error', 'Failed to reconnect after multiple attempts. Please refresh the page.');
        }
    };
}

function handleWebSocketMessage(message) {
    switch (message.type) {
        case 'initial_status':
            updateStatus(message.data.state, message.data.error_message);
            if (message.data.progress) {
                updateProgress(message.data.progress);
            }
            break;

        case 'progress':
            updateProgress(message.data);
            break;

        case 'log':
            addLog(message.data.level, message.data.message);
            break;

        case 'state_change':
            updateStatus(message.data.state, message.data.error_message);
            break;

        case 'pong':
            // Heartbeat response
            break;

        default:
            console.warn('Unknown message type:', message.type);
    }
}

// Send periodic ping to keep connection alive
setInterval(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send('ping');
    }
}, 30000);
