// Application state
let autoScroll = true;

// Initialize application
document.addEventListener('DOMContentLoaded', async () => {
    // Load parameter definitions and build form
    await loadParameters();

    // Initialize WebSocket connection
    initWebSocket();

    // Setup event listeners
    setupEventListeners();

    // Load initial status
    await loadStatus();
});

function setupEventListeners() {
    document.getElementById('configForm').addEventListener('submit', handleStartTraining);
    document.getElementById('stopBtn').addEventListener('click', handleStopTraining);
    document.getElementById('unloadBtn').addEventListener('click', handleUnloadModel);
    document.getElementById('resetBtn').addEventListener('click', handleResetForm);
    document.getElementById('clearLogsBtn').addEventListener('click', clearLogs);
    document.getElementById('autoScrollBtn').addEventListener('click', toggleAutoScroll);
}

async function handleStartTraining(e) {
    e.preventDefault();

    const config = getFormData();

    // Validate that hidden_size is divisible by n_heads
    if (config.hidden_size % config.n_heads !== 0) {
        alert(`Error: hidden_size (${config.hidden_size}) must be divisible by n_heads (${config.n_heads})`);
        return;
    }

    try {
        const response = await fetch('/api/training/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });

        if (!response.ok) {
            const error = await response.json();
            alert(`Failed to start training: ${error.detail}`);
            return;
        }

        const result = await response.json();
        addLog('info', result.message);

    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}

async function handleStopTraining() {
    if (!confirm('Are you sure you want to stop training?')) {
        return;
    }

    try {
        const response = await fetch('/api/training/stop', {
            method: 'POST'
        });

        if (!response.ok) {
            const error = await response.json();
            alert(`Failed to stop training: ${error.detail}`);
            return;
        }

        const result = await response.json();
        addLog('warning', result.message);

    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}

async function handleUnloadModel() {
    if (!confirm('Are you sure you want to unload the model from VRAM? This will free up GPU memory.')) {
        return;
    }

    try {
        const response = await fetch('/api/training/unload', {
            method: 'POST'
        });

        if (!response.ok) {
            const error = await response.json();
            alert(`Failed to unload model: ${error.detail}`);
            return;
        }

        const result = await response.json();
        addLog('info', result.message);

    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}

async function loadStatus() {
    try {
        const response = await fetch('/api/training/status');
        const status = await response.json();

        updateStatus(status.state, status.error_message);
        if (status.progress) {
            updateProgress(status.progress);
        }
    } catch (error) {
        console.error('Failed to load status:', error);
    }
}

function updateStatus(state, errorMessage = null) {
    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');

    // Update status indicator
    statusDot.className = 'status-dot status-' + state;

    switch (state) {
        case 'idle':
            statusText.textContent = 'Idle';
            startBtn.disabled = false;
            stopBtn.disabled = true;
            break;
        case 'starting':
            statusText.textContent = 'Starting...';
            startBtn.disabled = true;
            stopBtn.disabled = false;
            break;
        case 'running':
            statusText.textContent = 'Running';
            startBtn.disabled = true;
            stopBtn.disabled = false;
            break;
        case 'stopping':
            statusText.textContent = 'Stopping...';
            startBtn.disabled = true;
            stopBtn.disabled = true;
            break;
        case 'completed':
            statusText.textContent = 'Completed';
            startBtn.disabled = false;
            stopBtn.disabled = true;
            break;
        case 'error':
            statusText.textContent = 'Error';
            startBtn.disabled = false;
            stopBtn.disabled = true;
            if (errorMessage) {
                addLog('error', `Training failed: ${errorMessage}`);
            }
            break;
    }
}

function updateProgress(progressData) {
    // Debug logging
    console.log('Progress update received:', {
        current_step: progressData.current_step,
        total_steps: progressData.total_steps,
        percentage: progressData.total_steps > 0 ? (progressData.current_step / progressData.total_steps) * 100 : 0
    });

    // Update stat cards
    document.getElementById('currentStep').textContent = progressData.current_step || 0;
    document.getElementById('totalSteps').textContent = progressData.total_steps || 0;
    document.getElementById('currentEpoch').textContent = progressData.current_epoch || 0;
    document.getElementById('loss').textContent = (progressData.loss || 0).toFixed(4);
    document.getElementById('learningRate').textContent =
        (progressData.metrics?.learning_rate || 0).toExponential(3);

    // Update elapsed time
    if (progressData.elapsed_time !== undefined) {
        document.getElementById('elapsedTime').textContent = formatTime(progressData.elapsed_time);
    }

    // Update ETA
    if (progressData.metrics?.eta_seconds !== undefined) {
        document.getElementById('eta').textContent = formatTime(progressData.metrics.eta_seconds);
    }

    // Update speed
    if (progressData.metrics?.steps_per_sec !== undefined) {
        const speed = progressData.metrics.steps_per_sec;
        document.getElementById('speed').textContent = `${speed.toFixed(2)} steps/s`;
    }

    // Update progress bar
    const percentage = progressData.total_steps > 0
        ? (progressData.current_step / progressData.total_steps) * 100
        : 0;

    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');

    if (!progressBar || !progressText) {
        console.error('Progress bar elements not found!');
        return;
    }

    console.log('Setting progress bar width to:', `${percentage}%`);
    console.log('Progress bar element:', progressBar);
    progressBar.style.width = `${percentage}%`;
    progressText.textContent = `${percentage.toFixed(1)}%`;

    // Change text color based on progress bar position
    // White text when bar is past 50%, black otherwise
    if (percentage >= 50) {
        progressText.style.color = '#ffffff';
    } else {
        progressText.style.color = '#495057';
    }
}

function addLog(level, message) {
    const logsContainer = document.getElementById('logsContainer');
    const timestamp = new Date().toLocaleTimeString();

    const logEntry = document.createElement('div');
    logEntry.className = `log-entry log-${level}`;
    logEntry.innerHTML = `
        <span class="log-timestamp">[${timestamp}]</span>
        <span class="log-level">[${level.toUpperCase()}]</span>
        <span class="log-message">${escapeHtml(message)}</span>
    `;

    logsContainer.appendChild(logEntry);

    // Keep only last 500 log entries in DOM
    while (logsContainer.children.length > 500) {
        logsContainer.removeChild(logsContainer.firstChild);
    }

    if (autoScroll) {
        logsContainer.scrollTop = logsContainer.scrollHeight;
    }
}

function clearLogs() {
    const logsContainer = document.getElementById('logsContainer');
    logsContainer.innerHTML = '';
    addLog('info', 'Logs cleared');
}

function toggleAutoScroll() {
    autoScroll = !autoScroll;
    const btn = document.getElementById('autoScrollBtn');
    btn.classList.toggle('active', autoScroll);

    if (autoScroll) {
        const logsContainer = document.getElementById('logsContainer');
        logsContainer.scrollTop = logsContainer.scrollHeight;
    }
}

function formatTime(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
