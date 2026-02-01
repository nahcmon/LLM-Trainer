// Resource estimation functionality
let estimationTimeout = null;

function setupResourceEstimator() {
    // Create estimation display panel
    createEstimationPanel();

    // Add listeners to relevant form inputs
    const relevantInputs = [
        'model_size', 'vocab_size', 'n_layers', 'hidden_size', 'n_heads',
        'seq_length', 'batch_size', 'precision', 'gradient_checkpointing',
        'epochs', 'max_steps'
    ];

    relevantInputs.forEach(inputName => {
        const input = document.querySelector(`[name="${inputName}"]`);
        if (input) {
            input.addEventListener('change', debounceEstimation);
            input.addEventListener('input', debounceEstimation);
        }
    });

    // Run initial estimation
    updateEstimation();
}

function createEstimationPanel() {
    const rightPanel = document.querySelector('.right-panel');
    if (!rightPanel) return;

    const estimationSection = document.createElement('div');
    estimationSection.className = 'estimation-section';
    estimationSection.style.marginBottom = '20px';
    estimationSection.innerHTML = `
        <h2>Resource Estimates</h2>
        <div class="estimation-grid">
            <div class="estimation-card">
                <div class="estimation-label">Model Parameters</div>
                <div class="estimation-value" id="estimatedParams">Calculating...</div>
            </div>
            <div class="estimation-card">
                <div class="estimation-label">VRAM Required</div>
                <div class="estimation-value" id="estimatedVRAM">Calculating...</div>
            </div>
            <div class="estimation-card">
                <div class="estimation-label">Estimated Time</div>
                <div class="estimation-value" id="estimatedTime">Calculating...</div>
            </div>
        </div>
        <div id="estimationDetails" class="estimation-details"></div>
        <div id="estimationWarning" class="estimation-warning" style="display: none;"></div>
    `;

    // Insert before progress section
    const progressSection = rightPanel.querySelector('.progress-section');
    if (progressSection) {
        rightPanel.insertBefore(estimationSection, progressSection);
    } else {
        rightPanel.prepend(estimationSection);
    }

    // Add styles
    if (!document.getElementById('estimation-styles')) {
        const style = document.createElement('style');
        style.id = 'estimation-styles';
        style.textContent = `
            .estimation-section {
                background: white;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .estimation-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-top: 15px;
            }
            .estimation-card {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 6px;
                border-left: 4px solid #007bff;
            }
            .estimation-label {
                font-size: 12px;
                color: #666;
                margin-bottom: 5px;
                font-weight: 500;
            }
            .estimation-value {
                font-size: 20px;
                font-weight: bold;
                color: #333;
            }
            .estimation-details {
                margin-top: 15px;
                padding: 12px;
                background: #f8f9fa;
                border-radius: 4px;
                font-size: 13px;
                color: #666;
                font-family: monospace;
                white-space: pre-line;
            }
            .estimation-warning {
                margin-top: 10px;
                padding: 12px;
                background: #fff3cd;
                border: 1px solid #ffc107;
                border-radius: 4px;
                color: #856404;
            }
        `;
        document.head.appendChild(style);
    }
}

function debounceEstimation() {
    clearTimeout(estimationTimeout);
    estimationTimeout = setTimeout(updateEstimation, 500);
}

async function updateEstimation() {
    try {
        const config = getFormData();

        const response = await fetch('/api/training/estimate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });

        if (!response.ok) {
            throw new Error('Failed to get estimation');
        }

        const estimation = await response.json();
        displayEstimation(estimation);
    } catch (error) {
        console.error('Estimation error:', error);
        document.getElementById('estimatedParams').textContent = 'Error';
        document.getElementById('estimatedVRAM').textContent = 'Error';
        document.getElementById('estimatedTime').textContent = 'Error';
    }
}

function displayEstimation(estimation) {
    // Display parameter count
    const paramCount = estimation.parameter_count;
    document.getElementById('estimatedParams').textContent = formatParamCount(paramCount);

    // Display VRAM
    const totalVRAM = estimation.vram.total;
    document.getElementById('estimatedVRAM').textContent = `${totalVRAM} GB`;

    // Display time estimate
    document.getElementById('estimatedTime').textContent = estimation.time.formatted;

    // Display detailed breakdown
    const detailsEl = document.getElementById('estimationDetails');
    detailsEl.textContent = `VRAM Breakdown:
Model: ${estimation.vram.model} GB
Optimizer: ${estimation.vram.optimizer} GB
Gradients: ${estimation.vram.gradients} GB
Activations: ${estimation.vram.activations} GB
Overhead: ${estimation.vram.overhead} GB

${estimation.time.note}`;

    // Check for warnings
    const warningEl = document.getElementById('estimationWarning');
    if (totalVRAM > 24) {
        warningEl.style.display = 'block';
        warningEl.innerHTML = `<strong>⚠️ Warning:</strong> Estimated VRAM (${totalVRAM} GB) exceeds typical consumer GPU memory. Consider:
<ul style="margin: 5px 0 0 20px;">
<li>Reducing batch size</li>
<li>Enabling gradient checkpointing</li>
<li>Using smaller model size</li>
<li>Reducing sequence length</li>
</ul>`;
    } else {
        warningEl.style.display = 'none';
    }
}
