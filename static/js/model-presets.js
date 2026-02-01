// Model size presets - sync with utils/model_presets.py
const MODEL_SIZE_PRESETS = {
    "tiny-10M": {
        description: "~10M parameters - Testing/debugging",
        n_layers: 6,
        hidden_size: 384,
        n_heads: 6,
        n_inner: 1536,
        params: 10_000_000
    },
    "tiny-50M": {
        description: "~50M parameters - Small experiments",
        n_layers: 8,
        hidden_size: 512,
        n_heads: 8,
        n_inner: 2048,
        params: 50_000_000
    },
    "small-100M": {
        description: "~100M parameters - Similar to GPT-2 Small",
        n_layers: 12,
        hidden_size: 768,
        n_heads: 12,
        n_inner: 3072,
        params: 100_000_000
    },
    "small-300M": {
        description: "~300M parameters - Similar to GPT-2 Medium",
        n_layers: 24,
        hidden_size: 1024,
        n_heads: 16,
        n_inner: 4096,
        params: 300_000_000
    },
    "medium-500M": {
        description: "~500M parameters - Educational/research",
        n_layers: 24,
        hidden_size: 1280,
        n_heads: 20,
        n_inner: 5120,
        params: 500_000_000
    },
    "medium-1B": {
        description: "~1B parameters - Similar to GPT-2 Large",
        n_layers: 36,
        hidden_size: 1536,
        n_heads: 24,
        n_inner: 6144,
        params: 1_000_000_000
    },
    "large-2B": {
        description: "~2B parameters - Serious training",
        n_layers: 32,
        hidden_size: 2048,
        n_heads: 32,
        n_inner: 8192,
        params: 2_000_000_000
    },
    "large-7B": {
        description: "~7B parameters - Similar to LLaMA 7B",
        n_layers: 32,
        hidden_size: 4096,
        n_heads: 32,
        n_inner: 11008,
        params: 7_000_000_000
    },
    "xl-13B": {
        description: "~13B parameters - Similar to LLaMA 13B",
        n_layers: 40,
        hidden_size: 5120,
        n_heads: 40,
        n_inner: 13824,
        params: 13_000_000_000
    },
    "xl-30B": {
        description: "~30B parameters - Large scale training",
        n_layers: 60,
        hidden_size: 6656,
        n_heads: 52,
        n_inner: 17920,
        params: 30_000_000_000
    },
    "xl-70B": {
        description: "~70B parameters - Similar to LLaMA 70B",
        n_layers: 80,
        hidden_size: 8192,
        n_heads: 64,
        n_inner: 28672,
        params: 70_000_000_000
    },
    "xxl-175B": {
        description: "~175B parameters - Similar to GPT-3",
        n_layers: 96,
        hidden_size: 12288,
        n_heads: 96,
        n_inner: 49152,
        params: 175_000_000_000
    },
    "xxl-500B": {
        description: "~500B parameters - Mega scale",
        n_layers: 128,
        hidden_size: 16384,
        n_heads: 128,
        n_inner: 65536,
        params: 500_000_000_000
    },
    "ultra-1T": {
        description: "~1T parameters - Ultra scale",
        n_layers: 160,
        hidden_size: 20480,
        n_heads: 160,
        n_inner: 81920,
        params: 1_000_000_000_000
    },
    "ultra-2T": {
        description: "~2T parameters - Maximum scale",
        n_layers: 200,
        hidden_size: 25600,
        n_heads: 200,
        n_inner: 102400,
        params: 2_000_000_000_000
    },
    "custom": {
        description: "Custom configuration - Set manually",
        n_layers: null,
        hidden_size: null,
        n_heads: null,
        n_inner: null,
        params: null
    }
};

function applyModelPreset(presetName) {
    const preset = MODEL_SIZE_PRESETS[presetName];
    if (!preset) {
        console.warn('Unknown preset:', presetName);
        return;
    }

    // If custom, enable manual editing
    const isCustom = presetName === 'custom';
    const architectureInputs = ['n_layers', 'hidden_size', 'n_heads', 'n_inner'];

    architectureInputs.forEach(inputName => {
        const input = document.querySelector(`[name="${inputName}"]`);
        if (input) {
            if (!isCustom && preset[inputName] !== null) {
                input.value = preset[inputName];
                input.disabled = true;
                input.style.opacity = '0.6';
            } else {
                input.disabled = false;
                input.style.opacity = '1';
            }
        }
    });

    // Update parameter count display
    updateParameterCount();
}

function updateParameterCount() {
    const vocabSize = parseInt(document.querySelector('[name="vocab_size"]')?.value) || 32000;
    const seqLength = parseInt(document.querySelector('[name="seq_length"]')?.value) || 1024;
    const nLayers = parseInt(document.querySelector('[name="n_layers"]')?.value) || 12;
    const hiddenSize = parseInt(document.querySelector('[name="hidden_size"]')?.value) || 768;

    const paramCount = calculateModelParams(vocabSize, nLayers, hiddenSize, seqLength);
    const formatted = formatParamCount(paramCount);

    // Create or update parameter count display
    let display = document.getElementById('paramCountDisplay');
    if (!display) {
        const modelSizeInput = document.querySelector('[name="model_size"]');
        if (modelSizeInput) {
            display = document.createElement('div');
            display.id = 'paramCountDisplay';
            display.style.marginTop = '8px';
            display.style.padding = '8px';
            display.style.backgroundColor = '#f0f0f0';
            display.style.borderRadius = '4px';
            display.style.fontSize = '14px';
            display.style.fontWeight = 'bold';
            modelSizeInput.parentElement.appendChild(display);
        }
    }

    if (display) {
        display.textContent = `Estimated parameters: ${formatted}`;
    }
}

function calculateModelParams(vocabSize, nLayers, hiddenSize, seqLength) {
    // Token embeddings + position embeddings
    const embeddingParams = vocabSize * hiddenSize + seqLength * hiddenSize;

    // Transformer layers (attention + FFN + layer norms)
    const layerParams = 12 * (hiddenSize ** 2);
    const totalLayerParams = nLayers * layerParams;

    // Output layer (LM head)
    const outputParams = vocabSize * hiddenSize;

    return embeddingParams + totalLayerParams + outputParams;
}

function formatParamCount(paramCount) {
    if (paramCount >= 1_000_000_000_000) {
        return `${(paramCount / 1_000_000_000_000).toFixed(2)}T`;
    } else if (paramCount >= 1_000_000_000) {
        return `${(paramCount / 1_000_000_000).toFixed(2)}B`;
    } else if (paramCount >= 1_000_000) {
        return `${(paramCount / 1_000_000).toFixed(2)}M`;
    } else if (paramCount >= 1_000) {
        return `${(paramCount / 1_000).toFixed(2)}K`;
    } else {
        return paramCount.toString();
    }
}

// Setup listeners when form is ready
function setupModelPresetListeners() {
    const modelSizeSelect = document.querySelector('[name="model_size"]');
    if (modelSizeSelect) {
        modelSizeSelect.addEventListener('change', (e) => {
            applyModelPreset(e.target.value);
        });

        // Apply initial preset
        applyModelPreset(modelSizeSelect.value);
    }

    // Update param count when architecture params change
    const architectureInputs = ['vocab_size', 'seq_length', 'n_layers', 'hidden_size'];
    architectureInputs.forEach(inputName => {
        const input = document.querySelector(`[name="${inputName}"]`);
        if (input) {
            input.addEventListener('input', updateParameterCount);
        }
    });
}
