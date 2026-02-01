let parameterCategories = null;
let parameterDefinitions = null;

async function loadParameters() {
    try {
        const response = await fetch('/api/parameters');
        const data = await response.json();

        parameterCategories = data.categories;
        parameterDefinitions = data.definitions;

        // Build form from parameter definitions
        buildConfigForm(parameterCategories, parameterDefinitions);

        // Setup model preset listeners after form is built
        if (typeof setupModelPresetListeners === 'function') {
            setupModelPresetListeners();
        }

        // Setup resource estimator after form is built
        if (typeof setupResourceEstimator === 'function') {
            setupResourceEstimator();
        }
    } catch (error) {
        console.error('Failed to load parameters:', error);
        addLog('error', 'Failed to load parameter definitions');
    }
}

function buildConfigForm(categories, definitions) {
    const container = document.getElementById('parameterSections');
    const tabs = document.getElementById('configTabs');

    // Clear existing content
    container.innerHTML = '';
    tabs.innerHTML = '';

    // Build tabs
    Object.entries(categories).forEach(([categoryId, category], index) => {
        const tab = document.createElement('button');
        tab.type = 'button';
        tab.className = 'tab' + (index === 0 ? ' active' : '');
        tab.textContent = `${category.icon} ${category.label}`;
        tab.dataset.category = categoryId;
        tab.addEventListener('click', () => switchTab(categoryId));
        tabs.appendChild(tab);
    });

    // Build parameter sections
    Object.entries(categories).forEach(([categoryId, category], index) => {
        const section = document.createElement('div');
        section.className = 'parameter-section' + (index === 0 ? ' active' : '');
        section.dataset.category = categoryId;

        const title = document.createElement('h3');
        title.textContent = `${category.icon} ${category.label}`;
        section.appendChild(title);

        // Build parameter fields
        category.parameters.forEach(paramName => {
            const paramDef = definitions[paramName];
            if (!paramDef) return;

            const field = buildParameterField(paramName, paramDef);
            section.appendChild(field);
        });

        container.appendChild(section);
    });
}

function buildParameterField(name, definition) {
    // Special handling for dataset field
    if (name === 'dataset') {
        return buildDatasetListField(name, definition);
    }

    const fieldGroup = document.createElement('div');
    fieldGroup.className = 'form-group';

    // Label with tooltip icon
    const label = document.createElement('label');
    label.htmlFor = name;
    label.textContent = definition.label;

    if (definition.required) {
        const required = document.createElement('span');
        required.className = 'required';
        required.textContent = '*';
        label.appendChild(required);
    }

    const tooltipIcon = document.createElement('span');
    tooltipIcon.className = 'tooltip-icon';
    tooltipIcon.textContent = '?';
    tooltipIcon.dataset.tooltip = definition.tooltip;
    label.appendChild(tooltipIcon);

    fieldGroup.appendChild(label);

    // Input field based on type
    let input;

    switch (definition.type) {
        case 'text':
            input = document.createElement('input');
            input.type = 'text';
            input.id = name;
            input.name = name;
            input.value = definition.default || '';
            input.placeholder = definition.placeholder || '';
            input.required = definition.required;
            break;

        case 'number':
            input = document.createElement('input');
            input.type = 'number';
            input.id = name;
            input.name = name;
            input.value = definition.default !== null && definition.default !== undefined ? definition.default : '';
            input.step = definition.step || 'any';
            if (definition.min !== undefined) input.min = definition.min;
            if (definition.max !== undefined) input.max = definition.max;
            input.required = definition.required;
            break;

        case 'select':
            input = document.createElement('select');
            input.id = name;
            input.name = name;
            input.required = definition.required;

            definition.options.forEach(option => {
                const opt = document.createElement('option');
                opt.value = option;
                opt.textContent = option;
                if (option === definition.default) {
                    opt.selected = true;
                }
                input.appendChild(opt);
            });
            break;

        case 'checkbox':
            input = document.createElement('input');
            input.type = 'checkbox';
            input.id = name;
            input.name = name;
            input.checked = definition.default;
            fieldGroup.classList.add('checkbox-group');
            break;
    }

    input.className = 'form-control';
    fieldGroup.appendChild(input);

    return fieldGroup;
}

function switchTab(categoryId) {
    // Update active tab
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.category === categoryId);
    });

    // Update visible section
    document.querySelectorAll('.parameter-section').forEach(section => {
        section.classList.toggle('active', section.dataset.category === categoryId);
    });
}

function getFormData() {
    const formData = {};
    const form = document.getElementById('configForm');
    const formElements = form.elements;

    for (let element of formElements) {
        if (element.name && parameterDefinitions[element.name]) {
            if (element.type === 'checkbox') {
                formData[element.name] = element.checked;
            } else if (element.type === 'number') {
                const value = element.value;
                if (value === '') {
                    formData[element.name] = null;
                } else {
                    formData[element.name] = parseFloat(value);
                }
            } else {
                formData[element.name] = element.value || null;
            }
        }
    }

    // Special handling for dataset list
    if (window.datasetList && window.datasetList.length > 0) {
        formData.dataset = window.datasetList.join(', ');
    } else if (!formData.dataset) {
        formData.dataset = parameterDefinitions.dataset.default;
    }

    return formData;
}

function handleResetForm() {
    if (!confirm('Reset all parameters to default values?')) {
        return;
    }

    // Reload page to reset form
    window.location.reload();
}

// Dataset list management
window.datasetList = [];

function buildDatasetListField(name, definition) {
    const fieldGroup = document.createElement('div');
    fieldGroup.className = 'form-group dataset-list-group';

    // Label with tooltip
    const label = document.createElement('label');
    label.textContent = definition.label;

    if (definition.required) {
        const required = document.createElement('span');
        required.className = 'required';
        required.textContent = '*';
        label.appendChild(required);
    }

    const tooltipIcon = document.createElement('span');
    tooltipIcon.className = 'tooltip-icon';
    tooltipIcon.textContent = '?';
    tooltipIcon.dataset.tooltip = definition.tooltip;
    label.appendChild(tooltipIcon);

    fieldGroup.appendChild(label);

    // Input container with add button
    const inputContainer = document.createElement('div');
    inputContainer.className = 'dataset-input-container';

    const input = document.createElement('input');
    input.type = 'text';
    input.id = 'datasetInput';
    input.placeholder = definition.placeholder || '';
    input.className = 'form-control';

    const addButton = document.createElement('button');
    addButton.type = 'button';
    addButton.textContent = '+ Add';
    addButton.className = 'btn btn-small btn-primary';
    addButton.onclick = () => addDataset();

    inputContainer.appendChild(input);
    inputContainer.appendChild(addButton);
    fieldGroup.appendChild(inputContainer);

    // Dataset list container
    const listContainer = document.createElement('div');
    listContainer.id = 'datasetList';
    listContainer.className = 'dataset-list';
    fieldGroup.appendChild(listContainer);

    // Hidden input to store the value for form submission
    const hiddenInput = document.createElement('input');
    hiddenInput.type = 'hidden';
    hiddenInput.id = name;
    hiddenInput.name = name;
    fieldGroup.appendChild(hiddenInput);

    // Initialize with default dataset
    if (definition.default) {
        window.datasetList = [definition.default];
        updateDatasetList();
    }

    return fieldGroup;
}

function addDataset() {
    const input = document.getElementById('datasetInput');
    const dataset = input.value.trim();

    if (!dataset) {
        alert('Please enter a dataset name');
        return;
    }

    if (window.datasetList.includes(dataset)) {
        alert('Dataset already added');
        return;
    }

    window.datasetList.push(dataset);
    input.value = '';
    updateDatasetList();
}

function removeDataset(index) {
    window.datasetList.splice(index, 1);
    updateDatasetList();
}

function updateDatasetList() {
    const container = document.getElementById('datasetList');
    const hiddenInput = document.getElementById('dataset');

    if (!container) return;

    container.innerHTML = '';

    if (window.datasetList.length === 0) {
        container.innerHTML = '<div class="dataset-empty">No datasets added</div>';
        hiddenInput.value = '';
        return;
    }

    window.datasetList.forEach((dataset, index) => {
        const item = document.createElement('div');
        item.className = 'dataset-item';

        const nameSpan = document.createElement('span');
        nameSpan.textContent = dataset;
        nameSpan.className = 'dataset-name';

        const removeBtn = document.createElement('button');
        removeBtn.type = 'button';
        removeBtn.innerHTML = '🗑️';
        removeBtn.className = 'dataset-remove-btn';
        removeBtn.title = 'Remove dataset';
        removeBtn.onclick = () => removeDataset(index);

        item.appendChild(nameSpan);
        item.appendChild(removeBtn);
        container.appendChild(item);
    });

    // Update hidden input
    hiddenInput.value = window.datasetList.join(', ');
}
