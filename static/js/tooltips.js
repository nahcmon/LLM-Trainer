const tooltip = document.getElementById('tooltip');
let currentTooltipElement = null;

document.addEventListener('mouseover', (e) => {
    if (e.target.classList.contains('tooltip-icon')) {
        showTooltip(e.target);
    }
});

document.addEventListener('mouseout', (e) => {
    if (e.target.classList.contains('tooltip-icon')) {
        hideTooltip();
    }
});

function showTooltip(element) {
    const text = element.dataset.tooltip;
    if (!text) return;

    tooltip.textContent = text;
    tooltip.style.display = 'block';

    // Position tooltip
    const rect = element.getBoundingClientRect();
    const tooltipRect = tooltip.getBoundingClientRect();

    let left = rect.left;
    let top = rect.bottom + 5;

    // Adjust if tooltip goes off-screen
    if (left + tooltipRect.width > window.innerWidth) {
        left = window.innerWidth - tooltipRect.width - 10;
    }

    if (top + tooltipRect.height > window.innerHeight) {
        top = rect.top - tooltipRect.height - 5;
    }

    tooltip.style.left = left + 'px';
    tooltip.style.top = top + window.scrollY + 'px';

    currentTooltipElement = element;
}

function hideTooltip() {
    tooltip.style.display = 'none';
    currentTooltipElement = null;
}

// Update tooltip position on scroll
document.addEventListener('scroll', () => {
    if (currentTooltipElement) {
        const rect = currentTooltipElement.getBoundingClientRect();
        tooltip.style.left = rect.left + 'px';
        tooltip.style.top = (rect.bottom + 5 + window.scrollY) + 'px';
    }
});
