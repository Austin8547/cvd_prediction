document.addEventListener('DOMContentLoaded', () => {
    // 1. File Upload and Drag & Drop Elements
    setupDropZone('right-eye-zone', 'right-eye-input', 'right-filename', 'right-preview');
    setupDropZone('left-eye-zone', 'left-eye-input', 'left-filename', 'left-preview');

    // 2. Tab switcher logic
    setupTabs();

    // 3. Form Submission Handling
    const form = document.getElementById('predictor-form');
    const submitBtn = document.getElementById('submit-button');
    const btnText = submitBtn.querySelector('.btn-text');
    const btnSpinner = document.getElementById('btn-spinner');

    const emptyResults = document.getElementById('empty-results');
    const activeResults = document.getElementById('active-results');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const rightInput = document.getElementById('right-eye-input');
        const leftInput = document.getElementById('left-eye-input');
        const ageInput = document.getElementById('age-input');
        const gender = document.querySelector('input[name="gender"]:checked').value;

        if (!rightInput.files[0] || !leftInput.files[0]) {
            alert("Please select both Right and Left eye fundus images.");
            return;
        }

        // Set Loading State
        submitBtn.disabled = true;
        btnText.style.display = 'none';
        btnSpinner.style.display = 'block';

        const formData = new FormData();
        formData.append('right_eye', rightInput.files[0]);
        formData.append('left_eye', leftInput.files[0]);
        formData.append('age', ageInput.value);
        formData.append('gender', gender);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || "Server returned an error");
            }

            const data = await response.json();

            // Populate dashboard
            updateDashboard(data, ageInput.value, gender);

            // Toggle screens
            emptyResults.style.display = 'none';
            activeResults.style.display = 'block';

        } catch (error) {
            console.error(error);
            alert("Prediction failed: " + error.message);
        } finally {
            // Reset Loading State
            submitBtn.disabled = false;
            btnText.style.display = 'block';
            btnSpinner.style.display = 'none';
        }
    });
});

/**
 * Binds click, change, drag and preview listeners to an upload zone.
 */
function setupDropZone(zoneId, inputId, filenameId, previewId) {
    const zone = document.getElementById(zoneId);
    const input = document.getElementById(inputId);
    const filenameEl = document.getElementById(filenameId);
    const previewEl = document.getElementById(previewId);
    const contentEl = zone.querySelector('.drop-zone-content');

    // Trigger click on click
    zone.addEventListener('click', () => input.click());

    // File change handler
    input.addEventListener('change', () => {
        if (input.files.length > 0) {
            handleSelectedFile(input.files[0], filenameEl, previewEl, contentEl);
        }
    });

    // Drag-over handlers
    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.style.borderColor = 'var(--accent-cyan)';
    });

    zone.addEventListener('dragleave', () => {
        zone.style.borderColor = 'var(--border-color)';
    });

    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.style.borderColor = 'var(--border-color)';
        if (e.dataTransfer.files.length > 0) {
            input.files = e.dataTransfer.files;
            handleSelectedFile(e.dataTransfer.files[0], filenameEl, previewEl, contentEl);
        }
    });
}

function handleSelectedFile(file, filenameEl, previewEl, contentEl) {
    filenameEl.textContent = file.name;

    const reader = new FileReader();
    reader.onload = (e) => {
        previewEl.src = e.target.result;
        previewEl.style.display = 'block';
        contentEl.style.opacity = '0';
    };
    reader.readAsDataURL(file);
}

/**
 * Handles toggling tabs in the explainability visualization card.
 */
function setupTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');

            // Deactivate all
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.style.display = 'none');

            // Activate current
            btn.classList.add('active');
            document.getElementById(targetTab).style.display = 'block';
        });
    });
}

/**
 * Fills the metrics and visualizations in the results dashboard.
 */
function updateDashboard(data, age, gender) {
    const probPercent = (data.probability * 100).toFixed(1);
    
    // 1. Update risk metrics and bar widths
    document.getElementById('risk-percentage').textContent = `${probPercent}%`;
    
    const riskBar = document.getElementById('risk-bar');
    riskBar.style.width = `${probPercent}%`;

    // Reset gauge color class
    riskBar.style.background = 'linear-gradient(90deg, var(--accent-cyan), var(--accent-blue))';
    if (data.probability >= data.threshold_youden) {
        riskBar.style.background = 'linear-gradient(90deg, var(--accent-rose), #e11d48)';
    }

    // 2. Set classification status
    const categoryEl = document.getElementById('risk-category');
    const explanationEl = document.getElementById('risk-explanation');
    
    categoryEl.className = 'classification-value'; // Reset classes

    if (data.prediction_youden === "Thickened") {
        categoryEl.textContent = "Thickened CIMT";
        categoryEl.classList.add('state-thickened');
        explanationEl.textContent = `Predicted probability (${probPercent}%) exceeds the optimal validation threshold of ${(data.threshold_youden * 100).toFixed(1)}%. Patient displays elevated risks of arterial wall thickening.`;
    } else {
        categoryEl.textContent = "Normal CIMT";
        categoryEl.classList.add('state-normal');
        explanationEl.textContent = `Predicted probability (${probPercent}%) lies within normal bounds (below threshold of ${(data.threshold_youden * 100).toFixed(1)}%). Patient displays standard arterial wall thickness.`;
    }

    // 3. Set badge values
    const badgeYouden = document.getElementById('badge-youden');
    const badgeSens85 = document.getElementById('badge-sens85');

    badgeYouden.className = 'mini-badge';
    badgeSens85.className = 'mini-badge';

    if (data.prediction_youden === "Thickened") {
        badgeYouden.textContent = "Youden J: Thickened";
        badgeYouden.classList.add('badge-thickened');
    } else {
        badgeYouden.textContent = "Youden J: Normal";
        badgeYouden.classList.add('badge-normal');
    }

    if (data.prediction_sens85 === "Thickened") {
        badgeSens85.textContent = "Sens85%: Thickened";
        badgeSens85.classList.add('badge-thickened');
    } else {
        badgeSens85.textContent = "Sens85%: Normal";
        badgeSens85.classList.add('badge-normal');
    }

    // 4. Update GradCAM images
    document.getElementById('right-display-original').src = data.right_original;
    document.getElementById('right-display-cam').src = data.right_gradcam;
    document.getElementById('left-display-original').src = data.left_original;
    document.getElementById('left-display-cam').src = data.left_gradcam;
}
