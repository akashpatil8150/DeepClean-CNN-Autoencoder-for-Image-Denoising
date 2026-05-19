// DOM Elements
const uploadBox = document.getElementById('uploadBox');
const imageInput = document.getElementById('imageInput');
const denoiseBtn = document.getElementById('denoiseBtn');
const resultsSection = document.getElementById('resultsSection');
const loading = document.getElementById('loading');
const error = document.getElementById('error');
const originalImg = document.getElementById('originalImg');
const denoisedImg = document.getElementById('denoisedImg');
const samplesGrid = document.getElementById('samplesGrid');

let selectedFile = null;

// ── Load sample images on page load ──────────────────────────────────────────
async function loadSamples() {
    try {
        const res = await fetch('/api/samples');
        const samples = await res.json();
        samplesGrid.innerHTML = '';
        samples.forEach(s => {
            const card = document.createElement('div');
            card.className = 'sample-card';
            card.innerHTML = `
                <img src="${s.thumbnail}" alt="${s.label}" title="Click to denoise">
                <span>${s.label}</span>
            `;
            card.addEventListener('click', () => denoiseSample(s.filename, card));
            samplesGrid.appendChild(card);
        });
    } catch (e) {
        samplesGrid.innerHTML = '<p style="color:#999">Could not load samples.</p>';
    }
}

async function denoiseSample(filename, card) {
    // Highlight selected card
    document.querySelectorAll('.sample-card').forEach(c => c.classList.remove('active'));
    card.classList.add('active');

    showLoading();
    hideError();

    try {
        const res = await fetch('/api/denoise-sample', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Failed');
        showResults(data.original, data.denoised);
    } catch (err) {
        hideLoading();
        showError(err.message);
    }
}

// ── Upload flow ───────────────────────────────────────────────────────────────
uploadBox.addEventListener('click', () => imageInput.click());

imageInput.addEventListener('change', (e) => handleFile(e.target.files[0]));

uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.classList.add('dragover');
});
uploadBox.addEventListener('dragleave', () => uploadBox.classList.remove('dragover'));
uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.classList.remove('dragover');
    handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
    if (!file) return;
    if (!file.type.startsWith('image/')) { showError('Please upload an image file'); return; }
    selectedFile = file;
    denoiseBtn.disabled = false;
    const uploadContent = uploadBox.querySelector('.upload-content');
    uploadContent.innerHTML = `
        <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"></path>
            <polyline points="13 2 13 9 20 9"></polyline>
        </svg>
        <p style="color:#667eea;font-weight:600;">${file.name}</p>
        <span>Click to change file</span>
    `;
    hideError();
    resultsSection.style.display = 'none';
}

denoiseBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    showLoading();
    hideError();
    denoiseBtn.disabled = true;

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
        const res = await fetch('/denoise', { method: 'POST', body: formData });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Failed to denoise image');
        showResults(data.original, data.denoised);
        denoiseBtn.disabled = false;
    } catch (err) {
        hideLoading();
        showError(err.message);
        denoiseBtn.disabled = false;
    }
});

// ── Helpers ───────────────────────────────────────────────────────────────────
function showResults(original, denoised) {
    originalImg.src = original;
    denoisedImg.src = denoised;
    hideLoading();
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function showLoading() {
    loading.style.display = 'block';
    resultsSection.style.display = 'none';
}

function hideLoading() { loading.style.display = 'none'; }

function showError(message) {
    error.textContent = message;
    error.style.display = 'block';
    setTimeout(hideError, 5000);
}

function hideError() { error.style.display = 'none'; }

// Init
loadSamples();
