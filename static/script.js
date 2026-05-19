// DOM Elements
const uploadBox = document.getElementById('uploadBox');
const imageInput = document.getElementById('imageInput');
const denoiseBtn = document.getElementById('denoiseBtn');
const resultsSection = document.getElementById('resultsSection');
const loading = document.getElementById('loading');
const error = document.getElementById('error');
const originalImg = document.getElementById('originalImg');
const denoisedImg = document.getElementById('denoisedImg');

let selectedFile = null;

// Click to upload
uploadBox.addEventListener('click', () => {
    imageInput.click();
});

// File selection
imageInput.addEventListener('change', (e) => {
    handleFile(e.target.files[0]);
});

// Drag and drop
uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.classList.add('dragover');
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.classList.remove('dragover');
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.classList.remove('dragover');
    handleFile(e.dataTransfer.files[0]);
});

// Handle file selection
function handleFile(file) {
    if (!file) return;
    
    if (!file.type.startsWith('image/')) {
        showError('Please upload an image file');
        return;
    }
    
    selectedFile = file;
    denoiseBtn.disabled = false;
    
    // Update upload box to show file name
    const uploadContent = uploadBox.querySelector('.upload-content');
    uploadContent.innerHTML = `
        <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"></path>
            <polyline points="13 2 13 9 20 9"></polyline>
        </svg>
        <p style="color: #667eea; font-weight: 600;">${file.name}</p>
        <span>Click to change file</span>
    `;
    
    hideError();
    resultsSection.style.display = 'none';
}

// Denoise button click
denoiseBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    
    // Show loading
    loading.style.display = 'block';
    resultsSection.style.display = 'none';
    hideError();
    denoiseBtn.disabled = true;
    
    // Create form data
    const formData = new FormData();
    formData.append('image', selectedFile);
    
    try {
        const response = await fetch('/denoise', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Failed to denoise image');
        }
        
        // Display results
        originalImg.src = data.original;
        denoisedImg.src = data.denoised;
        
        loading.style.display = 'none';
        resultsSection.style.display = 'block';
        denoiseBtn.disabled = false;
        
    } catch (err) {
        loading.style.display = 'none';
        showError(err.message);
        denoiseBtn.disabled = false;
    }
});

// Error handling
function showError(message) {
    error.textContent = message;
    error.style.display = 'block';
    setTimeout(() => {
        hideError();
    }, 5000);
}

function hideError() {
    error.style.display = 'none';
}
