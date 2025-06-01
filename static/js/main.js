/**
 * Main JavaScript file for DeepfakeSoundShield
 * Rebuilt file attachment component with improved drag and drop functionality
 */

document.addEventListener('DOMContentLoaded', function () {
    // File upload related elements
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file');
    const fileNameDisplay = document.getElementById('file-name');
    const submitBtn = document.getElementById('submit-btn');
    const uploadForm = document.getElementById('upload-form');
    const progressContainer = document.querySelector('.progress-container');
    const progressBar = document.querySelector('.progress-bar');
    const progressText = document.getElementById('progress-text');
    const loadingOverlay = document.getElementById('loading-overlay');
    const browseBtn = document.getElementById('browse-button');

    // Skip if elements don't exist (e.g., on results page)
    if (!dropArea || !fileInput) return;

    // File type and size validation constants
    const ALLOWED_EXTENSIONS = ['wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac', 'wma'];
    const MAX_FILE_SIZE_MB = 200;
    const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024;

    // Initialize the component
    initializeFileUpload();

    /**
     * Initialize file upload component with all event listeners
     */
    function initializeFileUpload() {
        // Prevent default drag behaviors for the drop area
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, preventDefaults, false);
            // Also prevent defaults on the document to improve usability
            document.addEventListener(eventName, preventDefaults, false);
        });

        // Highlight drop area when drag is over it
        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, highlight, false);
            // Also add document listeners to detect when dragging anywhere on the page
            document.addEventListener(eventName, documentDragEnter, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, unhighlight, false);
            // Remove highlight when dragging leaves the document
            document.addEventListener(eventName, documentDragLeave, false);
        });

        // Handle dropped files
        dropArea.addEventListener('drop', handleDrop, false);

        // Handle click on drop area (should open file browser)
        dropArea.addEventListener('click', function(e) {
            // Only trigger if clicked directly on the drop area (not on a button)
            if (e.target === dropArea || !e.target.closest('button')) {
                fileInput.click();
            }
        });

        // Handle file browse button click
        if (browseBtn) {
            browseBtn.addEventListener('click', function(e) {
                e.preventDefault();
                fileInput.click();
            });
        }

        // Handle file input change
        fileInput.addEventListener('change', function() {
            handleFiles(this.files);
        });

        // Handle form submission
        if (uploadForm) {
            uploadForm.addEventListener('submit', handleFormSubmit);
        }
    }

    /**
     * Prevent default behaviors for drag events
     */
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    /**
     * Handle document drag enter events for better UX
     */
    function documentDragEnter(e) {
        // Only respond to file dragging
        if (e.dataTransfer.types.includes('Files')) {
            dropArea.classList.add('dragover');
            // Display a message in the drop area
            dropArea.setAttribute('data-message', 'Drop your audio file here');
        }
    }

    /**
     * Handle document drag leave events
     */
    function documentDragLeave(e) {
        // Only check for leave if we're leaving the document
        if (e.currentTarget.contains(e.relatedTarget)) {
            return;
        }
        dropArea.classList.remove('dragover');
        dropArea.removeAttribute('data-message');
    }

    /**
     * Highlight the drop area when an item is being dragged over it
     */
    function highlight() {
        dropArea.classList.add('highlight');
    }

    /**
     * Remove highlight when drag leaves the drop area or item is dropped
     */
    function unhighlight() {
        dropArea.classList.remove('highlight');
        dropArea.classList.remove('dragover');
        dropArea.removeAttribute('data-message');
    }

    /**
     * Handle file drop event
     */
    function handleDrop(e) {
        unhighlight();
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    /**
     * Process files from drop or input
     */
    function handleFiles(files) {
        if (!files.length) return;
        
        const file = files[0];
        validateAndDisplayFile(file);
    }

    /**
     * Validate file and update UI based on validity
     */
    function validateAndDisplayFile(file) {
        const fileExt = file.name.split('.').pop().toLowerCase();
        const fileSizeMB = (file.size / (1024 * 1024)).toFixed(2);
        
        // Validate file extension
        if (!ALLOWED_EXTENSIONS.includes(fileExt)) {
            showError(`Unsupported file format (.${fileExt}). Allowed formats: ${ALLOWED_EXTENSIONS.join(', ')}`);
            fileInput.value = ''; // Clear invalid file
            return;
        }
        
        // Validate file size
        if (file.size > MAX_FILE_SIZE_BYTES) {
            showError(`File size (${fileSizeMB}MB) exceeds the maximum allowed size of ${MAX_FILE_SIZE_MB}MB`);
            fileInput.value = ''; // Clear invalid file
            return;
        }
        
        // File is valid, update UI
        fileNameDisplay.innerHTML = `
            <div class="alert alert-info">
                <i class="fas fa-file-audio me-2"></i>
                <strong>${file.name}</strong> (${fileSizeMB} MB)
                <button type="button" class="btn-close float-end" id="clear-file" aria-label="Remove file"></button>
            </div>
        `;
        
        // Enable submit button
        submitBtn.disabled = false;
        
        // Add listener to clear button
        const clearBtn = document.getElementById('clear-file');
        if (clearBtn) {
            clearBtn.addEventListener('click', function(e) {
                e.preventDefault();
                fileInput.value = '';
                fileNameDisplay.innerHTML = '';
                submitBtn.disabled = true;
            });
        }
    }

    /**
     * Handle form submission
     */
    function handleFormSubmit(e) {
        e.preventDefault();
        
        if (!fileInput.files.length) {
            showError('Please select a file first');
            return;
        }

        const file = fileInput.files[0];
        
        // Revalidate file before submission
        const fileExt = file.name.split('.').pop().toLowerCase();
        
        if (!ALLOWED_EXTENSIONS.includes(fileExt)) {
            showError(`File type .${fileExt} is not supported. Allowed formats: ${ALLOWED_EXTENSIONS.join(', ')}`);
            return;
        }

        if (file.size > MAX_FILE_SIZE_BYTES) {
            showError(`File size (${(file.size / (1024 * 1024)).toFixed(2)}MB) exceeds the maximum allowed size of ${MAX_FILE_SIZE_MB}MB`);
            return;
        }

        // Prepare UI for upload
        submitBtn.disabled = true;
        progressContainer.style.display = 'block';
        updateProgressBar(0, 'Starting upload...');
        
        // Show loading overlay with a slight delay
        setTimeout(() => {
            if (loadingOverlay) loadingOverlay.classList.remove('d-none');
        }, 500);
        
        // Prepare form data
        const formData = new FormData();
        formData.append('file', file);
        
        // Send the request with progress monitoring
        const xhr = new XMLHttpRequest();
        
        // Track upload progress
        xhr.upload.addEventListener('progress', function(e) {
            if (e.lengthComputable) {
                const percentComplete = Math.round((e.loaded / e.total) * 100);
                updateProgressBar(percentComplete, 'Uploading...');
            }
        });
        
        // Handle completion
        xhr.addEventListener('load', function() {
            if (xhr.status >= 200 && xhr.status < 400) {
                try {
                    const response = JSON.parse(xhr.responseText);
                    if (response.error) {
                        handleUploadError(response.error);
                    } else if (response.redirect) {
                        window.location.href = response.redirect;
                    } else {
                        updateProgressBar(100, 'Upload complete!');
                    }
                } catch (e) {
                    handleUploadError('Invalid response from server');
                }
            } else {
                handleUploadError('Server error: ' + xhr.status);
            }
        });
        
        // Handle network errors
        xhr.addEventListener('error', function() {
            handleUploadError('Network error occurred');
        });
        
        // Handle aborted uploads
        xhr.addEventListener('abort', function() {
            handleUploadError('Upload cancelled');
        });
        
        // Start the upload
        xhr.open('POST', '/upload', true);
        xhr.setRequestHeader('X-Requested-With', 'XMLHttpRequest');
        xhr.send(formData);
    }
    
    /**
     * Update progress bar display
     */
    function updateProgressBar(percent, message) {
        progressBar.style.width = `${percent}%`;
        progressBar.setAttribute('aria-valuenow', percent);
        progressBar.textContent = `${percent}%`;
        progressText.textContent = message || 'Uploading...';
    }
    
    /**
     * Handle upload errors
     */
    function handleUploadError(message) {
        showError(message);
        submitBtn.disabled = false;
        progressContainer.style.display = 'none';
        if (loadingOverlay) loadingOverlay.classList.add('d-none');
    }

    /**
     * Display error message
     */
    function showError(message) {
        fileNameDisplay.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle me-2"></i>
                ${message}
            </div>
        `;
    }
});

// Progress polling for processing page
if (document.getElementById('main-progress-bar')) {
    const SESSION_ID = document.URL.split('/').pop();
    const progressBar = document.getElementById('main-progress-bar');
    const progressStatus = document.getElementById('progress-status');
    let pollTimer;

    function checkProgress() {
        fetch(`/progress/${SESSION_ID}`)
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    document.getElementById('error-container').classList.remove('d-none');
                    document.getElementById('error-message').textContent = data.error;
                    clearInterval(pollTimer);
                    return;
                }

                // Update progress bar
                progressBar.style.width = `${data.progress}%`;
                progressBar.setAttribute('aria-valuenow', data.progress);
                progressBar.textContent = `${data.progress}%`;
                
                // Update status message
                if (data.message) {
                    progressStatus.textContent = data.message;
                }
                
                // Check for trimming status
                if (data.status === 'trimming' && data.message) {
                    document.getElementById('trimming-notice').classList.remove('d-none');
                    document.getElementById('trimming-message').textContent = data.message;
                }
                
                // Update step status
                updateSteps(data.progress);
                
                // Check if processing is complete
                if (data.progress >= 100 && data.results_url) {
                    document.getElementById('completion-message').classList.remove('d-none');
                    document.getElementById('results-link').href = data.results_url;
                    clearInterval(pollTimer);
                    
                    // Auto-redirect after a short delay
                    setTimeout(() => {
                        window.location.href = data.results_url;
                    }, 2000);
                }
            })
            .catch(error => {
                console.error('Error checking progress:', error);
            });
    }

    function updateSteps(progress) {
        const steps = {
            'feature': {element: document.getElementById('step-feature'), threshold: 15},
            'pattern': {element: document.getElementById('step-pattern'), threshold: 35},
            'model': {element: document.getElementById('step-model'), threshold: 75},
            'result': {element: document.getElementById('step-result'), threshold: 95}
        };
        
        for (const [key, step] of Object.entries(steps)) {
            if (!step.element) continue; // Skip if element doesn't exist
            
            const statusElement = step.element.querySelector('.analysis-status');
            const iconElement = step.element.querySelector('.analysis-icon i');
            
            if (progress >= step.threshold) {
                // Completed step
                if (statusElement) {
                    statusElement.textContent = 'Completed';
                    statusElement.className = 'analysis-status text-success';
                }
                step.element.classList.add('completed');
                step.element.classList.remove('in-progress');
                
                // Make sure icon remains visible
                if (iconElement) {
                    iconElement.style.display = 'inline-block';
                }
            } else if (progress > step.threshold - 15) {
                // In progress step
                if (statusElement) {
                    statusElement.textContent = 'In Progress';
                    statusElement.className = 'analysis-status text-info';
                }
                step.element.classList.add('in-progress');
                
                // Make sure icon remains visible
                if (iconElement) {
                    iconElement.style.display = 'inline-block';
                }
            } else {
                // Pending step
                if (statusElement) {
                    statusElement.textContent = 'Pending';
                    statusElement.className = 'analysis-status text-secondary';
                }
                
                // Make sure icon remains visible
                if (iconElement) {
                    iconElement.style.display = 'inline-block';
                }
            }
        }
    }

    // Start polling when on processing page
    checkProgress(); // Check immediately
    pollTimer = setInterval(checkProgress, 2000); // Then check every 2 seconds
}