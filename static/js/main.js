document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('image-input');
    const previewImage = document.getElementById('preview-image');
    const processingSteps = document.getElementById('processing-steps');
    const resultSection = document.getElementById('result-section');
    const loadingSpinner = document.getElementById('loading-spinner');

    if (uploadForm) {
        uploadForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData();
            formData.append('image', fileInput.files[0]);
            
            try {
                loadingSpinner.classList.remove('d-none');
                resultSection.classList.add('d-none');
                
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.error) {
                    throw new Error(data.error);
                }
                
                // Display processing steps
                processingSteps.innerHTML = '';
                Object.entries(data.processed_images).forEach(([step, imageData]) => {
                    const stepDiv = document.createElement('div');
                    stepDiv.className = 'col-md-6 mb-4';
                    stepDiv.innerHTML = `
                        <div class="card">
                            <img src="${imageData}" class="card-img-top" alt="${step}">
                            <div class="card-body">
                                <h5 class="card-title text-capitalize">${step}</h5>
                            </div>
                        </div>
                    `;
                    processingSteps.appendChild(stepDiv);
                });
                
                // Display results
                document.getElementById('cancer-type').textContent = data.cancer_type;
                document.getElementById('cancer-stage').textContent = data.cancer_stage;
                document.getElementById('confidence').textContent = 
                    `${(data.confidence * 100).toFixed(1)}%`;
                
                resultSection.classList.remove('d-none');
            } catch (error) {
                alert('Error processing image: ' + error.message);
            } finally {
                loadingSpinner.classList.add('d-none');
            }
        });
    }

    if (fileInput) {
        fileInput.addEventListener('change', function() {
            const file = this.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    previewImage.src = e.target.result;
                    previewImage.classList.remove('d-none');
                };
                reader.readAsDataURL(file);
            }
        });
    }
});
