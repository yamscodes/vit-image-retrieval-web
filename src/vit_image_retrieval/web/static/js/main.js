/**
 * Main JavaScript for ViT Image Retrieval Web Interface
 * Handles common functionality and utilities
 */

// Global configuration
const VIT_CONFIG = {
    apiBaseUrl: '',
    uploadUrl: '/api/upload-image',
    maxFileSize: 100 * 1024 * 1024, // 100MB
    supportedExtensions: ['.png', '.jpg', '.jpeg', '.webp']
};

// Utility functions
const VIT_Utils = {
    /**
     * Format file size in human readable format
     */
    formatFileSize: function(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },

    /**
     * Check if file has allowed extension
     */
    isValidFileType: function(filename) {
        const ext = filename.slice(filename.lastIndexOf('.')).toLowerCase();
        return VIT_CONFIG.supportedExtensions.includes(ext);
    },

    /**
     * Generate unique ID
     */
    generateId: function() {
        return 'id_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    },

    /**
     * Format timestamp
     */
    formatTimestamp: function(timestamp) {
        const date = new Date(timestamp);
        return date.toLocaleString();
    },

    /**
     * Escape HTML special characters
     */
    escapeHtml: function(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    },

    /**
     * Show notification
     */
    showNotification: function(message, type = 'info', duration = 5000) {
        // Create notification element
        const id = 'notification_' + Date.now();
        const notification = document.createElement('div');
        notification.id = id;
        notification.className = `alert alert-${type} alert-dismissible fade show`;
        notification.style.position = 'fixed';
        notification.style.top = '20px';
        notification.style.right = '20px';
        notification.style.zIndex = '9999';
        notification.style.maxWidth = '400px';
        notification.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)';
        
        notification.innerHTML = `
            <strong>${type.charAt(0).toUpperCase() + type.slice(1)}:</strong> ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;

        // Add to body
        document.body.appendChild(notification);

        // Auto remove after duration
        setTimeout(() => {
            const elem = document.getElementById(id);
            if (elem) {
                elem.remove();
            }
        }, duration);

        return id;
    },

    /**
     * Hide notification
     */
    hideNotification: function(id) {
        const elem = document.getElementById(id);
        if (elem) {
            elem.remove();
        }
    },

    /**
     * Confirm action with user
     */
    confirmAction: function(message, title = 'Confirm', confirmText = 'Yes', cancelText = 'No') {
        return new Promise((resolve) => {
            // Create modal
            const modalId = 'confirmModal_' + Date.now();
            const modal = document.createElement('div');
            modal.id = modalId;
            modal.className = 'modal fade';
            modal.tabIndex = -1;
            modal.style.zIndex = '1055';
            
            modal.innerHTML = `
                <div class="modal-dialog">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">${title}</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body">
                            <p>${message}</p>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">${cancelText}</button>
                            <button type="button" class="btn btn-primary confirm-btn">${confirmText}</button>
                        </div>
                    </div>
                </div>
            `;

            document.body.appendChild(modal);

            // Initialize modal
            const bsModal = new bootstrap.Modal(modal);
            bsModal.show();

            // Handle confirm
            modal.querySelector('.confirm-btn').addEventListener('click', () => {
                bsModal.hide();
                modal.remove();
                resolve(true);
            });

            // Handle cancel
            modal.querySelector('.btn-close').addEventListener('click', () => {
                bsModal.hide();
                modal.remove();
                resolve(false);
            });

            modal.querySelector('.btn-secondary').addEventListener('click', () => {
                bsModal.hide();
                modal.remove();
                resolve(false);
            });
        });
    },

    /**
     * Show loading spinner
     */
    showLoading: function(element, text = 'Loading...') {
        const id = 'loading_' + Date.now();
        const loading = document.createElement('div');
        loading.id = id;
        loading.className = 'text-center py-4';
        loading.innerHTML = `
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-2">${text}</p>
        `;

        element.appendChild(loading);
        return id;
    },

    /**
     * Hide loading spinner
     */
    hideLoading: function(id) {
        const elem = document.getElementById(id);
        if (elem) {
            elem.remove();
        }
    }
};

// API client
const VIT_API = {
    /**
     * Make GET request
     */
    get: function(url, data = null) {
        return this.request('GET', url, data);
    },

    /**
     * Make POST request
     */
    post: function(url, data = null, isFormData = false) {
        return this.request('POST', url, data, isFormData);
    },

    /**
     * Make PUT request
     */
    put: function(url, data = null) {
        return this.request('PUT', url, data);
    },

    /**
     * Make DELETE request
     */
    delete: function(url, data = null) {
        return this.request('DELETE', url, data);
    },

    /**
     * Generic request handler
     */
    request: function(method, url, data = null, isFormData = false) {
        return new Promise((resolve, reject) => {
            const options = {
                method: method,
                headers: {}
            };

            if (data) {
                if (isFormData) {
                    options.body = data;
                } else {
                    options.headers['Content-Type'] = 'application/json';
                    options.body = JSON.stringify(data);
                }
            }

            fetch(VIT_CONFIG.apiBaseUrl + url, options)
                .then(response => {
                    if (!response.ok) {
                        return response.json().then(err => {
                            throw new Error(err.error || 'Request failed');
                        }).catch(() => {
                            throw new Error('Request failed with status ' + response.status);
                        });
                    }
                    return response.json();
                })
                .then(data => resolve(data))
                .catch(error => reject(error));
        });
    }
};

// File uploader
const VIT_Uploader = {
    /**
     * Upload single file
     */
    uploadFile: function(file, onProgress = null) {
        return new Promise((resolve, reject) => {
            // Validate file
            if (!file) {
                reject(new Error('No file provided'));
                return;
            }

            if (file.size > VIT_CONFIG.maxFileSize) {
                reject(new Error(`File too large. Max size: ${VIT_Utils.formatFileSize(VIT_CONFIG.maxFileSize)}`));
                return;
            }

            if (!VIT_Utils.isValidFileType(file.name)) {
                reject(new Error('Invalid file type. Supported: ' + VIT_CONFIG.supportedExtensions.join(', ')));
                return;
            }

            // Create form data
            const formData = new FormData();
            formData.append('image', file);

            // Create XMLHttpRequest for progress tracking
            const xhr = new XMLHttpRequest();

            if (onProgress) {
                xhr.upload.addEventListener('progress', (e) => {
                    if (e.lengthComputable) {
                        const percent = Math.round((e.loaded / e.total) * 100);
                        onProgress(percent, e.loaded, e.total);
                    }
                });
            }

            xhr.addEventListener('load', () => {
                try {
                    const response = JSON.parse(xhr.responseText);
                    if (xhr.status >= 200 && xhr.status < 300) {
                        resolve(response);
                    } else {
                        reject(new Error(response.error || 'Upload failed'));
                    }
                } catch (e) {
                    reject(new Error('Invalid response'));
                }
            });

            xhr.addEventListener('error', () => {
                reject(new Error('Upload failed'));
            });

            xhr.addEventListener('abort', () => {
                reject(new Error('Upload cancelled'));
            });

            xhr.open('POST', VIT_CONFIG.apiBaseUrl + VIT_CONFIG.uploadUrl);
            xhr.send(formData);
        });
    },

    /**
     * Upload multiple files
     */
    uploadFiles: function(files, onProgress = null) {
        const promises = files.map(file => this.uploadFile(file, onProgress));
        return Promise.all(promises);
    }
};

// Export utilities
if (typeof window !== 'undefined') {
    window.VIT_Utils = VIT_Utils;
    window.VIT_API = VIT_API;
    window.VIT_Uploader = VIT_Uploader;
    window.VIT_CONFIG = VIT_CONFIG;
}
