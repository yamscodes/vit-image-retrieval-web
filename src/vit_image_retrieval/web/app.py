"""
Flask application for ViT Image Retrieval Web Interface
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
import werkzeug

from vit_image_retrieval.core.feature_extractor import ImageFeatureExtractor
from vit_image_retrieval.core.retrieval_system import ImageRetrievalSystem

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global retrieval system instance
retrieval_system = None

# Configure allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# UPLOAD_FOLDER and INDEX_FOLDER will be set in create_app() relative to app root

def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_retrieval_system():
    """Get or create the global retrieval system."""
    global retrieval_system
    if retrieval_system is None:
        try:
            # Try to create a model directory
            home = Path.cwd()
            model_dir = home / '.vit_image_retrieval' / 'models'
            model_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using model directory: {model_dir}")
            
            retrieval_system = ImageRetrievalSystem(
                feature_extractor=ImageFeatureExtractor(model_dir=str(model_dir))
            )
        except Exception as e:
            logger.error(f"Error creating retrieval system: {e}")
            # Fall back to temporary directory
            import tempfile
            model_dir = Path(tempfile.gettempdir()) / '.vit_image_retrieval' / 'models'
            model_dir.mkdir(parents=True, exist_ok=True)
            retrieval_system = ImageRetrievalSystem(
                feature_extractor=ImageFeatureExtractor(model_dir=str(model_dir))
            )
    return retrieval_system

def create_app(config=None):
    """Create and configure the Flask application."""
    app = Flask(__name__, 
                static_folder='static',
                template_folder='templates')
    
    # Configure upload folders - use current working directory (repository root)
    # This ensures uploads and indexes are created at the repo root, not inside the web package
    # os.getcwd() should be the repository root when running via run.py
    app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
    app.config['INDEX_FOLDER'] = os.path.join(os.getcwd(), 'indexes')
    
    # Log the cwd for debugging
    logger.info(f"Current working directory (cwd): {os.getcwd()}")
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload
    
    # Create directories if they don't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['INDEX_FOLDER'], exist_ok=True)
    
    # Log the absolute paths for debugging
    logger.info(f"UPLOAD_FOLDER: {app.config['UPLOAD_FOLDER']}")
    logger.info(f"INDEX_FOLDER: {app.config['INDEX_FOLDER']}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"app.root_path: {app.root_path}")
    logger.info(f"Index folder exists: {os.path.exists(app.config['INDEX_FOLDER'])}")
    if os.path.exists(app.config['INDEX_FOLDER']):
        logger.info(f"Index folder contents: {os.listdir(app.config['INDEX_FOLDER'])}")
    
    # Register blueprints or routes
    register_routes(app)
    
    return app

def register_routes(app):
    """Register all routes for the application."""
    
    @app.route('/')
    def index():
        """Main page with tabs for feature extraction and retrieval."""
        return render_template('index.html')
    
    @app.route('/feature-extraction')
    def feature_extraction():
        """Feature extraction page."""
        return render_template('feature_extraction.html')
    
    @app.route('/image-retrieval')
    def image_retrieval():
        """Image retrieval page."""
        return render_template('image_retrieval.html')
    
    # API Routes
    
    @app.route('/api/extract-features', methods=['POST'])
    def api_extract_features():
        """API endpoint for feature extraction from directory."""
        try:
            # Get parameters
            image_dir = request.form.get('image_dir')
            index_name = request.form.get('index_name', '').strip()
            
            if not image_dir:
                return jsonify({'error': 'No image directory provided'}), 400
            
            # Get retrieval system
            rs = ImageRetrievalSystem(
                feature_extractor=ImageFeatureExtractor()
            )
            
            # Process images
            def progress_callback(value):
                logger.info(f"Progress: {value}%")
            
            rs.index_images(image_dir, progress_callback=progress_callback)
            
            # Generate index filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if index_name:
                index_name = ''.join(c if c.isalnum() else '_' for c in index_name)
                index_path = os.path.join(app.config['INDEX_FOLDER'], f"index_{index_name}_{timestamp}.faiss")
                metadata_path = os.path.join(app.config['INDEX_FOLDER'], f"metadata_{index_name}_{timestamp}.json")
            else:
                index_path = os.path.join(app.config['INDEX_FOLDER'], f"index_{timestamp}.faiss")
                metadata_path = os.path.join(app.config['INDEX_FOLDER'], f"metadata_{timestamp}.json")
            
            # Save index
            rs.save(index_path, metadata_path)
            
            return jsonify({
                'success': True,
                'index_path': index_path,
                'metadata_path': metadata_path,
                'num_images': len(rs.metadata)
            })
        except Exception as e:
            logger.error(f"Error in feature extraction: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/extract-features-from-files', methods=['POST'])
    def api_extract_features_from_files():
        """API endpoint for feature extraction from uploaded files."""
        try:
            # Check if we have uploaded files
            if 'images' not in request.files:
                return jsonify({'error': 'No images uploaded'}), 400
            
            files = request.files.getlist('images')
            if not files or all(f.filename == '' for f in files):
                return jsonify({'error': 'No valid images uploaded'}), 400
            
            # Get index name
            index_name = request.form.get('index_name', '').strip()
            
            # Save uploaded files to a temporary directory
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix='vit_extract_')
            
            try:
                saved_paths = []
                for file in files:
                    if file.filename == '':
                        continue
                    if not allowed_file(file.filename):
                        continue
                    
                    # Save file to temp directory
                    filename = werkzeug.utils.secure_filename(file.filename)
                    save_path = os.path.join(temp_dir, filename)
                    file.save(save_path)
                    saved_paths.append(save_path)
                
                if not saved_paths:
                    return jsonify({'error': 'No valid image files found'}), 400
                
                # Create retrieval system
                rs = ImageRetrievalSystem(
                    feature_extractor=ImageFeatureExtractor()
                )
                
                # Process images
                def progress_callback(value):
                    logger.info(f"Progress: {value}%")
                
                rs.index_images(temp_dir, progress_callback=progress_callback)
                
                # Generate index filenames
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if index_name:
                    index_name = ''.join(c if c.isalnum() else '_' for c in index_name)
                    index_file = f"index_{index_name}_{timestamp}.faiss"
                    metadata_file = f"metadata_{index_name}_{timestamp}.json"
                else:
                    index_file = f"index_{timestamp}.faiss"
                    metadata_file = f"metadata_{timestamp}.json"
                
                # Save index to the configured folder
                index_path = os.path.join(app.config['INDEX_FOLDER'], index_file)
                metadata_path = os.path.join(app.config['INDEX_FOLDER'], metadata_file)
                
                # Ensure index folder exists
                os.makedirs(app.config['INDEX_FOLDER'], exist_ok=True)
                
                rs.save(index_path, metadata_path)
                
                # Verify files were created
                logger.info(f"Index saved to: {index_path}")
                logger.info(f"Metadata saved to: {metadata_path}")
                logger.info(f"Index folder: {app.config['INDEX_FOLDER']}")
                logger.info(f"Index folder exists: {os.path.exists(app.config['INDEX_FOLDER'])}")
                if os.path.exists(app.config['INDEX_FOLDER']):
                    logger.info(f"Index folder contents: {os.listdir(app.config['INDEX_FOLDER'])}")
                
                # Return relative paths for the client
                return_index_path = index_file
                return_metadata_path = metadata_file
                
                return jsonify({
                'success': True,
                'index_path': return_index_path,
                'metadata_path': return_metadata_path,
                'num_images': len(rs.metadata),
                'full_index_path': index_path,
                'full_metadata_path': metadata_path
            })
            finally:
                # Clean up temp directory
                import shutil
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning(f"Failed to clean up temp directory: {e}")
        except Exception as e:
            logger.error(f"Error in feature extraction from files: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return jsonify({'error': str(e)}), 500
            
        except Exception as e:
            logger.error(f"Error in feature extraction: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/load-index', methods=['POST'])
    def api_load_index():
        """API endpoint to load an index."""
        try:
            index_path = request.form.get('index_path')
            metadata_path = request.form.get('metadata_path')
            
            if not index_path or not metadata_path:
                return jsonify({'error': 'Both index_path and metadata_path are required'}), 400
            
            # Construct absolute paths - handle both /indexes/filename and filename formats
            index_folder = app.config['INDEX_FOLDER']
            
            # Remove leading slash if present
            index_filename = index_path.lstrip('/')
            metadata_filename = metadata_path.lstrip('/')
            
            # Construct full paths
            full_index_path = os.path.join(index_folder, index_filename)
            full_metadata_path = os.path.join(index_folder, metadata_filename)
            
            logger.info(f"Attempting to load index from: {full_index_path}")
            logger.info(f"Attempting to load metadata from: {full_metadata_path}")
            
            # Check if files exist
            if not os.path.exists(full_index_path):
                logger.error(f'Index file not found: {full_index_path}')
                logger.error(f'Index folder contents: {os.listdir(index_folder) if os.path.exists(index_folder) else "Index folder does not exist"}')
                return jsonify({'error': f'Index file not found: {index_filename}. Available files: {os.listdir(index_folder) if os.path.exists(index_folder) else []}'}), 404
            if not os.path.exists(full_metadata_path):
                logger.error(f'Metadata file not found: {full_metadata_path}')
                return jsonify({'error': f'Metadata file not found: {metadata_filename}'}), 404
            
            # Create new retrieval system with loaded index
            global retrieval_system
            retrieval_system = ImageRetrievalSystem(
                index_path=full_index_path,
                metadata_path=full_metadata_path
            )
            
            logger.info(f"Successfully loaded index with {retrieval_system.index.ntotal} images")
            
            return jsonify({
                'success': True,
                'num_images': retrieval_system.index.ntotal,
                'index_name': os.path.basename(index_filename)
            })
            
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/list-indexes')
    def api_list_indexes():
        """List available index files."""
        try:
            index_folder = app.config['INDEX_FOLDER']
            
            # Ensure index folder exists
            if not os.path.exists(index_folder):
                os.makedirs(index_folder, exist_ok=True)
                return jsonify({'indexes': []})
            
            # Get all files in index folder
            all_files = os.listdir(index_folder)
            faiss_files = [f for f in all_files if f.endswith('.faiss')]
            json_files = [f for f in all_files if f.endswith('.json')]
            
            # Pair .faiss and .json files
            indexes = []
            for faiss_file in faiss_files:
                base_name = os.path.splitext(faiss_file)[0]
                json_file = f"{base_name}.json"
                if json_file in json_files:
                    # Get file info
                    faiss_path = os.path.join(index_folder, faiss_file)
                    json_path = os.path.join(index_folder, json_file)
                    
                    # Verify both files exist and are valid
                    if os.path.exists(faiss_path) and os.path.exists(json_path):
                        stat = os.stat(faiss_path)
                        indexes.append({
                            'name': faiss_file,
                            'metadata': json_file,
                            'size': stat.st_size,
                            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                        })
            
            # Sort by modification date (newest first)
            indexes.sort(key=lambda x: x['modified'], reverse=True)
            
            logger.info(f"Found {len(indexes)} index pairs in {index_folder}")
            logger.info(f"Index folder contents: {all_files}")
            
            return jsonify({'indexes': indexes})
            
        except Exception as e:
            logger.error(f"Error listing indexes: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return jsonify({'error': str(e)}), 500

    @app.route('/api/index-info', methods=['POST'])
    def api_index_info():
        """Get information about a specific index."""
        try:
            index_path = request.form.get('index_path')
            metadata_path = request.form.get('metadata_path')
            
            if not index_path or not metadata_path:
                return jsonify({'error': 'Both index_path and metadata_path are required'}), 400
            
            # Construct full paths
            # Handle paths that might already be absolute or have leading slashes
            index_path_clean = index_path.lstrip('/').replace('\\', '/')
            metadata_path_clean = metadata_path.lstrip('/').replace('\\', '/')
            full_index_path = os.path.join(app.config['INDEX_FOLDER'], index_path_clean)
            full_metadata_path = os.path.join(app.config['INDEX_FOLDER'], metadata_path_clean)
            
            logger.info(f"Looking for index at: {full_index_path}")
            logger.info(f"Looking for metadata at: {full_metadata_path}")
            logger.info(f"Index folder: {app.config['INDEX_FOLDER']}")
            logger.info(f"Index folder exists: {os.path.exists(app.config['INDEX_FOLDER'])}")
            if os.path.exists(app.config['INDEX_FOLDER']):
                logger.info(f"Index folder contents: {os.listdir(app.config['INDEX_FOLDER'])}")
            
            if not os.path.exists(full_index_path):
                available = os.listdir(app.config['INDEX_FOLDER']) if os.path.exists(app.config['INDEX_FOLDER']) else []
                return jsonify({'error': f'Index file not found: {full_index_path}. Available: {available}'}), 404
            
            if not os.path.exists(full_metadata_path):
                available = os.listdir(app.config['INDEX_FOLDER']) if os.path.exists(app.config['INDEX_FOLDER']) else []
                return jsonify({'error': f'Metadata file not found: {full_metadata_path}. Available: {available}'}), 404
            
            # Get file sizes
            index_size = os.path.getsize(full_index_path)
            metadata_size = os.path.getsize(full_metadata_path)
            
            # Try to load metadata to get info
            try:
                import json
                with open(full_metadata_path, 'r') as f:
                    metadata = json.load(f)
                num_images = len(metadata)
            except:
                num_images = 0
            
            return jsonify({
                'success': True,
                'index_path': index_path,
                'metadata_path': metadata_path,
                'index_size': index_size,
                'metadata_size': metadata_size,
                'num_images': num_images,
                'feature_dimension': 768,  # ViT-B/16 feature dimension
                'created_at': datetime.fromtimestamp(os.path.getmtime(full_index_path)).isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error getting index info: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/search', methods=['POST'])
    def api_search():
        """API endpoint for image search."""
        try:
            # Check if we have a query image
            if 'query_image' not in request.files:
                return jsonify({'error': 'No query image provided'}), 400
            
            query_file = request.files['query_image']
            if query_file.filename == '':
                return jsonify({'error': 'No selected query image'}), 400
            
            if not allowed_file(query_file.filename):
                return jsonify({'error': 'Invalid file type'}), 400
            
            # Save temporary query image
            query_filename = werkzeug.utils.secure_filename(query_file.filename)
            query_path = os.path.join(app.config['UPLOAD_FOLDER'], query_filename)
            query_file.save(query_path)
            
            # Get parameters
            k = int(request.form.get('k', 5))
            
            # Get retrieval system
            rs = get_retrieval_system()
            if rs is None or rs.index is None:
                return jsonify({'error': 'No index loaded. Please load an index first.'}), 400
            
            # Perform search
            results = rs.search(query_path, k=k)
            
            # Format results - return file paths relative to uploads folder for web access
            formatted_results = []
            for path, similarity, metadata in results:
                # Make path relative to uploads folder for web serving
                if path.startswith(app.config['UPLOAD_FOLDER']):
                    web_path = '/uploads/' + os.path.basename(path)
                else:
                    # For paths from the index, use as-is
                    web_path = path
                
                formatted_results.append({
                    'path': web_path,
                    'similarity': similarity,
                    'distance': metadata['distance'],
                    'filename': metadata['filename'],
                    'indexed_at': metadata['indexed_at']
                })
            
            # Clean up temporary file
            try:
                os.remove(query_path)
            except:
                pass
            
            return jsonify({
                'success': True,
                'query_image': query_filename,
                'results': formatted_results
            })
            
        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/upload-image', methods=['POST'])
    def api_upload_image():
        """API endpoint for uploading images."""
        try:
            if 'image' not in request.files:
                return jsonify({'error': 'No image provided'}), 400
            
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No selected image'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type'}), 400
            
            # Save file
            filename = werkzeug.utils.secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            
            return jsonify({
                'success': True,
                'filename': filename,
                'path': save_path
            })
            
        except Exception as e:
            logger.error(f"Error uploading image: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/stats')
    def api_stats():
        """Get system statistics."""
        try:
            rs = get_retrieval_system()
            if rs:
                return jsonify(rs.get_stats())
            else:
                return jsonify({'error': 'Retrieval system not initialized'}), 500
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/uploads/<filename>')
    def uploaded_file(filename):
        """Serve uploaded files."""
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    
    @app.route('/indexes/<filename>')
    def index_file(filename):
        """Serve index files."""
        return send_from_directory(app.config['INDEX_FOLDER'], filename)
    
    # Health check
    @app.route('/health')
    def health():
        """Health check endpoint."""
        return jsonify({'status': 'healthy'})
