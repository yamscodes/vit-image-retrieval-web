"""
Flask application for ViT Image Retrieval Web Interface

This version fixes the index loading issue by:
1. Using absolute paths consistently
2. Storing the retrieval system globally after feature extraction
3. Properly loading indexes from the correct location
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

# Global flag to track if we're using the extracted index
using_extracted_index = False

# Configure allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

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
    
    # Get absolute path to repository root
    # This should be the directory where run_web_app.py is located
    repo_root = Path(__file__).parent.parent.parent.parent
    
    # Configure upload folders - use absolute paths
    app.config['UPLOAD_FOLDER'] = str(repo_root / 'uploads')
    app.config['INDEX_FOLDER'] = str(repo_root / 'indexes')
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload
    
    # Create directories if they don't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['INDEX_FOLDER'], exist_ok=True)
    
    # Log the paths for debugging
    logger.info(f"Repository root: {repo_root}")
    logger.info(f"UPLOAD_FOLDER: {app.config['UPLOAD_FOLDER']}")
    logger.info(f"INDEX_FOLDER: {app.config['INDEX_FOLDER']}")
    logger.info(f"Current working directory: {os.getcwd()}")
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
            
            # Create retrieval system
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
            os.makedirs(app.config['INDEX_FOLDER'], exist_ok=True)
            rs.save(index_path, metadata_path)
            
            # Set the global retrieval system to this newly created index
            global retrieval_system, using_extracted_index
            retrieval_system = rs
            using_extracted_index = True
            
            # Return filenames only
            index_file = os.path.basename(index_path)
            metadata_file = os.path.basename(metadata_path)
            
            return jsonify({
                'success': True,
                'index_path': index_file,
                'metadata_path': metadata_file,
                'num_images': len(rs.metadata)
            })
        except Exception as e:
            logger.error(f"Error in feature extraction: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
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
            
            # Save uploaded files to a temporary directory first
            import tempfile
            import shutil
            import uuid
            temp_dir = tempfile.mkdtemp(prefix='vit_extract_')
            
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
            
            # Process images from temp directory
            def progress_callback(value):
                logger.info(f"Progress: {value}%")
            
            rs.index_images(temp_dir, progress_callback=progress_callback)
            
            # Move files from temp to uploads directory
            upload_dir = app.config['UPLOAD_FOLDER']
            os.makedirs(upload_dir, exist_ok=True)
            for path in saved_paths:
                filename = os.path.basename(path)
                dest_path = os.path.join(upload_dir, filename)
                if not os.path.exists(dest_path):
                    shutil.move(path, dest_path)
                else:
                    # File already exists, just clean up temp
                    try:
                        os.remove(path)
                    except:
                        pass
            
            # Update metadata paths to point to uploads directory
            for idx in list(rs.metadata.keys()):
                old_path = rs.metadata[idx]['path']
                filename = os.path.basename(old_path)
                rs.metadata[idx]['path'] = os.path.join(upload_dir, filename)
            
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
            logger.info(f"Index folder contents: {os.listdir(app.config['INDEX_FOLDER'])}")
            
            # Set the global retrieval system to this newly created index
            global retrieval_system, using_extracted_index
            retrieval_system = rs
            using_extracted_index = True
            
            # Clean up temp directory
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory: {e}")
            
            # Return relative paths for the client
            return jsonify({
                'success': True,
                'index_path': index_file,
                'metadata_path': metadata_file,
                'num_images': len(rs.metadata),
                'full_index_path': index_path,
                'full_metadata_path': metadata_path
            })
        except Exception as e:
            logger.error(f"Error in feature extraction from files: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
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
            index_path_clean = index_path.lstrip('/').replace('\\', '/')
            metadata_path_clean = metadata_path.lstrip('/').replace('\\', '/')
            full_index_path = os.path.join(app.config['INDEX_FOLDER'], index_path_clean)
            full_metadata_path = os.path.join(app.config['INDEX_FOLDER'], metadata_path_clean)
            
            logger.info(f"Loading index from: {full_index_path}")
            logger.info(f"Loading metadata from: {full_metadata_path}")
            logger.info(f"Index folder: {app.config['INDEX_FOLDER']}")
            
            # Check if files exist
            if not os.path.exists(full_index_path):
                available = os.listdir(app.config['INDEX_FOLDER']) if os.path.exists(app.config['INDEX_FOLDER']) else []
                return jsonify({'error': f'Index file not found: {full_index_path}. Available files: {available}'}), 404
            if not os.path.exists(full_metadata_path):
                available = os.listdir(app.config['INDEX_FOLDER']) if os.path.exists(app.config['INDEX_FOLDER']) else []
                return jsonify({'error': f'Metadata file not found: {full_metadata_path}. Available files: {available}'}), 404
            
            # Create new retrieval system with loaded index
            global retrieval_system, using_extracted_index
            retrieval_system = ImageRetrievalSystem(
                index_path=full_index_path,
                metadata_path=full_metadata_path
            )
            using_extracted_index = False
            
            logger.info(f"Successfully loaded index with {retrieval_system.index.ntotal} images")
            
            return jsonify({
                'success': True,
                'num_images': retrieval_system.index.ntotal,
                'index_name': os.path.basename(index_path)
            })
            
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return jsonify({'error': str(e)}), 500

    @app.route('/api/list-indexes')
    def api_list_indexes():
        """List available index files.
        
        Index files are named: index_<identifier>.faiss
        Metadata files are named: metadata_<identifier>.json
        Where <identifier> can contain underscores (e.g., "abc_20261031_121805")
        
        Only searches in the configured INDEX_FOLDER to ensure consistency with
        api_load_index which also uses INDEX_FOLDER.
        """
        try:
            index_folder = app.config['INDEX_FOLDER']
            
            # Ensure the index folder exists
            if not os.path.exists(index_folder):
                os.makedirs(index_folder, exist_ok=True)
                logger.info(f"Created index folder at: {index_folder}")
                return jsonify({'indexes': []})
            
            logger.info(f"Searching for indexes in: {index_folder}")
            
            # Get all files from the index folder
            try:
                files = os.listdir(index_folder)
                logger.info(f"Found {len(files)} files in {index_folder}")
            except PermissionError:
                logger.error(f"Permission denied reading folder: {index_folder}")
                return jsonify({'error': 'Permission denied reading index folder'}), 500
            except Exception as e:
                logger.error(f"Error reading folder {index_folder}: {e}")
                return jsonify({'error': str(e)}), 500
            
            # Separate faiss and json files
            faiss_files = [f for f in files if f.endswith('.faiss')]
            json_files = [f for f in files if f.endswith('.json')]
            
            logger.info(f".faiss files: {faiss_files}")
            logger.info(f".json files: {json_files}")
            
            # Pair .faiss and .json files
            # Strategy: For each .faiss file starting with 'index_', 
            # extract the identifier (everything between 'index_' and '.faiss')
            # and look for a matching 'metadata_<identifier>.json'
            indexes = []
            paired_metadata = set()  # Track which metadata files we've already paired
            
            for faiss_file in faiss_files:
                if faiss_file.startswith('index_'):
                    # Remove 'index_' prefix (6 chars) and '.faiss' suffix (6 chars)
                    identifier = faiss_file[6:-6]  # index_<identifier>.faiss -> <identifier>
                    metadata_file = f"metadata_{identifier}.json"
                    
                    if metadata_file in json_files and metadata_file not in paired_metadata:
                        faiss_path = os.path.join(index_folder, faiss_file)
                        json_path = os.path.join(index_folder, metadata_file)
                        
                        # Both should exist since we got them from the folder listing
                        if os.path.exists(faiss_path) and os.path.exists(json_path):
                            try:
                                stat = os.stat(faiss_path)
                                indexes.append({
                                    'name': faiss_file,
                                    'metadata': metadata_file,
                                    'size': stat.st_size,
                                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                                })
                                paired_metadata.add(metadata_file)
                                logger.info(f"Paired: {faiss_file} + {metadata_file}")
                            except Exception as e:
                                logger.warning(f"Error getting stats for {faiss_file}: {e}")
                        else:
                            logger.warning(f"File exists check failed for {faiss_path} or {json_path}")
                    else:
                        if metadata_file not in json_files:
                            logger.warning(f"No matching metadata file for {faiss_file} (looking for {metadata_file})")
                        else:
                            logger.warning(f"Metadata file {metadata_file} already paired")
            
            # Log any unpaired files
            unpaired_json = [f for f in json_files if f not in paired_metadata]
            if unpaired_json:
                logger.warning(f"Unpaired metadata files: {unpaired_json}")
            
            unpaired_faiss = [f for f in faiss_files if not any(idx['name'] == f for idx in indexes)]
            if unpaired_faiss:
                logger.warning(f"Unpaired index files: {unpaired_faiss}")
            
            # Sort by modification date (newest first)
            indexes.sort(key=lambda x: x['modified'], reverse=True)
            
            logger.info(f"Found {len(indexes)} valid index pairs in {index_folder}")
            
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
            index_path_clean = index_path.lstrip('/').replace('\\', '/')
            metadata_path_clean = metadata_path.lstrip('/').replace('\\', '/')
            full_index_path = os.path.join(app.config['INDEX_FOLDER'], index_path_clean)
            full_metadata_path = os.path.join(app.config['INDEX_FOLDER'], metadata_path_clean)
            
            logger.info(f"Looking for index at: {full_index_path}")
            logger.info(f"Looking for metadata at: {full_metadata_path}")
            
            # Check if files exist
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
            
            # Save temporary query image with a unique name to avoid conflicts
            import uuid
            query_filename = werkzeug.utils.secure_filename(query_file.filename)
            temp_query_filename = f"temp_{uuid.uuid4().hex}_{query_filename}"
            query_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_query_filename)
            query_file.save(query_path)
            
            # Get parameters
            k = int(request.form.get('k', 5))
            
            # Get retrieval system
            rs = get_retrieval_system()
            if rs is None or rs.index is None or rs.index.ntotal == 0:
                return jsonify({'error': 'No index loaded. Please load an index first or extract features.'}), 400
            
            # Perform search
            results = rs.search(query_path, k=k)
            
            # Format results - return file paths relative to uploads folder for web access
            formatted_results = []
            upload_dir = app.config['UPLOAD_FOLDER']
            for path, similarity, metadata in results:
                # Get the filename from the path
                filename = os.path.basename(path)
                
                # Construct web-accessible path
                # The metadata path should already point to upload_dir after extraction
                # But we need to make it a URL path
                if path.startswith(upload_dir):
                    # Path is already in uploads directory
                    web_path = '/uploads/' + filename
                else:
                    # Try to find the file in uploads directory
                    upload_path = os.path.join(upload_dir, filename)
                    if os.path.exists(upload_path):
                        web_path = '/uploads/' + filename
                    else:
                        # File doesn't exist in uploads, use filename directly
                        # This handles cases where original files were indexed from elsewhere
                        web_path = '/uploads/' + filename
                
                formatted_results.append({
                    'path': web_path,
                    'similarity': similarity,
                    'distance': metadata['distance'],
                    'filename': metadata['filename'],
                    'indexed_at': metadata['indexed_at']
                })
            
            # Clean up temporary file (only the temp file we created, not the original)
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
