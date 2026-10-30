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

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
INDEX_FOLDER = 'indexes'
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
    
    # Configure upload folders
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['INDEX_FOLDER'] = INDEX_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload
    
    # Create directories if they don't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['INDEX_FOLDER'], exist_ok=True)
    
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
        """API endpoint for feature extraction."""
        try:
            # Get parameters
            image_dir = request.form.get('image_dir')
            index_name = request.form.get('index_name', '').strip()
            
            if not image_dir:
                return jsonify({'error': 'No image directory provided'}), 400
            
            # Get retrieval system
            rs = get_retrieval_system()
            
            # Create new retrieval system for this extraction
            rs = ImageRetrievalSystem(
                feature_extractor=ImageFeatureExtractor()
            )
            
            # Process images
            def progress_callback(value):
                # For now, just log progress
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
    
    @app.route('/api/load-index', methods=['POST'])
    def api_load_index():
        """API endpoint to load an index."""
        try:
            index_path = request.form.get('index_path')
            metadata_path = request.form.get('metadata_path')
            
            if not index_path or not metadata_path:
                return jsonify({'error': 'Both index_path and metadata_path are required'}), 400
            
            # Create new retrieval system with loaded index
            global retrieval_system
            retrieval_system = ImageRetrievalSystem(
                index_path=index_path,
                metadata_path=metadata_path
            )
            
            return jsonify({
                'success': True,
                'num_images': retrieval_system.index.ntotal,
                'index_name': os.path.basename(index_path)
            })
            
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/list-indexes')
    def api_list_indexes():
        """List available index files."""
        try:
            index_folder = app.config['INDEX_FOLDER']
            faiss_files = []
            json_files = []
            
            if os.path.exists(index_folder):
                for file in os.listdir(index_folder):
                    if file.endswith('.faiss'):
                        faiss_files.append(file)
                    elif file.endswith('.json'):
                        json_files.append(file)
            
            # Pair .faiss and .json files
            indexes = []
            for faiss_file in faiss_files:
                base_name = os.path.splitext(faiss_file)[0]
                json_file = f"{base_name}.json"
                if json_file in json_files:
                    # Get file info
                    faiss_path = os.path.join(index_folder, faiss_file)
                    stat = os.stat(faiss_path)
                    indexes.append({
                        'name': faiss_file,
                        'metadata': json_file,
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
            
            return jsonify({'indexes': indexes})
            
        except Exception as e:
            logger.error(f"Error listing indexes: {str(e)}")
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
            query_path = os.path.join(app.config['UPLOAD_FOLDER'], query_file.filename)
            query_file.save(query_path)
            
            # Get parameters
            k = int(request.form.get('k', 5))
            
            # Get retrieval system
            rs = get_retrieval_system()
            if rs is None or rs.index is None:
                return jsonify({'error': 'No index loaded. Please load an index first.'}), 400
            
            # Perform search
            results = rs.search(query_path, k=k)
            
            # Format results
            formatted_results = []
            for path, similarity, metadata in results:
                formatted_results.append({
                    'path': path,
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
                'query_image': query_file.filename,
                'results': formatted_results
            })
            
        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
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
