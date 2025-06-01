"""
Flask application for the Audio Deepfake Detection System

This file defines the routes and functionality of the web application.

Author: DeepfakeSoundShield Team
Date: May 2025
"""

import os
import uuid
import logging
import threading
import time
import random
import json
from datetime import datetime
from flask import (
    Flask, render_template, request, redirect, url_for, 
    flash, jsonify, session
)
from werkzeug.utils import secure_filename
from utils import is_allowed_file, create_directories, get_file_size_mb, safe_filename

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-deepfake-soundshield")

# Set up file upload configuration
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac', 'wma'}
MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200MB

# Ensure upload directories exist
create_directories([UPLOAD_FOLDER, RESULTS_FOLDER])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# In-memory storage for processing tasks
processing_tasks = {}


@app.route('/')
def index():
    """Render the home page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and start the processing workflow"""
    # Check if request is AJAX/XHR
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
    
    if 'file' not in request.files:
        if is_ajax:
            return jsonify({'error': 'No file part in the request'}), 400
        flash('No file part in the request', 'danger')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        if is_ajax:
            return jsonify({'error': 'No file selected'}), 400
        flash('No file selected', 'danger')
        return redirect(url_for('index'))
    
    if not file or not is_allowed_file(file.filename, ALLOWED_EXTENSIONS):
        error_msg = f'Invalid file type. Allowed formats: {", ".join(ALLOWED_EXTENSIONS)}'
        if is_ajax:
            return jsonify({'error': error_msg}), 400
        flash(error_msg, 'danger')
        return redirect(url_for('index'))
    
    try:
        # Check if this file was already processed in this session
        # to prevent double processing
        current_session_id = session.get('session_id')
        if current_session_id and current_session_id in processing_tasks:
            task = processing_tasks[current_session_id]
            safe_name = secure_filename(file.filename or "")
            if task.get('original_filename') == safe_name:
                # This appears to be a duplicate submission, use existing session
                logger.info(f"Preventing duplicate upload for file: {file.filename}")
                if is_ajax:
                    return jsonify({'redirect': url_for('processing', session_id=current_session_id)})
                return redirect(url_for('processing', session_id=current_session_id))
        
        # Generate unique session ID and secure filename
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        
        filename = secure_filename(file.filename or "")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{session_id}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save uploaded file
        file.save(file_path)
        file_size_mb = get_file_size_mb(file_path)
        logger.info(f"File saved: {file_path} ({file_size_mb:.2f} MB)")
        
        # Initialize progress tracking
        processing_tasks[session_id] = {
            'progress': 0,
            'status': 'initializing',
            'message': 'Starting analysis...',
            'file_path': file_path,
            'original_filename': filename,
            'file_size_mb': file_size_mb,
            'start_time': time.time(),
            'results': None,
            'error': None
        }
        
        # Start processing in background thread to simulate actual ML processing
        thread = threading.Thread(target=process_audio_file, args=(session_id, file_path, filename))
        thread.daemon = True
        thread.start()
        
        # Redirect to processing page
        if is_ajax:
            return jsonify({'redirect': url_for('processing', session_id=session_id)})
        return redirect(url_for('processing', session_id=session_id))
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        if is_ajax:
            return jsonify({'error': f'Error uploading file: {str(e)}'}), 500
        flash(f'Error uploading file: {str(e)}', 'danger')
        return redirect(url_for('index'))


@app.route('/processing/<session_id>')
def processing(session_id):
    """Render the processing page"""
    # Verify that this session_id exists in our processing tasks
    if session_id not in processing_tasks:
        flash('Invalid processing session. Please upload your file again.', 'danger')
        return redirect(url_for('index'))
    
    return render_template('processing.html', session_id=session_id)


@app.route('/progress/<session_id>')
def progress(session_id):
    """API endpoint for checking processing progress"""
    if session_id not in processing_tasks:
        return jsonify({'error': 'Invalid session ID'}), 404
    
    task = processing_tasks[session_id]
    
    # Check if processing has completed
    if task['status'] == 'completed' and task['results']:
        results_url = url_for('results', session_id=session_id)
        return jsonify({
            'progress': 100,
            'status': 'completed',
            'message': 'Processing complete',
            'results_url': results_url
        })
    
    # Check for errors
    if task['error']:
        return jsonify({
            'progress': task['progress'],
            'status': 'error',
            'error': task['error']
        })
    
    # Return current progress
    return jsonify({
        'progress': task['progress'],
        'status': task['status'],
        'message': task['message']
    })


def process_audio_file(session_id, file_path, original_filename):
    """
    Process the uploaded audio file in a background thread
    
    This function simulates the audio processing workflow:
    1. Audio preprocessing and feature extraction
    2. Model predictions
    3. Result visualization
    4. Cleanup
    """
    try:
        task = processing_tasks[session_id]
        
        # Simulate audio feature extraction (15%)
        update_progress(session_id, 15, "Extracting audio features...")
        time.sleep(2)  # Simulate processing time
        
        # Simulate pattern analysis (35%)
        update_progress(session_id, 35, "Analyzing audio patterns...")
        time.sleep(2)  # Simulate processing time
        
        # Simulate model predictions (75%)
        update_progress(session_id, 75, "Running neural network models...")
        time.sleep(2)  # Simulate processing time
        
        # Create simulated predictions
        predictions = {
            'ann_prediction': 'Fake' if random.random() > 0.5 else 'Real',
            'cnn_prediction': 'Fake' if random.random() > 0.5 else 'Real',
            'rnn_prediction': 'Fake' if random.random() > 0.5 else 'Real',
            'lstm_prediction': 'Fake' if random.random() > 0.5 else 'Real',
            'gan_prediction': 'Fake' if random.random() > 0.5 else 'Real',
            
            'ann_confidence': random.uniform(60, 90),
            'cnn_confidence': random.uniform(60, 90),
            'rnn_confidence': random.uniform(60, 90),
            'lstm_confidence': random.uniform(60, 90),
            'gan_confidence': random.uniform(60, 90),
        }
        
        # Calculate ensemble prediction
        real_count = sum(1 for key in ['ann_prediction', 'cnn_prediction', 'rnn_prediction', 
                                      'lstm_prediction', 'gan_prediction'] 
                        if predictions[key] == 'Real')
        
        predictions['ensemble_prediction'] = 'Real' if real_count >= 3 else 'Fake'
        predictions['ensemble_confidence'] = sum([
            predictions['ann_confidence'] if predictions['ann_prediction'] == 'Fake' else 100 - predictions['ann_confidence'],
            predictions['cnn_confidence'] if predictions['cnn_prediction'] == 'Fake' else 100 - predictions['cnn_confidence'],
            predictions['rnn_confidence'] if predictions['rnn_prediction'] == 'Fake' else 100 - predictions['rnn_confidence'],
            predictions['lstm_confidence'] if predictions['lstm_prediction'] == 'Fake' else 100 - predictions['lstm_confidence'],
            predictions['gan_confidence'] if predictions['gan_prediction'] == 'Fake' else 100 - predictions['gan_confidence']
        ]) / 5.0
        
        # Simulate results visualization (95%)
        update_progress(session_id, 95, "Generating visualization...")
        
        # Actually generate the visualization instead of just simulating it
        from visualizer import generate_visualization, generate_audio_features
        chart_path = generate_visualization(predictions, session_id)
        chart_path = chart_path.replace('static/', '')
        
        # Generate MFCC and spectrogram visualizations
        # Normally we would use actual audio features, but for demonstration we'll create simulated ones
        mfcc_path, spectrogram_path = generate_audio_features(session_id)
        mfcc_path = mfcc_path.replace('static/', '')
        spectrogram_path = spectrogram_path.replace('static/', '')
        
        # Prepare results
        results = {
            'filename': original_filename,
            'final_verdict': predictions['ensemble_prediction'],
            'was_trimmed': task['file_size_mb'] > 20,  # Simulate trimming for files > 20MB
            'original_size': task['file_size_mb'],
            'chart_path': chart_path,
            'mfcc_path': mfcc_path,
            'spectrogram_path': spectrogram_path,
        }
        
        # Add all predictions to results
        for key, value in predictions.items():
            results[key] = value
        
        # Mark as completed
        update_progress(session_id, 100, "Analysis complete", "completed")
        processing_tasks[session_id]['results'] = results
        
        # Schedule cleanup after 1 hour
        threading.Timer(3600, lambda: processing_tasks.pop(session_id, None)).start()
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        processing_tasks[session_id]['error'] = str(e)
        processing_tasks[session_id]['status'] = 'error'


def update_progress(session_id, progress, message, status=None):
    """Update the progress tracking for a session"""
    if session_id in processing_tasks:
        processing_tasks[session_id]['progress'] = progress
        processing_tasks[session_id]['message'] = message
        if status:
            processing_tasks[session_id]['status'] = status


@app.route('/results/<session_id>')
def results(session_id):
    """Display the analysis results"""
    if session_id not in processing_tasks:
        flash('Invalid or expired results session', 'danger')
        return redirect(url_for('index'))
    
    task = processing_tasks[session_id]
    
    if task['status'] != 'completed' or not task['results']:
        flash('Processing has not completed yet', 'warning')
        return redirect(url_for('processing', session_id=session_id))
    
    return render_template('result.html', results=task['results'], session_id=session_id)


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    flash('File too large. Maximum size is 200MB.', 'danger')
    return redirect(url_for('index'))


@app.errorhandler(404)
def page_not_found(error):
    """Handle 404 errors"""
    return render_template('error.html', error='Page not found'), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    return render_template('error.html', error='Server error. Please try again later.'), 500


if __name__ == "__main__":
    # Run the Flask application

    app.run(host="0.0.0.0", port=5000, debug=True)
