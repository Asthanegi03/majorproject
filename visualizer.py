"""
Visualization module for the Audio Deepfake Detection System

This module generates visualizations of model predictions and audio features.

Author: DeepfakeSoundShield Team
Date: May 2025
"""

import os
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import librosa
import librosa.display

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def generate_audio_features(session_id):
    """
    Generate MFCC and spectrogram visualizations
    
    Parameters:
        session_id (str): Session ID for the current processing job
        
    Returns:
        tuple: (mfcc_path, spectrogram_path) - Paths to the generated visualizations
    """
    try:
        # Create destination directory if it doesn't exist
        result_dir = os.path.join('static', 'results')
        os.makedirs(result_dir, exist_ok=True)
        
        # For demonstration, create simulated audio data
        # In a real scenario, you would use the actual processed audio data
        sr = 22050  # Sample rate
        duration = 5.0  # 5 seconds
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        
        # Generate a synthetic audio signal with harmonic content
        # A real deepfake detector would use actual audio features
        audio_signal = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz
        audio_signal += 0.3 * np.sin(2 * np.pi * 880 * t)  # 880 Hz  
        audio_signal += 0.1 * np.sin(2 * np.pi * 1760 * t)  # 1760 Hz
        audio_signal += 0.05 * np.random.normal(0, 1, len(audio_signal))  # Add some noise
        
        # ---- MFCC Visualization ----
        plt.figure(figsize=(10, 6))
        plt.style.use('dark_background')
        
        # Compute MFCC features
        mfccs = librosa.feature.mfcc(y=audio_signal, sr=sr, n_mfcc=13)
        
        # Plot MFCC features
        librosa.display.specshow(mfccs, x_axis='time', sr=sr, cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title('MFCC Features Analysis')
        plt.tight_layout()
        
        # Set background color
        plt.gca().set_facecolor('#1e1e1e')
        plt.gcf().set_facecolor('#1e1e1e')
        
        # Save figure
        mfcc_path = os.path.join('static', 'results', f'mfcc_{session_id}.png')
        plt.savefig(mfcc_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # ---- Spectrogram Visualization ----
        plt.figure(figsize=(10, 6))
        plt.style.use('dark_background')
        
        # Compute spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_signal)), ref=np.max)
        
        # Plot spectrogram
        librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr, cmap='magma')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram Analysis')
        plt.tight_layout()
        
        # Set background color
        plt.gca().set_facecolor('#1e1e1e')
        plt.gcf().set_facecolor('#1e1e1e')
        
        # Save figure
        spectrogram_path = os.path.join('static', 'results', f'spectrogram_{session_id}.png')
        plt.savefig(spectrogram_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Audio features visualizations saved to {mfcc_path} and {spectrogram_path}")
        
        return mfcc_path, spectrogram_path
        
    except Exception as e:
        logger.error(f"Error generating audio features: {str(e)}", exc_info=True)
        
        # Return placeholder image paths
        mfcc_path = os.path.join('static', 'img', 'mfcc_error.png')
        spectrogram_path = os.path.join('static', 'img', 'spectrogram_error.png')
        
        # Create basic error images
        for img_path, title in [(mfcc_path, 'MFCC Error'), (spectrogram_path, 'Spectrogram Error')]:
            plt.figure(figsize=(10, 6))
            plt.style.use('dark_background')
            plt.text(0.5, 0.5, f"{title}: Could not generate visualization", 
                    ha='center', va='center', fontsize=16)
            plt.axis('off')
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            plt.savefig(img_path)
            plt.close()
            
        return mfcc_path, spectrogram_path


def generate_visualization(predictions, session_id):
    """
    Generate visualization of model predictions
    
    Parameters:
        predictions (dict): Dictionary containing model predictions
        session_id (str): Session ID for the current processing job
        
    Returns:
        str: Path to the generated visualization image
    """
    try:
        # Create destination directory if it doesn't exist
        result_dir = os.path.join('static', 'results')
        os.makedirs(result_dir, exist_ok=True)
        
        # Create figure with dark background
        plt.figure(figsize=(12, 6))
        plt.style.use('dark_background')
        
        # Model names and confidences
        models = ['ANN', 'CNN', 'RNN', 'LSTM', 'GAN']
        confidences = [
            predictions['ann_confidence'],
            predictions['cnn_confidence'],
            predictions['rnn_confidence'],
            predictions['lstm_confidence'],
            predictions['gan_confidence']
        ]
        
        # Add ensemble result
        models.append('Ensemble')
        confidences.append(predictions['ensemble_confidence'])
        
        # Custom color palette that matches bootstrap theme
        fake_color = '#dc3545'  # Bootstrap danger
        real_color = '#28a745'  # Bootstrap success
        
        # Colors based on prediction
        colors = []
        for model in models:
            if model == 'ANN':
                colors.append(real_color if predictions['ann_prediction'] == 'Real' else fake_color)
            elif model == 'CNN':
                colors.append(real_color if predictions['cnn_prediction'] == 'Real' else fake_color)
            elif model == 'RNN':
                colors.append(real_color if predictions['rnn_prediction'] == 'Real' else fake_color)
            elif model == 'LSTM':
                colors.append(real_color if predictions['lstm_prediction'] == 'Real' else fake_color)
            elif model == 'GAN':
                colors.append(real_color if predictions['gan_prediction'] == 'Real' else fake_color)
            elif model == 'Ensemble':
                colors.append(real_color if predictions['ensemble_prediction'] == 'Real' else fake_color)
        
        # Plot horizontal bar chart
        y_pos = np.arange(len(models))
        bars = plt.barh(y_pos, confidences, color=colors, alpha=0.9, height=0.6)
        
        # Set y-ticks with model names
        plt.yticks(y_pos, models)
        
        # Customize y-tick labels with larger font for the Ensemble
        for tick_idx, tick in enumerate(plt.gca().get_yticklabels()):
            if models[tick_idx] == 'Ensemble':
                tick.set_fontweight('bold')
        
        # Add value labels to the bars
        for bar, conf in zip(bars, confidences):
            plt.text(
                min(conf + 3, 97),  # Avoid labels going off the chart
                bar.get_y() + bar.get_height()/2, 
                f"{conf:.1f}%", 
                va='center',
                fontweight='bold',
                color='white'
            )
        
        # Add a vertical line at 50%
        plt.axvline(x=50, color='#6c757d', linestyle='--', alpha=0.7, 
                    label='50% threshold')
        
        # Add labels and title
        plt.xlabel('Confidence (%)', fontsize=12)
        plt.title('Model Confidence: Higher values suggest AI-generated audio', 
                 fontsize=16, pad=20)
        
        # Set x-axis range and add some padding
        plt.xlim(0, 105)
        
        # Add a legend
        legend_elements = [
            Patch(facecolor=fake_color, label='Predicted as FAKE'),
            Patch(facecolor=real_color, label='Predicted as REAL')
        ]
        plt.legend(handles=legend_elements, loc='lower center', 
                  bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=10)
        
        # Add grid for better readability
        plt.grid(axis='x', alpha=0.2, linestyle='--')
        
        # Set darker background colors
        plt.gca().set_facecolor('#212529')  # Bootstrap dark
        plt.gcf().set_facecolor('#212529')
        
        # Add a box around the plot
        for spine in plt.gca().spines.values():
            spine.set_visible(True)
            spine.set_color('#495057')
        
        # Highlight the ensemble row with a subtle background
        # Draw a rectangle behind the Ensemble bar
        ens_idx = models.index('Ensemble')
        # Create a rectangle patch
        ensemble_rect = Rectangle(
            xy=(-5, ens_idx - 0.3), width=115, height=0.6, 
            alpha=0.2, color='#0d6efd', zorder=0
        )
        # Add the patch to the plot
        plt.gca().add_patch(ensemble_rect)
        
        # Tight layout
        plt.tight_layout(rect=(0, 0.05, 1, 0.9))
        
        # Save figure
        chart_path = os.path.join('static', 'results', f'prediction_{session_id}.png')
        plt.savefig(chart_path, dpi=120, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {chart_path}")
        
        return chart_path
        
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}", exc_info=True)
        
        # Create a basic error chart
        plt.figure(figsize=(10, 5))
        plt.style.use('dark_background')
        plt.text(0.5, 0.5, "Chart generation failed", 
                ha='center', va='center', fontsize=16)
        plt.axis('off')
        
        # Make sure the error directory exists
        os.makedirs(os.path.join('static', 'img'), exist_ok=True)
        error_path = os.path.join('static', 'img', 'error_chart.png')
        plt.savefig(error_path)
        plt.close()
        
        return error_path