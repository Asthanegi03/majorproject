"""
Audio processing module for the Deepfake Detection System

This module handles loading, preprocessing, and feature extraction from audio files,
including intelligent trimming for large files.

Author: DeepfakeSoundShield Team
Date: May 2025
"""

import os
import time
import logging
import numpy as np
import librosa
import soundfile as sf
from tempfile import mkdtemp

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class AudioProcessor:
    """Handles audio loading, preprocessing, and feature extraction"""
    
    def __init__(self):
        """Initialize the audio processor"""
        self.sample_rate = 22050  # Default sample rate
        self.n_mfcc = 13  # Number of MFCC coefficients to extract
        self.n_fft = 2048  # FFT window size
        self.hop_length = 512  # Hop length for STFT
        self.duration = 5  # Default duration in seconds for analysis segments
        self.max_file_duration = 90  # Maximum file duration to process without trimming
        self.temp_dir = mkdtemp()
        
        logger.info("AudioProcessor initialized")
    
    def load_audio(self, file_path, progress_callback=None):
        """
        Load audio file and apply intelligent trimming if necessary
        
        Parameters:
            file_path (str): Path to the audio file
            progress_callback (function): Callback for progress updates
        
        Returns:
            tuple: (audio_data, sample_rate, was_trimmed, original_size_mb)
        """
        try:
            # Start loading
            if progress_callback:
                progress_callback(0.1, "Loading audio file...")
            
            # Get file size in MB
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            logger.info(f"Loading audio file: {file_path} ({file_size_mb:.2f} MB)")
            
            # Check file extension and convert if needed
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # For non-WAV files, convert to WAV first
            if file_ext != '.wav':
                if progress_callback:
                    progress_callback(0.2, f"Converting {file_ext} to WAV format...")
                
                # Load audio with librosa (handles various formats)
                y, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
                
                # Create temporary WAV file
                temp_wav_path = os.path.join(self.temp_dir, f"temp_{int(time.time())}.wav")
                sf.write(temp_wav_path, y, sr)
                file_path = temp_wav_path
                
                if progress_callback:
                    progress_callback(0.3, "Conversion complete. Processing WAV file...")
            
            # Get audio duration
            y, sr = librosa.load(file_path, sr=None, duration=5)  # Load just a small segment to get SR
            duration = librosa.get_duration(filename=file_path, sr=sr)
            logger.info(f"Audio duration: {duration:.2f} seconds, Sample rate: {sr} Hz")
            
            # Check if intelligent trimming is needed
            was_trimmed = False
            original_size_mb = None
            
            if duration > self.max_file_duration:
                was_trimmed = True
                original_size_mb = file_size_mb
                
                if progress_callback:
                    progress_callback(0.4, f"Large file detected ({duration:.1f}s). Applying intelligent trimming...")
                
                # Trim file intelligently - extract beginning, middle, and end segments
                segment_duration = min(self.duration, duration / 4)  # Don't use more than 1/4 of the file for each segment
                
                # Extract beginning segment
                y_start, sr = librosa.load(file_path, sr=self.sample_rate, offset=0, duration=segment_duration)
                
                if progress_callback:
                    progress_callback(0.5, "Analyzing beginning segment...")
                
                # Extract middle segment
                middle_offset = duration / 2 - segment_duration / 2
                y_middle, _ = librosa.load(file_path, sr=self.sample_rate, offset=middle_offset, duration=segment_duration)
                
                if progress_callback:
                    progress_callback(0.7, "Analyzing middle segment...")
                
                # Extract end segment
                end_offset = max(0, duration - segment_duration)
                y_end, _ = librosa.load(file_path, sr=self.sample_rate, offset=end_offset, duration=segment_duration)
                
                if progress_callback:
                    progress_callback(0.9, "Analyzing end segment...")
                
                # Concatenate segments
                y = np.concatenate((y_start, y_middle, y_end))
                
                logger.info(f"Intelligent trimming applied. Original: {duration:.2f}s, Trimmed: {len(y)/sr:.2f}s")
                
                if progress_callback:
                    progress_callback(1.0, f"Trimming complete. Processing {len(y)/sr:.1f}s of audio...")
            
            else:
                # Load the entire audio file
                if progress_callback:
                    progress_callback(0.5, "Loading complete audio file...")
                
                y, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
                
                if progress_callback:
                    progress_callback(1.0, "Audio loading complete")
            
            return y, sr, was_trimmed, original_size_mb
        
        except Exception as e:
            logger.error(f"Error loading audio: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to load audio file: {str(e)}")
    
    def extract_features(self, y, sr, progress_callback=None):
        """
        Extract MFCC and spectrogram features from audio
        
        Parameters:
            y (numpy.ndarray): Audio time series
            sr (int): Sample rate
            progress_callback (function): Callback for progress updates
            
        Returns:
            dict: Dictionary containing extracted features
        """
        try:
            if progress_callback:
                progress_callback(0.1, "Extracting audio features...")
            
            # Ensure audio is mono
            if len(y.shape) > 1:
                y = librosa.to_mono(y)
            
            # Normalize audio
            y = librosa.util.normalize(y)
            
            # Extract MFCCs
            if progress_callback:
                progress_callback(0.3, "Calculating MFCC features...")
            
            mfccs = librosa.feature.mfcc(
                y=y, 
                sr=sr, 
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Extract spectrogram
            if progress_callback:
                progress_callback(0.6, "Calculating spectrogram...")
            
            spectrogram = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))
            
            # Apply log transformation to spectrogram
            if progress_callback:
                progress_callback(0.8, "Processing features...")
            
            log_spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)
            
            # Calculate additional features
            spectral_contrast = librosa.feature.spectral_contrast(
                y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
            )
            
            spectral_centroid = librosa.feature.spectral_centroid(
                y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
            )
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
            )
            
            # Pack features into a dictionary
            features = {
                'mfccs': mfccs,
                'spectrogram': spectrogram,
                'log_spectrogram': log_spectrogram,
                'spectral_contrast': spectral_contrast,
                'spectral_centroid': spectral_centroid,
                'spectral_bandwidth': spectral_bandwidth,
                'sample_rate': sr,
                'duration': len(y) / sr
            }
            
            if progress_callback:
                progress_callback(1.0, "Feature extraction complete")
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to extract audio features: {str(e)}")

    def process_audio(self, file_path, progress_callback=None):
        """
        Complete audio processing pipeline
        
        Parameters:
            file_path (str): Path to the audio file
            progress_callback (function): Callback for progress updates
            
        Returns:
            dict: Extracted features
        """
        try:
            # Load audio
            if progress_callback:
                progress_callback(0.1, "Loading audio file...")
            
            y, sr, was_trimmed, original_size_mb = self.load_audio(file_path, progress_callback)
            
            # Extract features
            if progress_callback:
                progress_callback(0.5, "Extracting audio features...")
            
            features = self.extract_features(y, sr, progress_callback)
            
            # Add file metadata to features
            features['was_trimmed'] = was_trimmed
            features['original_size_mb'] = original_size_mb
            features['file_path'] = file_path
            
            return features
            
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to process audio: {str(e)}")