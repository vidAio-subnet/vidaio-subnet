import os
import csv
import time # time is imported but not used in the provided snippet, consider removing if not needed elsewhere
import pandas as pd # pandas is imported but not used in the provided snippet, consider removing if not needed elsewhere
from datetime import datetime
import threading

class VideoProcessingLogger:
    """Comprehensive logger for video processing metrics and results."""
    
    def __init__(self, log_dir="logs", session_name=None, preserve_frames=False, reduced_logging=False):
        """
        Initialize the video processing logger.
        
        Args:
            log_dir (str): Directory to store log files
            session_name (str): Unique session identifier
            preserve_frames (bool): Whether to preserve frame files in logs
            reduced_logging (bool): Whether to use reduced logging for better performance
        """
        self.log_dir = os.path.abspath(log_dir) # Use absolute path for clarity
        self.preserve_frames = preserve_frames
        self.reduced_logging = reduced_logging
        
        # Create base log directory
        os.makedirs(self.log_dir, exist_ok=True)
        # print(f"Base log directory: {self.log_dir}") # Optional: for debugging init
        
        if session_name is None:
            self.session_name = datetime.now().strftime("%Y%m%d_%H%M%S_session")
        else:
            self.session_name = session_name
        # print(f"Session name: {self.session_name}") # Optional: for debugging init

        # Create session-specific subdirectory
        self.session_log_dir = os.path.join(self.log_dir, self.session_name)
        os.makedirs(self.session_log_dir, exist_ok=True)
        # print(f"Session log directory: {self.session_log_dir}") # Optional: for debugging init
        
        # Log file paths (now within session_log_dir)
        self.detailed_log_path = os.path.join(self.session_log_dir, f"{self.session_name}_detailed.csv")
        self.summary_log_path = os.path.join(self.session_log_dir, f"{self.session_name}_summary.csv")
        self.general_log_file_path = os.path.join(self.session_log_dir, f"{self.session_name}_general.log") # For general messages
      
        # Initialize storage (these might not be strictly necessary if writing directly to CSV)
        self.detailed_records = []
        self.summary_records = []
        
        # Initialize CSV files
        self._initialize_csv_files()
        # print("CSV files initialized") # Optional: for debugging init

    # --- Add General Purpose Logging Methods ---
    def _log_general_message(self, level: str, message: str):
        """Internal helper to write to the general log file and print."""
        log_entry = f"{datetime.now().isoformat()} - {level.upper()} - {message}"
        if not self.reduced_logging or level.upper() in ["ERROR", "WARNING"]: # Always log errors/warnings
            print(log_entry)
        
        try:
            with open(self.general_log_file_path, 'a', encoding='utf-8') as f:
                f.write(log_entry + "\n")
        except Exception as e:
            print(f"CRITICAL LOGGER ERROR: Failed to write to general log {self.general_log_file_path}: {e}")

    def info(self, message: str):
        """Logs an informational message."""
        self._log_general_message("info", message)

    def error(self, message: str):
        """Logs an error message."""
        self._log_general_message("error", message)

    def warning(self, message: str):
        """Logs a warning message."""
        self._log_general_message("warning", message)

    def debug(self, message: str):
        """Logs a debug message. Only prints if reduced_logging is False."""
        if not self.reduced_logging:
            self._log_general_message("debug", message)
        else: # Still write to file even if not printing to console in reduced mode
            try:
                with open(self.general_log_file_path, 'a', encoding='utf-8') as f:
                    f.write(f"{datetime.now().isoformat()} - DEBUG - {message}\n")
            except Exception as e:
                print(f"CRITICAL LOGGER ERROR: Failed to write debug to general log {self.general_log_file_path}: {e}")
    # --- End of General Purpose Logging Methods ---
    
    def _initialize_csv_files(self):
        """Initialize CSV files with comprehensive headers including VMAF model tracking."""
        try:
            # Enhanced headers with VMAF model tracking and prediction data
            # Enhanced headers with preprocessing timing tracking
            detailed_headers = [
            # Basic metadata
            'timestamp', 'session_id', 'video_file', 'scene_number', 'processing_stage',
            'duration_seconds', 'file_size_input_mb', 'file_size_output_mb',
            
            # Core video features for VMAF prediction model
            'metrics_resolution_width', 'metrics_resolution_height', 'metrics_frame_rate',
            'metrics_bit_depth', 'input_bitrate_kbps', 'bits_per_pixel',
            'metrics_avg_motion', 'metrics_avg_edge_density', 'metrics_avg_texture',
            'metrics_avg_color_complexity', 'metrics_avg_motion_variance', 'metrics_avg_grain_noise',
            'metrics_avg_spatial_information', 'metrics_avg_temporal_information',
            
            # Scene classification features (numerical only)
            'scene_type', 'confidence_score', 'contrast_value',
            'prob_screen_content', 'prob_animation', 'prob_faces', 
            'prob_gaming', 'prob_other', 'prob_unclear',
            
            # Quality prediction and results - FIXED AND ENHANCED
            'target_vmaf', 'predicted_vmaf_at_optimal_cq', 'actual_vmaf', 'vmaf_prediction_error',
            'optimal_cq', 'adjusted_cq',
            
            # VMAF Model tracking - NEW FIELDS
            'vmaf_model_used', 'vmaf_model_path_used', 'vmaf_model_type',
            
            #  Preprocessing timing fields
            'preprocessing_enabled', 'preprocessing_applied',
            'preprocessing_quality_analysis_time', 'preprocessing_filter_recommendation_time',
            'preprocessing_filter_application_time', 'preprocessing_total_time',
            'preprocessing_filters_count', 'preprocessing_percentage_of_scene_time',
            'preprocessing_intensity_level', 'preprocessing_filters_applied',
            
            # Hardware info
            'codec_used', 'hardware_acceleration', 'auto_detection_used', 'hardware_type',
            
            # Performance monitoring fields 
            'avg_cpu_percent', 'peak_cpu_percent', 'avg_memory_mb', 'peak_memory_mb',
            'avg_gpu_percent', 'peak_gpu_percent', 'disk_read_mb', 'disk_write_mb',
            'monitoring_duration', 'performance_samples',

            # Encoding results
            'encoding_success', 'compression_ratio', 'processing_time_seconds',
            
            # Training data quality indicators
            'has_motion_data', 'has_complexity_data', 'has_vmaf_data', 'training_quality_score'
            ]
        
            
            # Summary headers remain the same
            summary_headers = [
                'timestamp', 'session_id', 'video_file', 'total_scenes', 'successful_scenes',
            'avg_vmaf', 'total_processing_time', 'total_compression_ratio',
            'avg_cq_used', 'total_file_size_mb', 
            'vmaf_model_used', 'vmaf_model_path_used', 'vmaf_model_type',
            'preprocessing_enabled', 'scenes_with_preprocessing', 'total_preprocessing_time',
            'avg_preprocessing_time_per_scene', 'preprocessing_percentage_of_total'
        ]
            
            # Write headers
            self._write_csv_header(self.detailed_log_path, detailed_headers)
            self._write_csv_header(self.summary_log_path, summary_headers)
            
            print("✓ Enhanced CSV headers initialized with VMAF model tracking")
            
        except Exception as e:
            print(f"✗ ERROR initializing CSV files: {e}")
            raise

    def _write_csv_header(self, file_path, headers):
        """Write CSV header if file doesn't exist."""
        try:
            if not os.path.exists(file_path):
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
                #print(f"✓ Created CSV file with headers: {file_path}")
            else:
                print(f"✓ CSV file already exists: {file_path}")
        except Exception as e:
            print(f"✗ ERROR writing CSV header to {file_path}: {e}")
            raise
    
    def log_scene_processing(self, video_file, scene_number, scene_data, processing_stage="complete"):
        """Log numerical scene processing data with enhanced VMAF tracking."""
        try:
            timestamp = datetime.now().isoformat()
            
            # Validate and clean feature data before logging
            validated_data = self._validate_feature_data(scene_data)
            
            # Calculate VMAF prediction error if both values exist
            predicted_vmaf = validated_data.get('predicted_vmaf_at_optimal_cq', 0)
            actual_vmaf = validated_data.get('actual_vmaf', 0)
            vmaf_error = 0
            if predicted_vmaf > 0 and actual_vmaf > 0:
                vmaf_error = abs(actual_vmaf - predicted_vmaf)
            
            # ✅ Extract preprocessing timing information
            preprocessing_enabled = validated_data.get('preprocessing_enabled', False)
            preprocessing_applied = validated_data.get('preprocessing_applied', False)
            preprocessing_total_time = validated_data.get('preprocessing_total_time', 0.0)
            scene_processing_time = validated_data.get('processing_time_seconds', 0.0)
            
            # Calculate preprocessing percentage of scene processing time
            preprocessing_percentage = 0.0
            if scene_processing_time > 0 and preprocessing_total_time > 0:
                preprocessing_percentage = (preprocessing_total_time / scene_processing_time) * 100
            
            # Count preprocessing filters applied
            preprocessing_filters_count = 0
            preprocessing_filters_list = validated_data.get('preprocessing_filters_applied', [])
            if isinstance(preprocessing_filters_list, list):
                preprocessing_filters_count = len(preprocessing_filters_list)
            elif isinstance(preprocessing_filters_list, str):
                preprocessing_filters_count = len(preprocessing_filters_list.split(',')) if preprocessing_filters_list else 0
            
            record = {
                # Basic metadata
                'timestamp': timestamp,
                'session_id': self.session_name,
                'video_file': os.path.basename(video_file) if video_file else '',
                'scene_number': scene_number,
                'processing_stage': processing_stage,
                'duration_seconds': validated_data.get('duration_seconds', 0),
                'file_size_input_mb': validated_data.get('file_size_input_mb', 0),
                'file_size_output_mb': validated_data.get('file_size_output_mb', 0),
                
                # Core video features for VMAF prediction (validated)
                'metrics_resolution_width': validated_data.get('metrics_resolution_width', 0),
                'metrics_resolution_height': validated_data.get('metrics_resolution_height', 0),
                'metrics_frame_rate': validated_data.get('metrics_frame_rate', 0),
                'metrics_bit_depth': validated_data.get('metrics_bit_depth', 8),
                'input_bitrate_kbps': validated_data.get('input_bitrate_kbps', 0),
                'bits_per_pixel': validated_data.get('bits_per_pixel', 0),
                'metrics_avg_motion': validated_data.get('metrics_avg_motion', 0),
                'metrics_avg_edge_density': validated_data.get('metrics_avg_edge_density', 0),
                'metrics_avg_texture': validated_data.get('metrics_avg_texture', 0),
                'metrics_avg_color_complexity': validated_data.get('metrics_avg_color_complexity', 0),
                'metrics_avg_motion_variance': validated_data.get('metrics_avg_motion_variance', 0),
                'metrics_avg_grain_noise': validated_data.get('metrics_avg_grain_noise', 0),
                'metrics_avg_spatial_information': validated_data.get('metrics_avg_spatial_information', 0),
                'metrics_avg_temporal_information': validated_data.get('metrics_avg_temporal_information', 0),
                
                # Scene classification results (numerical)
                'scene_type': validated_data.get('scene_type', ''),
                'confidence_score': validated_data.get('confidence_score', 0),
                'contrast_value': validated_data.get('contrast_value', 0),
                'prob_screen_content': validated_data.get('prob_screen_content', 0),
                'prob_animation': validated_data.get('prob_animation', 0),
                'prob_faces': validated_data.get('prob_faces', 0),
                'prob_gaming': validated_data.get('prob_gaming', 0),
                'prob_other': validated_data.get('prob_other', 0),
                'prob_unclear': validated_data.get('prob_unclear', 0),
                
                # Quality prediction and results 
                'target_vmaf': validated_data.get('target_vmaf', 0),
                'predicted_vmaf_at_optimal_cq': predicted_vmaf,
                'actual_vmaf': actual_vmaf,
                'vmaf_prediction_error': vmaf_error,
                'optimal_cq': validated_data.get('optimal_cq', 0),
                'adjusted_cq': validated_data.get('adjusted_cq', 0),
                
                # VMAF Model tracking
                'vmaf_model_used': validated_data.get('vmaf_model_used', ''),
                'vmaf_model_path_used': validated_data.get('vmaf_model_path_used', ''),
                'vmaf_model_type': validated_data.get('vmaf_model_type', ''),
                
                # ✅ NEW: Preprocessing timing fields
                'preprocessing_enabled': preprocessing_enabled,
                'preprocessing_applied': preprocessing_applied,
                'preprocessing_quality_analysis_time': validated_data.get('preprocessing_quality_analysis_time', 0.0),
                'preprocessing_filter_recommendation_time': validated_data.get('preprocessing_filter_recommendation_time', 0.0),
                'preprocessing_filter_application_time': validated_data.get('preprocessing_filter_application_time', 0.0),
                'preprocessing_total_time': preprocessing_total_time,
                'preprocessing_filters_count': preprocessing_filters_count,
                'preprocessing_percentage_of_scene_time': preprocessing_percentage,
                'preprocessing_intensity_level': validated_data.get('preprocessing_intensity', ''),
                'preprocessing_filters_applied': str(preprocessing_filters_list) if preprocessing_filters_list else '',
                
                # Hardware info fields
                'codec_used': validated_data.get('codec_used', ''),
                'hardware_acceleration': validated_data.get('hardware_acceleration', False),
                'auto_detection_used': validated_data.get('auto_detection_used', False),
                'hardware_type': validated_data.get('hardware_type', ''),
                
                # Performance monitoring fields
                'avg_cpu_percent': validated_data.get('avg_cpu_percent', 0),
                'peak_cpu_percent': validated_data.get('peak_cpu_percent', 0),
                'avg_memory_mb': validated_data.get('avg_memory_mb', 0),
                'peak_memory_mb': validated_data.get('peak_memory_mb', 0),
                'avg_gpu_percent': validated_data.get('avg_gpu_percent', 0),
                'peak_gpu_percent': validated_data.get('peak_gpu_percent', 0),
                'disk_read_mb': validated_data.get('disk_read_mb', 0),
                'disk_write_mb': validated_data.get('disk_write_mb', 0),
                'monitoring_duration': validated_data.get('monitoring_duration', 0),
                'performance_samples': validated_data.get('performance_samples', 0),
                
                # Encoding results
                'encoding_success': validated_data.get('encoding_success', False),
                'compression_ratio': validated_data.get('compression_ratio', 0),
                'processing_time_seconds': scene_processing_time,
                
                # Training data quality indicators
                'has_motion_data': validated_data.get('metrics_avg_motion', 0) > 0,
                'has_complexity_data': validated_data.get('metrics_avg_texture', 0) > 0,
                'has_vmaf_data': actual_vmaf > 0,
                'training_quality_score': self._calculate_training_quality_score(validated_data)
            }
            
            # Add to records
            self.detailed_records.append(record)
            
            # Write to CSV immediately for real-time access
            self._append_to_csv(self.detailed_log_path, record)
            
            if processing_stage == "complete":
                preprocessing_info = f" | Preprocessing: {preprocessing_total_time:.2f}s" if preprocessing_applied else ""
                print(f"✓ Logged scene {scene_number} data (quality: {record['training_quality_score']:.1f}){preprocessing_info}")
            
        except Exception as e:
            print(f"✗ ERROR logging scene processing: {e}")
            import traceback
            traceback.print_exc()

    def _append_to_csv(self, file_path, record_dict):
        """Append a record to a CSV file with debug information."""
        try:
           
            # Read existing headers
            with open(file_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader, [])
        
            # Check for missing fields
            missing_fields = [field for field in headers if field not in record_dict]
            if missing_fields:
                print(f"⚠️ DEBUG: Missing fields in record: {missing_fields}")
                # Add missing fields with default values
                for field in missing_fields:
                    record_dict[field] = ''
            
            # Write record
            with open(file_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writerow(record_dict)
            
            print(f"✅ DEBUG: Successfully wrote record to {file_path}")
            print(f"🔍 DEBUG: File size now: {os.path.getsize(file_path)} bytes")
                
        except Exception as e:
            print(f"❌ DEBUG: Error writing to CSV {file_path}: {e}")
            import traceback
            traceback.print_exc()
    
    def log_video_summary(self, video_file, scenes_df, total_processing_time, 
                     vmaf_model_used=None, vmaf_model_path_used=None, vmaf_model_type=None):
        """Log video summary data with VMAF model information and preprocessing stats."""
        try:
            timestamp = datetime.now().isoformat()
            
            # FIXED: Use provided VMAF model info (don't fallback to scenes_df)
            if vmaf_model_used is None:
                vmaf_model_used = 'Default VMAF'  # Default fallback
            if vmaf_model_path_used is None:
                vmaf_model_path_used = ''
            if vmaf_model_type is None:
                vmaf_model_type = ''

            # ✅ Calculate preprocessing statistics from scenes_df
            preprocessing_enabled = False
            scenes_with_preprocessing = 0
            total_preprocessing_time_summary = 0.0
            
            if scenes_df is not None and not scenes_df.empty:
                if 'preprocessing_applied' in scenes_df.columns:
                    preprocessing_enabled = scenes_df['preprocessing_applied'].any()
                    scenes_with_preprocessing = scenes_df['preprocessing_applied'].sum()
                
                if 'preprocessing_total_time' in scenes_df.columns:
                    total_preprocessing_time_summary = scenes_df['preprocessing_total_time'].sum()
            
            # Calculate preprocessing percentage of total processing time
            preprocessing_percentage = 0.0
            if total_processing_time > 0 and total_preprocessing_time_summary > 0:
                preprocessing_percentage = (total_preprocessing_time_summary / total_processing_time) * 100

            summary_record = {
                'timestamp': timestamp,
                'session_id': self.session_name,
                'video_file': os.path.basename(video_file) if video_file else '',
                'total_scenes': len(scenes_df) if scenes_df is not None else 0,
                'successful_scenes': len(scenes_df[scenes_df['encoding_success'] == True]) if scenes_df is not None else 0,
                'avg_vmaf': scenes_df['actual_vmaf'].mean() if scenes_df is not None and 'actual_vmaf' in scenes_df.columns else 0,
                'total_processing_time': total_processing_time,
                'total_compression_ratio': scenes_df['compression_ratio'].mean() if scenes_df is not None and 'compression_ratio' in scenes_df.columns else 0,
                'avg_cq_used': scenes_df['adjusted_cq'].mean() if scenes_df is not None and 'adjusted_cq' in scenes_df.columns else 0,
                'total_file_size_mb': 0,  
                # VMAF model tracking
                'vmaf_model_used': vmaf_model_used,
                'vmaf_model_path_used': vmaf_model_path_used,
                'vmaf_model_type': vmaf_model_type,
                # ✅ NEW: Preprocessing summary fields
                'preprocessing_enabled': preprocessing_enabled,
                'scenes_with_preprocessing': scenes_with_preprocessing,
                'total_preprocessing_time': total_preprocessing_time_summary,
                'avg_preprocessing_time_per_scene': total_preprocessing_time_summary / scenes_with_preprocessing if scenes_with_preprocessing > 0 else 0,
                'preprocessing_percentage_of_total': preprocessing_percentage
            }
            
            self.summary_records.append(summary_record)
            self._append_to_csv(self.summary_log_path, summary_record)
            
            preprocessing_info = f" | Preprocessing: {total_preprocessing_time_summary:.1f}s ({preprocessing_percentage:.1f}%)" if preprocessing_enabled else ""
            print(f"✓ Logged video summary for {video_file}{preprocessing_info}")
            
        except Exception as e:
            print(f"✗ ERROR logging video summary: {e}")
            import traceback
            traceback.print_exc()
    
    def finalize(self):
        """Finalize logging session."""
        print(f"Logging session finalized: {self.session_name}")
        print(f"Total detailed records: {len(self.detailed_records)}")
        print(f"Total summary records: {len(self.summary_records)}")
        print(f"Detailed log: {self.detailed_log_path}")
        print(f"Summary log: {self.summary_log_path}")
    
    def _validate_feature_data(self, scene_data):
        """Enhanced validation with better VMAF field handling."""
        validated = scene_data.copy()
        
        # Check for zero values in critical features and log warnings
        critical_features = [
            'metrics_resolution_width', 'metrics_resolution_height', 
            'metrics_frame_rate', 'input_bitrate_kbps'
        ]
        
        zero_features = []
        for feature in critical_features:
            if validated.get(feature, 0) == 0:
                zero_features.append(feature)
        
        if zero_features:
            print(f"⚠️ Warning: Zero values detected in critical features: {zero_features}")
            validated['data_quality_warning'] = f"Zero values in: {', '.join(zero_features)}"
        
        # Ensure numeric types - ENHANCED with all VMAF fields
        numeric_features = [
            'duration_seconds', 'file_size_input_mb', 'file_size_output_mb',
            'metrics_resolution_width', 'metrics_resolution_height', 'metrics_frame_rate',
            'input_bitrate_kbps', 'bits_per_pixel', 'metrics_avg_motion', 'metrics_avg_edge_density',
            'metrics_avg_texture', 'metrics_avg_color_complexity', 'metrics_avg_motion_variance', 
            'metrics_avg_grain_noise', 'metrics_avg_spatial_information', 'metrics_avg_temporal_information',
            'contrast_value', 'optimal_cq', 'adjusted_cq', 'processing_time_seconds',
            'compression_ratio', 'confidence_score', 'target_vmaf', 
            'predicted_vmaf_at_optimal_cq', 'actual_vmaf', 'vmaf_prediction_error','avg_cpu_percent', 'peak_cpu_percent', 'avg_memory_mb', 'peak_memory_mb',
            'avg_gpu_percent', 'peak_gpu_percent', 'disk_read_mb', 'disk_write_mb',
            'monitoring_duration', 'performance_samples' 
            'prob_screen_content', 'prob_animation', 'prob_faces', 'prob_gaming', 
            'prob_other', 'prob_unclear'
        ]
        
        for feature in numeric_features:
            try:
                value = validated.get(feature, 0)
                if value is None:
                    validated[feature] = 0.0
                else:
                    validated[feature] = float(value)
            except (ValueError, TypeError):
                validated[feature] = 0.0
        
        # Ensure boolean types
        boolean_features = [
            'encoding_success', 'hardware_acceleration', 'auto_detection_used',
            'has_motion_data', 'has_complexity_data', 'has_vmaf_data'
        ]
        
        for feature in boolean_features:
            try:
                value = validated.get(feature, False)
                if value is None:
                    validated[feature] = False
                else:
                    validated[feature] = bool(value)
            except (ValueError, TypeError):
                validated[feature] = False
        
        # Ensure string types - ENHANCED
        string_features = [
            'scene_type', 'codec_used', 'hardware_type', 'input_codec',
            'vmaf_model_used', 'vmaf_model_path_used', 'vmaf_model_type'  # Added VMAF model fields
        ]
        
        for feature in string_features:
            value = validated.get(feature, '')
            validated[feature] = str(value) if value is not None else ''
        
        return validated
    
    def _calculate_training_quality_score(self, scene_data):
        """Calculate a quality score for training data completeness."""
        score = 0
        max_score = 10
        
        # Core features available
        if scene_data.get('metrics_avg_motion', 0) > 0: score += 1
        if scene_data.get('metrics_resolution_width', 0) > 0: score += 1
        if scene_data.get('input_bitrate_kbps', 0) > 0: score += 1
        if scene_data.get('metrics_avg_texture', 0) > 0: score += 1
        if scene_data.get('bits_per_pixel', 0) > 0: score += 1
        
        # Classification data available
        if scene_data.get('scene_type', '') not in ['', 'unclear']: score += 1
        if scene_data.get('confidence_score', 0) > 0.5: score += 1
        
        # Quality prediction data available - UPDATED FOR VMAF
        if scene_data.get('predicted_vmaf', 0) > 0: score += 1
        if scene_data.get('actual_vmaf', 0) > 0: score += 1
        
        # Encoding success
        if scene_data.get('encoding_success', False): score += 1
        
        return (score / max_score) * 10  # Return score out of 10
    
    def _initialize_detailed_logging(self):
        """Initialize detailed logging settings."""
        # In reduced logging mode, skip some detailed logging
        if self.reduced_logging:
            print("Reduced logging mode enabled. Skipping detailed logging initialization.")
            return
        
        # Existing detailed logging initialization logic...
        pass  # Replace with actual implementation if needed