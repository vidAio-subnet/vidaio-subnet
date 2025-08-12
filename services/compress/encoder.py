import os
import torch
import time
import subprocess
import traceback
import pickle 
from sklearn.utils.validation import check_is_fitted # Pipeline validation
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
from utils.processing_utils import (
    encode_scene_with_size_check,
    classify_scene_from_path 
)
from utils.encode_video import encode_video
from utils.classify_scene import load_scene_classifier_model, CombinedModel

# def get_cq_from_lookup_table(scene_type, config):
#     """
#     Gets a CQ value from a lookup table in the config based on the scene type.
#     """
#     # Default CQ values if not found in config
#     default_cq_map = {
#         'animation': 28,
#         'low-action': 26,
#         'medium-action': 24,
#         'high-action': 22,
#         'default': 25
#     }
    
#     # Get the lookup table from the config, or use the default
#     cq_lookup_table = config.get('video_processing', {}).get('basic_cq_lookup', default_cq_map)
    
#     # Return the CQ for the scene type, or the default value
#     return cq_lookup_table.get(scene_type, cq_lookup_table.get('default', 25))

def get_cq_from_lookup_table(scene_type, config, target_vmaf=None, target_quality_level=None):
    """
    Look up the recommended CQ (Constant Quality) value for a scene type and target quality.
    
    This is the "basic" approach: we use a simple lookup table instead of 
    running AI predictions for each scene. It's much faster but less precise.
    
    CQ values control encoding quality:
    - Lower CQ = Higher quality, bigger file size (CQ 15-20)
    - Higher CQ = Lower quality, smaller file size (CQ 30-35)
    
    The lookup table now supports two selection modes:
    - By explicit quality level via config['video_processing']['basic_cq_lookup_by_quality']
    - By VMAF-derived tiers via config['video_processing']['basic_cq_lookup_tiered']
    - High quality tier (VMAF 95+): Lower CQ values for better quality
    - Medium quality tier (VMAF ~90): Balanced CQ values
    - Low quality tier (VMAF ~85): Higher CQ values for smaller files
    
    Args:
        scene_type (str): Scene classification from AI model
        config (dict): Configuration containing CQ lookup tables
    target_vmaf (float, optional): Target VMAF score for quality tier (fallback)
    target_quality_level (str, optional): 'High'|'Medium'|'Low' to select from by-quality table
        
    Returns:
        int: Recommended CQ value for encoding this scene type at target quality
    """
    vp = config.get('video_processing', {})

    # Prefer direct quality level lookup when provided
    if target_quality_level:
        norm = {'high': 'High', 'medium': 'Medium', 'low': 'Low'}
        q_key = norm.get(str(target_quality_level).lower(), target_quality_level)
        by_quality = vp.get('basic_cq_lookup_by_quality', {})
        if isinstance(by_quality, dict):
            table = by_quality.get(q_key) or by_quality.get(q_key.lower()) or {}
            if isinstance(table, dict):
                return table.get(scene_type, table.get('default', 25))

    # Determine quality tier based on target VMAF (fallback)
    if target_vmaf is None:
        quality_tier = 'medium'  # Default to medium if no target specified
    elif target_vmaf >= 93:
        quality_tier = 'high'
    elif target_vmaf >= 88:
        quality_tier = 'medium'
    else:
        quality_tier = 'low'
    
    # Default CQ values if not found in config file, organized by quality tier
    default_cq_map = {
        # High quality tier (lower CQ = higher quality)
        'high': {
            'animation': 25,      # Cartoons compress well naturally
            'low-action': 23,     # Text/faces need moderate CQ for clarity
            'medium-action': 21,  # Balanced CQ for general content
            'high-action': 19,    # Gaming/sports need lower CQ for quality
            'default': 22         # Safe middle-ground when unsure
        },
        # Medium quality tier (balanced CQ)
        'medium': {
            'animation': 28,      # Cartoons compress well, can use higher CQ
            'low-action': 26,     # Text/faces need moderate CQ for clarity
            'medium-action': 24,  # Balanced CQ for general content
            'high-action': 22,    # Gaming/sports need lower CQ for quality
            'default': 25         # Safe middle-ground when unsure
        },
        # Low quality tier (higher CQ = smaller files)
        'low': {
            'animation': 31,      # Cartoons compress well, can use higher CQ
            'low-action': 29,     # Text/faces need moderate CQ for clarity
            'medium-action': 27,  # Balanced CQ for general content
            'high-action': 25,    # Gaming/sports need lower CQ for quality
            'default': 28         # Safe middle-ground when unsure
        }
    }
    
    # Get the lookup table from config, or use our defaults
    tier_cq_maps = vp.get('basic_cq_lookup_tiered', default_cq_map)
    
    # Use the appropriate tier based on target quality
    cq_lookup_table = tier_cq_maps.get(quality_tier, tier_cq_maps.get('medium', default_cq_map['medium']))
    
    # Return the CQ for this scene type within the selected quality tier, with fallback to default
    return cq_lookup_table.get(scene_type, cq_lookup_table.get('default', 25))

# def load_encoding_resources(config, logging_enabled=True):
#     """Load AI models and other resources needed for BASIC encoding."""
    
#     model_paths = config.get('model_paths', {})
#     scene_model_path = model_paths.get('scene_classifier_model', "services/compress/models/scene_classifier_model.pth")
    
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     # Load scene classifier
#     model_state_dict, available_metrics, class_mapping = load_scene_classifier_model(scene_model_path, device, logging_enabled)
#     scene_classifier_model = CombinedModel(num_classes=len(class_mapping), metrics_dim=len(available_metrics))
#     scene_classifier_model.load_state_dict(model_state_dict)
#     scene_classifier_model.to(device)
#     scene_classifier_model.eval()
    
#     return {
#         "scene_classifier_model": scene_classifier_model,
#         "available_metrics": available_metrics,
#         "device": device,
#         "class_mapping": class_mapping
#     }

def get_scalers_from_pipeline(pipeline_path='src/models/preprocessing_pipeline.pkl', 
                             verbose=True, logging_enabled=True):
    """
    Load preprocessing pipeline and extract individual components with verbosity control.
    
    This function loads a complete scikit-learn preprocessing pipeline and extracts
    the individual components needed for VMAF prediction:
    - Feature scaler for normalizing video metrics
    - VMAF scaler for converting quality scores to model range
    - CQ (Constant Quality) parameter bounds for optimization
    
    The function also sets verbosity levels for all pipeline components to control
    the amount of output during batch processing operations.
    
    Pipeline Structure Expected:
        - feature_scaler: StandardScaler or similar for video features
        - vmaf_scaler: Custom scaler for VMAF quality scores  
        - cq_scaler: Custom scaler containing CQ parameter bounds
        
    Args:
        pipeline_path (str): Path to the saved preprocessing pipeline pickle file
        verbose (bool): Whether individual pipeline steps should produce verbose output
        logging_enabled (bool): Master logging control (overrides verbose if False)
        
    Returns:
        tuple: (pipeline_obj, feature_scaler_step, vmaf_scaler, cq_min, cq_max)
            - pipeline_obj: Complete fitted preprocessing pipeline
            - feature_scaler_step: Feature scaling component
            - vmaf_scaler: VMAF scaling component  
            - cq_min: Minimum CQ value for optimization bounds
            - cq_max: Maximum CQ value for optimization bounds
            Returns (None, None, None, None, None) if loading fails
    """
    
    # Import custom preprocessing classes needed for unpickling
    try:
        from utils.data_preprocessing import (
            ColumnDropper, VMAFScaler, TargetExtractor, CQScaler,
            ResolutionTransformer, FeatureScaler, FrameRateTransformer
        )
    except ImportError as e:
        if logging_enabled:
            print(f"❌ ERROR: Could not import preprocessing classes: {e}")
        return None, None, None, None, None
    
    # Control verbose output with master logging flag
    actual_verbose = verbose and logging_enabled
    
    # Initialize return variables
    pipeline_obj = None
    feature_scaler_step = None
    vmaf_scaler = None
    cq_min_original, cq_max_original = None, None
    
    # =================================================================
    # PIPELINE FILE LOADING AND VALIDATION
    # =================================================================
    if not os.path.exists(pipeline_path):
        if actual_verbose:
            print(f"❌ ERROR: Pipeline file not found at '{pipeline_path}'")
            print(f"   💡 Suggestion: Ensure preprocessing pipeline exists in src/models/ directory")
        return None, None, None, None, None
    
    try:
        # Load the pickled preprocessing pipeline
        if actual_verbose:
            print(f"📖 Loading preprocessing pipeline from: {pipeline_path}")
            
        with open(pipeline_path, 'rb') as f:
            pipeline_obj = pickle.load(f)
            
        if actual_verbose:
            print(f"✅ Pipeline loaded successfully")

        # =================================================================
        # VERBOSITY CONTROL FOR ALL PIPELINE COMPONENTS
        # =================================================================
        # Recursively set verbose flags for all components in the pipeline
        # This prevents excessive output during batch processing while allowing
        # detailed output when needed for debugging
        
        def set_pipeline_verbosity(pipeline, verbose_flag):
            """
            Recursively set verbose parameter for all pipeline components.
            
            Handles nested pipelines, feature unions, column transformers,
            and custom transformer classes.
            """
            if hasattr(pipeline, 'steps'):
                # Handle sklearn Pipeline objects
                for step_name, step_obj in pipeline.steps:
                    set_step_verbosity(step_obj, verbose_flag, step_name)
            elif hasattr(pipeline, 'named_steps'):
                # Handle pipelines with named step access
                for step_name, step_obj in pipeline.named_steps.items():
                    set_step_verbosity(step_obj, verbose_flag, step_name)
            else:
                # Handle single transformer objects
                set_step_verbosity(pipeline, verbose_flag, "pipeline")
        
        def set_step_verbosity(step_obj, verbose_flag, step_name="unknown"):
            """
            Set verbosity for a single pipeline step.
            
            Handles various transformer types and their verbosity parameters.
            """
            try:
                # ==========================================================
                # STANDARD VERBOSITY PARAMETERS
                # ==========================================================
                # Common verbose parameters found in sklearn and custom transformers
                verbose_attrs = ['verbose', 'verbose_', 'logging_enabled']
                
                for attr in verbose_attrs:
                    if hasattr(step_obj, attr):
                        setattr(step_obj, attr, verbose_flag)
                        if actual_verbose and verbose_flag:
                            print(f"   🔧 Set {attr}={verbose_flag} for step '{step_name}'")
                
                # ==========================================================
                # NESTED PIPELINE HANDLING
                # ==========================================================
                # Handle Pipeline objects nested within other pipelines
                if hasattr(step_obj, 'steps') or hasattr(step_obj, 'named_steps'):
                    set_pipeline_verbosity(step_obj, verbose_flag)
                
                # Handle wrapped transformers
                if hasattr(step_obj, 'transformer') and step_obj.transformer is not None:
                    set_step_verbosity(step_obj.transformer, verbose_flag, f"{step_name}.transformer")
                
                # ==========================================================
                # COMPOSITE TRANSFORMER HANDLING
                # ==========================================================
                # Handle FeatureUnion and ColumnTransformer which contain lists of transformers
                if hasattr(step_obj, 'transformer_list'):
                    for name, transformer in step_obj.transformer_list:
                        set_step_verbosity(transformer, verbose_flag, f"{step_name}.{name}")
                
                # ==========================================================
                # CUSTOM TRANSFORMER SPECIFIC PARAMETERS
                # ==========================================================
                # Handle verbosity parameters specific to our custom transformers
                custom_transformer_attrs = ['debug', 'show_warnings', 'print_info']
                for attr in custom_transformer_attrs:
                    if hasattr(step_obj, attr):
                        setattr(step_obj, attr, verbose_flag)
                        if actual_verbose and verbose_flag:
                            print(f"   🎛️ Set custom {attr}={verbose_flag} for step '{step_name}'")
                            
            except Exception as e:
                if actual_verbose:
                    print(f"  ⚠️ Warning: Could not set verbosity for step '{step_name}': {e}")
        
        # Apply verbosity settings to the entire pipeline
        if actual_verbose:
            print(f"🔧 Setting verbosity to {actual_verbose} for all pipeline components...")
        
        set_pipeline_verbosity(pipeline_obj, actual_verbose)
        
        # =================================================================
        # PIPELINE FITNESS VALIDATION
        # =================================================================
        # Check if the loaded pipeline has been fitted (trained)
        # Unfitted pipelines cannot be used for transformation
        try:
            check_is_fitted(pipeline_obj)
            if actual_verbose:
                print("✅ Loaded pipeline is fitted and ready for use")
        except NotFittedError:
            if actual_verbose:
                print("⚠️ WARNING: Loaded pipeline is NOT fitted!")
                print("   🔧 Attempting to extract components anyway for fallback creation...")
        except Exception as e:
            if actual_verbose:
                print(f"⚠️ Warning: Could not verify pipeline fitness: {e}")

        # =================================================================
        # FEATURE SCALER EXTRACTION
        # =================================================================
        # Extract the feature scaling component used for normalizing video metrics
        feature_scaler_step = None
        scaler_step_name = 'feature_scaler'
        
        # Try to find by name in named_steps
        if hasattr(pipeline_obj, 'named_steps') and scaler_step_name in pipeline_obj.named_steps:
            feature_scaler_step = pipeline_obj.named_steps[scaler_step_name]
            if actual_verbose:
                print(f"✅ Found feature scaler step: '{scaler_step_name}'")
        
        # Fallback: search through all steps by name or type
        elif hasattr(pipeline_obj, 'steps'):
            for name, step in pipeline_obj.steps:
                if scaler_step_name in name or isinstance(step, (StandardScaler)):
                    feature_scaler_step = step
                    if actual_verbose:
                        print(f"✅ Found feature scaler step: '{name}'")
                    break

        # Create default if not found
        if feature_scaler_step is None:
            if actual_verbose:
                print("⚠️ Warning: Could not find feature scaler step. Creating default StandardScaler...")
            feature_scaler_step = StandardScaler()

        # =================================================================
        # VMAF SCALER EXTRACTION
        # =================================================================
        # Extract the VMAF scaling component for quality score normalization
        vmaf_scaler = None
        vmaf_scaler_step_name = 'vmaf_scaler'
        
        if hasattr(pipeline_obj, 'named_steps') and vmaf_scaler_step_name in pipeline_obj.named_steps:
            vmaf_scaler = pipeline_obj.named_steps[vmaf_scaler_step_name]
            if actual_verbose:
                print(f"✅ Found VMAF scaler step: '{vmaf_scaler_step_name}'")

        # Create default VMAF scaler if not found
        if vmaf_scaler is None:
            if actual_verbose:
                print("⚠️ Warning: Could not find VMAF scaler. Creating default...")
                
            class DefaultVMAFScaler:
                """Default VMAF scaler with typical VMAF range."""
                def __init__(self): 
                    self.min_val, self.max_val = 20.0, 100.0  # Typical VMAF range
                    
            vmaf_scaler = DefaultVMAFScaler()
            if actual_verbose:
                print(f"   📊 Using default VMAF range: {vmaf_scaler.min_val}-{vmaf_scaler.max_val}")

        # =================================================================
        # CQ PARAMETER BOUNDS EXTRACTION
        # =================================================================
        # Extract CQ (Constant Quality) parameter bounds for optimization
        default_cq_min, default_cq_max = 10, 51  # Conservative CQ range
        cq_min, cq_max = default_cq_min, default_cq_max
        cq_scaler_step_name = 'cq_scaler'

        # Try to find CQ scaler component
        if hasattr(pipeline_obj, 'named_steps') and cq_scaler_step_name in pipeline_obj.named_steps:
            cq_scaler = pipeline_obj.named_steps[cq_scaler_step_name]
            
            # Check for CQ bounds in various attribute naming conventions
            if hasattr(cq_scaler, 'min_cq') and hasattr(cq_scaler, 'max_cq'):
                cq_min, cq_max = cq_scaler.min_cq, cq_scaler.max_cq
                if actual_verbose:
                    print(f"✅ Found CQ range in step '{cq_scaler_step_name}': {cq_min}-{cq_max}")
            elif hasattr(cq_scaler, 'min_val') and hasattr(cq_scaler, 'max_val'):
                cq_min, cq_max = cq_scaler.min_val, cq_scaler.max_val
                if actual_verbose:
                    print(f"✅ Found CQ range in scaler attributes: {cq_min}-{cq_max}")

        if actual_verbose:
            print(f"📊 Using CQ optimization range: {cq_min}-{cq_max}")
            print(f"   💡 This range will constrain the binary search for optimal quality parameters")

        return pipeline_obj, feature_scaler_step, vmaf_scaler, int(cq_min), int(cq_max)

    except Exception as e:
        if actual_verbose:
            print(f"❌ ERROR: Failed to load or extract from pipeline '{pipeline_path}': {e}")
            print("🔍 Detailed error information:")
            traceback.print_exc()
        return None, None, None, None, None




def load_encoding_resources(config, logging_enabled=True):
    """
    Load AI models and other resources needed for BASIC encoding.
    
    The basic pipeline needs:
    1. Scene classifier AI model - to identify content type (animation, gaming, etc.)
    2. Preprocessing pipeline - to prepare video frames for the AI model
    3. GPU/CPU device selection - for running the AI inference
    
    This is lighter than the enhanced pipeline which also loads VMAF prediction models.
    
    Args:
        config (dict): Configuration containing model file paths
        logging_enabled (bool): Whether to print loading status messages
        
    Returns:
        tuple: (scene_model, preprocessing_pipeline, device) ready for encoding
    """
    # Get model file paths from config
    model_paths = config.get('model_paths', {})
    scene_model_path = model_paths.get('scene_classifier_model', "../models/scene_classifier_model.pth")
    preprocessing_pipeline_path = model_paths.get('preprocessing_pipeline', "src/models/preprocessing_pipeline.pkl")
    
    # Choose GPU if available, otherwise use CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load scene classifier AI model
    model_state_dict, available_metrics, class_mapping = load_scene_classifier_model(scene_model_path, device, logging_enabled)
    print(f"   ✅ Scene classifier model loaded from {scene_model_path} on {device}")
    scene_classifier_model = CombinedModel(num_classes=len(class_mapping), metrics_dim=len(available_metrics))
    scene_classifier_model.load_state_dict(model_state_dict)
    scene_classifier_model.to(device)
    scene_classifier_model.eval()
    
    # Load preprocessing pipeline for preparing video frames
    pipeline_obj, feature_scaler_step, vmaf_scaler, cq_min, cq_max = get_scalers_from_pipeline(preprocessing_pipeline_path, verbose=logging_enabled, logging_enabled=logging_enabled)
    print(f"   ✅ Preprocessing pipeline loaded from {preprocessing_pipeline_path}")
    
    # Return all the loaded resources in a dictionary
    return {
        "scene_classifier_model": scene_classifier_model,  # AI model for classifying scene content
        "available_metrics": available_metrics,             # Video metrics the model expects
        "device": device,                                   # GPU or CPU device
        "class_mapping": class_mapping,                     # Maps AI output numbers to scene names
        "pipeline_obj": pipeline_obj,                       # Preprocessing pipeline object
        "feature_scaler_step": feature_scaler_step,         # Scales video features for AI
        "vmaf_scaler": vmaf_scaler,                         # Scales VMAF values (not used in basic)
        "cq_min": cq_min,                                   # Minimum CQ value allowed
        "cq_max": cq_max                                    # Maximum CQ value allowed
    }


# def ai_encoding(scene_metadata, config, resources, target_vmaf=None, logging_enabled=True):
#     """
#     Part 3: BASIC AI-powered analysis and encoding for a single scene.
#     This version uses a scene classifier and a CQ lookup table.
#     """
#     logging_enabled=True
#     if not scene_metadata:
#         return None, {
#             'scene_number': 0,
#             'encoding_success': False,
#             'error_reason': 'Scene metadata is None or empty',
#             'processing_time_seconds': 0.0,
#             'encoded_path': None,
#             'path': None
#         }
    
#     def safe_float(value, default=0.0):
#         if value is None: return default
#         try:
#             result = float(value)
#             return result if not (result != result) else default
#         except (TypeError, ValueError):
#             return default
    
#     def safe_positive_float(value, default=1.0):
#         result = safe_float(value, default)
#         return max(result, 0.1)
    
#     scene_path = scene_metadata.get('path')
#     scene_number = int(safe_float(scene_metadata.get('scene_number', 1), 1))
#     start_time = safe_float(scene_metadata.get('start_time'), 0.0)
#     end_time = safe_float(scene_metadata.get('end_time'), 0.0)
#     scene_duration = safe_positive_float(scene_metadata.get('duration'), 1.0)
    
#     if end_time <= start_time or scene_duration <= 0:
#         if end_time > start_time:
#             scene_duration = end_time - start_time
#         else:
#             try:
#                 cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', scene_path]
#                 result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
#                 if result.returncode == 0 and result.stdout.strip():
#                     scene_duration = safe_positive_float(result.stdout.strip(), 1.0)
#                     end_time = start_time + scene_duration
#                 else:
#                     scene_duration = 1.0
#             except Exception:
#                 scene_duration = 1.0

#     original_video_metadata = scene_metadata.get('original_video_metadata', {})
    
#     if logging_enabled:
#         print(f"\n🎬 Processing Scene {scene_number} (Basic Mode)")
#         print(f"   📁 File: {os.path.basename(scene_path) if scene_path else 'None'}")
#         print(f"   ⏱️ Timing: {start_time:.1f}s - {end_time:.1f}s (duration: {scene_duration:.1f}s)")

#     if not scene_path or not os.path.exists(scene_path) or os.path.getsize(scene_path) == 0:
#         return None, {
#             'scene_number': scene_number,
#             'encoding_success': False,
#             'error_reason': 'Scene file is missing, empty, or inaccessible',
#             'processing_time_seconds': 0.0,
#             'encoded_path': None,
#             'path': scene_path
#         }

#     processing_start_time = time.time()
    
#     temp_dir = config.get('directories', {}).get('temp_dir', './videos/temp_scenes')
    
#     # BASIC MODE: Use a fixed target VMAF for reference if needed, but it doesn't drive CQ selection
#     target_vmaf = safe_float(target_vmaf or original_video_metadata.get('target_vmaf') or 
#                            config.get('video_processing', {}).get('target_vmaf', 93.0), 93.0)

#     # Codec selection logic from enhanced version
#     original_codec = original_video_metadata.get('original_codec', 'unknown')
#     target_codec_from_part1 = original_video_metadata.get('target_codec', 'auto')
#     current_codec = original_video_metadata.get('codec', original_codec)
#     config_codec = config.get('video_processing', {}).get('codec', 'auto')
    
#     # Define codec upgrade map outside the conditional blocks
#     codec_upgrade_map = {'h264': 'libsvtav1', 'hevc': 'libsvtav1', 'vp9': 'libsvtav1', 'av1': 'libsvtav1'}
    
#     if target_codec_from_part1 and target_codec_from_part1 != 'auto':
#         codec = target_codec_from_part1
#     elif config_codec != 'auto':
#         codec = config_codec
#     else:
#         codec = codec_upgrade_map.get(current_codec.lower(), 'libsvtav1')

#     if logging_enabled:
#         print(f"   🎥 Selected Codec: {codec}")

#     # BASIC ANALYSIS: Classify scene to select CQ from lookup table
#     if logging_enabled:
#         print(f"   🤖 Running basic scene classification...")

#     scene_type = 'default'
#     confidence_score = 0.0
#     try:        
#         scene_type, detailed_results = classify_scene_from_path(
#             scene_path=scene_path,
#             temp_dir=temp_dir,
#             scene_classifier_model=resources['scene_classifier_model'],
#             available_metrics=resources['available_metrics'],
#             device=resources['device'],
#             class_mapping=resources['class_mapping'],
#             logging_enabled=logging_enabled
#         )
#         confidence_score = detailed_results.get('confidence_score', 0.0)

#         if logging_enabled:
#             print(f"   🎭 Scene classified as: '{scene_type}' (Confidence: {confidence_score:.2f})")

#     except Exception as e:
#         if logging_enabled:
#             print(f"   ❌ Scene classification failed: {e}")
#             traceback.print_exc()
#         # Continue with default scene type
    
#     # Get CQ from lookup table
#     base_cq = get_cq_from_lookup_table(scene_type, config)
#     if logging_enabled:
#         print(f"   🎚️ Base CQ from lookup table for '{scene_type}': {base_cq}")

#     # Apply conservative adjustment from config
#     conservative_cq_adjustment = safe_float(config.get('video_processing', {}).get('conservative_cq_adjustment', 2), 2)
#     final_cq = min(base_cq + conservative_cq_adjustment, 51.0)
#     if logging_enabled:
#         print(f"   🔧 Applied conservative adjustment: +{conservative_cq_adjustment} -> Final CQ: {final_cq}")

#     # Create a placeholder scene_data object
#     scene_data = {
#         'path': scene_path,
#         'scene_number': scene_number,
#         'start_time': start_time,
#         'end_time': end_time,
#         'duration': scene_duration,
#         'original_video_metadata': original_video_metadata,
#         'scene_type': scene_type,
#         'confidence_score': confidence_score,
#         'optimal_cq': base_cq,
#         'adjusted_cq': final_cq,
#         'final_adjusted_cq': final_cq,
#         'codec_selection_process': {'final_selected_codec': codec},
#         'model_training_data': {} # Placeholder
#     }

#     # ENCODING
#     output_scene_path = os.path.join(
#         temp_dir,
#         f"encoded_scene_{scene_number:03d}_{start_time:.0f}s-{end_time:.0f}s_{codec.lower()}.mp4"
#     )
#     os.makedirs(temp_dir, exist_ok=True)

#     size_increase_protection = config.get('video_processing', {}).get('size_increase_protection', True)
#     max_encoding_retries = int(safe_float(config.get('video_processing', {}).get('max_encoding_retries', 2), 2))
    
#     encoding_start_time = time.time()
#     encoded_path = None
#     encoding_time = 0

#     if size_increase_protection:
#         if logging_enabled:
#             print(f"   🛡️ Size increase protection enabled")
#         try:
#             encoded_path, encoding_time = encode_scene_with_size_check(
#                 scene_path=scene_path,
#                 output_path=output_scene_path,
#                 codec=codec,
#                 adjusted_cq=final_cq,
#                 content_type=scene_type,
#                 contrast_value=0.5, # Default contrast for basic mode
#                 max_retries=max_encoding_retries,
#                 logging_enabled=logging_enabled
#             )
#         except Exception as e:
#             if logging_enabled:
#                 print(f"   ❌ Size-protected encoding failed: {e}")
#     else:
#         if logging_enabled:
#             print(f"   ⚡ Standard encoding (size protection disabled)")
#         try:
#             _, encoding_time = encode_video(
#                 input_path=scene_path,
#                 output_path=output_scene_path,
#                 codec=codec,
#                 rate=final_cq,
#                 scene_type=scene_type,
#                 contrast_value=0.5, # Default contrast
#                 logging_enabled=logging_enabled
#             )
#             encoded_path = output_scene_path
#         except Exception as e:
#             if logging_enabled:
#                 print(f"   ❌ Standard encoding failed: {e}")

#     # Final update of scene_data
#     encoding_success = bool(encoded_path and os.path.exists(encoded_path) and os.path.getsize(encoded_path) > 0)
#     scene_data['encoding_success'] = encoding_success
#     scene_data['encoded_path'] = encoded_path if encoding_success else None
#     scene_data['encoding_time'] = safe_float(encoding_time, 0)
#     scene_data['processing_time_seconds'] = time.time() - processing_start_time

#     if encoding_success:
#         if logging_enabled:
#             print(f"   ✅ Scene {scene_number} encoded successfully.")
#         return encoded_path, scene_data
#     else:
#         if logging_enabled:
#             print(f"   ❌ Encoding failed for scene {scene_number}.")
#         scene_data['error_reason'] = 'Encoding process failed or produced an empty file'
#         return None, scene_data



def ai_encoding(scene_metadata, config, resources, target_vmaf=None, target_quality_level=None, logging_enabled=True):
    """
    Part 3: BASIC AI-powered analysis and encoding for a single scene.
    
    This is the "basic" approach that prioritizes speed over precision:
    1. Use AI to classify the scene content (animation, gaming, text, etc.)
    2. Look up a pre-determined CQ value from a table based on scene type AND quality tier
    3. Encode the scene once with that CQ value
    
    This implementation now supports quality tiers similar to the enhanced miner:
    - High quality tier (VMAF ~95): Lower CQ values for better quality
    - Medium quality tier (VMAF ~90): Balanced CQ values
    - Low quality tier (VMAF ~85): Higher CQ values for smaller files
    
    The basic approach is still much faster than the "enhanced" approach which:
    - Runs multiple encoding attempts with different CQ values
    - Uses AI to predict VMAF quality for each attempt
    - Uses binary search to find the optimal CQ
    
    Args:
        scene_metadata (dict): Contains scene file path, duration, etc.
        config (dict): Configuration settings for encoding
        resources (dict): Pre-loaded AI models and preprocessing tools
        target_vmaf (float): Target quality used to select appropriate quality tier
        target_quality_level (str): Desired quality level ('High', 'Medium', 'Low') used for CQ lookup
        logging_enabled (bool): Whether to print progress messages
        
    Returns:
        tuple: (success_status, scene_data_dict)
            - success_status: True if encoding succeeded, False otherwise
            - scene_data_dict: Contains encoding results, file sizes, CQ used, etc.
    """
    logging_enabled=True  # Force logging for debugging
    
    # Handle missing scene metadata
    if not scene_metadata:
        return None, {
            'scene_number': 0,
            'encoding_success': False,
            'error_reason': 'Scene metadata is None or empty',
            'processing_time_seconds': 0.0,
            'encoded_path': None,
            'path': None
        }
    
    # Helper functions to safely handle potentially invalid numeric values
    def safe_float(value, default=0.0):
        """Convert value to float, return default if invalid"""
        if value is None: return default
        try:
            result = float(value)
            return result if not (result != result) else default  # Check for NaN
        except (TypeError, ValueError):
            return default
    
    def safe_positive_float(value, default=1.0):
        """Convert to float and ensure it's positive"""
        result = safe_float(value, default)
        return max(result, 0.1)  # Minimum of 0.1 to avoid zero values
    
    # Extract scene information from metadata
    scene_path = scene_metadata.get('path')
    scene_number = int(safe_float(scene_metadata.get('scene_number', 1), 1))
    start_time = safe_float(scene_metadata.get('start_time'), 0.0)
    end_time = safe_float(scene_metadata.get('end_time'), 0.0)
    scene_duration = safe_positive_float(scene_metadata.get('duration'), 1.0)
    
    # Validate scene timing and duration
    if end_time <= start_time or scene_duration <= 0:
        # Try to fix timing issues
        if end_time > start_time:
            scene_duration = end_time - start_time
        else:
            # Get duration from the video file directly
            try:
                cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', scene_path]
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
                if result.returncode == 0 and result.stdout.strip():
                    scene_duration = safe_positive_float(result.stdout.strip(), 1.0)
                    end_time = start_time + scene_duration
                else:
                    scene_duration = 1.0
            except Exception:
                scene_duration = 1.0

    # Get original video information for codec selection
    original_video_metadata = scene_metadata.get('original_video_metadata', {})
    
    if logging_enabled:
        print(f"\n🎬 Processing Scene {scene_number} (Basic Mode)")
        print(f"   📁 File: {os.path.basename(scene_path) if scene_path else 'None'}")
        print(f"   ⏱️ Timing: {start_time:.1f}s - {end_time:.1f}s (duration: {scene_duration:.1f}s)")

    # Verify scene file exists and is not empty
    if not scene_path or not os.path.exists(scene_path) or os.path.getsize(scene_path) == 0:
        return None, {
            'scene_number': scene_number,
            'encoding_success': False,
            'error_reason': 'Scene file is missing, empty, or inaccessible',
            'processing_time_seconds': 0.0,
            'encoded_path': None,
            'path': scene_path
        }

    processing_start_time = time.time()
    
    # Get temporary directory for intermediate files
    temp_dir = config.get('directories', {}).get('temp_dir', './videos/temp_scenes')
    
    # BASIC MODE WITH QUALITY TIERS: Prefer explicit quality level; keep VMAF as fallback/for logs
    if not target_quality_level:
        target_quality_level = original_video_metadata.get('target_quality') or config.get('video_processing', {}).get('target_quality')
    target_vmaf = safe_float(target_vmaf or original_video_metadata.get('target_vmaf') or 
                           config.get('video_processing', {}).get('target_vmaf', 0.0), 0.0)

    # Choose encoding codec (prioritize config over auto-detection)
    original_codec = original_video_metadata.get('original_codec', 'unknown')
    target_codec_from_part1 = original_video_metadata.get('target_codec', 'auto')
    current_codec = original_video_metadata.get('codec', original_codec)
    config_codec = config.get('video_processing', {}).get('codec', 'auto')
    
    if target_codec_from_part1 and target_codec_from_part1 != 'auto':
        codec = target_codec_from_part1
    elif config_codec != 'auto':
        codec = config_codec
    else:
        # Auto-upgrade older codecs to modern AV1
        codec_upgrade_map = {'h264': 'av1_nvenc', 'hevc': 'av1_nvenc', 'vp9': 'av1_nvenc', 'av1': 'av1_nvenc'}
        codec = codec_upgrade_map.get(current_codec.lower(), 'av1_nvenc')

    if logging_enabled:
        print(f"   🎥 Selected Codec: {codec}")

    # STEP 1: BASIC ANALYSIS - Classify scene and extract video features in one call
    if logging_enabled:
        print(f"   🤖 Running scene classification and feature extraction...")
    
    scene_type = 'default'
    confidence_score = 0.0
    video_features = {}
    detailed_results = {}
    try:
        # Run AI scene classification to determine content type
        # This also extracts video features, so we get both in one call
        classification_result = classify_scene_from_path(
            scene_path=scene_path,
            temp_dir=temp_dir,
            scene_classifier_model=resources['scene_classifier_model'],
            available_metrics=resources['available_metrics'],
            device=resources['device'],
            metrics_scaler=resources['feature_scaler_step'],
            class_mapping=resources['class_mapping'],
            logging_enabled=logging_enabled,
        )
        
        # Handle the tuple return from classify_scene_from_path (now returns 3 items)
        if isinstance(classification_result, tuple) and len(classification_result) == 3:
            scene_type, detailed_results, video_features = classification_result
            confidence_score = detailed_results.get('confidence_score', 0.0)
        elif isinstance(classification_result, tuple) and len(classification_result) == 2:
            # Fallback for older return format
            scene_type, detailed_results = classification_result
            confidence_score = detailed_results.get('confidence_score', 0.0)
            video_features = {}
        elif isinstance(classification_result, dict):
            # Fallback if it returns a dict instead of tuple
            scene_type = classification_result.get('scene_type', 'default')
            confidence_score = classification_result.get('confidence_score', 0.0)
            video_features = {}
        else:
            # Unknown return type, use defaults
            scene_type = 'default'
            confidence_score = 0.0
            video_features = {}

        if logging_enabled:
            print(f"   🎭 Scene classified as: '{scene_type}' (Confidence: {confidence_score:.2f})")
            if video_features:
                print(f"   📊 Video features extracted: {len(video_features)} metrics")

    except Exception as e:
        if logging_enabled:
            print(f"   ❌ Scene classification failed: {e}")
            traceback.print_exc()
        # Continue with default values
    
    # Map the scene type to lookup table key
    original_scene_type = scene_type
    mapped_scene_type = map_scene_type_to_lookup_key(scene_type)
    
    if logging_enabled and original_scene_type != mapped_scene_type:
        print(f"   🔄 Mapped scene type '{original_scene_type}' -> '{mapped_scene_type}' for CQ lookup")
    
    # Get quality tier information for logging
    print(f"   🎯 Target quality level: {target_quality_level}")
    # If only quality level provided, derive indicative VMAF for logging purposes
    if target_quality_level and not target_vmaf:
        target_vmaf = get_target_vmaf_from_quality(target_quality_level)
        if logging_enabled:
            print(f"   ℹ️ Indicative VMAF from quality level '{target_quality_level}': {target_vmaf}")
    
    # Get CQ from lookup table using mapped scene type and target quality/VMAF
    base_cq = get_cq_from_lookup_table(
        mapped_scene_type,
        config,
        target_vmaf=target_vmaf,
        target_quality_level=target_quality_level,
    )
    
    # Log quality tier and CQ selection
    if logging_enabled:
        # Determine label for logging
        if target_quality_level:
            tier_label = str(target_quality_level).upper()
        else:
            if target_vmaf >= 93:
                tier_label = "HIGH"
            elif target_vmaf >= 88:
                tier_label = "MEDIUM"
            else:
                tier_label = "LOW"

        print(f"   🎯 Using quality tier: {tier_label} (Indicative VMAF: {target_vmaf if target_vmaf else 'n/a'})")
        print(f"   🎚️ Base CQ from lookup table for '{mapped_scene_type}' at {tier_label} quality: {base_cq}")

    
    
    
    # Apply conservative adjustment from config
    conservative_cq_adjustment = safe_float(config.get('video_processing', {}).get('conservative_cq_adjustment', 2), 2)
    final_cq = min(base_cq + conservative_cq_adjustment, 51.0)
    if logging_enabled:
        print(f"   🔧 Applied conservative adjustment: +{conservative_cq_adjustment} -> Final CQ: {final_cq}")

    # Create a placeholder scene_data object
    scene_data = {
        'path': scene_path,
        'scene_number': scene_number,
        'start_time': start_time,
        'end_time': end_time,
        'duration': scene_duration,
        'original_video_metadata': original_video_metadata,
        'scene_type': original_scene_type,  # Keep original scene type from classifier
        'mapped_scene_type': mapped_scene_type,  # Add mapped scene type for CQ lookup
        'confidence_score': confidence_score,
    'optimal_cq': base_cq,
        'adjusted_cq': final_cq,
        'final_adjusted_cq': final_cq,
        'codec_selection_process': {'final_selected_codec': codec},
    'model_training_data': {}, # Placeholder
        'target_quality_level': target_quality_level,
    'base_cq_for_quality': base_cq,
    }

    # ENCODING
    output_scene_path = os.path.join(
        temp_dir,
        f"encoded_scene_{scene_number:03d}_{start_time:.0f}s-{end_time:.0f}s_{codec.lower()}.mp4"
    )
    os.makedirs(temp_dir, exist_ok=True)

    size_increase_protection = config.get('video_processing', {}).get('size_increase_protection', True)
    max_encoding_retries = int(safe_float(config.get('video_processing', {}).get('max_encoding_retries', 2), 2))
    
    encoding_start_time = time.time()
    encoded_path = None
    encoding_time = 0

    if size_increase_protection:
        if logging_enabled:
            print(f"   🛡️ Size increase protection enabled")
        try:
            encoded_path, encoding_time = encode_scene_with_size_check(
                scene_path=scene_path,
                output_path=output_scene_path,
                codec=codec,
                adjusted_cq=final_cq,
                content_type=scene_type,
                contrast_value=0.5, # Default contrast for basic mode
                max_retries=max_encoding_retries,
                logging_enabled=logging_enabled
            )
        except Exception as e:
            if logging_enabled:
                print(f"   ❌ Size-protected encoding failed: {e}")
    else:
        if logging_enabled:
            print(f"   ⚡ Standard encoding (size protection disabled)")
        try:
            _, encoding_time = encode_video(
                input_path=scene_path,
                output_path=output_scene_path,
                codec=codec,
                rate=final_cq,
                scene_type=scene_type,
                contrast_value=0.5, # Default contrast
                logging_enabled=logging_enabled
            )
            encoded_path = output_scene_path
        except Exception as e:
            if logging_enabled:
                print(f"   ❌ Standard encoding failed: {e}")

    # Final update of scene_data
    encoding_success = bool(encoded_path and os.path.exists(encoded_path) and os.path.getsize(encoded_path) > 0)
    scene_data['encoding_success'] = encoding_success
    scene_data['encoded_path'] = encoded_path if encoding_success else None
    scene_data['encoding_time'] = safe_float(encoding_time, 0)
    scene_data['processing_time_seconds'] = time.time() - processing_start_time
    
    # Add file size information and codec details
    if encoding_success:
        # Calculate file sizes in MB
        input_file_size = os.path.getsize(scene_path) if os.path.exists(scene_path) else 0
        output_file_size = os.path.getsize(encoded_path) if os.path.exists(encoded_path) else 0
        
        scene_data['input_size_mb'] = input_file_size / (1024 * 1024)
        scene_data['encoded_file_size_mb'] = output_file_size / (1024 * 1024)
        
        # Calculate compression ratio
        if input_file_size > 0:
            compression_ratio = ((output_file_size - input_file_size) / input_file_size) * 100
            scene_data['compression_ratio'] = compression_ratio
        else:
            scene_data['compression_ratio'] = 0
        
        # Store codec information for filename generation
        scene_data['codec_used'] = codec
    else:
        scene_data['input_size_mb'] = 0
        scene_data['encoded_file_size_mb'] = 0
        scene_data['compression_ratio'] = 0
        scene_data['codec_used'] = 'unknown'

    # Add model training data for comprehensive reporting (similar to enhanced miner)
    scene_data['model_training_data'] = {
        'raw_video_features': video_features,
        'processed_video_features': {},  # Not used in basic miner
        'vmaf_model_features': {},       # Not used in basic miner
        'scene_classifier_features': video_features,  # Same as raw features for basic miner
        'scene_classifier_probabilities': detailed_results,
        'processing_timings': {
            'total_processing_time': time.time() - processing_start_time,
            'encoding_time': safe_float(encoding_time, 0) if 'encoding_time' in locals() else 0
        }
    }

    if encoding_success:
        if logging_enabled:
            print(f"   ✅ Scene {scene_number} encoded successfully.")
        return encoded_path, scene_data
    else:
        if logging_enabled:
            print(f"   ❌ Encoding failed for scene {scene_number}.")
        scene_data['error_reason'] = 'Encoding process failed or produced an empty file'
        return None, scene_data



def map_scene_type_to_lookup_key(scene_type):
    """
    Convert AI scene classifier output to CQ lookup table keys.
    
    The AI scene classifier returns descriptive names like:
    - 'Screen Content / Text' 
    - 'Animation / Cartoon / Rendered Graphics'
    - 'Faces / People'
    - 'Gaming Content'
    - 'other'
    - 'unclear'
    
    But our CQ lookup table uses simpler keys:
    - 'animation' (for cartoons/rendered content)
    - 'low-action' (for text/faces - less motion)
    - 'medium-action' (for general content)
    - 'high-action' (for gaming/sports - lots of motion)
    - 'default' (fallback for unclear content)
    
    Args:
        scene_type (str): Output from AI scene classifier
        
    Returns:
        str: Corresponding lookup table key
    """
    # Convert to lowercase for consistent matching
    scene_lower = scene_type.lower() if scene_type else 'default'
    
    # Map classifier output to our lookup table keys
    scene_mapping = {
        'screen content / text': 'low-action',        # Text doesn't need high quality
        'animation / cartoon / rendered graphics': 'animation',  # Cartoons compress well
        'faces / people': 'low-action',               # Faces need clarity but less motion
        'gaming content': 'high-action',              # Games have lots of motion/detail
        'other': 'medium-action',                     # General content gets medium quality
        'unclear': 'default',                        # When unsure, use safe default
        'default': 'default'
    }
    
    # Try direct mapping first
    if scene_lower in scene_mapping:
        return scene_mapping[scene_lower]
    
    # Try partial matching for robustness (in case of slight differences)
    for key, value in scene_mapping.items():
        if key in scene_lower or scene_lower in key:
            return value
    
    # If nothing matches, use safe default
    return 'default'



def get_target_vmaf_from_quality(quality_level):
    """
    Map quality level to target VMAF score for Basic AI encoding.
    
    This function provides consistent VMAF targets across Basic and Enhanced AI engines,
    ensuring users get predictable quality results regardless of the selected engine.
    
    Quality Level Mappings:
    - High: 95 VMAF (Maximum quality preservation)
    - Medium: 90 VMAF (Balanced quality and file size)
    - Low: 85 VMAF (Aggressive compression for smaller files)
    
    Args:
        quality_level (str): Target quality level ('High', 'Medium', 'Low')
    
    Returns:
        float: Target VMAF score for the specified quality level
    """
    quality_vmaf_map = {
        'High': 95.0,     # Maximum quality preservation
        'Medium': 90.0,   # Balanced quality and file size
        'Low': 85.0       # Aggressive compression for smaller files
    }
    
    return quality_vmaf_map.get(quality_level, 90.0)  # Default to Medium if unknown


