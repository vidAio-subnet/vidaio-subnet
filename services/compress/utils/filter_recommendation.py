import numpy as np
import subprocess
import os
import tempfile

def recommend_preprocessing_filters(quality_metrics, scene_type, target_vmaf=93.0,
            quality_thresholds=None,filter_intensity_settings=None):
    """
    Recommend optimal preprocessing filters with configurable quality thresholds and intensity.
    Updated to use both configurable thresholds AND intensity settings from Streamlit/config.
    """

    # Use default thresholds if none provided
    if quality_thresholds is None:
        quality_thresholds = {
            'noise_high': 0.4,
            'sharpness_low': 0.4,
            'contrast_low': 0.3,
            'artifacts_high': 0.4
        }
    
    # Use default intensity settings if none provided
    if filter_intensity_settings is None:
        filter_intensity_settings = {
            'max_cas_strength': 0.5,
            'max_unsharp_strength': 0.8,
            'max_contrast_boost': 1.2,
            'max_vibrance': 0.25
        }
    
    recommendations = {
        'filters': [],
        'reasons': [],
        'expected_vmaf_gain': 0,
        'processing_time_increase': 0
    }

    noise_threshold = quality_thresholds.get('noise_high', 0.4)
    if quality_metrics['noise_level'] > noise_threshold:
        if quality_metrics['noise_level'] > (noise_threshold + 0.3):  # Heavy noise
            # ✅ Use intensity-based strength instead of hardcoded values
            base_strength = 4.0
            intensity_multiplier = filter_intensity_settings.get('max_cas_strength', 0.5) * 2  # Scale for denoising
            adjusted_strength = min(base_strength * intensity_multiplier, 8.0)  # Cap at reasonable max
            
            recommendations['filters'].append(f'hqdn3d={adjusted_strength:.1f}:3.0:6.0:4.5')
            recommendations['reasons'].append(f'Heavy noise detected (>{noise_threshold + 0.3:.1f}) - applying strong 3D denoising (strength: {adjusted_strength:.1f})')
            recommendations['expected_vmaf_gain'] += 3.0
        else:
            # ✅ Use intensity-based adaptive denoising parameters
            base_strength_a = 0.02
            base_strength_b = 0.04
            intensity_factor = filter_intensity_settings.get('max_cas_strength', 0.5)
            
            strength_a = base_strength_a * (1 + intensity_factor)
            strength_b = base_strength_b * (1 + intensity_factor)
            
            recommendations['filters'].append(f'atadenoise=0a={strength_a:.3f}:0b={strength_b:.3f}:1a={strength_a*1.5:.3f}:1b={strength_b*1.25:.3f}')
            recommendations['reasons'].append(f'Moderate noise detected (>{noise_threshold:.1f}) - applying adaptive temporal denoising (intensity: {intensity_factor:.1f})')
            recommendations['expected_vmaf_gain'] += 1.5
        recommendations['processing_time_increase'] += 15
    
    # ✅ FIXED: Use configurable sharpness threshold AND intensity
    sharpness_threshold = quality_thresholds.get('sharpness_low', 0.4)
    if quality_metrics['sharpness'] < sharpness_threshold:
        cas_strength = filter_intensity_settings.get('max_cas_strength', 0.5)
        unsharp_strength = filter_intensity_settings.get('max_unsharp_strength', 0.8)
        
        if scene_type in ["Screen Content / Text", "Animation / Cartoon / Rendered Graphics"]:
            # ✅ Use configurable CAS strength instead of hardcoded 0.7
            recommendations['filters'].append(f'cas={cas_strength:.1f}')
            recommendations['reasons'].append(f'Low sharpness in artificial content (<{sharpness_threshold:.1f}) - applying CAS (strength: {cas_strength:.1f})')
            recommendations['expected_vmaf_gain'] += 2.0
        else:
            # ✅ Use configurable unsharp strength instead of hardcoded 1.2
            recommendations['filters'].append(f'unsharp=5:5:{unsharp_strength:.1f}:5:5:0.0')
            recommendations['reasons'].append(f'Low sharpness detected (<{sharpness_threshold:.1f}) - applying unsharp mask (strength: {unsharp_strength:.1f})')
            recommendations['expected_vmaf_gain'] += 1.5
        recommendations['processing_time_increase'] += 10
    
    # ✅ FIXED: Use configurable contrast threshold AND intensity
    contrast_threshold = quality_thresholds.get('contrast_low', 0.3)
    if quality_metrics['contrast'] < contrast_threshold:
        contrast_boost = filter_intensity_settings.get('max_contrast_boost', 1.2)
        
        if quality_metrics['contrast'] < (contrast_threshold - 0.1):  # Very low contrast
            recommendations['filters'].append('normalize=smoothing=50')
            recommendations['reasons'].append(f'Very low contrast (<{contrast_threshold - 0.1:.1f}) - applying RGB normalization')
            recommendations['expected_vmaf_gain'] += 2.5
        else:
            # ✅ Use configurable contrast boost instead of hardcoded 1.3
            brightness_adjust = (contrast_boost - 1.0) * 0.1  # Scale brightness with contrast
            gamma_adjust = max(0.8, 1.0 - (contrast_boost - 1.0) * 0.2)  # Adjust gamma inversely
            
            recommendations['filters'].append(f'eq=contrast={contrast_boost:.1f}:brightness={brightness_adjust:.2f}:gamma={gamma_adjust:.2f}')
            recommendations['reasons'].append(f'Low contrast detected (<{contrast_threshold:.1f}) - applying contrast enhancement (boost: {contrast_boost:.1f})')
            recommendations['expected_vmaf_gain'] += 1.8
        recommendations['processing_time_increase'] += 5
    
    # ✅ FIXED: Use configurable artifacts threshold
    artifacts_threshold = quality_thresholds.get('artifacts_high', 0.4)
    if quality_metrics['compression_artifacts'] > artifacts_threshold:
        if quality_metrics['compression_artifacts'] > (artifacts_threshold + 0.2):  # Heavy artifacts
            recommendations['filters'].append('pp7=qp=2:mode=medium')
            recommendations['reasons'].append(f'Heavy compression artifacts (>{artifacts_threshold + 0.2:.1f}) - applying postprocessing filter')
            recommendations['expected_vmaf_gain'] += 2.5
        else:
            recommendations['filters'].append('deband=1thr=0.02:2thr=0.02:3thr=0.02:blur=3')
            recommendations['reasons'].append(f'Compression artifacts detected (>{artifacts_threshold:.1f}) - applying debanding')
            recommendations['expected_vmaf_gain'] += 1.5
        recommendations['processing_time_increase'] += 25
    
    # ✅ FIXED: Use configurable vibrance intensity for color enhancement
    if quality_metrics.get('color_saturation', 0.5) < 0.3 and scene_type == "Faces / People":
        vibrance_intensity = filter_intensity_settings.get('max_vibrance', 0.25)
        recommendations['filters'].append(f'vibrance=intensity={vibrance_intensity:.2f}')
        recommendations['reasons'].append(f'Low color saturation in faces - applying selective saturation boost (intensity: {vibrance_intensity:.2f})')
        recommendations['expected_vmaf_gain'] += 1.0
        recommendations['processing_time_increase'] += 5
    
    # Rest of the function remains the same...
    
    return recommendations

def should_apply_preprocessing(quality_metrics, scene_type, target_vmaf, processing_time_budget=None,quality_thresholds=None,filter_intensity_settings=None):
    """
    Determine if preprocessing is worth applying based on analysis.
    """
    recommendations = recommend_preprocessing_filters(
        quality_metrics, scene_type, target_vmaf, quality_thresholds, filter_intensity_settings
    )
    # Calculate benefit vs cost
    expected_gain = recommendations['expected_vmaf_gain']
    time_cost = recommendations['processing_time_increase']
    
    # Decision criteria
    decision = {
        'apply_preprocessing': False,
        'recommended_filters': [],
        'priority_filters': [],
        'skip_reasons': []
    }
    if expected_gain >= 3.0:
        decision['apply_preprocessing'] = True
        decision['recommended_filters'] = recommendations['filters']
        decision['reasons'] = recommendations['reasons']  # ✅ Add reasons to decision
        
    elif expected_gain >= 1.5 and (processing_time_budget is None or time_cost <= processing_time_budget):
        decision['apply_preprocessing'] = True
        high_priority = []
        high_priority_reasons = []
        for i, reason in enumerate(recommendations['reasons']):
            if any(keyword in reason.lower() for keyword in ['heavy', 'very low', 'text content']):
                high_priority.append(recommendations['filters'][i])
                high_priority_reasons.append(reason)
        decision['priority_filters'] = high_priority
        decision['reasons'] = high_priority_reasons  # ✅ Add reasons to decision
        
    else:
        decision['skip_reasons'].append(f"Low expected gain ({expected_gain:.1f} VMAF points)")
        if processing_time_budget and time_cost > processing_time_budget:
            decision['skip_reasons'].append(f"Time cost too high ({time_cost}% increase)")
    
    return decision

def apply_preprocessing_filters(input_file, filters, temp_dir):
    """
    Apply preprocessing filters to video before encoding.
    """
    if not filters:
        return input_file
    
    preprocessed_file = os.path.join(temp_dir, f"preprocessed_{os.path.basename(input_file)}")
    filter_chain = ",".join(filters)
    
    cmd = [
        'ffmpeg', '-y', '-i', input_file,
        '-vf', filter_chain,
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '15',
        '-c:a', 'copy',
        preprocessed_file
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(preprocessed_file):
            return preprocessed_file
        else:
            print(f"Preprocessing failed: {result.stderr}")
            return input_file
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return input_file

def get_content_specific_filters(scene_type, quality_metrics):
    """
    Get content-specific filter recommendations using supported filters.
    """
    filters = []
    
    if scene_type == "Screen Content / Text":
        # For text content, prioritize sharpness
        if quality_metrics.get('sharpness', 0.5) < 0.6:
            filters.append('cas=0.8')  # Strong contrast adaptive sharpening
        if quality_metrics.get('noise_level', 0) > 0.3:
            filters.append('hqdn3d=1.0:0.5:2.0:1.0')  # Light denoising to preserve text
            
    elif scene_type == "Faces / People":
        # For faces, prioritize skin tone and detail preservation
        if quality_metrics.get('noise_level', 0) > 0.2:
            filters.append('vaguedenoiser=threshold=2.0:method=soft')  # Gentle denoising
        if quality_metrics.get('color_saturation', 0.5) < 0.4:
            filters.append('vibrance=intensity=0.2')  # Enhance skin tones
            
    elif scene_type == "Animation / Cartoon / Rendered Graphics":
        # For animation, enhance edges and colors
        if quality_metrics.get('edge_density', 0.5) < 0.4:
            filters.append('cas=0.6')  # Edge enhancement
        if quality_metrics.get('color_saturation', 0.5) < 0.5:
            filters.append('vibrance=intensity=0.3')  # Color enhancement
            
    elif scene_type == "Gaming Content":
        # For gaming, enhance details and reduce compression artifacts
        if quality_metrics.get('compression_artifacts', 0) > 0.3:
            filters.append('deband=1thr=0.02:2thr=0.02:3thr=0.02')
        if quality_metrics.get('sharpness', 0.5) < 0.5:
            filters.append('unsharp=5:5:0.8:5:5:0.0')
    
    return filters