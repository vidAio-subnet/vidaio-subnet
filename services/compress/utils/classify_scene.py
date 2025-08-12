import os
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    EfficientNet_V2_S_Weights,
    ResNet18_Weights,
    SqueezeNet1_1_Weights
)

CLASS_MAPPING = {
    'Screen Content / Text': 0,
    'Animation / Cartoon / Rendered Graphics': 1,
    'Faces / People': 2,
    'Gaming Content': 3,
    'other': 4,
    'unclear': 5
}
INV_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}
VIDEO_METRICS = [
    'metrics_avg_motion',
    'metrics_avg_edge_density',
    'metrics_avg_texture',
    'metrics_avg_temporal_information',
    'metrics_avg_spatial_information',
    'metrics_avg_color_complexity',
    'metrics_avg_motion_variance',
    'metrics_avg_grain_noise'
]

class CombinedModel(nn.Module):
    def __init__(self, num_classes, model_type='mobilenet_v3_small', use_pretrained=True, metrics_dim=10):
        super(CombinedModel, self).__init__()

        # Image feature extractor
        if model_type == 'mobilenet_v3_small':
            weights = MobileNet_V3_Small_Weights.DEFAULT if use_pretrained else None
            self.image_model = models.mobilenet_v3_small(weights=weights)
            self.image_features = nn.Sequential(*list(self.image_model.children())[:-1])
            image_feature_dim = 576  # MobileNetV3 Small feature dimension

        elif model_type == 'efficientnet_v2_s':
            weights = EfficientNet_V2_S_Weights.DEFAULT if use_pretrained else None
            self.image_model = models.efficientnet_v2_s(weights=weights)
            self.image_features = nn.Sequential(*list(self.image_model.children())[:-1])
            image_feature_dim = 1280  # EfficientNet v2 Small feature dimension

        elif model_type == 'resnet18':
            weights = ResNet18_Weights.DEFAULT if use_pretrained else None
            self.image_model = models.resnet18(weights=weights)
            self.image_features = nn.Sequential(*list(self.image_model.children())[:-1])
            image_feature_dim = 512  # ResNet18 feature dimension

        elif model_type == 'squeezenet1_1':
            weights = SqueezeNet1_1_Weights.DEFAULT if use_pretrained else None
            self.image_model = models.squeezenet1_1(weights=weights)
            self.image_features = nn.Sequential(*list(self.image_model.children())[:-1])
            image_feature_dim = 512  # SqueezeNet feature dimension

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Video metrics feature extractor
        self.metrics_encoder = nn.Sequential(
            nn.Linear(metrics_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        metrics_feature_dim = 128

        # Combined classifier
        combined_dim = image_feature_dim + metrics_feature_dim
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, metrics):
        # Extract image features
        image_feats = self.image_features(image)
        image_feats = image_feats.mean([2, 3])  # Global average pooling

        # Extract metrics features
        metrics_feats = self.metrics_encoder(metrics)

        # Combine features
        combined = torch.cat((image_feats, metrics_feats), dim=1)

        # Classify
        output = self.classifier(combined)
        return output

def load_scene_classifier_model(model_path, device='cpu', logging_enabled=True):
    """
    Load the trained scene classifier model

    Args:
        model_path: Path to the saved model (.pth file)
        device: Device to load the model on ('cpu' or 'cuda')
        logging_enabled (bool): If True, print loading info.

    Returns:
        Loaded model, metrics scaler, and available metrics
    """
    # Load checkpoint, explicitly allowing non-weight objects
    checkpoint = torch.load(
        model_path,
        map_location=torch.device(device),
        weights_only=False
    )
    if 'model_state_dict' not in checkpoint:
        raise ValueError(f"Invalid model checkpoint: {model_path}")

    # Get available metrics and class mapping (scaler is removed)
    available_metrics = checkpoint.get('available_metrics', VIDEO_METRICS)
    class_mapping = checkpoint.get('class_mapping', CLASS_MAPPING)

    if logging_enabled:
        print(f"Loaded scene classifier with {len(available_metrics)} metrics")

    # Create a new model with the same architecture
    model = CombinedModel(
        num_classes=len(class_mapping) if class_mapping else 6,
        model_type='mobilenet_v3_small',
        use_pretrained=False,
        metrics_dim=len(available_metrics)
    )

    # Load the state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return checkpoint['model_state_dict'], available_metrics, class_mapping

def extract_frames_from_scene(video_path, start_time, end_time, num_frames=3, output_dir=None):
    """
    Extract multiple frames in parallel using FFmpeg

    Args:
        video_path: Path to video file
        start_time: Start time of scene in seconds
        end_time: End time of scene in seconds
        num_frames: Number of frames to extract
        output_dir: Directory to save frames

    Returns:
        List of paths to extracted frames
    """
    import subprocess
    import tempfile
    import concurrent.futures

    # Create temporary directory if output_dir is not provided
    temp_dir = None
    if output_dir is None:
        temp_dir = tempfile.mkdtemp()
        output_dir = temp_dir
    else:
        os.makedirs(output_dir, exist_ok=True)

    scene_duration = end_time - start_time

    # Calculate frame intervals
    if num_frames > 1:
        intervals = np.linspace(start_time, end_time - 0.1, num_frames)
    else:
        intervals = [start_time + scene_duration / 2]  # Middle frame

    def extract_single_frame(index, time_pos):
        output_frame = os.path.join(output_dir, f"scene_frame_{index}.jpg")

        cmd = [
            'ffmpeg', '-y',
            '-ss', f"{time_pos:.3f}",
            '-i', video_path,
            '-frames:v', '1',
            '-q:v', '2', # Good quality JPEG
            '-hide_banner', '-loglevel', 'error', # Reduce FFmpeg noise
            output_frame
        ]

        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

            if os.path.exists(output_frame) and os.path.getsize(output_frame) > 0:
                return output_frame
        except subprocess.CalledProcessError as e:
            # Try with a slightly different timestamp as fallback
            fallback_time = max(start_time, min(time_pos + 0.1, end_time - 0.1))
            cmd[2] = f"{fallback_time:.3f}"

            try:
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

                if os.path.exists(output_frame) and os.path.getsize(output_frame) > 0:
                    return output_frame
            except subprocess.CalledProcessError as e_fallback:
                 print(f"Warning: Failed to extract frame at {time_pos:.3f} (and fallback {fallback_time:.3f}) for {video_path}. Error: {e_fallback.stderr.decode()}")
                 pass # Continue even if a frame fails
            except Exception as e_fallback_other:
                 print(f"Warning: Unexpected error during fallback frame extraction for {video_path}: {e_fallback_other}")
                 pass
        except Exception as e_other:
             print(f"Warning: Unexpected error during frame extraction for {video_path}: {e_other}")
             pass

        return None

    # Extract frames in parallel
    frame_paths = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_frames, 8)) as executor:
        futures = []
        for i, time_pos in enumerate(intervals):
            futures.append(executor.submit(extract_single_frame, i, time_pos))

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                frame_paths.append(result)

    return sorted(frame_paths)  # Sort to maintain frame order

def classify_scene_with_model(frame_paths, video_features, scene_classifier, metrics_scaler, available_metrics, device='cpu', logging_enabled=True):
    """
    Classify a scene using multiple frames and the trained classifier

    Args:
        frame_paths: List of paths to frames from the scene
        video_features: Dictionary of video features
        scene_classifier: Trained scene classifier model
        metrics_scaler: Scaler for metrics features
        available_metrics: List of available metrics for the model
        device: Device to run inference on ('cpu' or 'cuda')
        logging_enabled (bool): If True, print classification details.

    Returns:
        tuple: (classification_label, detailed_results)
        classification_label (str): Most common prediction across frames
        detailed_results (dict): Contains confidence scores and probabilities for all classes
    """
    # Image transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Extract metrics and scale them
    metrics_values = []
    for metric in available_metrics:
        if metric in video_features:
            metrics_values.append(video_features[metric])
        else:
            # Default to 0 if metric not available
            if logging_enabled:
                print(f"Warning: Metric '{metric}' not found in video features")
            metrics_values.append(0.0)

    metrics_array = np.array([metrics_values])
    # --- Use the passed metrics_scaler (expecting DataFrame input) ---
    scaled_metrics = metrics_array # Default to unscaled if scaler fails
    # Scale metrics if scaler is available
    if metrics_scaler:
        metrics_df = pd.DataFrame(metrics_array, columns=available_metrics)
        try:
            scaled_metrics_df = metrics_scaler.transform(metrics_df)
            scaled_metrics = scaled_metrics_df[available_metrics].values
        except Exception as e:
             print(f"Error applying metrics scaler: {e}. Using unscaled metrics.")
    else:
        if logging_enabled: print("Warning: No metrics scaler provided. Using unscaled metrics.")

    metrics_tensor = torch.FloatTensor(scaled_metrics).to(device)

    # Process each frame and collect predictions
    predictions = []
    confidences = []
    all_probabilities = []

    for frame_path in frame_paths:
        try:
            # Load and transform image
            image = Image.open(frame_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)

            # Make prediction
            with torch.no_grad():
                outputs = scene_classifier(image_tensor, metrics_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, prediction = torch.max(probabilities, 1)

                # Convert to numpy
                pred_class = prediction.item()
                pred_confidence = confidence.item()
                full_probs = probabilities.cpu().numpy()[0]

                predictions.append(pred_class)
                confidences.append(pred_confidence)
                all_probabilities.append(full_probs)
        except Exception as e:
            print(f"Error classifying frame {frame_path}: {e}") # Keep error prints

    # If no valid predictions, return "unclear"
    if not predictions:
        return "unclear", {
            'confidence_score': 0.0,
            'prob_screen_content': 0.0,
            'prob_animation': 0.0,
            'prob_faces': 0.0,
            'prob_gaming': 0.0,
            'prob_other': 0.0,
            'prob_unclear': 1.0,
            'frame_predictions': []
        }
    
    # Calculate weighted average probabilities across all frames
    avg_probabilities = np.mean(all_probabilities, axis=0)
    # Majority vote (with confidence weighting)
    prediction_counts = {}
    for pred, conf in zip(predictions, confidences):
        if pred not in prediction_counts:
            prediction_counts[pred] = 0
        prediction_counts[pred] += conf
    
    # Get majority class and its confidence
    majority_class = max(prediction_counts, key=prediction_counts.get)
    majority_confidence = prediction_counts[majority_class] / len(predictions)  # Normalize by number of frames

    # Convert to label string
    label = INV_CLASS_MAPPING.get(majority_class, "unclear")

    #Create detailed results with individual class probabilities
    detailed_results = {
        'confidence_score': majority_confidence,
        'prob_screen_content': float(avg_probabilities[CLASS_MAPPING.get('Screen Content / Text', 0)]),
        'prob_animation': float(avg_probabilities[CLASS_MAPPING.get('Animation / Cartoon / Rendered Graphics', 1)]),
        'prob_faces': float(avg_probabilities[CLASS_MAPPING.get('Faces / People', 2)]),
        'prob_gaming': float(avg_probabilities[CLASS_MAPPING.get('Gaming Content', 3)]),
        'prob_other': float(avg_probabilities[CLASS_MAPPING.get('other', 4)]),
        'prob_unclear': float(avg_probabilities[CLASS_MAPPING.get('unclear', 5)]),
        'frame_predictions': [INV_CLASS_MAPPING.get(p, "unclear") for p in predictions],
        'prediction_counts': {INV_CLASS_MAPPING.get(k, "unclear"): v for k, v in prediction_counts.items()}
    }

    # Log the results if enabled
    if logging_enabled:
        frame_preds = [INV_CLASS_MAPPING.get(p, "?") for p in predictions]
        print(f"Frame predictions: {frame_preds}")
        print(f"Final classification: {label} (confidence: {prediction_counts[majority_class]:.4f})")

    return label,detailed_results


