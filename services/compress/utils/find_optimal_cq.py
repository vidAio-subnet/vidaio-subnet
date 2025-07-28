"""
Optimal CQ (Constant Quality) Parameter Optimization using Machine Learning

This module provides intelligent video encoding parameter optimization through PyTorch-based
neural networks that predict VMAF (Video Multi-Method Assessment Fusion) quality scores.
The system uses binary search algorithms to find optimal CQ values that achieve target
quality metrics while minimizing file size.

Key Features:
- PyTorch neural network models for VMAF prediction
- Binary search optimization for CQ parameter selection
- Multiple model architectures (basic and improved with residual blocks)
- Comprehensive preprocessing pipeline integration
- Feature matching and validation
- Global model caching for performance
- Extensive error handling and fallback strategies

Model Architectures:
1. VMAFPredictionModel: Basic feedforward network with dropout
2. ImprovedVMAFModel: Advanced architecture with residual blocks and batch normalization

Usage:
    from find_optimal_cq import find_optimal_cq
    from analyze_video_fast import analyze_video_fast
    
    # Extract video features
    features = analyze_video_fast('video.mp4')
    
    # Find optimal CQ for target VMAF of 90
    optimal_cq = find_optimal_cq(features, target_vmaf_original=90.0)

Dependencies:
    - PyTorch for neural network inference
    - scikit-learn for preprocessing pipelines
    - pandas for data manipulation
    - numpy for numerical operations

Author: VIDAIO Development Team
Version: 2.1.0
Last Updated: 2025-05-30
"""

# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================
import os           # Operating system interface for file operations
import logging      # Logging facility for Python
import sys          # System-specific parameters and functions
import pickle       # Python object serialization for loading saved models
import traceback    # Detailed error printing and debugging

# =============================================================================
# THIRD-PARTY LIBRARY IMPORTS
# =============================================================================
# Data manipulation and analysis libraries
import pandas as pd             # Data structures and data analysis tools
import numpy as np              # Fundamental package for scientific computing
import matplotlib.pyplot as plt # Plotting library for data visualization
import seaborn as sns           # Statistical data visualization

# Machine learning and preprocessing libraries
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler    # Feature scaling
from sklearn.utils.validation import check_is_fitted # Pipeline validation
from sklearn.exceptions import NotFittedError       # Pipeline state checking

# PyTorch deep learning framework
import torch                    # Core PyTorch library
import torch.nn as nn          # Neural network modules
import torch.nn.functional as F # Neural network functions
from pathlib import Path       # Object-oriented filesystem paths
# =============================================================================
# PATH CONFIGURATION FOR MODULE IMPORTS
# =============================================================================
# Configure Python path to allow importing custom modules from the src directory
# This enables the system to find and import custom transformer classes and utilities

# Get current directory (should be src/ when running from main project)
current_dir = Path(__file__).parent
# Construct path to utilities directory
utils_dir = os.path.join(current_dir, 'utils')

# Add source directory to Python path for module imports
# This allows importing custom preprocessing transformers and utilities
if str(utils_dir) not in sys.path:
    sys.path.insert(0, str(utils_dir))
from encoder_configs import CODEC_CQ_LIMITS
# =============================================================================
# GLOBAL CACHE VARIABLES FOR PERFORMANCE OPTIMIZATION
# =============================================================================
# These global variables cache loaded models and feature names to avoid
# repeated file I/O operations during batch processing

# Cache for the loaded PyTorch VMAF prediction model
# Set to None initially, loaded on first use, then reused
_PT_MODEL = None

# Cache for the list of expected feature names for the model
# Loaded from model checkpoint or external file, cached for reuse
_FEATURE_NAMES = None

# =============================================================================
# PYTORCH NEURAL NETWORK MODEL DEFINITIONS
# =============================================================================
# Define neural network architectures for VMAF prediction
# These models must match the architectures used during training

class VMAFPredictionModel(nn.Module):
    """
    Basic feedforward neural network for VMAF quality prediction.
    
    This model uses a simple but effective architecture with multiple fully connected
    layers, ReLU activation functions, and dropout for regularization. It's designed
    to predict VMAF scores (0-100) from video complexity features.
    
    Architecture:
        Input Layer → 64 neurons (ReLU, 30% dropout)
        Hidden Layer → 32 neurons (ReLU, 20% dropout)  
        Hidden Layer → 16 neurons (ReLU)
        Output Layer → 1 neuron (Sigmoid activation)
    
    The sigmoid output produces values in range [0,1] which are then scaled
    to VMAF range [0,100] or custom ranges based on training data.
    
    Attributes:
        feature_names (list): Expected input feature names for validation
        model (nn.Sequential): The neural network layers
    """
    
    def __init__(self, input_dim, feature_names=None):
        """
        Initialize the VMAF prediction model.
        
        Args:
            input_dim (int): Number of input features (e.g., 12 for standard feature set)
            feature_names (list, optional): Names of expected input features
        """
        super(VMAFPredictionModel, self).__init__()
        
        # Store feature names for input validation during inference
        self.feature_names = feature_names
        
        # Define the neural network architecture
        # Sequential model allows easy layer-by-layer definition
        self.model = nn.Sequential(
            # First hidden layer: input_dim → 64 neurons
            nn.Linear(input_dim, 64),    # Fully connected layer
            nn.ReLU(),                   # ReLU activation for non-linearity
            nn.Dropout(0.3),             # 30% dropout for regularization
            
            # Second hidden layer: 64 → 32 neurons
            nn.Linear(64, 32),           # Fully connected layer
            nn.ReLU(),                   # ReLU activation
            nn.Dropout(0.2),             # 20% dropout (less aggressive)
            
            # Third hidden layer: 32 → 16 neurons
            nn.Linear(32, 16),           # Fully connected layer
            nn.ReLU(),                   # ReLU activation (no dropout on final hidden)
            
            # Output layer: 16 → 1 neuron
            nn.Linear(16, 1),            # Single output for VMAF prediction
            nn.Sigmoid()                 # Sigmoid to constrain output to [0,1]
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: VMAF prediction in range [0,1], shape (batch_size, 1)
        """
        return self.model(x)


class ResidualBlock(nn.Module):
    """
    Residual block with skip connections for improved gradient flow.
    
    This block implements the residual connection concept where the input
    is added to the output of the transformation layers. This helps with
    training deeper networks by mitigating the vanishing gradient problem.
    
    Architecture:
        Input → Linear → BatchNorm → ReLU → Dropout → Linear → BatchNorm → Add Input → ReLU
                 ↓                                                          ↑
                 └─────────────── Skip Connection ──────────────────────────┘
    
    The skip connection allows gradients to flow directly through the block,
    making it easier to train deeper networks effectively.
    """
    
    def __init__(self, in_features, out_features):
        """
        Initialize the residual block.
        
        Args:
            in_features (int): Number of input features
            out_features (int): Number of output features
        """
        super(ResidualBlock, self).__init__()
        
        # First transformation layer
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)  # Batch normalization for training stability
        
        # Second transformation layer  
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
        # Skip connection handling
        # If input and output dimensions differ, we need a projection layer
        self.shortcut = nn.Identity()  # Default: direct connection
        if in_features != out_features:
            # Projection layer to match dimensions
            self.shortcut = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        """
        Forward pass with residual connection.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor with residual connection applied
        """
        # Store input for skip connection
        identity = self.shortcut(x)
        
        # First transformation: Linear → BatchNorm → ReLU → Dropout
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        
        # Second transformation: Linear → BatchNorm
        out = self.bn2(self.fc2(out))
        
        # Add skip connection (residual)
        out += identity
        
        # Final activation
        out = F.relu(out)
        
        return out


class ImprovedVMAFModel(nn.Module):
    """
    Advanced VMAF prediction model with residual connections and batch normalization.
    
    This improved architecture incorporates modern deep learning techniques:
    - Residual blocks for better gradient flow
    - Batch normalization for training stability
    - Deeper architecture for increased model capacity
    - Skip connections to enable training of deeper networks
    
    Architecture:
        Input → Linear(64) → BatchNorm → ReLU
              → ResidualBlock(64→64) 
              → ResidualBlock(64→32)
              → Linear(32→16) → ReLU
              → Linear(16→1) → Sigmoid
    
    This architecture can capture more complex relationships between video
    features and VMAF scores compared to the basic model.
    """
    
    def __init__(self, input_dim, feature_names=None):
        """
        Initialize the improved VMAF prediction model.
        
        Args:
            input_dim (int): Number of input features
            feature_names (list, optional): Names of expected input features
        """
        super(ImprovedVMAFModel, self).__init__()
        
        # Store feature names for validation
        self.feature_names = feature_names
        
        # Input processing layer
        self.input_layer = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        
        # Residual blocks for deep feature learning
        self.res_block1 = ResidualBlock(64, 64)  # Same dimension residual block
        self.res_block2 = ResidualBlock(64, 32)  # Dimension reduction residual block
        
        # Final prediction layers
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)
        
        # Output activation
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass through the improved model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: VMAF prediction in range [0,1]
        """
        # Input processing with batch normalization
        x = F.relu(self.bn1(self.input_layer(x)))
        
        # Pass through residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        # Final prediction layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Apply sigmoid activation for final output
        x = self.activation(x)
        
        return x


# =============================================================================
# MODEL LOADING AND CACHING FUNCTIONS
# =============================================================================

def load_pytorch_model_if_needed(model_full_path=None, logging_enabled=True): # Changed parameter
    """
    Load PyTorch VMAF prediction model with global caching and automatic architecture detection.
    
    Args:
        model_full_path (str): Full, absolute path to the model file.
        logging_enabled (bool): Whether to print loading progress and errors
    """
    global _PT_MODEL
    
    if _PT_MODEL is None:
        try:
            # Use the provided full model path directly
            if not model_full_path:
                if logging_enabled:
                    print("❌ ERROR: No model path provided to load_pytorch_model_if_needed.")
                return None

            # Verify model file exists before attempting to load
            if not os.path.exists(model_full_path):
                if logging_enabled:
                    print(f"❌ ERROR: Model file not found at {model_full_path}")
                    print(f"       Please ensure the model file exists at the specified path.")
                return None

            if logging_enabled:
                print(f"🧠 Loading PyTorch model from: {model_full_path}") # Log the full path
            
            checkpoint = torch.load(model_full_path, map_location=torch.device('cpu'))
            
            # Extract model components from checkpoint
            state_dict = checkpoint['model_state_dict']      # Trained weights
            input_dim = checkpoint.get('input_dim')          # Model input dimension
            feature_names = checkpoint.get('feature_names')  # Expected feature names

            # Determine input dimension from feature names if not explicitly saved
            if input_dim is None and feature_names:
                input_dim = len(feature_names)
                if logging_enabled:
                    print(f"Determined input dimension from feature names: {input_dim}")

            # Validate that we can determine the input dimension
            if input_dim is None:
                if logging_enabled:
                    print("ERROR: Could not determine model input dimension from checkpoint.")
                    print("       Checkpoint must contain either 'input_dim' or 'feature_names'")
                return None

            # =================================================================
            # AUTOMATIC ARCHITECTURE DETECTION
            # =================================================================
            # Examine state dictionary keys to determine which model architecture was used
            # Different architectures have distinct layer naming patterns
            
            # Check for keys specific to ImprovedVMAFModel
            improved_model_indicators = [
                'input_layer',    # Specific to improved model
                'res_block',      # Residual blocks only in improved model
                'bn1'            # Batch normalization layers
            ]
            
            is_improved_model = any(
                any(indicator in key for indicator in improved_model_indicators)
                for key in state_dict.keys()
            )
            
            # Initialize the appropriate model architecture
            if is_improved_model:
                model = ImprovedVMAFModel(input_dim=input_dim, feature_names=feature_names)
                if logging_enabled:
                    print("Detected and loading ImprovedVMAFModel architecture")
            else:
                model = VMAFPredictionModel(input_dim=input_dim, feature_names=feature_names)
                if logging_enabled:
                    print("Detected and loading VMAFPredictionModel architecture")

            # Load the trained weights into the model
            model.load_state_dict(state_dict)
            
            # Set model to evaluation mode (disables dropout, sets batch norm to eval mode)
            model.eval()
            
            # Cache the loaded model globally for future use
            _PT_MODEL = model
            
            if logging_enabled:
                print(f"✅ PyTorch model loaded successfully")
                print(f"   📊 Input dimension: {input_dim}")
                print(f"   🏷️ Feature names: {len(feature_names) if feature_names else 'Not available'}")
                print(f"   🏗️ Architecture: {'Improved' if is_improved_model else 'Basic'}")
                
        except Exception as e:
            if logging_enabled:
                print(f"❌ ERROR: Failed to load PyTorch model from {model_full_path if model_full_path else 'unknown path'}: {e}")
                print("   🔍 Detailed error information:")
                traceback.print_exc()
            return None
    
    return _PT_MODEL


def load_feature_names_if_needed(feature_filename='feature_names.txt', logging_enabled=True):
    """
    Load expected feature names with multiple fallback strategies and global caching.
    
    This function implements a robust feature name loading system with multiple
    fallback strategies to ensure the model receives inputs in the correct order:
    
    1. First, try to get feature names from already-loaded model checkpoint
    2. If that fails, try to load from external feature_names.txt file
    3. Cache the result globally to avoid repeated file operations
    
    Feature names are critical for ensuring that video features are passed to
    the model in the exact order expected during training. Incorrect feature
    order can lead to poor prediction accuracy.
    
    Args:
        feature_filename (str): Name of the feature names file in src/model/ directory
        logging_enabled (bool): Whether to print loading progress and warnings
        
    Returns:
        list: List of feature name strings in the order expected by the model,
              or None if loading failed from all sources
              
    Global State:
        Updates _FEATURE_NAMES cache variable for subsequent calls
        
    Example:
        feature_names = load_feature_names_if_needed('feature_names.txt')
        if feature_names:
            print(f"Model expects these features: {feature_names}")
    """
    global _FEATURE_NAMES
    
    # Check if feature names are already loaded and cached
    if _FEATURE_NAMES is None:
        # =================================================================
        # STRATEGY 1: EXTRACT FROM LOADED MODEL CHECKPOINT
        # =================================================================
        # If a model is already loaded, try to get feature names from it
        # This is the most reliable source since it's saved with the model
        if _PT_MODEL and hasattr(_PT_MODEL, 'feature_names') and _PT_MODEL.feature_names:
            _FEATURE_NAMES = _PT_MODEL.feature_names
            if logging_enabled:
                print(f"✅ Loaded {len(_FEATURE_NAMES)} feature names from model checkpoint")
                print(f"   📋 Features: {', '.join(_FEATURE_NAMES[:3])}{'...' if len(_FEATURE_NAMES) > 3 else ''}")
            return _FEATURE_NAMES

        # =================================================================
        # STRATEGY 2: LOAD FROM EXTERNAL FEATURE NAMES FILE
        # =================================================================
        # Fallback to loading from separate feature_names.txt file
        try:
            # Construct path to feature names file
            cwd = os.getcwd()
            path = os.path.join(cwd, 'src', 'models')
            if not feature_filename:
                if logging_enabled:
                    print("⚠️ Warning: feature_filename is None.")
                return None
            feature_path = os.path.join(path, feature_filename)

            if os.path.exists(feature_path):
                if logging_enabled:
                    print(f"📖 Loading feature names from file: {feature_path}")
                
                # Read feature names from file (one per line)
                with open(feature_path, 'r', encoding='utf-8') as f:
                    _FEATURE_NAMES = [line.strip() for line in f if line.strip()]
                
                if logging_enabled:
                    print(f"✅ Loaded {len(_FEATURE_NAMES)} feature names from file")
                    print(f"   📋 Features: {', '.join(_FEATURE_NAMES[:3])}{'...' if len(_FEATURE_NAMES) > 3 else ''}")
            else:
                if logging_enabled:
                    print(f"⚠️ Warning: Feature names file not found at {feature_path}")
                    print("   🔍 Model might require specific feature order for accurate predictions")
                _FEATURE_NAMES = None
                return None

        except Exception as e:
            if logging_enabled:
                print(f"❌ ERROR: Failed to load feature names from file: {e}")
                print("   💡 This may affect model prediction accuracy if feature order is incorrect")
            _FEATURE_NAMES = None
            return None

    return _FEATURE_NAMES


# =============================================================================
# PREPROCESSING PIPELINE LOADING AND CONFIGURATION
# =============================================================================

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

def load_scalers_if_needed(logging_enabled=True):
    """
    Fallback function to load preprocessing pipeline and scalers when not provided.
    
    This function serves as a fallback mechanism when the main processing pipeline
    doesn't have access to pre-loaded preprocessing components. It attempts to
    load the complete pipeline from the standard location and extract components.
    
    Args:
        logging_enabled (bool): Whether to show loading progress and errors
        
    Returns:
        tuple: (pipeline_obj, feature_scaler_step, vmaf_scaler, cq_min, cq_max)
               All components or None values if loading fails
    """
    # Construct standard pipeline path
    cwd = os.getcwd()
    path = os.path.join(cwd, 'src', 'models')
    pipeline_path = os.path.join(path, 'preprocessing_pipeline.pkl')
    
    if logging_enabled:
        print("🔄 Attempting fallback loading of pipeline and scalers...")
        print(f"   📁 Loading from: {pipeline_path}")
    
    # Use main pipeline loading function with logging control
    pipeline_obj, feature_scaler_step, vmaf_scaler, cq_min, cq_max = get_scalers_from_pipeline(
        pipeline_path, verbose=True, logging_enabled=logging_enabled
    )

    # Validate that critical components were loaded
    if pipeline_obj is None or feature_scaler_step is None or vmaf_scaler is None:
        if logging_enabled:
            print("❌ Fallback loading failed - missing critical components")
        return None, None, None, None, None

    if logging_enabled:
        print("✅ Pipeline and scalers loaded successfully via fallback")
    return pipeline_obj, feature_scaler_step, vmaf_scaler, cq_min, cq_max


# =============================================================================
# VMAF PREDICTION FUNCTION CREATION
# =============================================================================

def create_pytorch_vmaf_prediction_function(model, vmaf_scaler=None, logging_enabled=True):
    """
    Create a specialized VMAF prediction function with comprehensive error handling.
    
    This function creates a closure that encapsulates a PyTorch model for VMAF prediction
    with all the necessary preprocessing, validation, and error handling. The returned
    function can be used directly by the CQ optimization algorithm.
    
    Key Features:
    - Feature order validation against model expectations
    - Data type conversion and tensor creation
    - Error handling with graceful fallbacks
    - VMAF scaling and inverse transformation
    - Comprehensive logging for debugging
    
    Args:
        model (torch.nn.Module): Trained PyTorch model for VMAF prediction
        vmaf_scaler (object): Scaler for converting VMAF values to/from model range
        logging_enabled (bool): Whether to log prediction details and errors
        
    Returns:
        function: Prediction function with signature predict_vmaf(video_features_df, cq_value_scaled)
                 Returns dict with 'vmaf_scaled' and 'vmaf_original' keys
                 Returns None if model is invalid
    """
    if model is None:
        if logging_enabled:
            print("❌ ERROR: Cannot create prediction function without a valid model")
        return None

    def predict_vmaf(video_features_df, cq_value_scaled=None):
        """
        Predict VMAF value for video features with a specific CQ value.
        
        This is the actual prediction function returned by the factory function.
        It handles all the complexity of feature validation, data type conversion,
        model inference, and result scaling.
        
        Args:
            video_features_df (pd.DataFrame): Video features (already scaled by pipeline)
            cq_value_scaled (float, optional): CQ value to use for prediction (0-1 range)
                                             If None, expects 'cq' column in dataframe
            
        Returns:
            dict: {
                'vmaf_scaled': float,    # VMAF prediction in model's output range (0-1)
                'vmaf_original': float   # VMAF prediction in original range (0-100 or custom)
            }
            
        Raises:
            Exception: Re-raises any prediction errors for debugging
        """
        try:
            # =============================================================
            # INPUT PREPARATION AND VALIDATION
            # =============================================================
            # Create a copy to avoid modifying the original dataframe
            features_with_cq = video_features_df.copy()

            # Validate that CQ column exists or can be set
            if 'cq' not in features_with_cq.columns:
                raise KeyError("Required column 'cq' not found in input features DataFrame for prediction")
            
            # =============================================================
            # CQ COLUMN IDENTIFICATION AND SETTING
            # =============================================================
            # Handle various naming conventions for CQ column
            cq_col_name = 'cq'
            potential_cq_cols = [col for col in features_with_cq.columns if 'cq' in col.lower()]
            
            if len(potential_cq_cols) == 1:
                cq_col_name = potential_cq_cols[0]
            elif 'cq' in features_with_cq.columns:
                cq_col_name = 'cq'
            else:
                if logging_enabled:
                    print(f"⚠️ Warning: Could not definitively identify CQ column in pipeline output")
                    print(f"   Available columns: {list(features_with_cq.columns)}")
                    print(f"   Assuming column name: '{cq_col_name}'")

            # Set the CQ value for prediction
            if cq_value_scaled is not None:
                features_with_cq[cq_col_name] = cq_value_scaled
            elif cq_col_name not in features_with_cq.columns:
                if logging_enabled:
                    print(f"❌ ERROR: cq_value_scaled not provided and CQ column '{cq_col_name}' missing")
                return {'vmaf_scaled': 0.0, 'vmaf_original': 0.0}

            # =============================================================
            # FEATURE ORDER VALIDATION AND MATCHING
            # =============================================================
            # Ensure features are in the order expected by the model
            model_features = None
            
            # Try to get expected features from model
            if hasattr(model, 'feature_names') and model.feature_names:
                model_features = model.feature_names
            elif _FEATURE_NAMES:
                model_features = _FEATURE_NAMES

            final_features_df = features_with_cq

            # Validate and reorder features if model expectations are known
            if model_features:
                # Check for missing features
                missing_features = [f for f in model_features if f not in final_features_df.columns]
                if missing_features:
                    if logging_enabled:
                        print(f"❌ ERROR: Pipeline output missing features required by model: {missing_features}")
                        print(f"   Pipeline output columns: {list(final_features_df.columns)}")
                        print(f"   Model expected features: {model_features}")
                    return {'vmaf_scaled': 0.0, 'vmaf_original': 0.0}
                
                try:
                    # Reorder columns to match model expectations
                    final_features_df = final_features_df[model_features]
                    if logging_enabled and len(model_features) <= 10:  # Only log for reasonable number of features
                        print(f"   ✅ Features reordered to match model expectations: {model_features}")
                except KeyError as e:
                    if logging_enabled:
                        print(f"❌ ERROR: Could not select/reorder pipeline output features to match model")
                        print(f"   KeyError: {e}")
                        print(f"   Model expects: {model_features}")
                        print(f"   Pipeline output has: {list(final_features_df.columns)}")
                    return {'vmaf_scaled': 0.0, 'vmaf_original': 0.0}
            else:
                if logging_enabled:
                    print("⚠️ Warning: Cannot verify feature order against model's expected input")
                    print("   Using all pipeline output columns in current order")

            # =============================================================
            # DATA TYPE CONVERSION AND VALIDATION
            # =============================================================
            # Ensure all data is numeric and compatible with PyTorch tensors
            object_cols = final_features_df.select_dtypes(include=['object']).columns
            if not object_cols.empty:
                if logging_enabled:
                    print(f"❌ ERROR: Object dtype columns found before tensor conversion: {list(object_cols)}")
                    print("   DataFrame sample with object columns:")
                    print(final_features_df[object_cols].head())
                
                try:
                    # Attempt automatic conversion to float32
                    final_features_df = final_features_df.astype(np.float32)
                    if logging_enabled:
                        print("   🔧 Attempted automatic conversion to float32")
                except ValueError as e:
                    if logging_enabled:
                        print(f"   ❌ Automatic conversion failed: {e}")
                        print("   Cannot create tensor from non-numeric data")
                    return {'vmaf_scaled': 0.0, 'vmaf_original': 0.0}
            else:
                # Ensure consistent float32 dtype for PyTorch compatibility
                final_features_df = final_features_df.astype(np.float32)

            # =============================================================
            # TENSOR CREATION AND VALIDATION
            # =============================================================
            # Convert preprocessed features to PyTorch tensor
            try:
                features_tensor = torch.FloatTensor(final_features_df.values)
                
                # Ensure tensor has correct dimensions (add batch dimension if needed)
                if features_tensor.ndim == 1:
                    features_tensor = features_tensor.unsqueeze(0)  # Add batch dimension
                
                # Validate tensor shape
                expected_input_size = len(model_features) if model_features else final_features_df.shape[1]
                if features_tensor.shape[1] != expected_input_size:
                    if logging_enabled:
                        print(f"❌ ERROR: Tensor shape mismatch")
                        print(f"   Expected: (batch_size, {expected_input_size})")
                        print(f"   Got: {features_tensor.shape}")
                    return {'vmaf_scaled': 0.0, 'vmaf_original': 0.0}
                    
            except Exception as e:
                if logging_enabled:
                    print(f"❌ ERROR: Could not convert final features to tensor: {e}")
                    print("   Final DataFrame dtypes:")
                    print(final_features_df.dtypes)
                return {'vmaf_scaled': 0.0, 'vmaf_original': 0.0}

            # =============================================================
            # MODEL INFERENCE
            # =============================================================
            # Perform VMAF prediction using the PyTorch model
            with torch.no_grad():  # Disable gradient computation for inference
                vmaf_prediction = model(features_tensor).item()  # Get scalar value

            # Clamp prediction to valid range [0, 1] for stability
            vmaf_scaled = max(0.0, min(1.0, vmaf_prediction))

            # =============================================================
            # VMAF SCALING AND INVERSE TRANSFORMATION
            # =============================================================
            # Convert model output back to original VMAF scale
            vmaf_original = vmaf_scaled * 100.0  # Default: scale to 0-100 range
            
            if vmaf_scaler and hasattr(vmaf_scaler, 'min_val') and hasattr(vmaf_scaler, 'max_val'):
                try:
                    # Use custom scaler to inverse transform VMAF
                    vmaf_range = vmaf_scaler.max_val - vmaf_scaler.min_val
                    if vmaf_range <= 0: 
                        raise ValueError("VMAF scaler range is zero or negative")
                    
                    # Apply inverse scaling: scaled_value * range + min_value
                    vmaf_original = vmaf_scaled * vmaf_range + vmaf_scaler.min_val
                    
                    # Clamp to reasonable VMAF bounds
                    vmaf_original = max(0.0, min(100.0, vmaf_original))
                    
                except Exception as e:
                    if logging_enabled:
                        print(f"⚠️ Warning: Could not inverse transform VMAF using custom scaler: {e}")
                        print("   Falling back to simple scaling (* 100)")
                    vmaf_original = vmaf_scaled * 100.0

            return {
                'vmaf_scaled': float(vmaf_scaled),
                'vmaf_original': float(vmaf_original)
            }
            
        except Exception as e:
            if logging_enabled:
                print(f"❌ Error during VMAF prediction call: {e}")
                print("🔍 Detailed error traceback:")
                traceback.print_exc()
            raise  # Re-raise for debugging

    return predict_vmaf


# =============================================================================
# CQ OPTIMIZATION SEARCH ALGORITHM
# =============================================================================

def search_for_cq(predict_vmaf_fn, video_features_df, target_vmaf_scaled,
                  min_cq=0.0, max_cq=1.0, min_cq_original=10, max_cq_original=63,
                  tolerance=0.01, max_iterations=10, logging_enabled=True):
    """
    Find optimal CQ value using binary search to achieve target VMAF score.
    
    This function implements an intelligent binary search algorithm to find the
    optimal Constant Quality (CQ) parameter that achieves a target VMAF score.
    The algorithm includes several optimizations:
    
    1. Boundary checking: Tests min/max CQ values first for early termination
    2. Binary search: Efficiently narrows the search space
    3. Convergence detection: Stops when the search space becomes sufficiently small
    4. Result tracking: Records all tested CQ/VMAF pairs for analysis
    
    The search operates on scaled CQ values [0,1] internally but reports
    results in both scaled and original CQ ranges for convenience.
    
    Args:
        predict_vmaf_fn (function): VMAF prediction function that takes (features_df, cq_scaled)
        video_features_df (pd.DataFrame): Video features for prediction (preprocessed)
        target_vmaf_scaled (float): Target VMAF score in scaled range [0,1]
        min_cq (float): Minimum CQ value in scaled range [0,1]
        max_cq (float): Maximum CQ value in scaled range [0,1]
        min_cq_original (int): Minimum CQ value in original encoder range
        max_cq_original (int): Maximum CQ value in original encoder range
        tolerance (float): Search precision tolerance (smaller = more precise)
        max_iterations (int): Maximum number of search iterations
        logging_enabled (bool): Whether to log search progress
        
    Returns:
        dict: {
            'optimal_cq': float,           # Best CQ value in scaled range [0,1]
            'optimal_cq_original': float,  # Best CQ value in original range
            'predicted_vmaf': float,       # VMAF prediction at optimal CQ (scaled)
            'predicted_vmaf_original': float, # VMAF prediction at optimal CQ (original)
            'iterations': int,             # Number of search iterations performed
            'all_tested': list            # All tested (cq_scaled, cq_original, vmaf_scaled, vmaf_original) tuples
        }
        
    Algorithm Details:
        1. Test boundary values (min_cq and max_cq) first
        2. If target is outside boundaries, return appropriate boundary
        3. Otherwise, perform binary search within boundaries
        4. Continue until convergence or max iterations reached
        5. Return the CQ value that best achieves the target VMAF
    """
    
    # Validate prediction function
    if predict_vmaf_fn is None:
        if logging_enabled:
            print("❌ ERROR: predict_vmaf_fn is None. Cannot search for CQ.")
        return {
            'optimal_cq': min_cq, 
            'optimal_cq_original': min_cq_original, 
            'predicted_vmaf': None, 
            'predicted_vmaf_original': None, 
            'iterations': 0, 
            'all_tested': []
        }

    # Initialize search variables
    low, high = min_cq, max_cq           # Search bounds in scaled range
    best_cq_scaled = low                 # Best CQ found so far
    best_vmaf_pred_scaled = None         # Best VMAF prediction (scaled)
    best_vmaf_pred_original = None       # Best VMAF prediction (original)
    iterations = 0                       # Number of iterations performed
    all_results = []                     # All tested CQ/VMAF combinations

    if logging_enabled:
        print(f"🔍 Starting CQ optimization search")
        print(f"   🎯 Target VMAF (scaled): {target_vmaf_scaled:.3f}")
        print(f"   📊 CQ search range (scaled): {min_cq:.3f} - {max_cq:.3f}")
        print(f"   📊 CQ search range (original): {min_cq_original} - {max_cq_original}")

    # =================================================================
    # BOUNDARY VALUE TESTING
    # =================================================================
    # Test the minimum and maximum CQ values first to check if target
    # is achievable within the given range
    
    if logging_enabled:
        print(f"   🧪 Testing boundary values...")
    
    # Test lowest CQ (highest quality, largest file size)
    vmaf_at_low = predict_vmaf_fn(video_features_df, low)
    low_original = scaled_to_original_cq(low, min_cq_original, max_cq_original)
    all_results.append((low, low_original, vmaf_at_low['vmaf_scaled'], vmaf_at_low['vmaf_original']))
    
    # Test highest CQ (lowest quality, smallest file size)
    vmaf_at_high = predict_vmaf_fn(video_features_df, high)
    high_original = scaled_to_original_cq(high, min_cq_original, max_cq_original)
    all_results.append((high, high_original, vmaf_at_high['vmaf_scaled'], vmaf_at_high['vmaf_original']))

    if logging_enabled:
        print(f"   📈 VMAF at CQ {low_original} (lowest): {vmaf_at_low['vmaf_original']:.2f}")
        print(f"   📉 VMAF at CQ {high_original} (highest): {vmaf_at_high['vmaf_original']:.2f}")

    # =================================================================
    # BOUNDARY CONDITION HANDLING
    # =================================================================
    # Check if target VMAF is achievable within the CQ range
    
    if target_vmaf_scaled >= vmaf_at_low['vmaf_scaled']:
        # Target is higher than achievable with lowest CQ - use lowest CQ
        if logging_enabled:
            print(f"   ⬆️ Target VMAF ({target_vmaf_scaled:.3f}) >= VMAF at lowest CQ ({vmaf_at_low['vmaf_scaled']:.3f})")
            print(f"   🎯 Using lowest CQ for maximum achievable quality")
        best_cq_scaled = low
        best_vmaf_pred_scaled = vmaf_at_low['vmaf_scaled']
        best_vmaf_pred_original = vmaf_at_low['vmaf_original']
        
    elif target_vmaf_scaled <= vmaf_at_high['vmaf_scaled']:
        # Target is lower than achievable with highest CQ - use highest CQ
        if logging_enabled:
            print(f"   ⬇️ Target VMAF ({target_vmaf_scaled:.3f}) <= VMAF at highest CQ ({vmaf_at_high['vmaf_scaled']:.3f})")
            print(f"   🎯 Using highest CQ for maximum compression")
        best_cq_scaled = high
        best_vmaf_pred_scaled = vmaf_at_high['vmaf_scaled']
        best_vmaf_pred_original = vmaf_at_high['vmaf_original']
        
    else:
        # =============================================================
        # BINARY SEARCH ALGORITHM
        # =============================================================
        # Target is achievable within range - perform binary search
        if logging_enabled:
            print(f"   🔄 Target is within achievable range - starting binary search")
            print(f"   ⚙️ Search tolerance: {tolerance}, Max iterations: {max_iterations}")
        
        while iterations < max_iterations:
            iterations += 1
            
            # Calculate midpoint of current search range
            mid_scaled = (low + high) / 2.0
            
            # Predict VMAF at midpoint
            result = predict_vmaf_fn(video_features_df, mid_scaled)
            predicted_vmaf_scaled = result['vmaf_scaled']
            predicted_vmaf_original = result['vmaf_original']

            # Convert to original CQ for logging
            mid_original = scaled_to_original_cq(mid_scaled, min_cq_original, max_cq_original)
            all_results.append((mid_scaled, mid_original, predicted_vmaf_scaled, predicted_vmaf_original))

            if logging_enabled:
                print(f"   🔄 Iteration {iterations}: CQ {mid_original:.1f} → VMAF {predicted_vmaf_original:.2f}")

            # ==========================================================
            # BINARY SEARCH LOGIC
            # ==========================================================
            if predicted_vmaf_scaled >= target_vmaf_scaled:
                # VMAF is at or above target - we can increase CQ (reduce quality/size)
                best_cq_scaled = mid_scaled
                best_vmaf_pred_scaled = predicted_vmaf_scaled
                best_vmaf_pred_original = predicted_vmaf_original
                low = mid_scaled  # Search higher CQ values (lower quality)
            else:
                # VMAF is below target - we need to decrease CQ (increase quality/size)
                high = mid_scaled  # Search lower CQ values (higher quality)

            # ==========================================================
            # CONVERGENCE CHECK
            # ==========================================================
            # Stop searching when the range becomes sufficiently small
            if high - low < 0.005:  # Convergence threshold
                if logging_enabled:
                    print(f"   ✅ Converged: search range {high - low:.6f} < 0.005")
                break

    # =================================================================
    # RESULT PREPARATION AND LOGGING
    # =================================================================
    # Sort all tested results by CQ value for analysis
    all_results.sort(key=lambda x: x[0])
    
    # Convert final result to original CQ scale
    best_cq_original = scaled_to_original_cq(best_cq_scaled, min_cq_original, max_cq_original)

    if logging_enabled:
        print(f"✅ CQ search completed successfully")
        print(f"   🎯 Final CQ: {best_cq_original:.1f} (scaled: {best_cq_scaled:.4f})")
        print(f"   📊 Predicted VMAF: {best_vmaf_pred_original:.2f} (scaled: {best_vmaf_pred_scaled:.3f})")
        print(f"   🔄 Iterations used: {iterations}")

    return {
        'optimal_cq': best_cq_scaled,
        'optimal_cq_original': best_cq_original,
        'predicted_vmaf': best_vmaf_pred_scaled,
        'predicted_vmaf_original': best_vmaf_pred_original,
        'iterations': iterations,
        'all_tested': all_results
    }

def scaled_to_original_cq(scaled_cq, min_cq_original, max_cq_original):
    """
    Convert scaled CQ value [0, 1] back to original encoder range.
    
    Args:
        scaled_cq (float): CQ value in range [0, 1]
        min_cq_original (int): Minimum CQ in original encoder range
        max_cq_original (int): Maximum CQ in original encoder range
        
    Returns:
        float: CQ value in original encoder range
    """
    return scaled_cq * (max_cq_original - min_cq_original) + min_cq_original


def original_to_scaled_cq(original_cq, min_cq_original, max_cq_original):
    """
    Convert original CQ value to scaled range [0, 1].
    
    Args:
        original_cq (float): CQ value in original encoder range
        min_cq_original (int): Minimum CQ in original encoder range
        max_cq_original (int): Maximum CQ in original encoder range
        
    Returns:
        float: CQ value in range [0, 1]
    """
    if max_cq_original == min_cq_original: 
        return 0.5  # Avoid division by zero
    return (original_cq - min_cq_original) / (max_cq_original - min_cq_original)

def get_codec_cq_limits(codec_name, conservative=True, logging_enabled=True):
    """
    Get CQ limits for a specific codec with fallback to defaults.
    
    Args:
        codec_name (str): Name of the codec (e.g., 'av1_nvenc', 'h264_nvenc')
        conservative (bool): Whether to use recommended_max (True) or max_cq (False)
        logging_enabled (bool): Whether to log the selected limits
        
    Returns:
        dict: Codec limits with keys 'max_cq', 'recommended_max', 'quality_range', 'description'
    """
    # Normalize codec name (handle variations)
    codec_normalized = codec_name.lower().strip()
    
    # Check for exact match first
    if codec_normalized in CODEC_CQ_LIMITS:
        limits = CODEC_CQ_LIMITS[codec_normalized]
    else:
        # Try partial matching for codec families
        for known_codec, limits in CODEC_CQ_LIMITS.items():
            if known_codec in codec_normalized or codec_normalized in known_codec:
                if logging_enabled:
                    print(f"   🔍 Partial codec match: '{codec_name}' → '{known_codec}'")
                limits = CODEC_CQ_LIMITS[known_codec]
                break
        else:
            # No match found, use default
            if logging_enabled:
                print(f"   ⚠️ Unknown codec '{codec_name}', using default limits")
            limits = CODEC_CQ_LIMITS['default']
    
    # Choose between conservative and aggressive limits
    effective_max_cq = limits['recommended_max'] if conservative else limits['max_cq']
    
    if logging_enabled:
        print(f"   🎯 Codec: {limits['description']}")
        print(f"   📊 CQ limits: range {limits['quality_range']}, max {effective_max_cq} ({'conservative' if conservative else 'aggressive'})")
    
    return {
        **limits,
        'effective_max_cq': effective_max_cq
    }
# =============================================================================
# MAIN CQ OPTIMIZATION FUNCTION
# =============================================================================
def find_optimal_cq(video_features, target_vmaf_original: float = 85.0,
                    pipeline_obj=None, vmaf_scaler=None, cq_min=None, cq_max=None,
                    required_feature_names=None, 
                    vmaf_prediction_model_path: str = None,
                    codec_name: str = None, conservative_cq_limits: bool = True,
                    logging_enabled=True,logger=None):
    """
    Find the optimal CQ (Constant Quality) value to achieve target VMAF using PyTorch ML models.
    
    This is the main entry point for CQ optimization in the AI video compression pipeline.
    It orchestrates the complete optimization workflow: component loading, feature preprocessing,
    model inference, and binary search optimization to find the CQ value that best achieves
    the target VMAF quality score.
    
    The function implements a robust optimization system with multiple fallback strategies,
    comprehensive error handling, and detailed logging for production use.
    
    WORKFLOW OVERVIEW:
    ==================
    1. Component Validation & Fallback Loading
       - Validates provided preprocessing components
       - Loads missing components using fallback mechanisms
       - Ensures all required scalers and bounds are available
    
    2. Model & Feature Name Loading
       - Loads PyTorch VMAF prediction model with caching
       - Determines expected feature names for input validation
       - Handles multiple model architectures automatically
    
    3. Input Data Preparation & Validation
       - Converts video features dictionary to DataFrame
       - Adds required CQ column for optimization
       - Validates and reorders features to match model expectations
       - Handles missing features with default values
    
    4. Preprocessing Pipeline Application
       - Applies complete scikit-learn preprocessing pipeline
       - Normalizes and scales all video features
       - Transforms data to model-expected format
    
    5. Target VMAF Scaling
       - Converts target VMAF from original range (0-100) to model range (0-1)
       - Uses custom VMAF scaler for proper normalization
       - Validates scaled target is within achievable bounds
    
    6. Binary Search Optimization
       - Creates specialized prediction function for the model
       - Performs intelligent binary search to find optimal CQ
       - Uses convergence detection and iteration limits
    
    7. Result Processing & Validation
       - Converts optimal CQ back to original encoder range
       - Clamps result to valid CQ bounds
       - Returns integer CQ value suitable for encoding
    
    PARAMETERS:
    ===========
    video_features (dict): Video feature dictionary extracted from analyze_video_fast()
        Expected keys include:
        - 'metrics_avg_motion': Motion complexity (0-1)
        - 'metrics_avg_edge_density': Edge detail level (0-1)  
        - 'metrics_avg_texture': Texture complexity
        - 'metrics_avg_temporal_information': Frame-to-frame changes
        - 'metrics_avg_spatial_information': Spatial detail
        - 'metrics_avg_color_complexity': Color variation
        - 'metrics_avg_motion_variance': Motion consistency
        - 'metrics_avg_grain_noise': Noise level
        - 'metrics_resolution_width': Video width in pixels
        - 'metrics_resolution_height': Video height in pixels
        - 'metrics_frame_rate': Frame rate in FPS
        - Additional codec and bitrate metrics
        
    target_vmaf_original (float, default=85): Target VMAF quality score in range [0-100]
        - Higher values = better quality, larger file size
        - Lower values = lower quality, smaller file size
        - Typical ranges: 80-95 for good quality, 95+ for premium quality
        
    pipeline_obj (sklearn.Pipeline, optional): Preprocessing pipeline for feature scaling
        - If None, loads from 'src/models/preprocessing_pipeline.pkl'
        - Should be fitted pipeline containing feature_scaler, vmaf_scaler, cq_scaler
        - Used to normalize video features to model-expected ranges
        
    vmaf_scaler (object, optional): VMAF scaling transformer
        - If None, extracted from pipeline_obj or created as default
        - Must have 'min_val' and 'max_val' attributes for range conversion
        - Used to convert VMAF scores between original and model ranges
        
    cq_min, cq_max (int, optional): CQ parameter bounds for optimization
        - If None, extracted from pipeline_obj or use defaults (10, 51)
        - Defines the search space for CQ optimization
        - Should match encoder's valid CQ range (e.g., 10-51 for AV1)
        
    required_feature_names (list, optional): Expected feature names for model input
        - If None, loads from model checkpoint or feature_names.txt
        - Ensures features are passed to model in correct order
        - Critical for model prediction accuracy
        
    vmaf_prediction_model_path (str, optional): Full path to the VMAF prediction model.
                                                    If None, attempts to load from a default.
        
    logging_enabled (bool, default=True): Whether to log detailed progress
        - True: Comprehensive logging for debugging and monitoring
        - False: Minimal output for batch processing
        
    RETURNS:
    ========
    int: Optimal CQ value in original encoder range, or -1 if optimization failed
        - Returns integer CQ value ready for use with video encoder
        - Value is clamped to [cq_min, cq_max] range for safety
        - Returns -1 on critical failures (model loading, pipeline errors, etc.)
        
    ERROR HANDLING:
    ===============
    The function implements comprehensive error handling with graceful degradation:
    
    1. Component Loading Errors:
       - Missing pipeline files → Fallback loading from default locations
       - Missing model files → Error return with detailed diagnostics
       - Missing feature names → Use default feature set or model-embedded names
       
    2. Data Processing Errors:
       - Invalid video features → Error return with feature validation details
       - Pipeline transformation errors → Detailed debugging information
       - Feature mismatch → Automatic default value insertion where possible
       
    3. Model Inference Errors:
       - Model loading failures → Multiple fallback strategies and clear error messages
       - Prediction errors → Graceful handling with diagnostic information
       - Architecture mismatch → Automatic model type detection
       
    4. Optimization Errors:
       - Search convergence issues → Boundary value returns
       - Invalid target VMAF → Automatic clamping to achievable range
       - CQ bound violations → Result clamping with warnings
    
    PERFORMANCE CHARACTERISTICS:
    ============================
    - Model Loading: ~0.5-2.0 seconds (first call), ~0.01 seconds (cached)
    - Feature Processing: ~0.1-0.3 seconds per video
    - CQ Optimization: ~0.2-0.8 seconds (typically 5-10 search iterations)
    - Memory Usage: ~50-100MB for loaded models
    - Thread Safety: Read-only operations after model loading
    
    USAGE EXAMPLES:
    ===============
    
    Basic Usage:
    ------------
    from analyze_video_fast import analyze_video_fast
    from find_optimal_cq import find_optimal_cq
    
    # Extract video features
    features = analyze_video_fast('video.mp4', max_frames=150)
    
    # Find optimal CQ for target VMAF of 90
    optimal_cq = find_optimal_cq(
        video_features=features,
        target_vmaf_original=90.0,
        logging_enabled=True
    )
    
    if optimal_cq != -1:
        print(f"Use CQ {optimal_cq} for encoding")
    else:
        print("CQ optimization failed")
    
    Advanced Usage with Custom Components:
    -------------------------------------
    # Load custom preprocessing components
    pipeline, _, vmaf_scaler, cq_min, cq_max = get_scalers_from_pipeline('custom_pipeline.pkl')
    
    # Specify required features explicitly
    required_features = [
        'metrics_avg_motion', 'metrics_avg_texture', 'metrics_frame_rate',
        'metrics_resolution_width', 'metrics_resolution_height', 'cq'
    ]
    
    # Optimize with custom components
    optimal_cq = find_optimal_cq(
        video_features=features,
        target_vmaf_original=93.0,
        pipeline_obj=pipeline,
        vmaf_scaler=vmaf_scaler,
        cq_min=cq_min,
        cq_max=cq_max,
        required_feature_names=required_features,
        logging_enabled=True
    )
    
    Batch Processing Usage:
    ----------------------
    videos = ['video1.mp4', 'video2.mp4', 'video3.mp4']
    target_vmaf = 88.0
    
    for video in videos:
        features = analyze_video_fast(video, logging_enabled=False)  # Reduce output
        optimal_cq = find_optimal_cq(
            video_features=features,
            target_vmaf_original=target_vmaf,
            logging_enabled=False  # Minimal logging for batch
        )
        print(f"{video}: CQ {optimal_cq}")
    
    INTEGRATION NOTES:
    ==================
    
    Pipeline Integration:
    - Called by process_short_video_as_single_scene() for single-scene optimization
    - Called by _process_individual_scenes() for multi-scene optimization
    - Results used by encode_scene_with_size_check() for actual encoding
    
    Model Dependencies:
    - Requires trained PyTorch VMAF prediction model in src/models/
    - Requires fitted preprocessing pipeline in src/models/
    - Optionally uses feature_names.txt for input validation
    
    Configuration Integration:
    - Target VMAF loaded from config.json
    - Model paths configurable via model_paths section
    - Logging behavior respects global logging settings
    
    Hardware Considerations:
    - Models run on CPU for compatibility (configurable)
    - Memory usage scales with model complexity
    - Can be parallelized across multiple videos
    
    QUALITY OPTIMIZATION STRATEGY:
    ==============================
    
    The CQ optimization uses a sophisticated strategy:
    
    1. Boundary Testing: Tests min/max CQ values first to check if target is achievable
    2. Early Termination: Returns boundary values if target is outside achievable range
    3. Binary Search: Efficiently narrows search space using predicted VMAF scores
    4. Convergence Detection: Stops when search range becomes sufficiently small
    5. Result Validation: Ensures returned CQ is within encoder's valid range
    
    This approach typically finds optimal CQ values within 5-10 iterations while
    ensuring the result achieves the target VMAF score as closely as possible.
    
    TROUBLESHOOTING:
    ================
    
    Common Issues and Solutions:
    
    1. Returns -1 (Optimization Failed):
       - Check that all model files exist in src/models/
       - Verify video_features contains expected keys
       - Ensure target_vmaf is reasonable (20-100 range)
       - Check preprocessing pipeline is fitted
    
    2. CQ Value Seems Wrong:
       - Verify target_vmaf is appropriate for content type
       - Check that video features were extracted correctly
       - Ensure model was trained on similar content
       - Consider content-specific CQ adjustments
    
    3. Slow Performance:
       - Set logging_enabled=False for batch processing
       - Ensure models are cached (avoid repeated loading)
       - Consider reducing max_iterations for faster search
    
    4. Feature Mismatch Errors:
       - Check that analyze_video_fast() extracted all required features
       - Verify required_feature_names matches model training
       - Ensure preprocessing pipeline handles all feature types
    
    VERSION HISTORY:
    ================
    v2.1.0: Added comprehensive error handling and feature name validation
    v2.0.0: Integrated PyTorch models with automatic architecture detection
    v1.5.0: Added custom VMAF scaler support and improved logging
    v1.0.0: Initial implementation with basic binary search optimization
    
    SEE ALSO:
    =========
    - analyze_video_fast(): Feature extraction for video analysis
    - get_scalers_from_pipeline(): Loading preprocessing components
    - search_for_cq(): Core binary search optimization algorithm
    - create_pytorch_vmaf_prediction_function(): Model prediction wrapper
    - enhanced_encoding_main.py: Main integration point for video processing
    
    AUTHOR: VIDAIO Development Team
    LAST UPDATED: 2025-05-30
    """
    
    # =================================================================
    # STEP 1: COMPONENT VALIDATION AND FALLBACK LOADING
    # =================================================================
    # Validate that all required components are available or can be loaded
    # This ensures the optimization can proceed with all necessary tools
    
    if logging_enabled:
        print("🔧 Validating optimization components...")
    
    # Store provided components, use fallback loading if any are missing
    current_pipeline = pipeline_obj
    current_vmaf_scaler = vmaf_scaler
    current_cq_min = cq_min
    current_cq_max = cq_max

    # Check if any critical components are missing
    missing_components = []
    if current_pipeline is None: missing_components.append("preprocessing pipeline")
    if current_vmaf_scaler is None: missing_components.append("VMAF scaler")
    if current_cq_min is None: missing_components.append("CQ minimum bound")
    if current_cq_max is None: missing_components.append("CQ maximum bound")
    
    if missing_components:
        if logging_enabled: 
            print(f"⚠️ Missing components: {', '.join(missing_components)}")
            print("🔄 Attempting fallback loading from default locations...")
        
        # Load missing components using fallback mechanism
        loaded_pipeline, loaded_feature_scaler, loaded_vmaf_scaler, loaded_cq_min, loaded_cq_max = load_scalers_if_needed(
            logging_enabled=logging_enabled
        )
        
        # Check if fallback loading was successful
        if loaded_pipeline is None:
            if logging_enabled: 
                print("❌ ERROR: Failed to load pipeline via fallback. Cannot proceed.")
                print("💡 Ensure preprocessing_pipeline.pkl exists in src/models/ directory")
            return -1, 0.0

        # Use loaded components to fill in missing values
        if current_pipeline is None: current_pipeline = loaded_pipeline
        if current_vmaf_scaler is None: current_vmaf_scaler = loaded_vmaf_scaler
        if current_cq_min is None: current_cq_min = loaded_cq_min
        if current_cq_max is None: current_cq_max = loaded_cq_max
        
        if logging_enabled:
            print("✅ All missing components loaded successfully")
    # =================================================================
    # STEP 1.5: CODEC-SPECIFIC CQ LIMIT ENFORCEMENT
    # =================================================================
    # Apply codec-specific maximum CQ values to prevent poor quality
    
    if codec_name and logging_enabled:
        print(f"🎥 Applying codec-specific CQ limits for: {codec_name}")
    
    # Get codec-specific limits
    codec_limits = None
    if codec_name:
        codec_limits = get_codec_cq_limits(
            codec_name=codec_name,
            conservative=conservative_cq_limits,
            logging_enabled=logging_enabled
        )
        
        # Apply codec-specific maximum CQ limit
        codec_max_cq = int(codec_limits['effective_max_cq'])
        
        if current_cq_max is None:
            current_cq_max = codec_max_cq
            if logging_enabled:
                print(f"   ✅ Set CQ max from codec limits: {current_cq_max}")
        elif current_cq_max > codec_max_cq:
            if logging_enabled:
                print(f"   🔧 Limiting CQ max: {current_cq_max} → {codec_max_cq} (codec-specific)")
            current_cq_max = codec_max_cq
        else:
            if logging_enabled:
                print(f"   ✅ Current CQ max ({current_cq_max}) within codec limits ({codec_max_cq})")
    
    # Apply final fallback defaults if still missing
    if current_cq_min is None: current_cq_min = 10
    if current_cq_max is None: current_cq_max = 40  # Conservative default
    
    if logging_enabled:
        print(f"📊 Final CQ optimization range: {current_cq_min}-{current_cq_max}")
    # =================================================================
    # STEP 2: PIPELINE FITNESS VALIDATION
    # =================================================================
    # Verify that the preprocessing pipeline has been fitted (trained)
    # Unfitted pipelines cannot transform new data
    try:
        if current_pipeline:
            check_is_fitted(current_pipeline)
        if logging_enabled: 
            print("✅ Preprocessing pipeline is fitted and ready for use")
    except NotFittedError:
        if logging_enabled: 
            print("❌ ERROR: Pipeline is NOT fitted before transforming features!")
            print("💡 The pipeline must be trained before it can process video features")
        return -1, 0.0
    except Exception as e:
        if logging_enabled: 
            print(f"❌ ERROR checking if pipeline is fitted: {e}")
        return -1, 0.0

    # =================================================================
    # STEP 3: MODEL AND FEATURE NAME LOADING
    # =================================================================
    # Load the PyTorch VMAF prediction model and determine expected features
    
    if logging_enabled:
        print("🧠 Loading PyTorch VMAF prediction model...")
    
    # Load model with caching for performance, passing the full path
    if vmaf_prediction_model_path:
        model = load_pytorch_model_if_needed(
            model_full_path=vmaf_prediction_model_path, # Pass the full path here
            logging_enabled=logging_enabled
        )
    else:
        model = None
    
    # Determine expected feature names for input validation
    if required_feature_names:
        if logging_enabled:
            print(f"📋 Using provided feature names: {len(required_feature_names)} features")
            if len(required_feature_names) <= 10:  # Show features if reasonable number
                print(f"   Features: {', '.join(required_feature_names)}")
    else:
        # Fall back to loading feature names from model or file
        required_feature_names = load_feature_names_if_needed(logging_enabled=logging_enabled)
        if logging_enabled:
            feature_count = len(required_feature_names) if required_feature_names else 0
            print(f"📋 Loaded feature names from model/file: {feature_count} features")

    # Validate that critical components were loaded
    if model is None:
        if logging_enabled: 
            print("❌ ERROR: Failed to load PyTorch model. Cannot proceed.")
            print("💡 Ensure vmaf_prediction_model.pt exists in src/models/ directory")
        return -1, 0.0

    # Use default feature names if none could be determined
    if not required_feature_names:
        if logging_enabled:
            print("⚠️ WARNING: Could not determine model's expected features")
            print("🔧 Using default feature set - this may affect prediction accuracy")
        
        # Default feature set based on common video analysis features
        required_feature_names = [
            'metrics_avg_motion', 'metrics_avg_edge_density', 'metrics_avg_texture',
            'metrics_avg_temporal_information', 'metrics_avg_spatial_information',
            'metrics_avg_color_complexity', 'metrics_avg_motion_variance', 'metrics_avg_grain_noise',
            'metrics_frame_rate', 'metrics_resolution', 'cq'
        ]
        
        if logging_enabled:
            print(f"   Using {len(required_feature_names)} default features")

    # =================================================================
    # STEP 4: INPUT DATA PREPARATION AND VALIDATION
    # =================================================================
    # Convert video features dictionary to DataFrame and prepare for processing
    
    if logging_enabled:
        print("📊 Preparing input features for optimization...")
    
    # Convert video features dictionary to pandas DataFrame
    try:
        input_data = pd.DataFrame([video_features])
        if logging_enabled:
            print(f"   ✅ Created DataFrame with {len(video_features)} features")
    except Exception as e:
        if logging_enabled: 
            print(f"❌ ERROR: Could not convert video_features dict to DataFrame: {e}")
            print("💡 Ensure video_features is a valid dictionary with numeric values")
        return -1, 0.0

    # Add CQ column required for optimization (will be varied during search)
    input_data['cq'] = current_cq_min  # Start with minimum CQ value
    
    # Check for missing required features and add defaults
    missing_features = [col for col in required_feature_names if col not in input_data.columns]
    if missing_features:
        if logging_enabled:
            print(f"⚠️ Missing {len(missing_features)} required features: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")
            print("🔧 Adding default values for missing features...")
        
        # Add default values for missing features
        feature_defaults = {
            'metrics_avg_motion': 0.1, 'metrics_avg_edge_density': 0.05,
            'metrics_avg_texture': 4.0, 'metrics_avg_temporal_information': 25.0,
            'metrics_avg_spatial_information': 50.0, 'metrics_avg_color_complexity': 3.0,
            'metrics_avg_motion_variance': 1.0, 'metrics_avg_grain_noise': 5.0,
            'metrics_frame_rate': 30.0, 'metrics_resolution': '(1920, 1080)',
            'metrics_resolution_width': 1920, 'metrics_resolution_height': 1080
        }
        
        for feature in missing_features:
            default_value = feature_defaults.get(feature, 0.0)
            input_data[feature] = default_value
            if logging_enabled:
                print(f"   Added {feature} = {default_value}")

    # Ensure features are in the correct order expected by the model
    try:
        input_data = input_data[required_feature_names]
        if logging_enabled:
            print(f"✅ Features reordered to match model expectations")
            #print(f"   Final feature order: {list(input_data.columns)}")
    except KeyError as e:
        if logging_enabled:
            print(f"❌ ERROR: Could not reorder features to match model requirements: {e}")
            print(f"   Required features: {required_feature_names}")
            print(f"   Available features: {list(input_data.columns)}")
        return -1, 0.0
    except Exception as e:
        if logging_enabled:
            print(f"❌ ERROR: Feature reordering failed: {e}")
        return -1, 0.0

    # =================================================================
    # STEP 5: PREPROCESSING PIPELINE APPLICATION
    # =================================================================
    # Apply the complete preprocessing pipeline to normalize and scale features
    
    if logging_enabled:
        print("⚙️ Applying preprocessing pipeline to features...")
    
    try:
        # Transform features through the preprocessing pipeline
        if current_pipeline:
            features_scaled_array = current_pipeline.transform(input_data)
        else:
            features_scaled_array = input_data.to_numpy()
        
        if logging_enabled: 
            print(f"✅ Pipeline transformation successful")
            print(f"   Input shape: {input_data.shape}")
            print(f"   Output shape: {features_scaled_array.shape}")
            
    except Exception as e:
        if logging_enabled:
            print(f"❌ ERROR: Pipeline transformation failed: {e}")
            print(f"   Input DataFrame shape: {input_data.shape}")
            print(f"   Input DataFrame columns: {list(input_data.columns)}")
            print("🔍 Detailed error traceback:")
            traceback.print_exc()
        return -1, 0.0

    # Convert pipeline output back to DataFrame with correct column names
    features_scaled_df = pd.DataFrame(features_scaled_array, columns=required_feature_names)
    
    if logging_enabled: 
        print(f"✅ Features successfully preprocessed and scaled")
        print(f"   Scaled feature sample: {features_scaled_df.iloc[0][:3].values}")

    # =================================================================
    # STEP 6: CQ COLUMN IDENTIFICATION AND VALIDATION
    # =================================================================
    # Ensure the CQ column is present and can be modified during optimization
    
    if 'cq' in features_scaled_df.columns:
        cq_column_name = 'cq'
        if logging_enabled: 
            print(f"✅ CQ column identified: '{cq_column_name}'")
    else:
        if logging_enabled: 
            print(f"❌ ERROR: CQ column not found in pipeline output")
            print(f"   Available columns: {list(features_scaled_df.columns)}")
            print("💡 Pipeline must preserve the 'cq' column for optimization")
        return -1, 0.0

    # =================================================================
    # STEP 7: TARGET VMAF SCALING AND VALIDATION
    # =================================================================
    # Convert target VMAF from original range (0-100) to model range (0-1)
    
    if logging_enabled:
        print(f"🎯 Scaling target VMAF: {target_vmaf_original}")
    
    try:
        # Validate VMAF scaler has required attributes
        if current_vmaf_scaler and hasattr(current_vmaf_scaler, 'min_val') and hasattr(current_vmaf_scaler, 'max_val'):
            # Calculate VMAF range and validate it's positive
            vmaf_range = current_vmaf_scaler.max_val - current_vmaf_scaler.min_val
            if vmaf_range <= 0: 
                raise ValueError(f"VMAF scaler range is invalid: {vmaf_range}")
            
            # Scale target VMAF to model's expected range
            target_vmaf_scaled = (target_vmaf_original - current_vmaf_scaler.min_val) / vmaf_range
            
            # Clamp to valid range [0, 1] for safety
            target_vmaf_scaled = max(0.0, min(1.0, target_vmaf_scaled))
            
            if logging_enabled:
                print(f"✅ Target VMAF scaled: {target_vmaf_original} → {target_vmaf_scaled:.3f}")
                print(f"   VMAF scaler range: [{current_vmaf_scaler.min_val}, {current_vmaf_scaler.max_val}]")
        else:
            target_vmaf_scaled = target_vmaf_original / 100.0
            if logging_enabled:
                print(f"⚠️ Using default VMAF scaling (0-100 range)")

    except (AttributeError, ValueError, ZeroDivisionError) as e:
        if logging_enabled: 
            print(f"❌ ERROR: VMAF scaling failed: {e}")
            print("💡 Check VMAF scaler configuration and target VMAF value")
        return -1, 0.0

    # =================================================================
    # STEP 8: PREDICTION FUNCTION CREATION
    # =================================================================
    # Create a specialized prediction function for the binary search algorithm
    
    if logging_enabled:
        print("🔮 Creating VMAF prediction function...")
    
    predict_vmaf_fn = create_pytorch_vmaf_prediction_function(
        model, 
        current_vmaf_scaler, 
        logging_enabled=logging_enabled
    )
    
    if predict_vmaf_fn is None:
        if logging_enabled: 
            print("❌ ERROR: Failed to create prediction function")
            print("💡 Check model loading and VMAF scaler configuration")
        return -1, 0.0
    
    if logging_enabled:
        print("✅ Prediction function created successfully")

    # =================================================================
    # STEP 9: BINARY SEARCH OPTIMIZATION
    # =================================================================
    # Perform binary search to find optimal CQ value for target VMAF
    
    if logging_enabled:
        print(f"🔍 Starting CQ optimization search...")
        print(f"   🎯 Target VMAF: {target_vmaf_original} (scaled: {target_vmaf_scaled:.3f})")
        print(f"   📊 CQ search range: [{current_cq_min}, {current_cq_max}]")

    try:
        # Prepare base features for optimization (single row)
        base_features_scaled_df = features_scaled_df.iloc[[0]].copy()

        # Perform binary search optimization
        search_result = search_for_cq(
            predict_vmaf_fn=predict_vmaf_fn,
            video_features_df=base_features_scaled_df,
            target_vmaf_scaled=target_vmaf_scaled,
            min_cq=0.0,  # Search in normalized range
            max_cq=1.0,
            min_cq_original=current_cq_min,  # Original encoder range
            max_cq_original=current_cq_max,
            tolerance=0.01,  # Search precision
            max_iterations=10,  # Iteration limit
            logging_enabled=logging_enabled
        )
        
    except Exception as e:
        if logging_enabled: 
            print(f"❌ ERROR: CQ search failed: {e}")
            print("🔍 Detailed error traceback:")
            traceback.print_exc()
        return -1, 0.0

    # =================================================================
    # STEP 10: RESULT PROCESSING AND VALIDATION
    # =================================================================
    # Process search results and prepare final CQ value
    
    if logging_enabled:
        print("📋 Processing optimization results...")
        print(f"✅ Search completed in {search_result['iterations']} iterations")
        print(f"   Optimal CQ (scaled): {search_result['optimal_cq']:.4f}")
        print(f"   Optimal CQ (original): {search_result['optimal_cq_original']:.2f}")

        # Display prediction accuracy
        predicted_vmaf_original = search_result.get('predicted_vmaf_original')
        if predicted_vmaf_original is not None:
            prediction_error = abs(predicted_vmaf_original - target_vmaf_original)
            print(f"   Predicted VMAF: {predicted_vmaf_original:.2f}")
            print(f"   Prediction error: ±{prediction_error:.2f} VMAF points")
        else:
            # Show boundary VMAF values if search hit boundaries
            if search_result['all_tested']:
                vmaf_at_low = search_result['all_tested'][0][3]
                vmaf_at_high = search_result['all_tested'][1][3]
                print(f"   VMAF at lowest CQ ({current_cq_min}): {vmaf_at_low:.2f}")
                print(f"   VMAF at highest CQ ({current_cq_max}): {vmaf_at_high:.2f}")
            else:
                print("   ⚠️ No VMAF predictions available")

    # Convert result to integer CQ value suitable for encoding
    optimal_original_cq_float = search_result['optimal_cq_original']
    optimal_original_cq_int = int(round(optimal_original_cq_float))
    
    # Clamp result to valid CQ bounds for safety
    final_cq = max(current_cq_min, min(current_cq_max, optimal_original_cq_int))
    
    # Log any clamping that occurred
    if logging_enabled and final_cq != optimal_original_cq_int:
        print(f"⚠️ CQ clamped to valid range [{current_cq_min}-{current_cq_max}]: {final_cq}")
        
    if logging_enabled:
        print(f"🎯 Final optimal CQ: {final_cq}")
        print("✅ CQ optimization completed successfully")
    predicted_vmaf = search_result.get('predicted_vmaf_original', 0.0)
    return final_cq, predicted_vmaf

# --- Standalone Execution Example ---
if __name__ == "__main__":
    print("Running find_optimal_cq standalone test...")
    
    # Import required modules for standalone execution
    try:
        from src.utils.analyze_video_fast import analyze_video_fast
        from utils.preprocessing_w_augmentation import (
            ColumnDropper, ResolutionTransformer, FrameRateTransformer,
            CQScaler, VMAFScaler, FeatureScaler, TargetExtractor
        )
        print("✅ All required imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running from the correct directory")
        sys.exit(1)

    video_path = './videos/720p50_shields_ter_full.mp4' # Ensure this test video exists or change path
    # For standalone test, define the model path explicitly or ensure config.json is read by a helper
    # This example assumes a direct path for testing.
    # In a real scenario, this path would come from the orchestrator.
    test_vmaf_model_path = os.path.abspath("src/models/vmaf_prediction_model.pt") 
    print(f"Test VMAF model path: {test_vmaf_model_path}")


    print("Testing comprehensive video analysis...")
    metrics = analyze_video_fast(video_path, max_frames=200, logging_enabled=False)  # Disable for cleaner output

    target_vmaf = 92.5
    print(f"\n--- Test Case: Target VMAF = {target_vmaf} ---")
    result = find_optimal_cq( # Expecting two return values now
        metrics, 
        target_vmaf_original=target_vmaf, 
        vmaf_prediction_model_path=test_vmaf_model_path, # Pass the path
        logging_enabled=True
    )
    if isinstance(result, tuple) and len(result) == 2:
        optimal_cq_result, predicted_vmaf_at_cq = result
    else:
        optimal_cq_result = result
        predicted_vmaf_at_cq = -1
    print(f"\n---> Final Optimal CQ for Target {target_vmaf}: {optimal_cq_result}, Predicted VMAF: {predicted_vmaf_at_cq}")
    '''
    target_vmaf = 80.0
    print(f"\n--- Test Case: Target VMAF = {target_vmaf} ---")
    optimal_cq_result = find_optimal_cq(metrics, target_vmaf_original=target_vmaf)
    print(f"\n---> Final Optimal CQ for Target {target_vmaf}: {optimal_cq_result}")
    '''
