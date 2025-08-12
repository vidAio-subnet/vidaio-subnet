import pandas as pd
import numpy as np
import re
import sys
import os
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import NotFittedError, check_is_fitted # Import for checking fit status

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop
        
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self
        
    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns_to_drop:
            if col in X_copy.columns:
                X_copy.drop(col, axis=1, inplace=True)
        return X_copy
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names after dropping columns."""
        if input_features is None:
             raise ValueError("input_features is required for get_feature_names_out")

        dropped_cols_set = set(self.columns_to_drop)
        return [col for col in input_features if col not in dropped_cols_set]
    
class VMAFScaler(BaseEstimator, TransformerMixin):
    def __init__(self, target_column='vmaf', clip_values=True,verbose=True):
        self.target_column = target_column
        self.verbose = verbose 
        self.min_val = None
        self.max_val = None
        self.clip_values = clip_values
        
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        if self.target_column in X.columns:
            X_target = pd.to_numeric(X[self.target_column], errors='coerce')
            self.min_val = X_target.min()
            self.max_val = X_target.max()
            if self.verbose:
                print(f"VMAF scale - discovered range: min={self.min_val:.2f}, max={self.max_val:.2f}")
        return self
        
    def transform(self, X):
        X_copy = X.copy()
        if self.target_column in X_copy.columns and self.min_val is not None and self.max_val is not None:
            X_copy[self.target_column] = pd.to_numeric(X_copy[self.target_column], errors='coerce')

            if self.verbose: 
                print(f"Original {self.target_column} values (first 3): {X_copy[self.target_column].head(3).tolist()}")

            range_diff = self.max_val - self.min_val
            if range_diff > 1e-6:
                scaled_values = (X_copy[self.target_column] - self.min_val) / range_diff
                X_copy[self.target_column] = scaled_values
                if self.clip_values:
                    X_copy[self.target_column] = X_copy[self.target_column].clip(0, 1)
            else:
                X_copy[self.target_column] = 0.5
                if self.verbose:
                    print("Warning: VMAF min and max too close, using default value")

            if self.verbose:
                print(f"Scaled {self.target_column} values (first 3): {X_copy[self.target_column].head(3).tolist()}")
                final_min = X_copy[self.target_column].min()
                final_max = X_copy[self.target_column].max()
                print(f"Final scaled {self.target_column} range: min={final_min:.6f}, max={final_max:.6f}")
                if abs(final_min - final_max) < 1e-6:
                    print(f"ERROR: All {self.target_column} values collapsed to {final_min}!")
        return X_copy
        
    def inverse_transform(self, X):
        """Convert scaled VMAF back to original scale"""
        X_copy = X.copy()
        if self.target_column in X_copy.columns and self.min_val is not None and self.max_val is not None:
            X_copy[self.target_column] = X_copy[self.target_column] * (self.max_val - self.min_val) + self.min_val
        return X_copy
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names, which are unchanged by this transformer."""
        if input_features is None:
             raise ValueError("input_features is required for get_feature_names_out")
        return list(input_features)
    
class TargetExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, target_column='vmaf', vmaf_scaler=None):
        self.target_column = target_column
        self.vmaf_scaler = vmaf_scaler
        
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self
        
    def transform(self, X):
        return X
        
    def get_target(self, X):
        if self.target_column in X.columns:
            y = pd.to_numeric(X[self.target_column], errors='coerce')
            return y
        return None
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names, which are unchanged by this transformer."""
        if input_features is None:
             raise ValueError("input_features is required for get_feature_names_out")
        return list(input_features)
    
class CQScaler(BaseEstimator, TransformerMixin):
    """
    Specialized scaler for CQ (Constant Quality) values that adapts to different codecs.
    Different encoders use different CQ/QP ranges:
    - AV1: typically 10-63
    - x264/x265: typically 0-51
    - NVENC: typically 0-51 (but may have different perceptual mapping)
    - VP9: typically 0-63
    """
    def __init__(self, cq_column='cq', codec='auto', min_cq=None, max_cq=None,verbose=True):
        self.cq_column = cq_column
        self.codec = codec
        self.custom_min_cq = min_cq
        self.custom_max_cq = max_cq
        self.verbose = verbose
        
        # Default ranges for common codecs
        self.codec_ranges = {
            'av1': (10, 63),
            'aom-av1': (10, 63),
            'libaom': (10, 63),
            'x264': (0, 51),
            'x265': (0, 51),
            'h264': (0, 51),
            'h265': (0, 51),
            'hevc': (0, 51),
            'nvenc': (0, 51),
            'nvenc_h264': (0, 51),
            'nvenc_hevc': (0, 51),
            'qsv': (1, 51),
            'vp9': (0, 63),
            'vp8': (0, 63)
        }
        
        self.min_cq = None
        self.max_cq = None
        self.fitted_min = None
        self.fitted_max = None
        self.detected_codec = None
        
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        if self.cq_column in X.columns:
            cq_values = pd.to_numeric(X[self.cq_column], errors='coerce')
            self.fitted_min = cq_values.min()
            self.fitted_max = cq_values.max()

            if self.codec == 'auto':
                self.detected_codec = self._detect_codec(self.fitted_min, self.fitted_max)
                if self.verbose: print(f"Auto-detected codec: {self.detected_codec}")
            else:
                self.detected_codec = self.codec

            if self.custom_min_cq is not None and self.custom_max_cq is not None:
                self.min_cq = self.custom_min_cq
                self.max_cq = self.custom_max_cq
                if self.verbose: print(f"Using custom CQ range: [{self.min_cq}, {self.max_cq}]")
            else:
                codec_key = self.detected_codec.lower() if self.detected_codec else 'av1'
                if codec_key in self.codec_ranges:
                    self.min_cq, self.max_cq = self.codec_ranges[codec_key]
                else:
                    self.min_cq, self.max_cq = self.codec_ranges['av1']
                    if self.verbose: print(f"Unknown codec '{codec_key}', falling back to AV1 range: [{self.min_cq}, {self.max_cq}]")

            if self.verbose:
                print(f"CQ scaling - discovered range: min={self.fitted_min}, max={self.fitted_max}")
                print(f"CQ scaling - using range: min={self.min_cq}, max={self.max_cq} (codec: {self.detected_codec})")

            if self.fitted_min < self.min_cq:
                if self.verbose: print(f"Warning: CQ values below minimum expected ({self.min_cq}) found in data. Using {self.fitted_min} as minimum.")
                self.min_cq = self.fitted_min
            if self.fitted_max > self.max_cq:
                if self.verbose: print(f"Warning: CQ values above maximum expected ({self.max_cq}) found in data. Using {self.fitted_max} as maximum.")
                self.max_cq = self.fitted_max
        return self
    
    
    def _detect_codec(self, min_val, max_val):
        """Try to detect codec from CQ value range"""
        # Check common value ranges
        if 0 <= min_val <= 10 and 50 <= max_val <= 51:
            return 'x264/x265'  # Likely H.264/H.265 (also possibly NVENC)
        elif 10 <= min_val <= 15 and 60 <= max_val <= 63:
            return 'av1'  # Likely AV1
        elif 0 <= min_val <= 5 and 60 <= max_val <= 63:
            return 'vp9'  # Likely VP9
        elif 0 <= min_val <= 5 and 30 <= max_val <= 40:
            return 'nvenc'  # Possibly NVENC with restricted range
        else:
            # Examine the distribution to make a best guess
            if max_val <= 51:
                return 'h264/h265'
            else:
                return 'av1'  # Default to AV1
    
    def transform(self, X):
        X_copy = X.copy()
        if self.cq_column in X_copy.columns:
            X_copy[self.cq_column] = pd.to_numeric(X_copy[self.cq_column], errors='coerce')

            if self.verbose: # Wrap print
                print(f"Original {self.cq_column} values (first 3): {X_copy[self.cq_column].head(3).tolist()}")

            # Apply scaling only if range is valid
            if self.min_cq is not None and self.max_cq is not None and self.max_cq > self.min_cq:
                 X_copy[self.cq_column] = (X_copy[self.cq_column] - self.min_cq) / (self.max_cq - self.min_cq)
                 X_copy[self.cq_column] = X_copy[self.cq_column].clip(0, 1)
            elif self.min_cq is not None: # Handle case where min == max
                 X_copy[self.cq_column] = 0.5 # Map to middle
                 if self.verbose: print(f"Warning: CQ min equals max ({self.min_cq}), setting scaled value to 0.5") # Wrap print
            else: # Handle case where fit wasn't called or failed
                 if self.verbose: print(f"Warning: CQ scaler not properly fitted, skipping transform for {self.cq_column}") # Wrap print


            if self.verbose: # Wrap print
                print(f"Scaled {self.cq_column} values (first 3): {X_copy[self.cq_column].head(3).tolist()}")
        return X_copy
    
    def inverse_transform(self, X):
        """Convert scaled CQ back to original scale"""
        X_copy = X.copy()
        
        if self.cq_column in X_copy.columns:
            X_copy[self.cq_column] = X_copy[self.cq_column] * (self.max_cq - self.min_cq) + self.min_cq
            X_copy[self.cq_column] = np.round(X_copy[self.cq_column]).astype(int)

        return X_copy
    
    def get_cq_range(self):
        """Return the current CQ range"""
        return self.min_cq, self.max_cq

    def get_feature_names_out(self, input_features=None):
        """Return feature names, which are unchanged by this transformer."""
        if input_features is None:
             raise ValueError("input_features is required for get_feature_names_out")
       
        return list(input_features)

class ResolutionTransformer(BaseEstimator, TransformerMixin):
    """
    Enhanced transformer for video resolution that extracts total pixel count
    and converts to an ordinal scale.
    """
    def __init__(self, resolution_column='metrics_resolution', verbose=True):
        self.resolution_column = resolution_column
        self.verbose = verbose
        self.resolution_mapping = {}
        self.resolution_values = []
        self.standard_resolutions = [ #TODO: check if these are enough or more combinations need to be added
            (640, 360),    # 360p
            (640, 480),    # 480p
            (854, 480),    # 480p widescreen
            (1280, 720),   # 720p
            (1920, 1080),  # 1080p
            (2560, 1440),  # 1440p
            (3840, 2160),  # 4K
            (7680, 4320)   # 8K
        ]
        # Convert to pixel counts
        self.standard_pixels = [width * height for width, height in self.standard_resolutions]
        # Create friendly names for debugging
        self.resolution_names = ["360p", "480p", "480p wide", "720p", "1080p", "1440p", "4K", "8K"]
        
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        if self.resolution_column in X.columns:
            if self.verbose: # Wrap print
                sample_values = X[self.resolution_column].head(5).tolist()
                print(f"Sample resolution values: {sample_values}")

            resolution_values = []

            for val in X[self.resolution_column]:
                res = self._extract_resolution(val)
                if res > 0:
                    resolution_values.append(res)
            
            # If we found valid resolutions, create mapping
            if resolution_values:
                # Get unique resolutions
                unique_resolutions = sorted(set(resolution_values))
                self.resolution_values = unique_resolutions
                
                # Create ordinal mapping (1-based)
                self.resolution_mapping = {res: i+1 for i, res in enumerate(unique_resolutions)}
                
                # Map resolutions to closest standard resolution names for display
                name_mapping = {}
                for res in unique_resolutions:
                    closest_idx = self._find_closest_standard_resolution(res)
                    name_mapping[res] = self.resolution_names[closest_idx]
                
                if self.verbose: # Wrap print block
                    print(f"Found {len(unique_resolutions)} unique resolutions")
                    print("Resolution mapping:")
                    for res, idx in self.resolution_mapping.items():
                        name = name_mapping.get(res, "Unknown")
                        print(f"  {res} pixels ({name}): {idx}")
            else:
                # If no valid resolutions found, use standard mapping
                if self.verbose: print("No valid resolutions found, using standard mapping")
                self.resolution_values = self.standard_pixels
                self.resolution_mapping = {res: i+1 for i, res in enumerate(self.standard_pixels)}
        
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        if self.resolution_column in X_copy.columns:
            # Extract resolution values and convert to ordinal
            pixels = []
            ordinal_values = []
            scaled_values = []
            
            for val in X_copy[self.resolution_column]:
                # Get pixel count
                res = self._extract_resolution(val)
                pixels.append(res)
                
                # Convert to ordinal value
                ord_val = self._get_ordinal_value(res)
                ordinal_values.append(ord_val)
                
                # Scale to [0,1] range
                scaled = (ord_val - 1) / (len(self.resolution_mapping) - 1) if len(self.resolution_mapping) > 1 else 0.5
                scaled_values.append(scaled)
            
            if self.verbose: # Wrap print block
                print(f"Resolution conversion examples (first 3):")
                for i in range(min(3, len(X_copy))):
                    pixel_val = pixels[i]
                    ord_val = ordinal_values[i]
                    scaled_val = scaled_values[i]
                    closest_idx = self._find_closest_standard_resolution(pixel_val)
                    res_name = self.resolution_names[closest_idx]
                    print(f"  '{X_copy[self.resolution_column].iloc[i]}' → {pixel_val} pixels ({res_name}) → ordinal {ord_val} → scaled {scaled_val:.4f}")

            X_copy[self.resolution_column] = scaled_values
        
        return X_copy
    
    def _extract_resolution(self, res_input):
        """Extract total pixel count from resolution string or tuple"""

        # --- ADD THIS TUPLE HANDLING ---
        if isinstance(res_input, (tuple, list)) and len(res_input) == 2:
            try:
                width = int(res_input[0])
                height = int(res_input[1])
                return width * height
            except (ValueError, TypeError):
                if self.verbose:
                    print(f"Warning: Could not parse resolution tuple/list: '{res_input}'")
                return 0 # Return 0 if tuple elements aren't numbers
        # --- END TUPLE HANDLING ---

        # Now, proceed with string handling if it wasn't a tuple
        if not isinstance(res_input, str):
            if self.verbose:
                 print(f"Warning: Unexpected resolution type: {type(res_input)}, value: '{res_input}'")
            return 0 # Return 0 if not a string or handled tuple

        res_str = res_input

        res_str = res_str.strip(" '\"")

        match = re.search(r'^\(?(\d+),\s*(\d+)\)?$', res_str) # Adjusted regex slightly for robustness
        if match:
            width = int(match.group(1))
            height = int(match.group(2))
            return width * height

        # Try to match patterns like 1280x720
        match = re.search(r'^(\d+)[xX×](\d+)$', res_str) # Anchored regex
        if match:
            width = int(match.group(1))
            height = int(match.group(2))
            return width * height

        # Try patterns like "width: 1920, height: 1080"
        match = re.search(r'width:?\s*(\d+).*?height:?\s*(\d+)', res_str, re.IGNORECASE)
        if match:
            width = int(match.group(1))
            height = int(match.group(2))
            return width * height

        # Try to match common resolution names
        resolution_names = {
            "360p": 640*360,
            "480p": 640*480,
            "720p": 1280*720,
            "1080p": 1920*1080,
            "1440p": 2560*1440,
            "2k": 2560*1440,
            "4k": 3840*2160,
            "uhd": 3840*2160,
            "8k": 7680*4320
        }

        # Check lower case for name matching
        res_str_lower = res_str.lower()
        for name, pixels in resolution_names.items():
            if name == res_str_lower: # Match exact name after stripping/lowercasing
                return pixels

        # If no pattern matches, return default
        if self.verbose:
             print(f"Warning: Could not parse resolution string: '{res_str}'") # Log unparsed strings
        return 0
    
    def _get_ordinal_value(self, pixel_count):
        """Get ordinal value for a resolution, using nearest neighbor for new values"""
        # If pixel count is invalid, use the smallest valid value
        if pixel_count <= 0:
            return 1
        
        # If resolution exists in mapping, use the predefined value
        if pixel_count in self.resolution_mapping:
            return self.resolution_mapping[pixel_count]
        
        # If it's a new value, find the closest known resolution
        if self.resolution_values:
            closest_resolution = min(self.resolution_values, 
                                    key=lambda x: abs(x - pixel_count))
            
            closest_value = self.resolution_mapping[closest_resolution]
            print(f"New resolution value {pixel_count} mapped to closest known value {closest_resolution} (ordinal: {closest_value})")
            return closest_value
        
        # If we have no mapping at all, return index of closest standard resolution + 1
        closest_idx = self._find_closest_standard_resolution(pixel_count)
        return closest_idx + 1
    
    def _find_closest_standard_resolution(self, pixel_count):
        """Find the index of the closest standard resolution by pixel count"""
        if pixel_count <= 0:
            return 0  # Default to lowest resolution
        
        # Find closest standard resolution
        diffs = [abs(pixel_count - std_res) for std_res in self.standard_pixels]
        return diffs.index(min(diffs))
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names, which are unchanged by this transformer."""
        if input_features is None:
             raise ValueError("input_features is required for get_feature_names_out")
        # These transformers modify values but don't change the set of columns names
        return list(input_features)

class FeatureScaler(BaseEstimator, TransformerMixin):
    """
    Enhanced version of FeatureScaler with better handling of edge cases and logging.
    """
    def __init__(self, columns_to_scale, scaling_type='minmax', excluded_columns=None,verbose=True):
        self.columns_to_scale = columns_to_scale
        self.scaling_type = scaling_type
        self.excluded_columns = excluded_columns or []
        self.verbose = verbose
        # Create the underlying scikit-learn scaler
        if scaling_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaling_type == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaling type: {scaling_type}")
            
        # Store fit parameters
        self.feature_min = {}
        self.feature_max = {}
        self.feature_mean = {}
        self.feature_std = {}
        
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        # Get columns to scale that exist in the dataframe
        existing_columns = [col for col in self.columns_to_scale 
                           if col in X.columns and col not in self.excluded_columns]
        
        if existing_columns:
            
            if self.verbose:
                print(f"Fitting {self.scaling_type} scaler to: {existing_columns}")
            
            # Store stats for each feature independently
            for col in existing_columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
                if X[col].isna().any():
                    if self.verbose: print(f"  Warning: {col} contains {X[col].isna().sum()} NaN values that will be replaced with 0") # Wrap print
                    X[col] = X[col].fillna(0)
                
                if X[col].nunique() > 1:
                    self.feature_min[col] = float(X[col].min())
                    self.feature_max[col] = float(X[col].max())
                    self.feature_mean[col] = float(X[col].mean())
                    self.feature_std[col] = float(X[col].std())
                    if self.verbose: print(f"  {col}: min={self.feature_min[col]}, max={self.feature_max[col]}, mean={self.feature_mean[col]:.4f}, std={self.feature_std[col]:.4f}") # Wrap print
                else:
                    single_value = float(X[col].iloc[0])
                    self.feature_min[col] = single_value - 0.5
                    self.feature_max[col] = single_value + 0.5
                    self.feature_mean[col] = single_value
                    self.feature_std[col] = 1.0
                    if self.verbose: print(f"  {col}: only one unique value ({single_value}), using artificial range [{self.feature_min[col]}, {self.feature_max[col]}]") # Wrap print

            X_to_scale = X[existing_columns].fillna(0)
            try:
                self.scaler.fit(X_to_scale)
            except Exception as e:
                if self.verbose: 
                    print(f"Warning: Error fitting scaler: {e}")
                    print("Using manual scaling parameters instead")
        return self
        
    def transform(self, X):
        X_copy = X.copy()
        existing_columns = [col for col in self.columns_to_scale 
                           if col in X_copy.columns and col not in self.excluded_columns]
        
        if existing_columns:
            if self.verbose: # Wrap print block
                for col in existing_columns[:3]:
                    print(f"{col} before {self.scaling_type} scaling: {X_copy[col].head(3).tolist()}")

            for col in existing_columns:
                X_copy[col] = pd.to_numeric(X_copy[col], errors='coerce')
                if X_copy[col].isna().any():
                    if self.verbose: print(f"  Warning: {col} contains {X_copy[col].isna().sum()} NaN values that will be replaced with 0") # Wrap print
                    X_copy[col] = X_copy[col].fillna(0)

                if col in self.feature_min and col in self.feature_max:
                    min_val = self.feature_min[col]
                    max_val = self.feature_max[col]

                    out_of_range = (X_copy[col] < min_val).any() or (X_copy[col] > max_val).any()
                    if out_of_range and self.verbose: # Wrap print
                        print(f"Warning: {col} has values outside of training range [{min_val}, {max_val}]")

                    if min_val == max_val:
                        if self.verbose: print(f"Warning: {col} has min=max={min_val}, setting to 0.5") # Wrap print
                        X_copy[col] = 0.5
                    else:
                        X_copy[col] = (X_copy[col] - min_val) / (max_val - min_val)
                        X_copy[col] = X_copy[col].clip(0, 1)
                else:
                    if self.verbose: print(f"Warning: No min/max values for {col}, skipping scaling") # Wrap print

            if self.verbose: # Wrap print block
                for col in existing_columns[:3]:
                    print(f"{col} after {self.scaling_type} scaling: {X_copy[col].head(3).tolist()}")
        return X_copy
    
    def inverse_transform(self, X):
        """Convert scaled features back to original scale"""
        X_copy = X.copy()
        existing_columns = [col for col in self.columns_to_scale 
                           if col in X_copy.columns and col not in self.excluded_columns]
        
        if existing_columns:
            for col in existing_columns:
                if col in self.feature_min and col in self.feature_max:
                    min_val = self.feature_min[col]
                    max_val = self.feature_max[col]
                    
                    if min_val != max_val:
                        # Apply inverse min-max scaling
                        X_copy[col] = X_copy[col] * (max_val - min_val) + min_val
        
        return X_copy
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names, which are unchanged by this transformer."""
        if input_features is None:
             raise ValueError("input_features is required for get_feature_names_out")
        # These transformers modify values but don't change the set of columns names
        return list(input_features)

class FrameRateTransformer(BaseEstimator, TransformerMixin):
    """
    Specialized transformer for frame rate values.
    Handles common frame rates and scales to [0,1].
    """
    def __init__(self, frame_rate_column='metrics_frame_rate',verbose=True):
        self.frame_rate_column = frame_rate_column
        self.common_frame_rates = [24, 25, 30, 50, 60, 120]
        self.verbose = verbose
        self.min_rate = None
        self.max_rate = None
        
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        if self.frame_rate_column in X.columns:
            # Convert to numeric
            rates = pd.to_numeric(X[self.frame_rate_column], errors='coerce')
            
            # Get min/max
            self.min_rate = rates.min()
            self.max_rate = rates.max()
            
            # Make sure min/max cover common frame rates
            if self.min_rate > min(self.common_frame_rates):
                if self.verbose: print(f"Adjusting minimum frame rate to {min(self.common_frame_rates)}") # Wrap print
                self.min_rate = min(self.common_frame_rates)
            if self.max_rate < max(self.common_frame_rates):
                if self.verbose: print(f"Adjusting maximum frame rate to {max(self.common_frame_rates)}") # Wrap print
                self.max_rate = max(self.common_frame_rates)

            if self.verbose: # Wrap print block
                print(f"Frame rate range: {self.min_rate} to {self.max_rate}")
                print(f"Common values: {rates.value_counts().head()}")
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        if self.frame_rate_column in X_copy.columns and self.min_rate is not None and self.max_rate is not None:
            # Convert to numeric
            X_copy[self.frame_rate_column] = pd.to_numeric(X_copy[self.frame_rate_column], errors='coerce')
            
            # Handle missing values
            if X_copy[self.frame_rate_column].isna().any():
                median_rate = (self.min_rate + self.max_rate) / 2
                X_copy[self.frame_rate_column] = X_copy[self.frame_rate_column].fillna(median_rate)
            
            if self.verbose: # Wrap print
                print(f"Original frame rates (first 3): {X_copy[self.frame_rate_column].head(3).tolist()}")
            
            # Scale only if range is valid
            if self.max_rate > self.min_rate:
                 X_copy[self.frame_rate_column] = (X_copy[self.frame_rate_column] - self.min_rate) / (self.max_rate - self.min_rate)
                 X_copy[self.frame_rate_column] = X_copy[self.frame_rate_column].clip(0, 1)
            else: # Handle min == max
                 X_copy[self.frame_rate_column] = 0.5
                 if self.verbose: print(f"Warning: Frame rate min equals max ({self.min_rate}), setting scaled value to 0.5") # Wrap print
            if self.verbose: # Wrap print
                    print(f"Scaled frame rates (first 3): {X_copy[self.frame_rate_column].head(3).tolist()}")
            return X_copy
        
    def get_feature_names_out(self, input_features=None):
        """Return feature names, which are unchanged by this transformer."""
        if input_features is None:
             raise ValueError("input_features is required for get_feature_names_out")
        return list(input_features)
    