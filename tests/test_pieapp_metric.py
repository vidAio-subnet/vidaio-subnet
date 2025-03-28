# #### 4k


# import pyiqa
# import torch
# import cv2
# import os
# import numpy as np
# import matplotlib.pyplot as plt

# # List available models
# print(pyiqa.list_models())

# # Set device
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# # Create metric
# iqa_metric = pyiqa.create_metric('pieapp', device=device)
# print(f"Lower better: {iqa_metric.lower_better}")

# def extract_frames(video_path, output_dir):
#     """Extract frames from a video file"""
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     cap = cv2.VideoCapture(video_path)
#     frame_count = 0
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Save frame as PNG
#         output_path = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
#         cv2.imwrite(output_path, frame)
#         frame_count += 1
    
#     cap.release()
#     return frame_count

# def calculate_video_quality(ref_video, dist_video, temp_dir_suffix=""):
#     """Calculate quality metrics between reference and distorted videos for every frame"""
#     # Create temporary directories for frames with unique suffixes
#     ref_frames_dir = f"temp_ref_frames_{temp_dir_suffix}"
#     dist_frames_dir = f"temp_dist_frames_{temp_dir_suffix}"
    
#     # Extract frames
#     print(f"Extracting frames from reference video for {temp_dir_suffix}...")
#     ref_frame_count = extract_frames(ref_video, ref_frames_dir)
#     print(f"Extracting frames from distorted video {temp_dir_suffix}...")
#     dist_frame_count = extract_frames(dist_video, dist_frames_dir)
    
#     # Ensure both videos have the same number of frames
#     frame_count = min(ref_frame_count, dist_frame_count)
    
#     # Calculate scores for every frame
#     scores = []
#     frame_indices = []
    
#     for i in range(frame_count):
#         ref_frame_path = os.path.join(ref_frames_dir, f"frame_{i:04d}.png")
#         dist_frame_path = os.path.join(dist_frames_dir, f"frame_{i:04d}.png")
        
#         if os.path.exists(ref_frame_path) and os.path.exists(dist_frame_path):
#             score = iqa_metric(dist_frame_path, ref_frame_path)
#             scores.append(score.item())
#             frame_indices.append(i)
            
#             if i % 10 == 0:  # Print progress every 10 frames
#                 print(f"Processed frame {i}/{frame_count} for {temp_dir_suffix}")
    
#     # Calculate average score
#     avg_score = np.mean(scores)
#     print(f"Average score for {temp_dir_suffix}: {avg_score}")
    
#     # Clean up temporary files
#     import shutil
#     shutil.rmtree(ref_frames_dir, ignore_errors=True)
#     shutil.rmtree(dist_frames_dir, ignore_errors=True)
    
#     return frame_indices, scores, avg_score

# def compare_multiple_videos(original_video, processed_videos, labels=None):
#     """Compare multiple processed videos against the original"""
#     if labels is None:
#         labels = [f"Processed {i+1}" for i in range(len(processed_videos))]
    
#     results = []
    
#     for i, (proc_video, label) in enumerate(zip(processed_videos, labels)):
#         print(f"\nComparing {label} with original...")
#         frame_indices, scores, avg_score = calculate_video_quality(original_video, proc_video, temp_dir_suffix=label)
#         results.append((frame_indices, scores, avg_score, label))
    
#     # Plot results
#     plt.figure(figsize=(12, 7))
    
#     for frame_indices, scores, avg_score, label in results:
#         plt.plot(frame_indices, scores, label=f"{label} (Avg: {avg_score:.4f})")
    
#     plt.title("Quality Comparison Against Original Video")
#     plt.xlabel("Frame Number")
#     plt.ylabel("Quality Score")
#     plt.grid(True, alpha=0.3)
#     plt.legend()
    
#     # Add horizontal line for average scores
#     for _, _, avg_score, label in results:
#         plt.axhline(y=avg_score, linestyle='--', alpha=0.5, color='gray')
    
#     plt.tight_layout()
#     plt.savefig("quality_comparison.png", dpi=300)
#     plt.show()
    
#     return results

# # Example usage
# if __name__ == "__main__":
#     original_video = "path/to/original_video.mp4"  # Video C
#     processed_video_a = "path/to/processed_video_a.mp4"  # Video A
#     processed_video_b = "path/to/processed_video_b.mp4"  # Video B
    
#     # Compare both processed videos against the original
#     results = compare_multiple_videos(
#         original_video,
#         [processed_video_a, processed_video_b],
#         labels=["Processed A", "Processed B"]
#     )
    
#     # You can also save the detailed results to CSV
#     import pandas as pd
    
#     for frame_indices, scores, avg_score, label in results:
#         df = pd.DataFrame({
#             'Frame': frame_indices,
#             'Score': scores
#         })
#         df.to_csv(f"quality_scores_{label.replace(' ', '_')}.csv", index=False)
    
#     print("\nAnalysis complete. Results saved to CSV files and quality_comparison.png")
















#### 8k video specific

import pyiqa
import torch
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import gc
from PIL import Image
import time

# Memory optimization settings
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
torch.backends.cudnn.benchmark = True  # Optimize CUDA operations [[2]](#__2)

class MemoryEfficientPieAPP:
    def __init__(self, device=None, tile_size=1024, overlap=128):
        """Initialize memory-efficient PieAPP processor"""
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.tile_size = tile_size
        self.overlap = overlap
        
        # Create metric once to avoid reloading model weights
        self.metric = pyiqa.create_metric('pieapp', device=self.device)
        
        # Set up mixed precision if available
        self.amp_enabled = torch.cuda.is_available() and hasattr(torch.cuda, 'amp')
        
        print(f"Initialized MemoryEfficientPieAPP on {self.device}")
        print(f"Mixed precision available: {self.amp_enabled}")
        print(f"Using tile size: {self.tile_size}, overlap: {self.overlap}")
        
        # Print GPU memory info
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    def process_single_frame(self, ref_frame_path, dist_frame_path):
        """Process a single frame with tiling and memory optimization"""
        # Check if files exist
        if not os.path.exists(ref_frame_path) or not os.path.exists(dist_frame_path):
            print(f"Error: One or both image files do not exist")
            return None
            
        # Get image dimensions without loading full image
        with Image.open(ref_frame_path) as img:
            width, height = img.size
        
        # For small images, process directly
        if width <= self.tile_size and height <= self.tile_size:
            return self._process_direct(ref_frame_path, dist_frame_path)
        else:
            return self._process_tiled(ref_frame_path, dist_frame_path)
    
    def _process_direct(self, ref_frame_path, dist_frame_path):
        """Process images directly if they're small enough"""
        try:
            # Clear cache before processing
            torch.cuda.empty_cache()
            
            # Use mixed precision if available
            if self.amp_enabled:
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        score = self.metric(dist_frame_path, ref_frame_path)
                        return score.item()
            else:
                with torch.no_grad():
                    score = self.metric(dist_frame_path, ref_frame_path)
                    return score.item()
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("CUDA OOM in direct processing, falling back to tiled approach")
                return self._process_tiled(ref_frame_path, dist_frame_path)
            else:
                raise e
    
    def _process_tiled(self, ref_frame_path, dist_frame_path):
        """Process images using tiling approach"""
        # Load images
        ref_img = cv2.imread(ref_frame_path)
        dist_img = cv2.imread(dist_frame_path)
        
        if ref_img is None or dist_img is None:
            print(f"Error loading images")
            return None
        
        h, w = ref_img.shape[:2]
        scores = []
        weights = []
        
        # Create temporary directory for tiles
        temp_dir = "temp_pieapp_tiles"
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Process tiles
            for y in range(0, h, self.tile_size - self.overlap):
                for x in range(0, w, self.tile_size - self.overlap):
                    # Extract tile coordinates
                    y_end = min(y + self.tile_size, h)
                    x_end = min(x + self.tile_size, w)
                    
                    # Skip small edge tiles
                    if (y_end - y) < 512 or (x_end - x) < 512:
                        continue
                    
                    # Extract and save tiles
                    ref_tile = ref_img[y:y_end, x:x_end]
                    dist_tile = dist_img[y:y_end, x:x_end]
                    
                    ref_tile_path = os.path.join(temp_dir, f"ref_tile_{y}_{x}.jpg")
                    dist_tile_path = os.path.join(temp_dir, f"dist_tile_{y}_{x}.jpg")
                    
                    cv2.imwrite(ref_tile_path, ref_tile)
                    cv2.imwrite(dist_tile_path, dist_tile)
                    
                    # Free memory
                    del ref_tile, dist_tile
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    # Process tile
                    try:
                        # Use mixed precision if available
                        if self.amp_enabled:
                            with torch.cuda.amp.autocast():
                                with torch.no_grad():
                                    score = self.metric(dist_tile_path, ref_tile_path)
                                    tile_score = score.item()
                        else:
                            with torch.no_grad():
                                score = self.metric(dist_tile_path, ref_tile_path)
                                tile_score = score.item()
                        
                        # Calculate weight based on tile area
                        tile_weight = (y_end - y) * (x_end - x)
                        
                        scores.append(tile_score)
                        weights.append(tile_weight)
                        
                    except Exception as e:
                        print(f"Error processing tile at ({x},{y}): {e}")
                    
                    # Clean up
                    os.remove(ref_tile_path)
                    os.remove(dist_tile_path)
                    torch.cuda.empty_cache()
                    
                    # Print progress
                    print(f"Processed tile at ({x},{y}) to ({x_end},{y_end})")
            
            # Calculate weighted average
            if scores:
                weighted_score = np.average(scores, weights=weights)
                return weighted_score
            else:
                return None
                
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                try:
                    os.rmdir(temp_dir)
                except:
                    pass
    
    def calculate_video_quality(self, ref_video, dist_video, frame_interval=1, max_frames=None):
        """Calculate quality metrics between reference and distorted videos"""
        print(f"Comparing videos...")
        
        # Create temporary directories for frames
        ref_frames_dir = "temp_ref_frames"
        dist_frames_dir = "temp_dist_frames"
        
        os.makedirs(ref_frames_dir, exist_ok=True)
        os.makedirs(dist_frames_dir, exist_ok=True)
        
        try:
            # Open video captures
            ref_cap = cv2.VideoCapture(ref_video)
            dist_cap = cv2.VideoCapture(dist_video)
            
            if not ref_cap.isOpened() or not dist_cap.isOpened():
                print("Error: Could not open video files.")
                return [], [], 0
            
            # Get video info
            ref_frame_count = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            dist_frame_count = int(dist_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = min(ref_frame_count, dist_frame_count)
            
            if max_frames is not None:
                frame_count = min(frame_count, max_frames)
            
            # Process frames
            scores = []
            frame_indices = []
            
            for i in range(0, frame_count, frame_interval):
                # Skip to the correct frame position
                if i > 0 and frame_interval > 1:
                    ref_cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    dist_cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                
                # Read frames
                ref_ret, ref_frame = ref_cap.read()
                dist_ret, dist_frame = dist_cap.read()
                
                if not ref_ret or not dist_ret:
                    break
                
                # Save frames to disk temporarily
                ref_frame_path = os.path.join(ref_frames_dir, f"frame_{i:04d}.jpg")
                dist_frame_path = os.path.join(dist_frames_dir, f"frame_{i:04d}.jpg")
                
                cv2.imwrite(ref_frame_path, ref_frame)
                cv2.imwrite(dist_frame_path, dist_frame)
                
                # Clear frame variables to free memory
                del ref_frame, dist_frame
                gc.collect()
                
                # Process this frame
                print(f"\nProcessing frame {i}/{frame_count}")
                score = self.process_single_frame(ref_frame_path, dist_frame_path)
                
                if score is not None:
                    scores.append(score)
                    frame_indices.append(i)
                    print(f"Frame {i} score: {score:.6f}")
                
                # Remove temporary files immediately after processing
                os.remove(ref_frame_path)
                os.remove(dist_frame_path)
                
                # Force garbage collection
                gc.collect()
                torch.cuda.empty_cache()
            
            # Release video captures
            ref_cap.release()
            dist_cap.release()
            
            # Calculate average score
            avg_score = np.mean(scores) if scores else 0
            print(f"Average score: {avg_score:.6f}")
            
            return frame_indices, scores, avg_score
            
        finally:
            # Clean up temporary directories
            for dir_path in [ref_frames_dir, dist_frames_dir]:
                if os.path.exists(dir_path):
                    try:
                        # Remove any remaining files
                        for file in os.listdir(dir_path):
                            os.remove(os.path.join(dir_path, file))
                        os.rmdir(dir_path)
                    except:
                        pass

    def compare_multiple_pairs(self, video_pairs, frame_interval=1, max_frames=None, labels=None):
        """Compare multiple video pairs and return results for plotting"""
        results = []
        
        for i, (ref_video, dist_video) in enumerate(video_pairs):
            print(f"\nProcessing video pair {i+1}/{len(video_pairs)}")
            print(f"Reference: {ref_video}")
            print(f"Distorted: {dist_video}")
            
            # Process this video pair
            frame_indices, scores, avg_score = self.calculate_video_quality(
                ref_video, dist_video, 
                frame_interval=frame_interval,
                max_frames=max_frames
            )
            
            # Use provided label or generate a default one
            label = labels[i] if labels and i < len(labels) else f"Video {i+1}"
            
            results.append({
                'frame_indices': frame_indices,
                'scores': scores,
                'avg_score': avg_score,
                'label': label
            })
            
            print(f"Completed video pair {i+1}: Average score = {avg_score:.6f}")
        
        return results

    def plot_comparison_results(self, results, title="PieAPP Quality Comparison", save_path=None):
        """Plot comparison results for multiple video pairs on a single graph"""
        plt.figure(figsize=(12, 7))
        
        # Plot each result set
        for result in results:
            plt.plot(
                result['frame_indices'], 
                result['scores'], 
                label=f"{result['label']} (Avg: {result['avg_score']:.4f})"
            )
        
        plt.title(title)
        plt.xlabel("Frame Number")
        plt.ylabel("PieAPP Score")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        
        return plt

# Example usage
if __name__ == "__main__":
    # Initialize the memory-efficient PieAPP processor
    pieapp_processor = MemoryEfficientPieAPP(tile_size=1024, overlap=128)
    
    # Define video pairs to compare
    video_pairs = [
        ("/workspace/vidaio-subnet/original.mp4", "/workspace/vidaio-subnet/output.mp4"),
        ("/workspace/vidaio-subnet/original.mp4", "/workspace/vidaio-subnet/output2.mp4")
    ]
    
    # Define labels for the pairs
    labels = ["Method A", "Method B"]
    
    # Process all pairs
    results = pieapp_processor.compare_multiple_pairs(
        video_pairs,
        frame_interval=1,  # Process every 5th frame
        max_frames=100,    # Process up to 100 frames
        labels=labels
    )
    
    # Plot results
    plt_figure = pieapp_processor.plot_comparison_results(
        results,
        title="PieAPP Quality Comparison of Different Methods",
        save_path="pieapp_comparison.png"
    )
    
    plt_figure.show()
