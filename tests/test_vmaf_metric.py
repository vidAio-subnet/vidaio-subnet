import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import os

def parse_vmaf_xml(xml_file):
    """
    Parse the VMAF XML file to extract frame-level VMAF scores.
    
    Args:
        xml_file (str): Path to the VMAF XML output file.
        
    Returns:
        tuple: (frame_numbers, vmaf_scores)
    """
    try:
        # Check if file exists
        if not os.path.exists(xml_file):
            print(f"Error: File {xml_file} not found")
            return [], []
        
        # Parse XML file
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Extract frame information
        frames = root.findall('.//frame')
        
        frame_numbers = []
        vmaf_scores = []
        
        for frame in frames:
            frame_num = int(frame.get('frameNum'))
            vmaf_score = float(frame.get('vmaf'))
            
            frame_numbers.append(frame_num)
            vmaf_scores.append(vmaf_score)
        
        return frame_numbers, vmaf_scores
    
    except Exception as e:
        print(f"Error parsing XML file {xml_file}: {e}")
        return [], []

def compare_vmaf_scores(xml_file1, xml_file2, label1="Method 1", label2="Method 2"):
    """
    Compare VMAF scores from two XML files.
    
    Args:
        xml_file1 (str): Path to the first VMAF XML output file.
        xml_file2 (str): Path to the second VMAF XML output file.
        label1 (str): Label for the first method.
        label2 (str): Label for the second method.
    """
    # Parse VMAF results
    frames1, scores1 = parse_vmaf_xml(xml_file1)
    frames2, scores2 = parse_vmaf_xml(xml_file2)
    
    if not frames1 or not frames2:
        print("Failed to parse one or both XML files.")
        return
    
    # Calculate statistics
    avg1 = np.mean(scores1) if scores1 else 0
    avg2 = np.mean(scores2) if scores2 else 0
    
    # Create visualization
    plt.figure(figsize=(14, 8))
    
    # Plot first method's VMAF scores
    if frames1 and scores1:
        plt.plot(frames1, scores1, 'b-', marker='o', alpha=0.7, label=f'{label1} (Avg: {avg1:.2f})')
        plt.axhline(y=avg1, color='b', linestyle='--', alpha=0.5)
    
    # Plot second method's VMAF scores
    if frames2 and scores2:
        plt.plot(frames2, scores2, 'r-', marker='x', alpha=0.7, label=f'{label2} (Avg: {avg2:.2f})')
        plt.axhline(y=avg2, color='r', linestyle='--', alpha=0.5)
    
    # Add labels and title
    plt.title(f'VMAF Quality Comparison: {label1} vs {label2}')
    plt.xlabel('Frame Number')
    plt.ylabel('VMAF Score')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set y-axis to start from 0 and end at 100 (VMAF range)
    plt.ylim(0, 100)
    
    # Add statistics text boxes if we have data
    if scores1:
        stats1 = f"{label1} Stats:\nAvg: {avg1:.2f}\nMin: {min(scores1):.2f}\nMax: {max(scores1):.2f}"
        plt.annotate(stats1, xy=(0.02, 0.15), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.5", fc="lightskyblue", alpha=0.8))
    
    if scores2:
        stats2 = f"{label2} Stats:\nAvg: {avg2:.2f}\nMin: {min(scores2):.2f}\nMax: {max(scores2):.2f}"
        plt.annotate(stats2, xy=(0.02, 0.02), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.5", fc="lightcoral", alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Print comparison summary
    print("\nVMAF Comparison Summary:")
    print(f"{label1} Average VMAF: {avg1:.2f}")
    print(f"{label2} Average VMAF: {avg2:.2f}")
    
    if avg1 != 0 and avg2 != 0:
        diff = avg1 - avg2
        print(f"Difference: {diff:.2f}")
        if diff > 0:
            print(f"{label1} produced better quality output on average")
        elif diff < 0:
            print(f"{label2} produced better quality output on average")
        else:
            print("Both methods produced the same quality on average")

def plot_single_vmaf(xml_file, label="Video"):
    """
    Plot VMAF scores from a single XML file.
    
    Args:
        xml_file (str): Path to the VMAF XML output file.
        label (str): Label for the video.
    """
    # Parse VMAF results
    frames, scores = parse_vmaf_xml(xml_file)
    
    if not frames:
        print(f"Failed to parse XML file: {xml_file}")
        return
    
    # Calculate statistics
    avg = np.mean(scores)
    min_score = min(scores)
    max_score = max(scores)
    
    # Create visualization
    plt.figure(figsize=(14, 8))
    
    # Plot VMAF scores
    plt.plot(frames, scores, 'g-', marker='o', linewidth=2, alpha=0.8, label=f'VMAF (Avg: {avg:.2f})')
    
    # Add horizontal line for average
    plt.axhline(y=avg, color='g', linestyle='--', alpha=0.5)
    
    # Add labels and title
    plt.title(f'VMAF Quality Scores: {label}')
    plt.xlabel('Frame Number')
    plt.ylabel('VMAF Score')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set y-axis to start from 0 and end at 100 (VMAF range)
    plt.ylim(0, 100)
    
    # Add statistics text box
    stats = f"VMAF Stats:\nAvg: {avg:.2f}\nMin: {min_score:.2f}\nMax: {max_score:.2f}"
    plt.annotate(stats, xy=(0.02, 0.02), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", fc="lightgreen", alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\nVMAF Summary for {label}:")
    print(f"Average VMAF: {avg:.2f}")
    print(f"Minimum VMAF: {min_score:.2f}")
    print(f"Maximum VMAF: {max_score:.2f}")

# Example usage
if __name__ == "__main__":
    # For a single file analysis
    # plot_single_vmaf("/path/to/your/vmaf_output.xml", "Your Video")
    
    # For comparing two methods
    compare_vmaf_scores(
        "/workspace/vidaio-subnet/vmaf_video2x.xml", 
        "/workspace/vidaio-subnet/vmaf_ffmpeg.xml",
    )
    
    print("Run this script with your VMAF XML files as arguments")
    print("Example usage:")
    print("1. For single file: python vmaf_analyzer.py --single /path/to/vmaf.xml 'Video Label'")
    print("2. For comparison: python vmaf_analyzer.py --compare /path/to/method1.xml /path/to/method2.xml 'Method 1' 'Method 2'")
