import os
import pandas as pd
from datetime import datetime

def clear_test_logs():
    """Remove test entries from log files."""
    log_dir = "./output/logs/metadata"
    
    if not os.path.exists(log_dir):
        print("No log directory found.")
        return
    
    for filename in os.listdir(log_dir):
        if filename.startswith("detailed_log_") and filename.endswith(".csv"):
            filepath = os.path.join(log_dir, filename)
            
            try:
                # Read the CSV
                df = pd.read_csv(filepath)
                
                # Remove test entries
                df_clean = df[~df['video_file'].str.contains('test_video', na=False)]
                df_clean = df_clean[df_clean['scene_number'] != 999]
                
                # If we have real data, save it back
                if len(df_clean) > 0:
                    df_clean.to_csv(filepath, index=False)
                    print(f"✅ Cleaned {filepath} - removed {len(df) - len(df_clean)} test entries")
                else:
                    # If only test data, create a new file with just headers
                    df.iloc[0:0].to_csv(filepath, index=False)
                    print(f"✅ Reset {filepath} - was only test data")
                    
            except Exception as e:
                print(f"❌ Error cleaning {filepath}: {e}")

if __name__ == "__main__":
    clear_test_logs()