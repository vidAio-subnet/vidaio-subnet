import random

def get_video_segment_url(video_url: str, duration: int, segment_length: int = 30) -> str:
    """
    Generate a video URL with a random time window based on the video duration.
    
    Args:
        video_url (str): Original video URL
        duration (int): Total duration of the video in seconds
        segment_length (int, optional): Length of desired video segment in seconds. Defaults to 30.
    
    Returns:
        str: Video URL with time window parameters
    """
    if duration <= segment_length:
        return video_url
        
    # Calculate maximum start time to ensure we don't exceed video duration
    max_start_time = duration - segment_length
    
    # Generate random start time
    start_time = random.randint(0, max_start_time)
    end_time = start_time + segment_length
    
    # Add time parameters to URL
    if '?' in video_url:
        time_window_url = f"{video_url}&t_start={start_time}&t_end={end_time}"
    else:
        time_window_url = f"{video_url}?t_start={start_time}&t_end={end_time}"
    
    return time_window_url
