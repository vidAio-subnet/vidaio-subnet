import json
import requests
from typing import List, Dict, Any, Union
from datetime import datetime
from loguru import logger
from services.dashboard.model import UpscalingMinerData, CompressionMinerData
from vidaio_subnet_core import CONFIG

config = CONFIG.dashboard

def send_upscaling_data_to_dashboard(
    data: UpscalingMinerData, 
    dashboard_endpoint: str = config.endpoint,
    timeout: int = config.timeout,
    max_retries: int = config.max_retries
) -> bool:
    """
    Send upscaling data to the dashboard endpoint.
    
    Args:
        data: The upscaling data to send
        dashboard_endpoint: The URL of the dashboard endpoint
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        
    Returns:
        bool: True if the data was successfully sent, False otherwise
    """
    return _send_data_to_dashboard(data, dashboard_endpoint, timeout, max_retries, "upscaling")

def send_compression_data_to_dashboard(
    data: CompressionMinerData, 
    dashboard_endpoint: str = config.endpoint,
    timeout: int = config.timeout,
    max_retries: int = config.max_retries
) -> bool:
    """
    Send compression data to the dashboard endpoint.
    
    Args:
        data: The compression data to send
        dashboard_endpoint: The URL of the dashboard endpoint
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        
    Returns:
        bool: True if the data was successfully sent, False otherwise
    """
    return _send_data_to_dashboard(data, dashboard_endpoint, timeout, max_retries, "compression")

def _send_data_to_dashboard(
    data: Union[UpscalingMinerData, CompressionMinerData], 
    dashboard_endpoint: str,
    timeout: int,
    max_retries: int,
    task_type: str
) -> bool:
    """
    Internal function to send data to the dashboard endpoint.
    
    Args:
        data: The data to send
        dashboard_endpoint: The URL of the dashboard endpoint
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        task_type: The type of processing task for logging
        
    Returns:
        bool: True if the data was successfully sent, False otherwise
    """
    if isinstance(data, UpscalingMinerData):
        payload = data.to_dict()

    if isinstance(data, CompressionMinerData):
        payload = data.to_dict()

    payload = data
    
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "MinerDataSender/1.0"
    }
    
    for attempt in range(max_retries):
        try:
            # logger.info(f"Sending {task_type} data to dashboard (attempt {attempt + 1}/{max_retries})")
            response = requests.post(
                dashboard_endpoint,
                headers=headers,
                data=json.dumps(payload),
                timeout=timeout
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully sent {task_type} data to dashboard")
                return True
            
            else:
                logger.warning(f"Failed to send {task_type} data. Status code: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {task_type}. Check dashboard configuration")
        
        if attempt < max_retries - 1:
            logger.info(f"Retrying {task_type} data in 2 seconds...")
            import time
            time.sleep(2)
    
    logger.error(f"Failed to send {task_type} data after {max_retries} attempts")
    return False

def main():
    """Example usage of the send_data_to_dashboard functions"""
    
    # Example 1: Upscaling data
    upscaling_data = UpscalingMinerData(
        validator_uid=0,
        validator_hotkey="0x1234567890abcdef1234567890abcdef12345678",
        request_type="video",
        miner_uids=[211, 102, 103],
        miner_hotkeys=[
            "0xabcdef1234567890abcdef1234567890abcdef12",
            "0xbcdef1234567890abcdef1234567890abcdef123",
            "0xcdef1234567890abcdef1234567890abcdef1234"
        ],
        timestamp=datetime.now().isoformat(),
        vmaf_scores=[95.2, 87.6, 92.1],
        final_scores=[91.95, 83.45, 89.0],
        accumulate_scores=[7650.2, 320.7, 410.5],
        applied_multipliers = [201, 101, 102],
        status=["completed", "completed", "completed"]
    )
    
    # Example 2: Compression data
    compression_data = CompressionMinerData(
        validator_uid=0,
        validator_hotkey="0x876543210fedcba9876543210fedcba987654321",
        request_type="video",
        miner_uids=[222, 202],
        miner_hotkeys=[
            "0xfedcba9876543210fedcba9876543210fedcba98",
            "0xedcba9876543210fedcba9876543210fedcba987"
        ],
        timestamp=datetime.now().isoformat(),
        vmaf_scores=[88.5, 92.3],
        compression_rates=[0.65, 0.72],
        final_scores=[86.85, 91.0],
        accumulate_scores=[1234.1, 420.8],
        applied_multipliers = [9999, 107],
        status=["completed", "completed"]
    )
    
    # Send upscaling data
    logger.info("Sending upscaling data...")
    upscaling_success = send_upscaling_data_to_dashboard(upscaling_data)
    if upscaling_success:
        logger.info("Upscaling data successfully sent to dashboard")
    else:
        logger.error("Failed to send upscaling data to dashboard")
    
    # Send compression data
    logger.info("Sending compression data...")
    compression_success = send_compression_data_to_dashboard(compression_data)
    if compression_success:
        logger.info("Compression data successfully sent to dashboard")
    else:
        logger.error("Failed to send compression data to dashboard")


if __name__ == "__main__":
    main()
