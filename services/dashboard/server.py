import json
import requests
from typing import List, Dict, Any, Union
from datetime import datetime
from loguru import logger
from services.dashboard.model import MinerData
from vidaio_subnet_core import CONFIG

config = CONFIG.dashboard

def send_data_to_dashboard(
    data: Union[MinerData, Dict[str, Any]], 
    dashboard_endpoint: str = config.endpoint,
    timeout: int = config.timeout,
    max_retries: int = config.max_retries
) -> bool:
    """
    Send request data to the dashboard endpoint.
    
    Args:
        data: The request data to send, either as a MinerData object or a dictionary
        dashboard_url: The URL of the dashboard endpoint
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        
    Returns:
        bool: True if the data was successfully sent, False otherwise
    """
    # Convert to dictionary if data is a MinerData object
    if isinstance(data, MinerData):
        payload = data.to_dict()
    else:
        payload = data
    
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "MinerDataSender/1.0"
    }
    
    # Try to send the data with retries
    for attempt in range(max_retries):
        try:
            # logger.info(f"Sending data to dashboard (attempt {attempt + 1}/{max_retries})")
            response = requests.post(
                dashboard_endpoint,
                headers=headers,
                data=json.dumps(payload),
                timeout=timeout
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully sent data to dashboard. Response: {response.text}")
                return True
            
            else:
                logger.warning(f"Failed to send data. Status code: {response.status_code}, Response: {response.text}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
        
        if attempt < max_retries - 1:
            logger.info(f"Retrying in 2 seconds...")
            import time
            time.sleep(2)
    
    logger.error(f"Failed to send data after {max_retries} attempts")
    return False


def main():
    """Example usage of the send_data_to_dashboard function"""
    # Example data
    sample_data = MinerData(
        validator_uid=0,
        validator_hotkey="0x1234567890abcdef1234567890abcdef12345678",
        miner_uids=[101, 102, 103],
        miner_hotkeys=[
            "0xabcdef1234567890abcdef1234567890abcdef12",
            "0xbcdef1234567890abcdef1234567890abcdef123",
            "0xcdef1234567890abcdef1234567890abcdef1234"
        ],
        timestamp=datetime.now().isoformat(),
        request_type="test dashboard api",
        vmaf_scores=[95.2, 87.6, 92.1],
        pieapp_scores=[88.7, 79.3, 85.9],
        final_scores=[91.95, 83.45, 89.0],
        accumulate_scores=[450.2, 320.7, 410.5],
        status=["completed", "completed", "completed"],
        task_urls = ["https://storage.example.com/tasks/image123.jpg"] * 3,
        p_time=[10.0, 15.0, 12.0],
        processed_urls=[
            "https://storage.example.com/processed/image123_miner101.jpg",
            "https://storage.example.com/processed/image123_miner102.jpg",
            "https://storage.example.com/processed/image123_miner103.jpg"
        ]
    )
    
    # Configuration
    # dashboard_url = "https://dashboard-api.example.com/requests"
    
    # Send the data
    success = send_data_to_dashboard(sample_data)
    if success:
        logger.info("Data successfully sent to dashboard")
    else:
        logger.info("Failed to send data to dashboard")


if __name__ == "__main__":
    main()