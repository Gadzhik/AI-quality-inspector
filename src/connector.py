import requests
import json
from .logger import setup_logger

logger = setup_logger(name="TelemetryConnector")

class TelemetryConnector:
    """
    Class to connect to the telemetry system and report defects.
    """
    def __init__(self, endpoint_url: str = "http://localhost:8080/telemetry/defect"):
        """
        Initialize the connector.
        
        Args:
            endpoint_url: The URL of the telemetry defect reporting endpoint.
        """
        self.endpoint_url = endpoint_url
        logger.info(f"Initialized TelemetryConnector with endpoint: {self.endpoint_url}")

    def send_defect(self, data: dict):
        """
        Send a defect report to the telemetry system.
        
        Args:
            data: The defect data (e.g., {"defect": True, "confidence": 95}).
        """
        try:
            headers = {'Content-Type': 'application/json'}
            # In a real scenario, we might want to send more info like timestamp, camera ID, etc.
            payload = json.dumps(data)
            
            # Using a simplified synchronous request here as per requirements, 
            # though async could be beneficial for high throughput.
            # Assuming the endpoint is responsive.
            response = requests.post(self.endpoint_url, data=payload, headers=headers, timeout=5)
            
            if response.status_code == 200 or response.status_code == 201:
                logger.info(f"Successfully sent defect report to Telemetry: {data}")
            else:
                logger.error(f"Failed to send defect report. Status: {response.status_code}, Response: {response.text}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending defect report to Telemetry: {e}")
