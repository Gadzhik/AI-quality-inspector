import os
import cv2
import json
from datetime import datetime
from src.logger import setup_logger

logger = setup_logger(name="DatasetCollector")

class DatasetCollector:
    def __init__(self, base_dir="dataset"):
        self.base_dir = base_dir
        self.pos_dir = os.path.join(base_dir, "positive")
        self.neg_dir = os.path.join(base_dir, "negative")
        
        os.makedirs(self.pos_dir, exist_ok=True)
        os.makedirs(self.neg_dir, exist_ok=True)
        logger.info(f"DatasetCollector initialized. Saving to: {self.base_dir}")

    def save_positive(self, frame, metadata):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        basename = f"frame_{timestamp}"
        
        # Save original frame
        img_path = os.path.join(self.pos_dir, f"{basename}.jpg")
        cv2.imwrite(img_path, frame)
        
        # Ensure label matches
        metadata["label"] = "positive"
        metadata["timestamp"] = timestamp
        if "source" not in metadata:
            metadata["source"] = "manual_label"
        
        # Save metadata
        meta_path = os.path.join(self.pos_dir, f"{basename}.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)
            
        logger.info(f"Standard saved to positive/: {basename}")

    def save_negative(self, frame, metadata):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        basename = f"frame_{timestamp}"
        
        # Save original frame
        img_path = os.path.join(self.neg_dir, f"{basename}.jpg")
        cv2.imwrite(img_path, frame)
        
        # Ensure label matches
        metadata["label"] = "negative"
        metadata["timestamp"] = timestamp
        if "source" not in metadata:
            metadata["source"] = "manual_label"
        
        # Save metadata
        meta_path = os.path.join(self.neg_dir, f"{basename}.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)
            
        logger.info(f"Defect saved to negative/: {basename}")
