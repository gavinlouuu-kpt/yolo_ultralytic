import os
import json
import numpy as np
import shutil
from pathlib import Path
import random
from PIL import Image
import logging
from utils.parse_label_json import LabelParser, get_bbox_from_mask

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_directory_structure():
    """Create YOLO directory structure"""
    dirs = ['dataset/train/images', 'dataset/train/labels',
            'dataset/val/images', 'dataset/val/labels']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def convert_annotations(json_path, image_dir, train_split=0.8):
    """Convert Label Studio annotations to YOLO format"""
    # Create directory structure
    create_directory_structure()
    
    # Load annotations
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    # Parse annotations using LabelParser
    masks_dict = LabelParser.parse_json(json_data, parse_by_image_number=False)
    logger.info(f"Processing {len(masks_dict)} images from JSON")
    
    # Get list of actual images in directory
    existing_images = set(os.listdir(image_dir))
    logger.info(f"Found {len(existing_images)} images in directory")
    
    # Process each image
    valid_samples = []
    first_logged = False
    
    for filename, mask in masks_dict.items():
        if filename not in existing_images:
            logger.warning(f"Cannot find image file: {filename}")
            continue
            
        img_path = os.path.join(image_dir, filename)
        
        try:
            # Get bounding box from mask
            bbox = get_bbox_from_mask(mask, debug=not first_logged)
            if bbox is None:
                logger.warning(f"No valid bbox found for {filename}")
                continue
                
            valid_samples.append((img_path, mask, bbox))
            first_logged = True
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
            continue
    
    logger.info(f"Found {len(valid_samples)} valid samples")
    
    # Split into train/val
    random.shuffle(valid_samples)
    split_idx = int(len(valid_samples) * train_split)
    train_samples = valid_samples[:split_idx]
    val_samples = valid_samples[split_idx:]
    
    logger.info(f"Split: {len(train_samples)} training, {len(val_samples)} validation")
    
    # Process splits
    process_split(train_samples, 'train')
    process_split(val_samples, 'val')
    
    # Create data.yaml
    create_yaml(len(train_samples), len(val_samples))
    
    logger.info("Conversion completed successfully")

def process_split(samples, split):
    """Process a set of samples for a specific split (train/val)"""
    for img_path, mask, bbox in samples:
        # Copy image
        shutil.copy2(img_path, f'dataset/{split}/images/')
        
        # Create label file
        create_label_file(img_path, bbox, split)

def create_label_file(image_path, bbox, split):
    """Create YOLO format label file"""
    try:
        # Get image dimensions from mask
        image = Image.open(image_path)
        width, height = image.size
        
        # Convert bbox to YOLO format (normalized center coordinates and dimensions)
        xmin, ymin, xmax, ymax = bbox
        x_center = (xmin + xmax) / 2 / width
        y_center = (ymin + ymax) / 2 / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        
        # Write label file
        filename = os.path.splitext(os.path.basename(image_path))[0]
        label_path = f'dataset/{split}/labels/{filename}.txt'
        
        with open(label_path, 'w') as f:
            # Class index (0 for single class) followed by bbox coordinates
            f.write(f"0 {x_center} {y_center} {w} {h}\n")
            
    except Exception as e:
        logger.error(f"Error creating label file for {image_path}: {str(e)}")

def create_yaml(num_train, num_val):
    """Create data.yaml file"""
    yaml_content = f"""train: dataset/train/images
val: dataset/val/images
nc: 1  # number of classes
names: ['object']  # class names

# Dataset information
train_num: {num_train}
val_num: {num_val}
"""
    with open('data.yaml', 'w') as f:
        f.write(yaml_content)

if __name__ == "__main__":
    # Update these paths according to your setup
    json_path = "LS_data/project-3-at-2025-01-07-09-28-eb91e8e5/project-3-at-2025-01-07-09-29-eb91e8e5.json"
    image_dir = "LS_data/project-3-at-2025-01-07-09-28-eb91e8e5/images"
    
    convert_annotations(json_path, image_dir) 