import os
import json
import numpy as np
import shutil
from pathlib import Path
import random
from PIL import Image, ImageDraw
import logging
import glob
from utils.parse_label_json import LabelParser, get_bbox_from_mask

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_project_paths(project_name):
    """Find JSON and images paths for a given project name"""
    # Look for project directory
    project_dirs = glob.glob(f"LS_data/{project_name}*")
    if not project_dirs:
        raise ValueError(f"No project directory found matching: {project_name}")
    
    project_dir = project_dirs[0]  # Take the first matching directory
    logger.info(f"Found project directory: {project_dir}")
    
    # Find JSON file
    json_files = glob.glob(f"{project_dir}/*.json")
    if not json_files:
        raise ValueError(f"No JSON file found in {project_dir}")
    
    json_path = json_files[0]  # Take the first JSON file
    logger.info(f"Found annotation file: {json_path}")
    
    # Find images directory
    images_dir = os.path.join(project_dir, "images")
    if not os.path.exists(images_dir):
        raise ValueError(f"Images directory not found: {images_dir}")
    
    logger.info(f"Found images directory: {images_dir}")
    
    return json_path, images_dir

def create_directory_structure():
    """Create YOLO directory structure"""
    dirs = ['dataset/train/images', 'dataset/train/labels',
            'dataset/val/images', 'dataset/val/labels',
            'dataset_review/train', 'dataset_review/val']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def draw_bounding_boxes(image_path, bboxes, output_path):
    """Draw bounding boxes on image and save for review"""
    # Open image and convert to RGB to ensure color consistency
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    draw = ImageDraw.Draw(image)
    
    # Draw each bbox
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        # Draw rectangle with 2-pixel width red outline
        draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 0, 0), width=2)  # Pure red in RGB
    
    # Save the image with bounding boxes
    image.save(output_path)

def convert_annotations(project_name, train_split=0.8):
    """Convert Label Studio annotations to YOLO format"""
    # Find project paths
    json_path, image_dir = find_project_paths(project_name)
    
    # Create directory structure
    create_directory_structure()
    
    # Load annotations
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    # Parse annotations using LabelParser
    masks_dict = LabelParser.parse_json(json_data, parse_by_image_number=False)
    logger.info(f"Processing {len(masks_dict)} images with annotations from JSON")
    
    # Get list of all images in directory
    all_images = set(os.listdir(image_dir))
    logger.info(f"Found {len(all_images)} total images in directory")
    
    # Process each image
    valid_samples = []  # Images with annotations
    blank_samples = []  # Images without annotations
    first_logged = False
    
    # First process images with annotations
    for filename, masks in masks_dict.items():
        if filename not in all_images:
            logger.warning(f"Cannot find image file: {filename}")
            continue
            
        img_path = os.path.join(image_dir, filename)
        
        try:
            # Get bounding boxes from masks
            bboxes = []
            for mask in masks:
                bbox = get_bbox_from_mask(mask, debug=not first_logged)
                if bbox is not None:
                    bboxes.append(bbox)
            
            if bboxes:
                valid_samples.append((img_path, bboxes))
                first_logged = True
            else:
                logger.warning(f"No valid bboxes found for {filename}")
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
            continue
    
    # Then collect blank images (images without annotations)
    annotated_images = set(masks_dict.keys())
    blank_images = all_images - annotated_images
    
    for filename in blank_images:
        img_path = os.path.join(image_dir, filename)
        blank_samples.append((img_path, []))
    
    logger.info(f"Found {len(valid_samples)} samples with annotations")
    logger.info(f"Found {len(blank_samples)} blank samples")
    
    # Combine and shuffle all samples
    all_samples = valid_samples + blank_samples
    random.shuffle(all_samples)
    
    # Split into train/val
    split_idx = int(len(all_samples) * train_split)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    logger.info(f"Split: {len(train_samples)} training, {len(val_samples)} validation")
    logger.info(f"Training set: {sum(1 for _, bboxes in train_samples if bboxes)} with annotations, {sum(1 for _, bboxes in train_samples if not bboxes)} blank")
    logger.info(f"Validation set: {sum(1 for _, bboxes in val_samples if bboxes)} with annotations, {sum(1 for _, bboxes in val_samples if not bboxes)} blank")
    
    # Process splits
    process_split(train_samples, 'train')
    process_split(val_samples, 'val')
    
    # Create data.yaml
    create_yaml(len(train_samples), len(val_samples))
    
    logger.info("Conversion completed successfully")

def process_split(samples, split):
    """Process a set of samples for a specific split (train/val)"""
    for img_path, bboxes in samples:
        # Get filename
        filename = os.path.basename(img_path)
        
        # Copy image to dataset
        shutil.copy2(img_path, f'dataset/{split}/images/')
        
        # Create label file
        create_label_file(img_path, bboxes, split)
        
        # Create visualization for review
        review_path = f'dataset_review/{split}/{filename}'
        draw_bounding_boxes(img_path, bboxes, review_path)

def create_label_file(image_path, bboxes, split):
    """Create YOLO format label file"""
    try:
        # Get image dimensions
        image = Image.open(image_path)
        width, height = image.size
        
        # Create label file
        filename = os.path.splitext(os.path.basename(image_path))[0]
        label_path = f'dataset/{split}/labels/{filename}.txt'
        
        with open(label_path, 'w') as f:
            # Write each bbox
            for bbox in bboxes:
                # Convert bbox to YOLO format (normalized center coordinates and dimensions)
                xmin, ymin, xmax, ymax = bbox
                x_center = (xmin + xmax) / 2 / width
                y_center = (ymin + ymax) / 2 / height
                w = (xmax - xmin) / width
                h = (ymax - ymin) / height
                
                # Class index (0 for single class) followed by bbox coordinates
                f.write(f"0 {x_center} {y_center} {w} {h}\n")
            
    except Exception as e:
        logger.error(f"Error creating label file for {image_path}: {str(e)}")

def create_yaml(num_train, num_val):
    """Create data.yaml file"""
    # Get absolute path to current directory
    current_dir = os.path.abspath(os.path.dirname(__file__))
    
    yaml_content = f"""train: {os.path.join(current_dir, 'dataset/train/images')}
val: {os.path.join(current_dir, 'dataset/val/images')}
nc: 1  # number of classes
names: ['object']  # class names

# Dataset information
train_num: {num_train}
val_num: {num_val}
"""
    with open('data.yaml', 'w') as f:
        f.write(yaml_content)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert Label Studio annotations to YOLO format')
    parser.add_argument('project', help='Project name (directory name in LS_data/)')
    parser.add_argument('--split', type=float, default=0.8, help='Train/val split ratio (default: 0.8)')
    
    args = parser.parse_args()
    
    convert_annotations(args.project, args.split) 