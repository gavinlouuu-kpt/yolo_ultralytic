from typing import List, Dict
import numpy as np
import re
from label_studio_sdk.converter.brush import decode_from_annotation
import logging
import os

logger = logging.getLogger(__name__)


def get_bbox_from_mask(mask, padding=5, debug=False):
    """
    Get bounding box coordinates from mask with padding
    Args:
        mask: numpy array of shape (H, W)
        padding: number of pixels to pad around the mask
        debug: whether to print debug information
    Returns:
        tuple: (xmin, ymin, xmax, ymax) or None if no mask found
    """
    if debug:
        logger.info(f"Mask shape: {mask.shape}")
    
    # Find non-zero points
    y_indices, x_indices = np.nonzero(mask)
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        if debug:
            logger.warning("No non-zero points found in mask")
        return None
    
    # Get bounding box coordinates
    xmin = max(0, np.min(x_indices) - padding)
    ymin = max(0, np.min(y_indices) - padding)
    xmax = min(mask.shape[1], np.max(x_indices) + padding)
    ymax = min(mask.shape[0], np.max(y_indices) + padding)
    
    if debug:
        logger.info(f"Bbox coordinates: ({xmin}, {ymin}, {xmax}, {ymax})")
    
    return (xmin, ymin, xmax, ymax)

class LabelParser:
    @staticmethod
    def _get_image_number(filename: str) -> str:
        """Extract image number from filename."""
        match = re.search(r'image_(\d+)', filename)
        if match:
            return match.group(1)
        return filename

    @staticmethod
    def parse_json(json_data: List[Dict], parse_by_image_number: bool = False) -> Dict[str, List[np.ndarray]]:
        """
        Parse Label Studio JSON annotations into a dictionary of masks
        Args:
            json_data: List of annotation dictionaries from Label Studio
            parse_by_image_number: If True, use image number as key, else use filename
        Returns:
            Dict[str, List[np.ndarray]]: Dictionary mapping image filenames to lists of masks
        """
        masks_dict = {}
        
        for task in json_data:
            try:
                # Skip if no annotations
                if 'annotations' not in task or not task['annotations']:
                    continue
                
                # Get image filename
                image_filename = os.path.basename(task['data']['image'])
                key = LabelParser._get_image_number(image_filename) if parse_by_image_number else image_filename
                
                # Initialize list for this image's masks
                masks_dict[key] = []
                
                # Process annotations
                for annotation in task['annotations']:
                    for result in annotation['result']:
                        if 'value' in result and 'rle' in result['value']:
                            # Format result for the decoder
                            formatted_result = [{
                                'type': 'brushlabels',
                                'rle': result['value']['rle'],
                                'original_width': result['original_width'],
                                'original_height': result['original_height']
                            }]
                            
                            # Decode RLE to mask using label-studio-sdk
                            try:
                                mask_layers = decode_from_annotation('image', formatted_result)
                                # Get the first layer
                                if mask_layers:
                                    first_layer = next(iter(mask_layers.values()))
                                    mask = (first_layer > 0).astype(np.uint8)
                                    masks_dict[key].append(mask)
                            except Exception as e:
                                logger.error(f"Failed to decode mask for {key}: {str(e)}")
                                continue
            
            except Exception as e:
                logger.error(f"Error processing task: {str(e)}")
                continue
            
            # Remove images with no valid masks
            if not masks_dict[key]:
                del masks_dict[key]
        
        logger.info(f"Successfully parsed {len(masks_dict)} images with {sum(len(masks) for masks in masks_dict.values())} total masks")
        return masks_dict