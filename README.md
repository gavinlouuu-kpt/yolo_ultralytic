# Label Studio to YOLO Conversion Project

This project helps convert Label Studio segmentation annotations to:
1. YOLO format bounding boxes for object detection training
2. Binary masks for segmentation training

Both formats include handling of blank images (without annotations) to help reduce false positives during training.

## Prerequisites

1. Python environment with required packages:
```bash
pip install -r requirements.txt
```

2. Label Studio export data:
   - Segmentation masks in RLE format
   - JSON annotation file
   - Original images

## Project Structure

```
project/
├── LS_data/                      # Label Studio exported data
│   └── project-name/             # Your project directory
│       ├── images/              # Original images
│       └── project-*.json       # Annotation file
├── dataset/                     # Generated YOLO dataset (object detection)
│   ├── train/
│   │   ├── images/             # Training images
│   │   └── labels/             # YOLO format annotations
│   └── val/
│       ├── images/             # Validation images
│       └── labels/             # YOLO format annotations
├── dataset_seg/                 # Generated segmentation dataset
│   ├── train/
│   │   ├── images/             # Training images
│   │   └── masks/              # Binary segmentation masks
│   └── val/
│       ├── images/             # Validation images
│       └── masks/              # Binary segmentation masks
├── dataset_review/             # Visualization for review
│   ├── detection/              # Object detection visualizations
│   │   ├── train/             # Training images with bbox overlay
│   │   └── val/               # Validation images with bbox overlay
│   └── segmentation/          # Segmentation visualizations
│       ├── train/             # Training images with mask overlay
│       └── val/               # Validation images with mask overlay
├── utils/
│   └── parse_label_json.py     # Utility functions
├── convert_ls_to_yolo.py       # Object detection conversion
├── convert_ls_to_seg.py        # Segmentation conversion
├── data.yaml                   # YOLO configuration
├── data_seg.yaml               # Segmentation configuration
└── README.md
```

## Usage Steps

1. **Prepare Your Data**:
   - Export your Label Studio project with segmentation annotations
   - Place the exported data in the `LS_data` directory
   - The script expects the following structure:
     ```
     LS_data/
     └── your-project-name/
         ├── images/
         └── project-*.json
     ```

2. **Run Conversion**:

   For Object Detection (YOLO format):
   ```bash
   # Basic usage (uses default 80/20 train/val split)
   python convert_ls_to_yolo.py your-project-name

   # Custom split ratio (e.g., 90/10 split)
   python convert_ls_to_yolo.py your-project-name --split 0.9
   ```

   For Segmentation:
   ```bash
   # Basic usage (uses default 80/20 train/val split)
   python convert_ls_to_seg.py your-project-name

   # Custom split ratio (e.g., 90/10 split)
   python convert_ls_to_seg.py your-project-name --split 0.9
   ```

3. **Review the Conversion**:
   
   For Object Detection:
   - Check `dataset_review/detection/` directory
   - Images will show red bounding boxes
   - Blank images will have no boxes
   - Verify that boxes correctly encompass the objects

   For Segmentation:
   - Check `dataset_review/segmentation/` directory
   - Images will show mask overlays
   - Blank images will have no masks
   - Verify that masks correctly cover the objects

4. **Dataset Statistics**:
   Both scripts provide detailed statistics about your dataset:
   - Total number of images
   - Number of annotated vs blank images
   - Distribution in train/val splits
   - Number of annotations (boxes or masks)

## Training

### Object Detection with YOLO
```bash
yolo train model=yolov8n.pt data=data.yaml epochs=100 imgsz=640
```

Parameters explained:
- `model`: YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)
- `data`: Path to your data.yaml file
- `epochs`: Number of training epochs
- `imgsz`: Input image size (must be divisible by 32)

### Segmentation Training
The segmentation dataset follows a standard format compatible with most segmentation frameworks:
- Images in PNG format
- Binary masks in PNG format (0 for background, 255 for object)
- Clear train/val split
- Configuration in data_seg.yaml

## Notes

- Both conversion processes include handling of blank images
- All images are processed regardless of their original format
- Multiple objects per image are supported
- The scripts provide detailed logging of the conversion process
- Paths in YAML files are automatically set relative to the project directory
- Object detection includes padding around bounding boxes
- Segmentation masks maintain original RLE precision 