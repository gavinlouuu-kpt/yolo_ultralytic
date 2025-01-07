# Label Studio to YOLO Conversion Project

This project helps convert Label Studio segmentation annotations to YOLO format bounding boxes for object detection training.

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
│   └── project-*/
│       ├── images/              # Original images
│       └── project-*.json       # Annotation file
├── dataset/                     # Generated YOLO dataset
│   ├── train/
│   │   ├── images/             # Training images
│   │   └── labels/             # YOLO format annotations
│   └── val/
│       ├── images/             # Validation images
│       └── labels/             # YOLO format annotations
├── dataset_review/             # Visualization for review
│   ├── train/                  # Training images with bbox overlay
│   └── val/                    # Validation images with bbox overlay
├── utils/
│   └── parse_label_json.py     # Utility functions
├── convert_ls_to_yolo.py       # Main conversion script
├── data.yaml                   # YOLO configuration file
└── README.md
```

## Usage Steps

1. **Prepare Your Data**:
   - Export your Label Studio project with segmentation annotations
   - Place the exported data in the `LS_data` directory
   - Note the paths to your:
     - JSON annotation file
     - Images directory

2. **Update Paths**:
   - Open `convert_ls_to_yolo.py`
   - Update the paths at the bottom of the file:
     ```python
     json_path = "LS_data/your-project/annotations.json"
     image_dir = "LS_data/your-project/images"
     ```

3. **Run Conversion**:
   ```bash
   python convert_ls_to_yolo.py
   ```
   This will:
   - Create YOLO format dataset in `dataset/`
   - Generate visualization in `dataset_review/`
   - Create `data.yaml` configuration file

4. **Review the Conversion**:
   - Check `dataset_review/` directory
   - Images will show red bounding boxes
   - Verify that boxes correctly encompass the objects

5. **Prepare for Training**:
   - The script automatically creates `data.yaml`
   - Dataset is split into train/val sets (default 80/20)
   - All paths are configured for YOLO training

## Training with YOLO

Once the conversion is complete, you can train using:
```bash
yolo train model=yolov8n.pt data=data.yaml epochs=100 imgsz=640
```

Parameters explained:
- `model`: YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)
- `data`: Path to your data.yaml file
- `epochs`: Number of training epochs
- `imgsz`: Input image size (must be divisible by 32)

## Notes

- The conversion process includes padding around the bounding boxes
- All images are processed regardless of their original format
- Multiple objects per image are supported
- The script provides detailed logging of the conversion process 