# AutoAnnotation: A Pipeline for Automated Video Annotation with Manual Refinement

## Overview

AutoAnnotation is a powerful pipeline designed to significantly speed up the process of annotating objects in video streams, particularly for recycling facilities or similar scenarios. It combines state-of-the-art computer vision models to automatically generate initial bounding box and segmentation mask annotations, which can then be efficiently reviewed and corrected using CVAT.

The pipeline leverages:
- DINO-X (via DDS Cloud API) for robust object detection
  - Note: DINO-X API provides 20 yen free credits upon sign-up, after which usage is paid
  - Consider using test mode when running the code to optimize API credit usage
- SAM2 (Segment Anything Model 2) for high-quality segmentation masks
- CVAT for intuitive manual correction and refinement

For detailed information about the core annotation mechanism and DINO-X integration, please refer to the [Grounded-SAM-2 repository](https://github.com/IDEA-Research/Grounded-SAM-2), particularly the [DINO-X demo section](https://github.com/IDEA-Research/Grounded-SAM-2?tab=readme-ov-file#grounded-sam-2-image-demo-with-dino-x).

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Workflow Guide](#workflow-guide)
   - [Input Data Preparation](#input-data-preparation)
   - [Frame Extraction](#frame-extraction)
   - [Automated Annotation](#automated-annotation)
   - [Preparing for CVAT](#preparing-for-cvat)
   - [Manual Correction in CVAT](#manual-correction-in-cvat)
5. [Configuration Guide](#configuration-guide)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)
8. [Contributing](#contributing)

## System Requirements

### Hardware Requirements
- NVIDIA GPU with CUDA support (recommended for SAM2)
- Minimum 16GB RAM
- Sufficient disk space for:
  - Video files
  - Extracted frames (can be large)
  - Model checkpoints
  - Annotation files

### Software Requirements
- Linux operating system (tested on Ubuntu 20.04 LTS)
- Python 3.10 or later
- CUDA 12.1 or later
- Git

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/AutoAnnotation.git
   cd AutoAnnotation
   ```

2. **Set Up Python Environment**
   ```bash
   # Create and activate a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   # or
   .\venv\Scripts\activate  # On Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Grounded-SAM-2**
   ```bash
   # Clone the Grounded-SAM-2 repository
   git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
   cd Grounded-SAM-2
   
   # Install SAM2 (required for segmentation)
   pip install -e .
   
   # Download SAM2 checkpoints (required)
   cd checkpoints
   bash download_ckpts.sh
   cd ../gdino_checkpoints
   bash download_ckpts.sh
   cd ../..
   ```

5. **Set Up API Access**
   - Create a `.env` file in the project root:
     ```bash
     touch .env
     ```
   - Add your DDS Cloud API token (note: the variable name must be `API_TOKEN`):
     ```
     API_TOKEN=your_token_here
     ```
   - Get your API token from: https://deepdataspace.com/request_api

## Project Structure

```
AutoAnnotation/
├── scripts/                        # Python scripts for the pipeline
│   ├── extract_frames.py          # Frame extraction script
│   ├── nested_coco_sam_dinox_v2.py # Main annotation script
│   └── merge_coco_json.py         # COCO JSON merging script
│
├── input_videos/                  # Input video directory (create this)
│   ├── #1/                       # Camera/source 1
│   │   └── video1.MP4
│   └── #2/                       # Camera/source 2
│       └── videoA.MP4
│
├── frames/                        # Extracted frames (created automatically)
│   ├── #1/                       # Mirrors input_videos structure
│   │   └── video1_frame_0000.png
│   └── #2/
│       └── videoA_frame_0000.png
│
├── outputs/                       # Annotation outputs (created automatically)
│   ├── annotations/              # Per-frame COCO JSON annotations
│   │   ├── #1/
│   │   │   └── video1_frame_0000_coco_annotation.json
│   │   └── #2/
│   └── visualizations/           # Optional visualization images
│
├── annotations_per_video/         # Merged annotations (created automatically)
│   ├── #1_video1_coco_merged.json
│   └── #2_videoA_coco_merged.json
│
├── checkpoints/                   # Model checkpoints (created during setup)
│   ├── sam2/                     # SAM2 model weights
│   └── gdino_checkpoints/        # Grounding DINO model weights
│
├── sam2/                         # SAM2 model code (created during setup)
│   └── ...                      # Model implementation files
│
├── data/                         # Additional data files (if any)
│
├── slurm_logs/                   # Log files for SLURM jobs (if using cluster)
│
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup file
└── .env                         # API tokens (create this, not in git)
```

### Directory Descriptions

1. **User-Created Directories/Files** (you need to create these):
   - `input_videos/`: Place your input video files here
   - `.env`: Create this file to store your API tokens

2. **Automatically Created Directories** (created by the scripts):
   - `frames/`: Created by `extract_frames.py`
   - `outputs/`: Created by `nested_coco_sam_dinox_v2.py`
   - `annotations_per_video/`: Created by `merge_coco_json.py`

3. **Setup-Created Directories** (created during installation):
   - `checkpoints/`: Created when downloading model weights
   - `sam2/`: Created when installing SAM2
   - `slurm_logs/`: Created if using SLURM for cluster computing

Note: All automatically created directories will be generated when you run the respective scripts. You only need to create the `input_videos/` directory and the `.env` file before starting the pipeline.

## Workflow Guide

### Input Data Preparation

1. **Video File Organization**
   - Place your input videos in the `input_videos` directory
   - Organize videos by camera/source in subdirectories
   - Supported format: MP4 (other formats may work but untested)
   - Example structure:
     ```
     input_videos/
     ├── #1/
     │   ├── video1.MP4
     │   └── video2.MP4
     └── #2/
         ├── videoA.MP4
         └── videoB.MP4
     ```

### Frame Extraction

1. **Configure Frame Extraction**
   - Open `scripts/extract_frames.py`
   - Set the following parameters:
     ```python
     INPUT_ROOT_DIR = "input_videos"  # Input video directory
     OUTPUT_ROOT_DIR = "frames"        # Output frames directory
     FRAME_INTERVAL_N = 30             # Extract every Nth frame
     IMAGE_FORMAT = "png"              # Output image format
     VIDEO_EXTENSION = ".MP4"          # Input video extension
     ```

2. **Run Frame Extraction**
   ```bash
   python scripts/extract_frames.py
   ```
   - This will create a mirrored directory structure in `frames/`
   - Example output: `frames/#1/video1_frame_0000.png`

### Automated Annotation

1. **Configure Annotation Script**
   - Open `scripts/nested_coco_sam_dinox_v2.py`
   - Set key parameters:
     ```python
     # Core detection parameters
     TEXT_PROMPT = ["bottle", "can", "box"]  # Object classes to detect
     BOX_THRESHOLD = 0.35                    # Detection confidence threshold
     IOU_THRESHOLD = 0.5                     # IoU threshold for NMS
     
     # Processing parameters
     FRAMES_DIR = "frames"                   # Input frames directory
     FRAME_GLOB_PATTERN = "*.png"           # Frame file pattern
     WITH_SLICE_INFERENCE = True            # Enable for high-res images
     OUTPUT_DIR_BASE = "outputs"            # Output directory
     
     # Test mode parameters (recommended for parameter tuning)
     TEST_MODE = True                       # Enable test mode
     NUM_TEST_FRAMES = 10                   # Number of frames to process in test mode
     ```

2. **Parameter Tuning**
   - The annotation quality depends heavily on parameter configuration
   - We recommend using test mode (`TEST_MODE = True`) to experiment with different parameters
   - Key parameters to tune:
     - `TEXT_PROMPT`: Be specific and descriptive with object classes
       - Example: "clear plastic bottle" instead of just "bottle"
       - Multiple prompts can be combined: ["clear plastic bottle", "colored plastic bottle"]
     - `BOX_THRESHOLD`: Controls detection confidence
       - Higher values (e.g., 0.5) reduce false positives but may miss objects
       - Lower values (e.g., 0.3) catch more objects but may include false positives
     - `IOU_THRESHOLD`: Controls overlap between detections
       - Lower values (e.g., 0.3) for crowded scenes with overlapping objects
       - Higher values (e.g., 0.7) for scenes with well-separated objects
     - `WITH_SLICE_INFERENCE`: Recommended for high-resolution images
       - Helps detect small objects in large images
       - Increases processing time but improves detection quality
   - Test different parameter combinations on a small subset of frames
   - Monitor API credit usage during testing
   - Once optimal parameters are found, disable test mode for full processing

3. **Run Automated Annotation**
   ```bash
   python scripts/nested_coco_sam_dinox_v2.py
   ```
   - This will generate COCO JSON annotations for each frame
   - Output structure: `outputs/annotations/#1/video1_frame_0000_coco_annotation.json`
   - In test mode, only processes `NUM_TEST_FRAMES` frames to save API credits

### Preparing for CVAT

1. **Configure Merging Script**
   - Open `scripts/merge_coco_json.py`
   - Set parameters:
     ```python
     NESTED_PER_FRAME_ANNOTATIONS_ROOT = "outputs/annotations"
     ALL_FRAMES_IMAGES_ROOT = "frames"
     OUTPUT_CVAT_READY_MERGED_DIR = "annotations_per_video"
     ```

2. **Run Merging Script**
   ```bash
   python scripts/merge_coco_json.py
   ```
   - This creates one COCO JSON file per video
   - Output: `annotations_per_video/#1_video1_coco_merged.json`

### Manual Correction in CVAT

1. **Create CVAT Project**
   - Log in to your CVAT instance
   - Create a new project
   - Create a new task within the project
   - Set task type to "Instance Segmentation" (required for segmentation masks)

2. **Upload Data**
   - Upload all frames from one video as a task
   - Upload the corresponding merged COCO JSON file
   - Ensure frame filenames match those in the COCO JSON

3. **Correction Workflow**
   - Review automatically generated annotations
   - Use CVAT tools to:
     - Adjust bounding boxes
     - Refine segmentation masks
     - Correct class labels
     - Add missing annotations
     - Remove false positives

4. **Export Corrected Annotations**
   - Export in COCO 1.0 format
   - Save to a new directory (e.g., `corrected_annotations_cvat_export`)

## Configuration Guide

### Frame Extraction Parameters
- `FRAME_INTERVAL_N`: Higher values reduce processing time but may miss fast-moving objects
- `IMAGE_FORMAT`: PNG recommended for quality, JPG for space efficiency

### Annotation Parameters
- `TEXT_PROMPT`: Be specific and consistent with object classes
- `BOX_THRESHOLD`: Higher values (e.g., 0.5) reduce false positives but may miss objects
- `WITH_SLICE_INFERENCE`: Enable for high-resolution images with small objects
- `IOU_THRESHOLD`: Adjust based on object density (lower for crowded scenes)

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**
   - Ensure virtual environment is activated
   - Verify all dependencies are installed
   - Check Grounded-SAM-2 installation

2. **API Token Errors**
   - Verify `.env` file exists and contains valid token
   - Check token expiration
   - Ensure proper API access

3. **CUDA/GPU Issues**
   - Verify CUDA installation
   - Check GPU memory usage
   - Consider reducing batch size or image resolution

4. **File Not Found Errors**
   - Verify directory structure matches documentation
   - Check file permissions
   - Ensure consistent file extensions

5. **CVAT Import Issues**
   - Verify frame filenames match COCO JSON
   - Check JSON format validity
   - Ensure proper task type selection

### Performance Optimization

1. **Processing Speed**
   - Adjust `FRAME_INTERVAL_N` based on video content
   - Use appropriate image format and resolution
   - Consider batch processing for large datasets

2. **Detection Quality**
   - Fine-tune `TEXT_PROMPT` for specific objects
   - Adjust `BOX_THRESHOLD` and `IOU_THRESHOLD`
   - Enable `WITH_SLICE_INFERENCE` for high-res images

## Best Practices

1. **API Credit Management**
   - Start with test mode to experiment with parameters
   - Use a small subset of frames for initial testing
   - Monitor API credit usage through the DDS Cloud dashboard
   - Keep track of successful parameter combinations for different scenarios

2. **Video Quality**
   - Use high-resolution videos when possible
   - Ensure good lighting conditions
   - Minimize motion blur
   - Consider video preprocessing if needed (e.g., stabilization, denoising)

3. **Annotation Efficiency**
   - Start with conservative detection thresholds
   - Review a sample of frames before full processing
   - Use CVAT's AI-assisted tools when available
   - Document successful parameter combinations for different object types

4. **Data Organization**
   - Maintain consistent naming conventions
   - Keep clear directory structure
   - Regular backups of annotations
   - Track which parameters were used for each annotation batch

## Contributing

We welcome contributions to improve AutoAnnotation! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

For major changes, please open an issue first to discuss proposed improvements.

## License

[Add your chosen license here]

## Acknowledgments

- [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) for the core detection and segmentation models
  - See their [DINO-X demo section](https://github.com/IDEA-Research/Grounded-SAM-2?tab=readme-ov-file#grounded-sam-2-image-demo-with-dino-x) for detailed information about the annotation mechanism
- [CVAT](https://github.com/opencv/cvat) for the annotation interface
- [DDS Cloud API](https://deepdataspace.com) for DINO-X access
  - Note: 20 yen free credits upon sign-up, paid usage thereafter
  - Visit their [pricing page](https://deepdataspace.com/pricing) for current rates
