from dds_cloudapi_sdk import Config
from dds_cloudapi_sdk import Client
from dds_cloudapi_sdk.tasks.v2_task import V2Task

import os
import cv2
import json
import torch
import tempfile
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from dotenv import load_dotenv
from datetime import datetime
from tqdm import tqdm

env_path = "/home/vivora/AutoAnnotation/scripts/.env"
env_file = Path(env_path)
if not env_file.exists():
    print(f"Warning: Environment file '{env_path}' not found.")
load_dotenv(env_path)

"""
Hyper parameters
"""
API_TOKEN = os.environ.get("API_TOKEN")
TEXT_PROMPT = "paper . bottle . can . steel . plastic .food . waste . aluminium . cap . package . plastic bag ."
FRAMES_DIR = Path("/home/vivora/AutoAnnotation/frames")
FRAME_GLOB_PATTERN = "*.png"

SAM2_CHECKPOINT = "/home/vivora/AutoAnnotation/checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
BOX_THRESHOLD = 0.25
IOU_THRESHOLD = 0.8
WITH_SLICE_INFERENCE = False
SLICE_WH = (480, 480)
OVERLAP_RATIO = (0.2, 0.2)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_MODE = False
NUM_TEST_FRAMES = 1 # Process only one frame for testing

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR_BASE = Path("/home/vivora/AutoAnnotation/outputs/grounded_sam2_dinox_v2_coco_nested_demo") # New output name
OUTPUT_DIR = OUTPUT_DIR_BASE / timestamp
BASE_VISUALS_DIR = OUTPUT_DIR / "visualizations"
BASE_ANNOTATIONS_DIR = OUTPUT_DIR / "annotations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BASE_VISUALS_DIR.mkdir(parents=True, exist_ok=True)
BASE_ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

DUMP_JSON_RESULTS = True

"""
Initialize the DDS client and prepare class mappings for script internal use
"""
if not API_TOKEN:
    raise ValueError("API_TOKEN not found. Please set it in your environment or .env file.")
token = API_TOKEN
config = Config(token)
client = Client(config)

script_classes_list = [x.strip().lower() for x in TEXT_PROMPT.split('.') if x]
script_class_name_to_id = {name: id for id, name in enumerate(script_classes_list)}
script_id_to_class_name = {id: name for name, id in script_class_name_to_id.items()}

coco_categories = []
coco_class_name_to_category_id = {}
for i, class_name in enumerate(script_classes_list):
    category_id = i + 1
    coco_categories.append({"id": category_id, "name": class_name, "supercategory": "object"})
    coco_class_name_to_category_id[class_name] = category_id

"""
Find and Prepare Frame Files
"""
print(f"Searching for '{FRAME_GLOB_PATTERN}' files recursively in: {FRAMES_DIR}")
all_frame_files = sorted(list(FRAMES_DIR.rglob(FRAME_GLOB_PATTERN)))

if not all_frame_files:
    print(f"Error: No frame files found matching '{FRAME_GLOB_PATTERN}' in {FRAMES_DIR}.")
    exit()

if TEST_MODE:
    num_to_process = min(NUM_TEST_FRAMES, len(all_frame_files))
    frame_files_to_process = all_frame_files[:num_to_process]
    print(f"\n--- RUNNING IN TEST MODE ---")
    print(f"Processing the first {len(frame_files_to_process)} frames out of {len(all_frame_files)} found.")
else:
    frame_files_to_process = all_frame_files
    print(f"\n--- RUNNING IN FULL MODE ---")
    print(f"Processing all {len(frame_files_to_process)} found frames.")

print(f"\nLoading SAM2 model onto {DEVICE}...")

if DEVICE == "cuda" and torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("TF32 enabled for CUDA.")

with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=(DEVICE=="cuda")):
    sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)
print("SAM2 model loaded.")

"""
Main Annotation Loop
"""
print("\nStarting annotation process...")

for frame_loop_idx, frame_path_obj in enumerate(tqdm(frame_files_to_process, desc="Annotating Frames")):
    IMG_PATH = str(frame_path_obj)
    # For COCO JSON, each file is for one image, so image_id within that file can be 1.
    # If you merge them later, these will need to be globally unique.
    # For now, this '1' is the ID of the image *within this specific JSON file*.
    coco_image_id_in_file = 1

    # --- Determine relative path for saving outputs ---
    try:
        # frame_path_obj example: /home/vivora/AutoAnnotation/frames/#1/GX010012_MP4_frames/GX010012_MP4_frame_00000000.png
        # FRAMES_DIR example:     /home/vivora/AutoAnnotation/frames
        # relative_parent_dir will be: #1/GX010012_MP4_frames
        relative_parent_dir = frame_path_obj.parent.relative_to(FRAMES_DIR)
    except ValueError:
        tqdm.write(f"Warning: Could not determine relative path for {frame_path_obj}. Output might be flat.")
        specific_annotation_output_dir = BASE_ANNOTATIONS_DIR
        specific_visual_output_dir = BASE_VISUALS_DIR
    else:
        specific_annotation_output_dir = BASE_ANNOTATIONS_DIR / relative_parent_dir
        specific_visual_output_dir = BASE_VISUALS_DIR / relative_parent_dir

    specific_annotation_output_dir.mkdir(parents=True, exist_ok=True)
    specific_visual_output_dir.mkdir(parents=True, exist_ok=True)

    tqdm.write(f"\nProcessing frame {frame_loop_idx} ({frame_path_obj.relative_to(FRAMES_DIR)}): {IMG_PATH}")

    cv2_img = cv2.imread(IMG_PATH)
    if cv2_img is None:
        tqdm.write(f"Warning: Could not read image {IMG_PATH}. Skipping frame {frame_loop_idx}.")
        continue
    pil_image = Image.open(IMG_PATH)

    input_boxes_xyxy = np.array([])
    api_confidences = np.array([])
    api_class_ids_script_indexed = np.array([])
    api_class_names = []

    if WITH_SLICE_INFERENCE:
        def callback(image_slice: np.ndarray) -> sv.Detections:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
                temp_filename = tmpfile.name
            cv2.imwrite(temp_filename, image_slice)
            uploaded_image_url_slice = client.upload_file(temp_filename)
            task = V2Task(
                api_path="/v2/task/dinox/detection",
                api_body={
                    "model": "DINO-X-1.0", "image": uploaded_image_url_slice,
                    "prompt": {"type": "text", "text": TEXT_PROMPT},
                    "targets": ["bbox", "mask"], "bbox_threshold": BOX_THRESHOLD,
                    "iou_threshold": IOU_THRESHOLD,
                }
            )
            client.run_task(task)
            result_dict = task.result
            os.remove(temp_filename)
            cb_input_boxes, cb_confidences, cb_class_ids = [], [], []
            api_objects = result_dict.get("objects", [])
            if api_objects:
                for obj_dict in api_objects:
                    cb_input_boxes.append(obj_dict["bbox"])
                    cb_confidences.append(obj_dict["score"])
                    cls_name = obj_dict["category"].lower().strip()
                    if cls_name in script_class_name_to_id:
                        cb_class_ids.append(script_class_name_to_id[cls_name])
                    else:
                        tqdm.write(f"Warning: Unknown class '{cls_name}' from API in slice for {IMG_PATH}. Skipping.")
                        if cb_input_boxes: cb_input_boxes.pop()
                        if cb_confidences: cb_confidences.pop()
                        continue
            if not cb_input_boxes: return sv.Detections.empty()
            if not (len(cb_input_boxes) == len(cb_confidences) == len(cb_class_ids)):
                tqdm.write(f"Warning: Mismatch in slice components for {IMG_PATH}. Returning empty.")
                return sv.Detections.empty()
            return sv.Detections(
                xyxy=np.array(cb_input_boxes).reshape(-1, 4), 
                confidence=np.array(cb_confidences), 
                class_id=np.array(cb_class_ids)
            )
        slicer = sv.InferenceSlicer(
            callback=callback, slice_wh=SLICE_WH, overlap_ratio_wh=OVERLAP_RATIO,
            iou_threshold=0.5, 
            overlap_filter=sv.detection.overlap_filter.OverlapFilter.NON_MAX_SUPPRESSION
        )
        slicer_detections = slicer(cv2_img)
        if len(slicer_detections.xyxy) > 0:
            input_boxes_xyxy = slicer_detections.xyxy
            api_confidences = slicer_detections.confidence
            api_class_ids_script_indexed = slicer_detections.class_id
            api_class_names = [script_id_to_class_name[id_] for id_ in api_class_ids_script_indexed]
        else:
            tqdm.write(f"No detections found for frame {frame_loop_idx} ({frame_path_obj.name}) after slicing.")
    else: # Not WITH_SLICE_INFERENCE
        uploaded_image_url_full = client.upload_file(IMG_PATH)
        task = V2Task(
            api_path="/v2/task/dinox/detection",
            api_body={
                "model": "DINO-X-1.0", "image": uploaded_image_url_full,
                "prompt": {"type": "text", "text": TEXT_PROMPT},
                "targets": ["bbox", "mask"], "bbox_threshold": BOX_THRESHOLD,
                "iou_threshold": IOU_THRESHOLD,
            }
        )
        client.run_task(task)
        result_dict = task.result
        temp_input_boxes, temp_confidences, temp_class_names, temp_class_ids_script = [], [], [], []
        api_objects = result_dict.get("objects", [])
        if api_objects:
            for obj_dict in api_objects:
                cls_name = obj_dict["category"].lower().strip()
                if cls_name in script_class_name_to_id:
                    temp_input_boxes.append(obj_dict["bbox"])
                    temp_confidences.append(obj_dict["score"])
                    temp_class_names.append(cls_name)
                    temp_class_ids_script.append(script_class_name_to_id[cls_name])
                else:
                    tqdm.write(f"Warning: Unknown class '{cls_name}' from API for {IMG_PATH}. Skipping.")
        if temp_input_boxes:
            input_boxes_xyxy = np.array(temp_input_boxes)
            api_confidences = np.array(temp_confidences)
            api_class_names = temp_class_names
            api_class_ids_script_indexed = np.array(temp_class_ids_script)
        else:
            tqdm.write(f"No valid detections found for frame {frame_loop_idx} ({frame_path_obj.name}).")

    coco_output = {
        "info": {"year": datetime.now().year, "version": "1.0", "description": "GroundedSAM+SAM2 Annotations",
                 "contributor": "AutoAnnotation Script", "date_created": datetime.now().isoformat()},
        "licenses": [{"id": 1, "name": "CC0", "url": "https://creativecommons.org/publicdomain/zero/1.0/"}],
        "images": [{"id": coco_image_id_in_file, "file_name": frame_path_obj.name,
                    "width": pil_image.width, "height": pil_image.height,
                    "license": 1, "date_captured": datetime.now().isoformat()}],
        "categories": coco_categories,
        "annotations": []
    }
    
    annotation_json_filename = f"{frame_path_obj.stem}_coco_annotation.json"
    visualization_filename = f"{frame_path_obj.stem}_visualization.jpg"

    if input_boxes_xyxy.shape[0] == 0:
        tqdm.write(f"Skipping SAM for frame {frame_loop_idx} ({frame_path_obj.name}) due to no API detections.")
        if DUMP_JSON_RESULTS:
            json_output_path = specific_annotation_output_dir / annotation_json_filename
            with open(json_output_path, "w") as f: json.dump(coco_output, f, indent=4)
        continue

    with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=(DEVICE=="cuda")):
        sam2_predictor.set_image(np.array(pil_image.convert("RGB")))
        masks_sam, scores_sam, _ = sam2_predictor.predict(
            point_coords=None, point_labels=None, box=input_boxes_xyxy, multimask_output=False,
        )
        
    if masks_sam.ndim == 4: masks_sam = masks_sam.squeeze(1)

    if masks_sam.shape[0] != input_boxes_xyxy.shape[0]:
        tqdm.write(f"Warning: Mismatch DINO-X boxes and SAM masks for {frame_path_obj.name}. Skipping annotation.")
        if DUMP_JSON_RESULTS:
            json_output_path = specific_annotation_output_dir / annotation_json_filename
            with open(json_output_path, "w") as f: json.dump(coco_output, f, indent=4)
        continue
    
    coco_annotation_id_counter = 1
    for i in range(len(input_boxes_xyxy)):
        original_bbox_xyxy = input_boxes_xyxy[i]
        coco_bbox = [float(original_bbox_xyxy[0]), float(original_bbox_xyxy[1]),
                       float(original_bbox_xyxy[2] - original_bbox_xyxy[0]),
                       float(original_bbox_xyxy[3] - original_bbox_xyxy[1])]
        mask_for_rle = masks_sam[i].astype(np.uint8)
        rle_obj_for_coco = mask_util.encode(np.asfortranarray(mask_for_rle))
        final_rle_for_json = {}
        if isinstance(rle_obj_for_coco, list):
            if not rle_obj_for_coco: continue
            merged_rle = mask_util.merge(rle_obj_for_coco)
            final_rle_for_json = {"size": [int(merged_rle['size'][0]), int(merged_rle['size'][1])],
                                  "counts": merged_rle['counts'].decode('utf-8')}
            area = float(mask_util.area(merged_rle))
        else:
            final_rle_for_json = {"size": [int(rle_obj_for_coco['size'][0]), int(rle_obj_for_coco['size'][1])],
                                  "counts": rle_obj_for_coco['counts'].decode('utf-8')}
            area = float(mask_util.area(rle_obj_for_coco))
        current_class_name = api_class_names[i]
        coco_output["annotations"].append({
            "id": coco_annotation_id_counter, "image_id": coco_image_id_in_file,
            "category_id": coco_class_name_to_category_id[current_class_name],
            "segmentation": final_rle_for_json, "area": area, "bbox": coco_bbox, "iscrowd": 0,
            "attributes": {"score_SAM": scores_sam[i].item() if isinstance(scores_sam[i], torch.Tensor) else float(scores_sam[i]),
                           "score_DINOX": float(api_confidences[i]) if api_confidences.size > 0 else 0.0}
        })
        coco_annotation_id_counter += 1

    if input_boxes_xyxy.shape[0] > 0 :
        labels_vis = [f"{cname} {conf:.2f}" for cname, conf in zip(api_class_names, api_confidences)]
        detections_for_vis = sv.Detections(
            xyxy=input_boxes_xyxy, mask=masks_sam.astype(bool),
            class_id=api_class_ids_script_indexed, confidence=api_confidences
        )
        annotated_scene = cv2_img.copy()
        try:
            bbox_annotator = sv.BoundingBoxAnnotator()
            annotated_scene = bbox_annotator.annotate(scene=annotated_scene, detections=detections_for_vis)
            
            label_annotator = sv.LabelAnnotator()
            if len(labels_vis) != len(detections_for_vis):
                tqdm.write(f"Warning: Label/detection count mismatch for vis for {frame_path_obj.name}. Recreating.")
                current_vis_confidences = detections_for_vis.confidence if detections_for_vis.confidence is not None else [0.0]*len(detections_for_vis)
                labels_vis = [f"{script_id_to_class_name[cid]} {det_conf:.2f}" 
                              for cid, det_conf in zip(detections_for_vis.class_id, current_vis_confidences)]
            annotated_scene = label_annotator.annotate(scene=annotated_scene, detections=detections_for_vis, labels=labels_vis)
            
            mask_annotator = sv.MaskAnnotator()
            annotated_scene_with_masks = mask_annotator.annotate(scene=annotated_scene, detections=detections_for_vis)

            visual_image_path = specific_visual_output_dir / visualization_filename
            cv2.imwrite(str(visual_image_path), annotated_scene_with_masks)
        except Exception as e:
            tqdm.write(f"Error during visualization for {frame_path_obj.name}: {e}")

    if DUMP_JSON_RESULTS:
        json_output_path = specific_annotation_output_dir / annotation_json_filename
        with open(json_output_path, "w") as f: json.dump(coco_output, f, indent=4)

print("\nProcessing complete.")
if TEST_MODE: print(f"--- FINISHED IN TEST MODE ({len(frame_files_to_process)} frames processed) ---")
else: print(f"--- FINISHED IN FULL MODE ({len(frame_files_to_process)} frames processed) ---")
