import json
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

def merge_coco_jsons_per_video(
    # The output of annotation script will have subfolders like #1, #2, etc., and inside those will be individual frame JSONs
    nested_per_frame_annotations_root: Path,
    frames_images_root: Path, # To get image dimensions if not in per-frame JSON, or to verify filenames
    output_merged_dir: Path,
    dataset_info: dict = None,
    default_license: list = None
):
    output_merged_dir.mkdir(parents=True, exist_ok=True)

    if dataset_info is None:
        dataset_info = {
            "year": datetime.now().year, "version": "1.0",
            "description": "Merged Recycling Dataset - Per Video",
            "contributor": "Your Name/Organization",
            "url": "", "date_created": datetime.now().isoformat()
        }
    if default_license is None:
        default_license = [{"id": 1, "name": "CC0", "url": "https://creativecommons.org/publicdomain/zero/1.0/"}]

    # Iterate through camera folders (e.g., #1, #2) in the annotations directory
    for camera_ann_dir in tqdm(list(nested_per_frame_annotations_root.iterdir()), desc="Processing Cameras"):
        if not camera_ann_dir.is_dir():
            continue

        # Group JSON files within this camera folder by their original video stem
        jsons_by_video_stem = defaultdict(list)
        for json_file_path in camera_ann_dir.glob("*_coco_annotation.json"): # Assuming this naming from Step 2
            # Extract video stem. Example: GX010012_MP4_frame_00000000_coco_annotation.json
            # We want "GX010012.MP4" or just "GX010012" as the key
            parts = json_file_path.stem.split("_frame_")
            if len(parts) > 0:
                video_stem_key = parts[0] # e.g., GX010012.MP4 or GX010012
                jsons_by_video_stem[video_stem_key].append(json_file_path)
            else:
                print(f"Warning: Could not parse video stem from {json_file_path.name}. Skipping.")

        for video_stem, per_frame_json_paths in tqdm(jsons_by_video_stem.items(), desc=f"Cam: {camera_ann_dir.name}", leave=False):
            merged_coco = {
                "info": dataset_info, "licenses": default_license,
                "images": [], "categories": [], "annotations": []
            }
            global_image_id_counter_for_video = 1
            global_annotation_id_counter_for_video = 1
            categories_set_for_video = False
            
            output_json_filename = f"{camera_ann_dir.name}_{video_stem}_coco_merged.json"

            sorted_per_frame_json_paths = sorted(per_frame_json_paths) # Sort by full path

            for json_path in sorted_per_frame_json_paths:
                try:
                    with open(json_path, 'r') as f:
                        single_frame_coco = json.load(f)
                except Exception as e:
                    print(f"Error reading or parsing {json_path}: {e}. Skipping.")
                    continue

                if not categories_set_for_video and "categories" in single_frame_coco and single_frame_coco["categories"]:
                    merged_coco["categories"] = single_frame_coco["categories"]
                    categories_set_for_video = True
                
                for img_info in single_frame_coco.get("images", []): # Should be one image
                    new_img_info = img_info.copy()
                    new_img_info["id"] = global_image_id_counter_for_video
                    
                    new_img_info["file_name"] = img_info["file_name"]
                    
                    new_img_info["license"] = default_license[0]["id"] if default_license else 0
                    
                    if "width" not in new_img_info or "height" not in new_img_info or \
                       new_img_info["width"] == 0 or new_img_info["height"] == 0:
                        actual_image_physical_path = frames_images_root / relative_image_path
                        if actual_image_physical_path.exists():
                            try:
                                from PIL import Image
                                with Image.open(actual_image_physical_path) as pil_img_dim:
                                    new_img_info["width"] = pil_img_dim.width
                                    new_img_info["height"] = pil_img_dim.height
                            except Exception as e_img:
                                print(f"Warning: Could not get dims for {actual_image_physical_path}: {e_img}")
                                new_img_info["width"] = new_img_info.get("width", 0) # Fallback
                                new_img_info["height"] = new_img_info.get("height", 0) # Fallback
                        else:
                            print(f"Warning: Image file {actual_image_physical_path} not found for dimensions.")
                            new_img_info["width"] = new_img_info.get("width", 0) # Fallback
                            new_img_info["height"] = new_img_info.get("height", 0) # Fallback


                    merged_coco["images"].append(new_img_info)

                    # The image_id in the annotations of single_frame_coco should be 1 (or consistent)
                    # because each of those JSONs represents a single image.
                    original_image_id_in_file = img_info["id"] 

                    for ann_info in single_frame_coco.get("annotations", []):
                        if ann_info.get("image_id") == original_image_id_in_file:
                            new_ann_info = ann_info.copy()
                            new_ann_info["image_id"] = global_image_id_counter_for_video
                            new_ann_info["id"] = global_annotation_id_counter_for_video
                            new_ann_info["iscrowd"] = ann_info.get("iscrowd", 0)
                            merged_coco["annotations"].append(new_ann_info)
                            global_annotation_id_counter_for_video += 1
                    global_image_id_counter_for_video += 1
            
            if not merged_coco["categories"] and not categories_set_for_video :
                 print(f"CRITICAL WARNING: No categories found for video '{output_json_filename}'.")


            final_output_path = output_merged_dir / output_json_filename
            try:
                with open(final_output_path, 'w') as f:
                    json.dump(merged_coco, f, indent=4)
                tqdm.write(f"Aggregated COCO for {camera_ann_dir.name}/{video_stem} saved to {final_output_path}")
            except Exception as e:
                print(f"Error writing aggregated COCO for {camera_ann_dir.name}/{video_stem}: {e}")

    print("\nPer-video COCO aggregation complete.")

if __name__ == "__main__":
    NESTED_PER_FRAME_ANNOTATIONS_ROOT = Path("/home/vivora/AutoAnnotation/outputs/grounded_sam2_dinox_v2_coco_nested_demo/20250509_023422/annotations")

    ALL_FRAMES_IMAGES_ROOT = Path("/home/vivora/AutoAnnotation/frames") # Where actual .png frames are

    OUTPUT_CVAT_READY_MERGED_DIR = Path("/home/vivora/AutoAnnotation/annotations_per_video")

    if not NESTED_PER_FRAME_ANNOTATIONS_ROOT.exists():
        print(f"Error: Nested per-frame annotations root not found: {NESTED_PER_FRAME_ANNOTATIONS_ROOT}")
        exit()
    if not ALL_FRAMES_IMAGES_ROOT.exists():
        print(f"Error: Root for frame images not found: {ALL_FRAMES_IMAGES_ROOT}")
        exit()

    merge_coco_jsons_per_video(
        nested_per_frame_annotations_root=NESTED_PER_FRAME_ANNOTATIONS_ROOT,
        frames_images_root=ALL_FRAMES_IMAGES_ROOT,
        output_merged_dir=OUTPUT_CVAT_READY_MERGED_DIR
    )
