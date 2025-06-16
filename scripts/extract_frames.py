import cv2
from tqdm import tqdm
from pathlib import Path
import time

def extract_frames_robust(video_path: Path, output_base_dir: Path, n: int, image_format: str):
    """
    Extracts every nth frame from a SINGLE video and saves them into a specified
    directory, naming files clearly.

    Args:
        video_path (Path): Path object for the input video file.
        output_base_dir (Path): The specific directory WHERE frames for THIS video
                                and THIS 'n' value should be saved.
        n (int): Interval for frame extraction (extract every nth frame).
        image_format (str): Format to save images ('png' or 'jpg').
    """

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return 0, 0 # Return processed, saved counts

    total_frames = None
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = None
    except Exception:
        total_frames = None

    if total_frames is None:
        print(f"Warning: Could not determine total frames for {video_path.name}. Progress bar may lack ETA.")

    output_base_dir.mkdir(parents=True, exist_ok=True)

    # --- Frame Extraction Loop ---
    current_frame_index = 0
    saved_frame_count = 0
    video_stem = video_path.stem # Get filename without extension (e.g., "GX010012")

    with tqdm(total=total_frames, unit='frame', desc=f"Vid: {video_path.name} (n={n})", leave=False) as pbar:
        while True:
            success, frame = cap.read()
            if not success:
                break

            if current_frame_index % n == 0:
                filename = f"{video_stem}_frame_{current_frame_index:08d}.{image_format}"
                save_path = output_base_dir / filename

                try:
                    cv2.imwrite(str(save_path), frame)
                    saved_frame_count += 1
                except Exception as e:
                    print(f"\nError saving frame {current_frame_index} from {video_path.name} to {save_path}: {e}")

            pbar.update(1) # Update progress bar for each frame processed
            current_frame_index += 1

    cap.release()
    return current_frame_index, saved_frame_count

# --- Main Execution Logic ---
if __name__ == "__main__":
    INPUT_ROOT_DIR = Path(r'/home/vivora/AutoAnnotation/Sample_Videos')

    OUTPUT_ROOT_DIR = Path('/home/vivora/AutoAnnotation/frames')

    FRAME_INTERVAL_N = 15

    IMAGE_FORMAT = 'png'

    VIDEO_EXTENSION = '.MP4'

    # Process only specific subfolders (e.g., camera folders).
    #    Leave empty [] to process ALL found videos.
    #    Example: PROCESS_ONLY_FOLDERS = ['#1', '#3']
    PROCESS_ONLY_FOLDERS = []

    print(f"\nSearching for *{VIDEO_EXTENSION.lower()} files in {INPUT_ROOT_DIR}...")
    all_video_files = list(INPUT_ROOT_DIR.rglob(f'*{VIDEO_EXTENSION.lower()}'))
    all_video_files.extend(list(INPUT_ROOT_DIR.rglob(f'*{VIDEO_EXTENSION.upper()}')))
    all_video_files = sorted(list(set(all_video_files)))


    if not all_video_files:
        print("No video files found matching the extension.")
        exit()

    print(f"Found {len(all_video_files)} potential video files.")

    # Filter videos based on PROCESS_ONLY_FOLDERS if the list is not empty
    videos_to_process = []
    if PROCESS_ONLY_FOLDERS:
        for video_path in all_video_files:
            if video_path.parent.name in PROCESS_ONLY_FOLDERS:
                videos_to_process.append(video_path)
        print(f"After filtering, {len(videos_to_process)} videos will be processed.")
    else:
        videos_to_process = all_video_files

    if not videos_to_process:
        print("No videos left to process after filtering.")
        exit()

    total_processed_count = 0
    total_saved_count = 0
    start_time = time.time()

    # --- Process each selected video file ---
    print("\n--- Starting Extraction ---")
    for video_path in tqdm(videos_to_process, unit='video', desc="Overall Progress"):

        try:
            relative_path = video_path.relative_to(INPUT_ROOT_DIR)
            # Example: relative_path = Path('20241009/#1/GX010012.MP4')
        except ValueError:
            print(f"Warning: Could not determine relative path for {video_path}. Skipping.")
            continue

        # Example output: /home/vivora/AutoAnnotation/frames / 20241009 / #1
        specific_output_dir = OUTPUT_ROOT_DIR / relative_path.parent

        specific_output_dir.mkdir(parents=True, exist_ok=True)

        processed, saved = extract_frames_robust(
            video_path=video_path,
            output_base_dir=specific_output_dir,
            n=FRAME_INTERVAL_N,
            image_format=IMAGE_FORMAT
        )
        total_processed_count += processed
        total_saved_count += saved

    end_time = time.time()
    print("\n" + "="*40)
    print("Batch Frame Extraction Complete!")
    print(f"Processed {len(videos_to_process)} video files.")
    print(f"Total frames processed (read): {total_processed_count}")
    print(f"Total frames saved: {total_saved_count}")
    print(f"Output structure generated under: {OUTPUT_ROOT_DIR}")
    print(f"Frame interval used (n): {FRAME_INTERVAL_N}")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    print("="*40)
