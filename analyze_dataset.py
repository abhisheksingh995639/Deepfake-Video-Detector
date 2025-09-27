import os
import cv2
import numpy as np
import shutil
import argparse
from tqdm import tqdm

# --- Face Detection & Quality Configuration ---
# NOTE: These values are copied directly from train_model.py for consistency.
# If you change them in one file, change them here as well.
PROTOTXT_PATH = "deploy.prototxt"
CAFFEMODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"
FRAMES_PER_SECOND = 2
MIN_FACE_SIZE = 60
FACE_CONFIDENCE_THRESHOLD = 0.6
FACE_PADDING = 20

def is_good_face_crop(face, min_size=MIN_FACE_SIZE):
    """
    Check if the face crop is good quality.
    This function is identical to the one in train_model.py.
    """
    if face is None or face.size == 0:
        return False
    
    h, w = face.shape[:2]
    if h < min_size or w < min_size:
        return False
    
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 100:  # Too blurry
        return False
    
    return True

def find_usable_faces_in_video(video_path, face_net):
    """
    Analyzes a single video to see if at least one usable face can be extracted.
    Stops processing as soon as the first valid face is found for efficiency.
    
    Args:
        video_path (str): The full path to the video file.
        face_net: The loaded OpenCV DNN face detection model.

    Returns:
        bool: True if at least one usable face is found, False otherwise.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  - Warning: Could not open video file: {os.path.basename(video_path)}")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            return False
            
        capture_interval = int(fps / FRAMES_PER_SECOND) if FRAMES_PER_SECOND > 0 else 1
        if capture_interval == 0:
            capture_interval = 1

        frame_num = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            if frame_num % capture_interval == 0:
                (h, w) = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
                face_net.setInput(blob)
                detections = face_net.forward()

                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > FACE_CONFIDENCE_THRESHOLD:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        
                        padded_startX = max(0, startX - FACE_PADDING)
                        padded_startY = max(0, startY - FACE_PADDING)
                        padded_endX = min(w, endX + FACE_PADDING)
                        padded_endY = min(h, endY + FACE_PADDING)
                        
                        face = frame[padded_startY:padded_endY, padded_startX:padded_endX]
                        
                        if is_good_face_crop(face):
                            # Found a usable face, no need to check further
                            cap.release()
                            return True
            
            frame_num += 1

    except Exception as e:
        print(f"  - Error processing {os.path.basename(video_path)}: {e}")
        return False
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
            
    # If the loop completes without finding a face
    return False

def analyze_and_separate_videos(base_dataset_path):
    """
    Main function to iterate through the dataset, analyze videos,
    and move those with no usable faces to a separate directory.
    """
    print("--- 📹 Starting Video Dataset Analysis 📹 ---")
    
    # --- Path Setup ---
    video_dataset_path = os.path.join(base_dataset_path, "video_dataset")
    bad_videos_path = os.path.join(base_dataset_path, "video_dataset_no_faces")

    if not os.path.isdir(video_dataset_path):
        print(f"❌ ERROR: Dataset path not found: '{video_dataset_path}'")
        return

    # --- Model Loading ---
    print("🧠 Loading face detection model...")
    if not os.path.exists(PROTOTXT_PATH) or not os.path.exists(CAFFEMODEL_PATH):
        print("❌ ERROR: Face detector files (deploy.prototxt or .caffemodel) not found.")
        return
    face_net = cv2.dnn.readNet(PROTOTXT_PATH, CAFFEMODEL_PATH)
    print("✅ Model loaded successfully.")

    moved_count = 0
    total_count = 0

    # --- Main Analysis Loop ---
    for category in ['real', 'fake']:
        source_category_path = os.path.join(video_dataset_path, category)
        dest_category_path = os.path.join(bad_videos_path, category)
        
        if not os.path.isdir(source_category_path):
            print(f"- Skipping category '{category}', directory not found.")
            continue

        os.makedirs(dest_category_path, exist_ok=True)
        
        video_files = [f for f in os.listdir(source_category_path) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        
        print(f"\n🔬 Analyzing '{category}' videos...")
        
        # Use tqdm for a progress bar
        for video_name in tqdm(video_files, desc=f"Processing {category}", unit="video"):
            total_count += 1
            video_path = os.path.join(source_category_path, video_name)
            
            if not find_usable_faces_in_video(video_path, face_net):
                # No usable faces found, move the file
                destination_path = os.path.join(dest_category_path, video_name)
                print(f"  - 🚫 No faces found in '{video_name}'. Moving it.")
                shutil.move(video_path, destination_path)
                moved_count += 1

    print("\n" + "="*50)
    print("🎉 Analysis Complete! 🎉")
    print("="*50)
    print(f"📊 Total videos scanned: {total_count}")
    print(f"🗑️ Videos moved (no faces found): {moved_count}")
    print(f"👍 Usable videos remaining: {total_count - moved_count}")
    print(f"🗂️  Moved videos are located in: '{bad_videos_path}'")
    print("="*50)

if __name__ == '__main__':
    # Setup to allow running from the command line with an argument
    parser = argparse.ArgumentParser(
        description="Analyze a video dataset and separate videos with no detectable faces.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--path', 
        type=str, 
        default=r"E:\DeepFake",
        help="The base project path containing the 'video_dataset' folder.\n"
             "Example: E:\\DeepFake"
    )
    args = parser.parse_args()

    analyze_and_separate_videos(args.path)  