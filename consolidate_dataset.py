import os
import shutil
import random

# --- CONFIGURATION ---
# The path to your new FaceForensics++ dataset
SOURCE_DATASET_PATH = r"E:\DeepFake\FaceForensics++_C23"

# The path where your final 'video_dataset' folder will be created
DESTINATION_DATASET_PATH = r"E:\DeepFake\video_dataset"

# Define which folders contain which type of videos
REAL_FOLDERS = ['original']
FAKE_FOLDERS = ['DeepFakeDetection', 'Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures']

# --- NEW: Configuration for a Balanced, Stratified Sample ---
# Total number of videos to select for EACH main category (real and fake)
SAMPLES_PER_CATEGORY = 500 
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def consolidate_and_sample():
    """
    Creates a balanced dataset by sampling an equal number of real and fake videos.
    The fake videos are sampled proportionally from all sub-categories.
    """
    print("--- Starting Stratified & Balanced Dataset Consolidation ---")

    if os.path.isdir(DESTINATION_DATASET_PATH):
        print(f"Destination folder '{DESTINATION_DATASET_PATH}' already exists. Deleting for a fresh start.")
        shutil.rmtree(DESTINATION_DATASET_PATH)

    dest_real_path = os.path.join(DESTINATION_DATASET_PATH, 'real')
    dest_fake_path = os.path.join(DESTINATION_DATASET_PATH, 'fake')
    os.makedirs(dest_real_path, exist_ok=True)
    os.makedirs(dest_fake_path, exist_ok=True)

    # --- Process FAKE Videos with Stratified Sampling ---
    print("\nProcessing FAKE videos with Stratified Sampling...")
    fake_video_sources = []
    for folder_name in FAKE_FOLDERS:
        source_folder = os.path.join(SOURCE_DATASET_PATH, folder_name)
        if os.path.isdir(source_folder):
            videos = [os.path.join(source_folder, f) for f in os.listdir(source_folder) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
            fake_video_sources.append({'name': folder_name, 'videos': videos})
        else:
            print(f" - WARNING: Source folder not found, skipping: {source_folder}")
    
    total_fake_videos = sum(len(src['videos']) for src in fake_video_sources)
    
    for source in fake_video_sources:
        proportion = len(source['videos']) / total_fake_videos
        num_to_sample = int(proportion * SAMPLES_PER_CATEGORY)
        
        print(f" - Sampling {num_to_sample} videos from '{source['name']}'...")
        
        sampled_videos = random.sample(source['videos'], num_to_sample)
        
        for video_path in sampled_videos:
            video_file = os.path.basename(video_path)
            new_filename = f"{source['name']}_{video_file}"
            destination_path = os.path.join(dest_fake_path, new_filename)
            shutil.copy2(video_path, destination_path)

    # --- Process REAL Videos with Simple Random Sampling ---
    print("\nProcessing REAL videos (random sample)...")
    source_folder = os.path.join(SOURCE_DATASET_PATH, REAL_FOLDERS[0])
    if os.path.isdir(source_folder):
        all_real_videos = [f for f in os.listdir(source_folder) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        
        print(f" - Found {len(all_real_videos)} videos in '{REAL_FOLDERS[0]}'. Randomly sampling {SAMPLES_PER_CATEGORY}.")
        sampled_videos = random.sample(all_real_videos, SAMPLES_PER_CATEGORY)
        
        for video_file in sampled_videos:
            source_path = os.path.join(source_folder, video_file)
            destination_path = os.path.join(dest_real_path, video_file)
            shutil.copy2(source_path, destination_path)
    else:
        print(f" - WARNING: Source folder not found, skipping: {source_folder}")


    print("\n--- Dataset Consolidation Complete! ---")
    print(f"Your balanced dataset is ready in: '{DESTINATION_DATASET_PATH}'")

if __name__ == '__main__':
    if not os.path.isdir(SOURCE_DATASET_PATH):
        print(f"!!! FATAL ERROR: The source dataset path does not exist: '{SOURCE_DATASET_PATH}'")
    else:
        consolidate_and_sample()



