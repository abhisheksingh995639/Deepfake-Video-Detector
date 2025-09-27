import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import shutil
import random 
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# --- Main Configuration ---
BASE_PROJECT_PATH = r"E:\DeepFake"
VIDEO_DATASET_PATH = os.path.join(BASE_PROJECT_PATH, "video_dataset")
IMAGE_DATASET_PATH = os.path.join(BASE_PROJECT_PATH, "processed_images")

# --- Face Detection Configuration ---
PROTOTXT_PATH = "deploy.prototxt"
CAFFEMODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"

# --- FAST TRAINING CONFIGURATION ---
# BALANCED CONFIGURATION (2-3 hours training time)
FRAMES_PER_SECOND = 2 
VIDEO_SAMPLE_SIZE = 200       # Reduced from 500
MIN_FACE_SIZE = 60            # Slightly smaller for more faces
FACE_CONFIDENCE_THRESHOLD = 0.6  # Lower threshold to get more faces
FACE_PADDING = 20             # Padding around detected faces

# Model training parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 10           # Reduced from 20
FINE_TUNE_EPOCHS = 5          # Reduced from 15
LEARNING_RATE = 0.0005
FINE_TUNE_LEARNING_RATE = 0.00001

# --- ALTERNATIVE CONFIGURATIONS (uncomment to use) ---
# FAST CONFIGURATION (30-60 minutes):
# VIDEO_SAMPLE_SIZE = 20
# FRAMES_PER_SECOND = 1
# INITIAL_EPOCHS = 8
# FINE_TUNE_EPOCHS = 3

# ULTRA FAST CONFIGURATION (10-15 minutes):
# VIDEO_SAMPLE_SIZE = 10  
# FRAMES_PER_SECOND = 1
# INITIAL_EPOCHS = 5
# FINE_TUNE_EPOCHS = 2

tf.keras.mixed_precision.set_global_policy('mixed_float16')

def sanity_checks():
    """Runs checks to ensure the environment is set up correctly before starting."""
    print("--- Running Pre-Training Sanity Checks ---")
    if not os.path.isdir(VIDEO_DATASET_PATH):
        print(f"!!! ERROR: The main video dataset path does not exist: '{VIDEO_DATASET_PATH}'")
        return False
    
    # Check for face detection files
    if not os.path.exists(PROTOTXT_PATH) or not os.path.exists(CAFFEMODEL_PATH):
        print("!!! ERROR: Face detector files (prototxt/caffemodel) not found.")
        print("Please download:")
        print("- deploy.prototxt")
        print("- res10_300x300_ssd_iter_140000.caffemodel")
        print("From OpenCV GitHub repository")
        return False
        
    print("--- Sanity Checks Passed Successfully ---")
    return True

def is_good_face_crop(face, min_size=MIN_FACE_SIZE):
    """Check if the face crop is good quality for training."""
    if face is None or face.size == 0:
        return False
    
    h, w = face.shape[:2]
    if h < min_size or w < min_size:
        return False
    
    # Check if face is too blurry
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 100:  # Too blurry
        return False
    
    return True

def preprocess_videos_to_images():
    """
    Extract frames with improved face detection and quality filtering.
    """
    if os.path.isdir(IMAGE_DATASET_PATH):
        print("Existing 'processed_images' directory found. Deleting it to create a new random sample.")
        shutil.rmtree(IMAGE_DATASET_PATH)

    print("--- Starting Video Preprocessing with Face Detection ---")
    
    # Load face detector
    face_net = cv2.dnn.readNet(PROTOTXT_PATH, CAFFEMODEL_PATH)
    random.seed(42)

    stats = {"real": {"videos": 0, "frames": 0, "faces": 0}, 
             "fake": {"videos": 0, "frames": 0, "faces": 0}}

    for category in ['real', 'fake']:
        video_category_path = os.path.join(VIDEO_DATASET_PATH, category)
        image_category_path = os.path.join(IMAGE_DATASET_PATH, category)
        os.makedirs(image_category_path, exist_ok=True)
        
        all_videos = [f for f in os.listdir(video_category_path) if f.endswith(('.mp4', '.avi', '.mov'))]
        random.shuffle(all_videos)
        selected_videos = all_videos[:VIDEO_SAMPLE_SIZE]
        
        stats[category]["videos"] = len(selected_videos)
        print(f"Processing {len(selected_videos)} {category} videos...")

        for video_name in selected_videos:
            video_path = os.path.join(video_category_path, video_name)
            face_count = 0
            
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened(): 
                    continue
                    
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps == 0: 
                    continue
                    
                capture_interval = int(fps / FRAMES_PER_SECOND) if FRAMES_PER_SECOND > 0 else 1
                if capture_interval == 0: 
                    capture_interval = 1
                
                frame_num = 0
                while cap.isOpened():
                    success, frame = cap.read()
                    if not success: 
                        break
                    
                    stats[category]["frames"] += 1
                    
                    if frame_num % capture_interval == 0:
                        # Detect faces in frame
                        (h, w) = frame.shape[:2]
                        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
                        face_net.setInput(blob)
                        detections = face_net.forward()
                        
                        # Find best face detection
                        best_detection_idx = -1
                        max_confidence = FACE_CONFIDENCE_THRESHOLD
                        
                        for i in range(detections.shape[2]):
                            confidence = detections[0, 0, i, 2]
                            if confidence > max_confidence:
                                max_confidence = confidence
                                best_detection_idx = i
                        
                        if best_detection_idx != -1:
                            # Extract face with padding
                            box = detections[0, 0, best_detection_idx, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype("int")
                            
                            # Add padding
                            padded_startX = max(0, startX - FACE_PADDING)
                            padded_startY = max(0, startY - FACE_PADDING)
                            padded_endX = min(w, endX + FACE_PADDING)
                            padded_endY = min(h, endY + FACE_PADDING)
                            
                            face = frame[padded_startY:padded_endY, padded_startX:padded_endX]
                            
                            # Only save good quality faces
                            if is_good_face_crop(face):
                                face_filename = f"{os.path.splitext(video_name)[0]}_face_{face_count}.jpg"
                                output_path = os.path.join(image_category_path, face_filename)
                                cv2.imwrite(output_path, face)
                                face_count += 1
                                stats[category]["faces"] += 1
                    
                    frame_num += 1
                    
            except Exception as e:
                print(f"Error processing {video_name}: {e}")
            finally:
                if 'cap' in locals() and cap.isOpened(): 
                    cap.release()
            
            print(f"  - Extracted {face_count} faces from '{video_name}'")
    
    # Print statistics
    print("\n--- Extraction Statistics ---")
    for category in ['real', 'fake']:
        s = stats[category]
        print(f"{category.upper()}: {s['videos']} videos, {s['frames']} frames processed, {s['faces']} faces extracted")
    
    # Check for class imbalance
    real_faces = stats['real']['faces']
    fake_faces = stats['fake']['faces']
    if real_faces == 0 or fake_faces == 0:
        print("⚠️  CRITICAL ERROR: One class has no faces extracted!")
        print("Please check your video dataset and face detection settings.")
        return False
    
    if abs(real_faces - fake_faces) / max(real_faces, fake_faces) > 0.3:
        print(f"⚠️  WARNING: Significant class imbalance detected!")
        print(f"Real faces: {real_faces}, Fake faces: {fake_faces}")
        print("Consider balancing your dataset for better training results.")
    
    print("--- Video Preprocessing Complete ---")
    return True

def create_model():
    """Creates the model with improved architecture."""
    tf.keras.backend.clear_session()
    
    # Enhanced data augmentation
    data_augmentation = Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.15),
        RandomZoom(0.15),
    ], name='data_augmentation')
    
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.efficientnet_v2.preprocess_input(x)
    
    base_model = EfficientNetV2B0(weights='imagenet', include_top=False, name='efficientnetv2-b0')
    base_model.trainable = False
    
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)  # Reduced dropout to prevent overfitting to fake class
    x = Dense(128, activation='relu', name='feature_layer')(x)  # Additional layer for better feature learning
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid', dtype='float32', name='classification_layer')(x)
    
    model = Model(inputs, outputs)
    return model

def build_dataset(subset):
    """Builds a tf.data pipeline for training or validation."""
    dataset = tf.keras.utils.image_dataset_from_directory(
        IMAGE_DATASET_PATH,
        validation_split=0.2,
        subset=subset,
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    # Print class names to verify label mapping
    if subset == "training":
        class_names = dataset.class_names
        print(f"\n--- LABEL MAPPING ---")
        print(f"Class names: {class_names}")
        print(f"Label 0 = '{class_names[0]}'")
        print(f"Label 1 = '{class_names[1]}'")
        print("--- IMPORTANT: Remember this mapping for inference! ---\n")
    
    # Apply additional preprocessing
    def preprocess(image, label):
        image = tf.cast(image, tf.float32)
        return image, label
    
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

def calculate_class_weights():
    """Calculate class weights to handle imbalance - no sklearn needed."""
    real_path = os.path.join(IMAGE_DATASET_PATH, 'real')
    fake_path = os.path.join(IMAGE_DATASET_PATH, 'fake')
    
    # Count image files
    real_count = 0
    fake_count = 0
    
    if os.path.exists(real_path):
        real_count = len([f for f in os.listdir(real_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if os.path.exists(fake_path):
        fake_count = len([f for f in os.listdir(fake_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if real_count == 0 or fake_count == 0:
        print("Warning: One of the classes has no images. Using balanced weights.")
        return {0: 1.0, 1: 1.0}
    
    total = real_count + fake_count
    
    # Calculate inverse frequency weights
    class_weights = {
        0: total / (2.0 * real_count),   # Real class (label 0)
        1: total / (2.0 * fake_count)    # Fake class (label 1)
    }
    
    print(f"Dataset: {real_count} real images, {fake_count} fake images")
    print(f"Class weights: Real={class_weights[0]:.3f}, Fake={class_weights[1]:.3f}")
    return class_weights

def train_model():
    """Handles the entire model training and fine-tuning process."""
    train_ds = build_dataset("training")
    val_ds = build_dataset("validation")
    model = create_model()
    
    # Calculate class weights
    class_weights = calculate_class_weights()
    
    # Improved callbacks
    checkpoint = ModelCheckpoint(
        "best_deepfake_model_effnet.keras", 
        save_best_only=True, 
        monitor='val_accuracy', 
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=7, 
        restore_best_weights=True, 
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE), 
        loss='binary_crossentropy', 
        metrics=['accuracy', 'precision', 'recall']
    )
    
    print("\n--- MODEL ARCHITECTURE ---")
    model.summary()
    print("--- STARTING TRAINING ---\n")

    # Initial training
    print("--- PHASE 1: Initial Training ---")
    history = model.fit(
        train_ds, 
        epochs=INITIAL_EPOCHS, 
        validation_data=val_ds, 
        callbacks=[checkpoint, early_stopping, reduce_lr],
        class_weight=class_weights,
        verbose=1
    )

    print("\n--- PHASE 2: Fine-Tuning ---")
    model = tf.keras.models.load_model("best_deepfake_model_effnet.keras")
    
    base_model = model.get_layer('efficientnetv2-b0')
    base_model.trainable = True
    
    # Unfreeze top layers only
    fine_tune_at = int(len(base_model.layers) * 0.8)  # More conservative unfreezing
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    model.compile(
        optimizer=Adam(learning_rate=FINE_TUNE_LEARNING_RATE), 
        loss='binary_crossentropy', 
        metrics=['accuracy', 'precision', 'recall']
    )
    
    print(f"Fine-tuning from layer {fine_tune_at} onwards...")
    
    history_fine_tune = model.fit(
        train_ds,
        epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
        initial_epoch=len(history.history['loss']),
        validation_data=val_ds,
        callbacks=[checkpoint, early_stopping, reduce_lr],
        class_weight=class_weights,
        verbose=1
    )
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE! 🎉")
    print("="*60)
    print(f"✅ Model saved as: best_deepfake_model_effnet.keras")
    print(f"✅ Training epochs: {len(history.history['loss'])}")
    print(f"✅ Fine-tuning epochs: {len(history_fine_tune.history['loss'])}")
    
    # Get final metrics
    final_train_acc = history_fine_tune.history['accuracy'][-1]
    final_val_acc = history_fine_tune.history['val_accuracy'][-1]
    final_train_loss = history_fine_tune.history['loss'][-1]
    final_val_loss = history_fine_tune.history['val_loss'][-1]
    
    print(f"📊 Final Training Accuracy: {final_train_acc:.1%}")
    print(f"📊 Final Validation Accuracy: {final_val_acc:.1%}")
    print(f"📊 Final Training Loss: {final_train_loss:.4f}")
    print(f"📊 Final Validation Loss: {final_val_loss:.4f}")
    print("="*60)
    
    # Plot training history
    try:
        plot_training_history(history, history_fine_tune)
        print("✅ Training plots saved as: training_history.png")
    except Exception as e:
        print(f"⚠️  Could not create training plots: {e}")
    
    print("\n🚀 Ready to test with desktop_app.py!")
    print("📋 Remember the label mapping from above for correct inference logic.")

def plot_training_history(history1, history2):
    """Plot training metrics."""
    # Combine histories
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']
    
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.axvline(x=len(history1.history['accuracy']), color='r', linestyle='--', label='Fine-tuning Start')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.axvline(x=len(history1.history['loss']), color='r', linestyle='--', label='Fine-tuning Start')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    print("🤖 DEEPFAKE DETECTION MODEL TRAINING")
    print("="*50)
    print(f"📁 Dataset: {VIDEO_DATASET_PATH}")
    print(f"🎯 Videos per class: {VIDEO_SAMPLE_SIZE}")
    print(f"⚡ FPS sampling: {FRAMES_PER_SECOND}")
    print(f"🏋️  Training epochs: {INITIAL_EPOCHS} + {FINE_TUNE_EPOCHS}")
    print("="*50)
    
    if sanity_checks():
        print("\n🚀 Starting video preprocessing...")
        if preprocess_videos_to_images():
            print("\n🚀 Starting model training...")
            train_model()
        else:
            print("\n❌ Preprocessing failed. Please check your dataset.")
    else:
        print("\n❌ Sanity checks failed. Please fix the issues above.")