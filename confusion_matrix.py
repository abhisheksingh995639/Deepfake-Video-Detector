import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
from tensorflow.keras.preprocessing.image import img_to_array
import cv2

# --- Configuration ---
MODEL_PATH = "best_deepfake_model_effnet.keras"
TEST_DATA_PATH = "processed_images"  # Your test dataset
IMAGE_SIZE = (224, 224)
DECISION_THRESHOLD = 0.6  # Same as your desktop app

def load_test_data():
    """Load test data from your processed_images directory."""
    print("Loading test dataset...")
    
    # Create test dataset
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DATA_PATH,
        image_size=IMAGE_SIZE,
        batch_size=32,
        shuffle=False  # Important: don't shuffle for confusion matrix
    )
    
    class_names = test_dataset.class_names
    print(f"Classes found: {class_names}")
    
    return test_dataset, class_names

def evaluate_model():
    """Evaluate model and generate confusion matrix."""
    # Load model
    print("Loading trained model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Load test data
    test_dataset, class_names = load_test_data()
    
    # Get predictions
    print("Making predictions on test data...")
    y_pred_proba = model.predict(test_dataset, verbose=1)
    y_pred = (y_pred_proba > DECISION_THRESHOLD).astype(int).flatten()
    
    # Get true labels
    y_true = []
    for _, labels in test_dataset:
        y_true.extend(labels.numpy())
    y_true = np.array(y_true)
    
    # Ensure same length
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    print(f"\nEvaluation Summary:")
    print(f"Total samples: {len(y_true)}")
    print(f"True labels distribution: {np.bincount(y_true)}")
    print(f"Predicted labels distribution: {np.bincount(y_pred)}")
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate metrics
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    
    # True/False Positives and Negatives
    tn, fp, fn, tp = cm.ravel()
    
    precision_fake = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_fake = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision_real = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_real = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\nPerformance Metrics:")
    print(f"Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"Real Videos - Precision: {precision_real:.3f}, Recall: {recall_real:.3f}")
    print(f"Fake Videos - Precision: {precision_fake:.3f}, Recall: {recall_fake:.3f}")
    
    return cm, class_names, y_true, y_pred, y_pred_proba

def plot_confusion_matrix(cm, class_names, save_path="confusion_matrix.png"):
    """Create and save a beautiful confusion matrix plot."""
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title('Confusion Matrix - Raw Counts', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted Label', fontsize=12)
    ax1.set_ylabel('True Label', fontsize=12)
    
    # Plot 2: Normalized (percentages)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Oranges',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax2, cbar_kws={'label': 'Percentage'})
    ax2.set_title('Confusion Matrix - Normalized', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted Label', fontsize=12)
    ax2.set_ylabel('True Label', fontsize=12)
    
    # Add performance text
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    textstr = f'''Performance Summary:
    Total Samples: {cm.sum():,}
    Accuracy: {accuracy:.1%}
    
    True Negatives (Real→Real): {tn}
    False Positives (Real→Fake): {fp}
    False Negatives (Fake→Real): {fn}
    True Positives (Fake→Fake): {tp}'''
    
    fig.text(0.02, 0.02, textstr, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Confusion matrix saved as: {save_path}")

def detailed_analysis(y_true, y_pred, y_pred_proba, class_names):
    """Provide detailed analysis of model performance."""
    print("\n" + "="*60)
    print("DETAILED PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Prediction confidence analysis
    correct_predictions = y_true == y_pred
    correct_confidences = y_pred_proba[correct_predictions].flatten()
    incorrect_confidences = y_pred_proba[~correct_predictions].flatten()
    
    print(f"\nConfidence Analysis:")
    print(f"Correct predictions - Mean confidence: {np.mean(np.abs(correct_confidences - 0.5) + 0.5):.3f}")
    print(f"Incorrect predictions - Mean confidence: {np.mean(np.abs(incorrect_confidences - 0.5) + 0.5):.3f}")
    
    # Threshold analysis
    print(f"\nThreshold Analysis (current threshold: {DECISION_THRESHOLD}):")
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    for threshold in thresholds:
        y_pred_thresh = (y_pred_proba > threshold).astype(int).flatten()
        accuracy_thresh = np.sum(y_true == y_pred_thresh) / len(y_true)
        print(f"Threshold {threshold}: Accuracy = {accuracy_thresh:.3f} ({accuracy_thresh*100:.1f}%)")

def plot_prediction_distribution(y_pred_proba, y_true, class_names):
    """Plot distribution of prediction scores."""
    plt.figure(figsize=(12, 8))
    
    # Separate scores by true class
    real_scores = y_pred_proba[y_true == 0].flatten()
    fake_scores = y_pred_proba[y_true == 1].flatten()
    
    plt.subplot(2, 2, 1)
    plt.hist(real_scores, bins=30, alpha=0.7, label=f'True {class_names[0]}', color='green', edgecolor='black')
    plt.hist(fake_scores, bins=30, alpha=0.7, label=f'True {class_names[1]}', color='red', edgecolor='black')
    plt.axvline(x=DECISION_THRESHOLD, color='black', linestyle='--', label=f'Threshold ({DECISION_THRESHOLD})')
    plt.xlabel('Prediction Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Scores by True Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.boxplot([real_scores, fake_scores], labels=[f'True {class_names[0]}', f'True {class_names[1]}'])
    plt.ylabel('Prediction Score')
    plt.title('Prediction Score Box Plots')
    plt.grid(True, alpha=0.3)
    
    # ROC-like analysis at different thresholds
    plt.subplot(2, 2, 3)
    thresholds = np.linspace(0, 1, 100)
    accuracies = []
    precisions = []
    recalls = []
    
    for thresh in thresholds:
        y_pred_thresh = (y_pred_proba > thresh).astype(int).flatten()
        accuracy = np.sum(y_true == y_pred_thresh) / len(y_true)
        
        tp = np.sum((y_true == 1) & (y_pred_thresh == 1))
        fp = np.sum((y_true == 0) & (y_pred_thresh == 1))
        fn = np.sum((y_true == 1) & (y_pred_thresh == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
    
    plt.plot(thresholds, accuracies, label='Accuracy', linewidth=2)
    plt.plot(thresholds, precisions, label='Precision (Fake)', linewidth=2)
    plt.plot(thresholds, recalls, label='Recall (Fake)', linewidth=2)
    plt.axvline(x=DECISION_THRESHOLD, color='red', linestyle='--', label=f'Current Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Performance Metrics vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Statistics summary
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    stats_text = f'''Prediction Statistics:
    
    Real Videos ({class_names[0]}):
    - Count: {len(real_scores)}
    - Mean Score: {np.mean(real_scores):.3f}
    - Std Dev: {np.std(real_scores):.3f}
    - Min: {np.min(real_scores):.3f}
    - Max: {np.max(real_scores):.3f}
    
    Fake Videos ({class_names[1]}):
    - Count: {len(fake_scores)}
    - Mean Score: {np.mean(fake_scores):.3f}
    - Std Dev: {np.std(fake_scores):.3f}
    - Min: {np.min(fake_scores):.3f}
    - Max: {np.max(fake_scores):.3f}
    
    Threshold: {DECISION_THRESHOLD}
    Total Samples: {len(y_pred_proba)}'''
    
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('prediction_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Prediction analysis plots saved as: prediction_analysis.png")

def main():
    """Main function to run the complete analysis."""
    try:
        print("DEEPFAKE DETECTION MODEL EVALUATION")
        print("="*50)
        
        # Check if model exists
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model file '{MODEL_PATH}' not found!")
            print("Please ensure your trained model is in the same directory.")
            return
        
        # Check if test data exists
        if not os.path.exists(TEST_DATA_PATH):
            print(f"Error: Test data directory '{TEST_DATA_PATH}' not found!")
            print("Please ensure your processed_images directory exists.")
            return
        
        # Run evaluation
        cm, class_names, y_true, y_pred, y_pred_proba = evaluate_model()
        
        # Create visualizations
        plot_confusion_matrix(cm, class_names)
        plot_prediction_distribution(y_pred_proba, y_true, class_names)
        
        # Detailed analysis
        detailed_analysis(y_true, y_pred, y_pred_proba, class_names)
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETE!")
        print("Files generated:")
        print("- confusion_matrix.png")
        print("- prediction_analysis.png")
        print("="*60)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("Please check that all required files are in place and try again.")

if __name__ == "__main__":
    main()