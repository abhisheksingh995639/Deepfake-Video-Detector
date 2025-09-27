import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import cv2
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array


MODEL_FILENAME = "best_deepfake_model_effnet.keras"
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)
IMAGE_SIZE = (224, 224)


PROTOTXT_PATH = os.path.join(os.path.dirname(__file__), "deploy.prototxt")
CAFFEMODEL_PATH = os.path.join(os.path.dirname(__file__), "res10_300x300_ssd_iter_140000.caffemodel")


MIN_FACE_SIZE = 50  
FACE_CONFIDENCE_THRESHOLD = 0.7  
DECISION_THRESHOLD = 0.6  
FRAME_SKIP = 5  
MIN_FRAMES_FOR_DECISION = 10  

class DeepfakeDetectorApp(tk.Tk):
    def __init__(self):
        super().__init__()

        
        self.title("Advanced Deepfake Video Detector")
        self.geometry("900x700")
        self.configure(bg="#1e1e1e")
        self.resizable(True, True)

        
        self.model = None
        self.face_net = None
        self.video_path = None
        self.analysis_thread = None
        self.is_analyzing = False
        self.after_id = None

        # --- Style Configuration ---
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.configure_styles()

        # --- UI Creation ---
        self.create_widgets()
        
        # --- Deferred Model Loading ---
        self.after(100, self.load_models_thread)

    def configure_styles(self):
        """Configures the styles for the UI elements."""
        self.style.configure("TFrame", background="#1e1e1e")
        self.style.configure("TLabel", background="#1e1e1e", foreground="#e0e0e0", font=("Segoe UI", 10))
        self.style.configure("Header.TLabel", font=("Segoe UI", 18, "bold"))
        self.style.configure("Status.TLabel", font=("Segoe UI", 10, "italic"))
        self.style.configure("TButton", background="#3c3c3c", foreground="#e0e0e0", borderwidth=1, relief="flat", font=("Segoe UI", 10, "bold"))
        self.style.map("TButton", background=[("active", "#505050")], relief=[("pressed", "solid")])
        self.style.configure("TLabelframe", background="#2a2a2a", bordercolor="#444444")
        self.style.configure("TLabelframe.Label", background="#2a2a2a", foreground="#e0e0e0", font=("Segoe UI", 12, "bold"))
        self.style.configure("ResultHeader.TLabel", font=("Segoe UI", 14, "bold"), foreground="#4CAF50")
        self.style.configure("ResultValue.TLabel", font=("Segoe UI", 12))

    def create_widgets(self):
        """Creates and arranges all UI elements in the window."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # --- Header ---
        header_frame = ttk.Frame(self, padding="20 10")
        header_frame.grid(row=0, column=0, sticky="ew")
        header_frame.grid_columnconfigure(0, weight=1)
        
        header_label = ttk.Label(header_frame, text="Advanced Deepfake Detector", style="Header.TLabel")
        header_label.grid(row=0, column=0)
        self.status_label = ttk.Label(header_frame, text="Loading AI Models...", style="Status.TLabel")
        self.status_label.grid(row=1, column=0, sticky="w", pady=(5,0))
        
        # --- Main Content Area ---
        main_frame = ttk.Frame(self)
        main_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        main_frame.grid_columnconfigure(0, weight=2)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)

        # --- Left Panel (Video Preview) ---
        video_frame = ttk.LabelFrame(main_frame, text="Live Analysis")
        video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        video_frame.grid_rowconfigure(0, weight=1)
        video_frame.grid_columnconfigure(0, weight=1)
        
        self.video_label = ttk.Label(video_frame, background="black")
        self.video_label.grid(row=0, column=0, sticky="nsew")

        # --- Right Panel (Controls & Results) ---
        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        
        # Configure the controls_frame layout
        controls_frame.grid_columnconfigure(0, weight=1)
        controls_frame.grid_rowconfigure(0, weight=0)  # Top controls, don't expand
        controls_frame.grid_rowconfigure(1, weight=1)  # Middle results, EXPAND
        controls_frame.grid_rowconfigure(2, weight=0)  # Bottom buttons, don't expand

        # --- Top Controls Frame (for selection) ---
        top_controls = ttk.Frame(controls_frame)
        top_controls.grid(row=0, column=0, sticky="ew")
        top_controls.grid_columnconfigure(0, weight=1)

        self.select_button = ttk.Button(top_controls, text="Select Video File", command=self.select_file, state=tk.DISABLED)
        self.select_button.grid(row=0, column=0, sticky="ew", pady=(0, 5), ipady=5)
        
        self.file_label = ttk.Label(top_controls, text="No file selected.", style="Status.TLabel", wraplength=250)
        self.file_label.grid(row=1, column=0, sticky="ew", pady=(0, 15))

        # --- Results Display (in the expanding middle row) ---
        self.results_frame = ttk.LabelFrame(controls_frame, text="Analysis Results")
        self.results_frame.grid(row=1, column=0, sticky="nsew", pady=5)
        self.results_frame.grid_columnconfigure(0, weight=1)
        self.clear_results()

        # --- Bottom Controls Frame (for actions) ---
        bottom_controls = ttk.Frame(controls_frame)
        bottom_controls.grid(row=2, column=0, sticky="ew")
        bottom_controls.grid_columnconfigure(0, weight=1)

        self.analyze_button = ttk.Button(bottom_controls, text="Analyze Video", command=self.start_analysis, state=tk.DISABLED)
        self.analyze_button.grid(row=0, column=0, sticky="ew", pady=5, ipady=5)
        
        self.cancel_button = ttk.Button(bottom_controls, text="Cancel", command=self.cancel_analysis, state=tk.DISABLED)
        self.cancel_button.grid(row=1, column=0, sticky="ew", pady=5, ipady=5)


    def load_models_thread(self):
        """Loads both models in a separate thread to keep the UI responsive."""
        threading.Thread(target=self.load_all_models, daemon=True).start()

    def load_all_models(self):
        """Loads the pre-trained Keras model and the DNN face detector."""
        try:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Deepfake model not found at '{MODEL_PATH}'")
            
            if not os.path.exists(PROTOTXT_PATH) or not os.path.exists(CAFFEMODEL_PATH):
                 raise FileNotFoundError("Face detector files (prototxt/caffemodel) not found in the script's directory.")
            
            self.model = tf.keras.models.load_model(MODEL_PATH)
            self.face_net = cv2.dnn.readNet(PROTOTXT_PATH, CAFFEMODEL_PATH)
            
            self.status_label.config(text="All Models Loaded Successfully. Ready to Analyze.", foreground="#4CAF50")
            self.select_button.config(state=tk.NORMAL)

        except Exception as e:
            self.status_label.config(text="Error: Could not load AI models.", foreground="#F44336")
            messagebox.showerror("Model Load Error", f"Could not load required AI models.\n\nError: {e}\n\nPlease ensure all model files are in the same directory as the script.")

    def select_file(self):
        """Opens a file dialog to select a video file."""
        path = filedialog.askopenfilename(
            title="Select a Video File",
            filetypes=(("Video Files", "*.mp4 *.avi *.mov"), ("All files", "*.*"))
        )
        if path:
            self.video_path = path
            self.file_label.config(text=os.path.basename(path))
            self.analyze_button.config(state=tk.NORMAL)
            self.display_first_frame()
            self.clear_results()

    def display_first_frame(self):
        """Displays the first frame of the selected video as a thumbnail."""
        if self.video_path:
            cap = cv2.VideoCapture(self.video_path)
            if cap.isOpened():
                success, frame = cap.read()
                if success:
                    self.update_video_label(frame)
                cap.release()

    def start_analysis(self):
        """Starts the video analysis in a separate thread."""
        if not self.video_path or not self.model or not self.face_net:
            messagebox.showwarning("Warning", "Please select a video file and ensure models are loaded.")
            return

        self.is_analyzing = True
        self.analyze_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.NORMAL)
        self.clear_results()
        self.status_label.config(text="Analyzing...", foreground="#FFC107")

        self.analysis_thread = threading.Thread(target=self.run_video_analysis, daemon=True)
        self.analysis_thread.start()

    def is_good_face_crop(self, face, min_size=MIN_FACE_SIZE):
        """Check if the face crop is good quality for analysis."""
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

    def run_video_analysis(self):
        """The core video analysis logic with improved processing."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.after(0, self.analysis_finished, {"error": "Could not open video file."})
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        predictions = []  # Store all valid predictions
        processed_frames = 0
        (h, w) = (None, None)
        
        frame_num = 0
        while cap.isOpened() and self.is_analyzing:
            success, frame = cap.read()
            if not success:
                break
            
            frame_num += 1
            
            # Skip frames to match training data sampling
            if frame_num % FRAME_SKIP != 0:
                continue
                
            if w is None or h is None:
                (h, w) = frame.shape[:2]

            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
            self.face_net.setInput(blob)
            detections = self.face_net.forward()

            prediction_text = "No Face"
            box_color = (255, 255, 0)  # Yellow
            
            # Find the detection with the highest confidence
            best_detection_idx = -1
            max_confidence = FACE_CONFIDENCE_THRESHOLD
            if detections.shape[2] > 0:
                for i in range(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > max_confidence:
                        max_confidence = confidence
                        best_detection_idx = i

            if best_detection_idx != -1:
                processed_frames += 1
                box = detections[0, 0, best_detection_idx, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Ensure the bounding box is within the frame bounds
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                if startX < endX and startY < endY:
                    face = frame[startY:endY, startX:endX]
                    
                    # Only process good quality face crops
                    if self.is_good_face_crop(face):
                        try:
                            # Add padding around face for better context
                            padding = 20
                            padded_startY = max(0, startY - padding)
                            padded_endY = min(h, endY + padding)
                            padded_startX = max(0, startX - padding)
                            padded_endX = min(w, endX + padding)
                            
                            padded_face = frame[padded_startY:padded_endY, padded_startX:padded_endX]
                            
                            face_resized = cv2.resize(padded_face, IMAGE_SIZE)
                            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                            face_array = img_to_array(face_rgb)
                            face_processed = tf.keras.applications.efficientnet_v2.preprocess_input(np.expand_dims(face_array, axis=0))
                            
                            prediction = self.model.predict(face_processed, verbose=0)[0][0]
                            predictions.append(prediction)

                            # CORRECT LOGIC: High score = Real, Low score = Fake
                            if prediction > DECISION_THRESHOLD:
                                prediction_text = "REAL"
                                box_color = (0, 255, 0)  # Green
                            else:
                                prediction_text = "FAKE"
                                box_color = (0, 0, 255)  # Red
                                
                        except Exception as e:
                            print(f"Error processing face: {e}")
                            prediction_text = "Error"
                            box_color = (0, 165, 255)  # Orange
                    else:
                        prediction_text = "FAKE (Low Quality)"
                        box_color = (0, 0, 255)  # Red for Fake
                        predictions.append(0.0) # Add a strong 'FAKE' score to the list

                    cv2.rectangle(frame, (startX, startY), (endX, endY), box_color, 2)
                    cv2.putText(frame, f"{prediction_text} ({max_confidence:.2%})", (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

            self.update_status_label(f"Analyzing frame {frame_num}/{total_frames} (Processed: {processed_frames})")
            self.update_video_label(frame)

        cap.release()
        
        # Calculate results based on collected predictions
        results = self.calculate_final_results(predictions, processed_frames, total_frames)
        
        if self.is_analyzing:
            self.after(0, self.analysis_finished, results)

    def calculate_final_results(self, predictions, processed_frames, total_frames):
        """Calculate final results using statistical analysis of predictions."""
        if len(predictions) < MIN_FRAMES_FOR_DECISION:
            return {
                "verdict": "INCONCLUSIVE",
                "confidence": 0,
                "reason": f"Too few valid faces detected ({len(predictions)} < {MIN_FRAMES_FOR_DECISION})",
                "processed_frames": processed_frames,
                "total_frames": total_frames,
                "predictions": predictions
            }
        
        # Calculate statistics
        predictions_array = np.array(predictions)
        mean_prediction = np.mean(predictions_array)
        std_prediction = np.std(predictions_array)
        
        # Count predictions on each side of threshold
        fake_count = np.sum(predictions_array < DECISION_THRESHOLD)  # Lower values = fake
        real_count = np.sum(predictions_array >= DECISION_THRESHOLD)  # Higher values = real
        
        # Use mean prediction with confidence based on consistency
        if mean_prediction > DECISION_THRESHOLD:
            verdict = "REAL"
            confidence = mean_prediction * 100  # Higher confidence for higher values
            # Reduce confidence if predictions are inconsistent
            if std_prediction > 0.2:
                confidence *= (1 - std_prediction)
        else:
            verdict = "FAKE"
            confidence = (1 - mean_prediction) * 100  # Higher confidence for lower values
            # Reduce confidence if predictions are inconsistent
            if std_prediction > 0.2:
                confidence *= (1 - std_prediction)
        
        # Cap confidence at reasonable levels
        confidence = min(confidence, 95)
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "mean_prediction": mean_prediction,
            "std_prediction": std_prediction,
            "fake_count": fake_count,
            "real_count": real_count,
            "processed_frames": processed_frames,
            "total_frames": total_frames,
            "predictions": predictions
        }

    def cancel_analysis(self):
        """Stops the analysis thread."""
        if self.is_analyzing:
            self.is_analyzing = False
            self.cancel_button.config(state=tk.DISABLED)
            self.analyze_button.config(state=tk.NORMAL)
            self.status_label.config(text="Analysis cancelled.", foreground="#e0e0e0")

    def analysis_finished(self, results):
        """Updates the UI when analysis is complete."""
        self.is_analyzing = False
        self.analyze_button.config(state=tk.NORMAL)
        self.cancel_button.config(state=tk.DISABLED)
        
        if "error" in results:
            messagebox.showerror("Analysis Error", results["error"])
            self.status_label.config(text="Analysis failed.", foreground="#F44336")
            return
            
        self.status_label.config(text="Analysis Complete.", foreground="#4CAF50")
        self.display_results(results)

    def display_results(self, results):
        """Displays the final analysis results in the UI."""
        # ####################################################### #
        # ##                  UPDATE IS HERE                   ## #
        # ####################################################### #
        # Manually clear the results frame first. This removes the
        # "Results will be displayed here" placeholder before adding new widgets.
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        verdict = results["verdict"]
        confidence = results["confidence"]
        
        if verdict == "INCONCLUSIVE":
            verdict_color = "#FFC107"  # Yellow
        elif verdict == "FAKE":
            verdict_color = "#F44336"  # Red
        else:  # REAL
            verdict_color = "#4CAF50"  # Green

        # Verdict
        verdict_label = ttk.Label(self.results_frame, text=verdict, font=("Segoe UI", 24, "bold"), foreground=verdict_color)
        verdict_label.grid(row=0, column=0, pady=(10,5))
        
        # Confidence
        confidence_text = f"Confidence: {confidence:.2f}%"
        confidence_label = ttk.Label(self.results_frame, text=confidence_text, style="ResultValue.TLabel")
        confidence_label.grid(row=1, column=0, pady=5)
        
        # Show reason for inconclusive results
        if verdict == "INCONCLUSIVE" and "reason" in results:
            reason_label = ttk.Label(self.results_frame, text=results["reason"], style="Status.TLabel", wraplength=200)
            reason_label.grid(row=2, column=0, pady=5)

        # Separator
        separator = ttk.Separator(self.results_frame, orient='horizontal')
        separator.grid(row=3, column=0, sticky='ew', pady=10)

        # Detailed Stats
        stats_frame = ttk.Frame(self.results_frame)
        stats_frame.grid(row=4, column=0, sticky="nsew", padx=10)
        stats_frame.grid_columnconfigure(0, weight=1)
        stats_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(stats_frame, text="Total Frames:", style="ResultValue.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(stats_frame, text=str(results["total_frames"]), style="ResultValue.TLabel").grid(row=0, column=1, sticky="e")
        
        ttk.Label(stats_frame, text="Processed Frames:", style="ResultValue.TLabel").grid(row=1, column=0, sticky="w")
        ttk.Label(stats_frame, text=str(results["processed_frames"]), style="ResultValue.TLabel").grid(row=1, column=1, sticky="e")

        if "fake_count" in results and "real_count" in results:
            ttk.Label(stats_frame, text="Fake Predictions:", style="ResultValue.TLabel").grid(row=2, column=0, sticky="w")
            ttk.Label(stats_frame, text=str(results["fake_count"]), style="ResultValue.TLabel").grid(row=2, column=1, sticky="e")

            ttk.Label(stats_frame, text="Real Predictions:", style="ResultValue.TLabel").grid(row=3, column=0, sticky="w")
            ttk.Label(stats_frame, text=str(results["real_count"]), style="ResultValue.TLabel").grid(row=3, column=1, sticky="e")

        if "mean_prediction" in results:
            ttk.Label(stats_frame, text="Mean Score:", style="ResultValue.TLabel").grid(row=4, column=0, sticky="w")
            ttk.Label(stats_frame, text=f"{results['mean_prediction']:.3f}", style="ResultValue.TLabel").grid(row=4, column=1, sticky="e")

    def clear_results(self):
        """Clears the results from the UI and adds a placeholder."""
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        placeholder = ttk.Label(self.results_frame, text="Results will be displayed here.", style="Status.TLabel", justify=tk.CENTER)
        placeholder.grid(row=0, column=0, sticky="nsew", padx=10, pady=20)
        self.results_frame.grid_rowconfigure(0, weight=1)
        self.results_frame.grid_columnconfigure(0, weight=1)

    def update_video_label(self, frame):
        """Updates the video display label with a new frame."""
        if self.after_id:
            self.after_cancel(self.after_id)
            
        label_width = self.video_label.winfo_width()
        label_height = self.video_label.winfo_height()

        if label_width <= 1 or label_height <= 1:
            self.after_id = self.after(50, self.update_video_label, frame)
            return

        h, w, _ = frame.shape
        scale = min(label_width/w, label_height/h)
        new_w, new_h = int(w*scale), int(h*scale)
        
        if new_w > 0 and new_h > 0:
            resized_frame = cv2.resize(frame, (new_w, new_h))
            img = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            self.photo = ImageTk.PhotoImage(image=img)
            self.video_label.config(image=self.photo)
            self.video_label.image = self.photo
    
    def update_status_label(self, text):
        """Thread-safe way to update a label from the analysis thread."""
        self.after(0, lambda: self.status_label.config(text=text))

if __name__ == "__main__":
    app = DeepfakeDetectorApp()
    app.mainloop()