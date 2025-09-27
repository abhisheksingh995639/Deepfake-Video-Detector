# Deepfake Video Classification using EfficientNetV2B0

A deep learning system for detecting deepfake videos using the EfficientNetV2B0 architecture with a user-friendly desktop application interface.

## Overview

This project addresses the critical challenge of detecting synthetically generated video content by implementing a high-performance deep learning system. The solution provides an end-to-end pipeline from data processing to a polished desktop application that effectively classifies videos as either authentic ('real') or manipulated ('fake').

## Features

- **High-Performance Model**: Achieves 89.5% validation accuracy using EfficientNetV2B0 architecture
- **Automated Face Detection**: Uses OpenCV DNN-based face detector for precise face localization
- **Quality Filtering**: Intelligent filtering based on face size and blur detection (Laplacian variance)
- **Transfer Learning**: Leverages pre-trained ImageNet weights with fine-tuning strategy
- **Desktop Application**: User-friendly GUI built with Tkinter
- **Real-time Analysis**: Live video preview with frame-by-frame prediction display

## Architecture

The system consists of several modular components:

1. **Dataset Consolidation Module** (`consolidate_dataset.py`)
2. **Preprocessing and Face Extraction** (within `train_model.py`)
3. **Model Training Module** (`train_model.py`)
4. **Desktop Application** (`desktop_app.py`)

### Model Architecture

- **Input**: 224x224x3 images with data augmentation
- **Base Model**: Pre-trained EfficientNetV2B0 (frozen during initial training)
- **Classification Head**:
  - GlobalAveragePooling2D
  - Dropout (0.3)
  - Dense layer (128 neurons, ReLU)
  - Dropout (0.2)
  - Output layer (1 neuron, sigmoid)

## Requirements

### Hardware Requirements
- **GPU**: NVIDIA RTX 3060 Ti (8GB VRAM) or equivalent
- **RAM**: 16 GB
- **Storage**: 50 GB available space

### Software Requirements
- **OS**: Windows 10/11, macOS, or Linux
- **Python**: 3.9+
- **CUDA**: 11.2+ (for GPU acceleration)

### Dependencies
```
tensorflow>=2.8.0
opencv-python>=4.5.0
numpy>=1.21.0
Pillow>=8.3.0
tkinter (included with Python)
```

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd deepfake-detection
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Download required models**:
   - Download the pre-trained face detection model files
   - Place them in the appropriate directory as specified in the code

## Usage

### Training the Model

1. **Prepare the dataset**:
```bash
python consolidate_dataset.py
```
This will create a balanced dataset structure from the FaceForensics++ source data.

2. **Train the model**:
```bash
python train_model.py
```
This will:
- Extract faces from videos
- Apply quality filtering
- Train the EfficientNetV2B0 model
- Save the best model as `best_deepfake_model_effnet.keras`

### Running the Desktop Application

```bash
python desktop_app.py
```

**Using the Application**:
1. Click "Select Video" to choose a video file
2. Click "Analyze Video" to start the detection process
3. View real-time analysis with bounding boxes and predictions
4. Review the final verdict and detailed statistics

## Dataset

The project uses the **FaceForensics++** dataset (C23 compression quality):
- **Source**: Contains original videos and various deepfake generation methods
- **Training Subset**: 200 videos per class (real/fake)
- **Final Dataset**: 12,090 extracted face images (224x224 pixels)
- **Split**: 80% training, 20% validation

## Performance

- **Validation Accuracy**: 89.5%
- **Model Size**: Optimized for balance between accuracy and efficiency
- **Processing Speed**: Analyzes every 5th frame for real-time performance

## Results

The trained model demonstrates:
- Strong generalization capability (validation accuracy closely tracks training accuracy)
- Effective learning of deepfake artifacts
- Robust performance on the validation dataset
- Successful integration into a practical application

## Limitations

- **Dataset Scope**: Trained exclusively on FaceForensics++ dataset
- **Generalization**: May have reduced performance on newer deepfake techniques not in training data
- **Face Dependency**: Requires clear, well-defined faces for analysis
- **Single Modality**: Analyzes only visual content (no audio analysis)

## Future Enhancements

- **Expanded Training Data**: Include multiple datasets and "in-the-wild" examples
- **Temporal Analysis**: Incorporate RNNs or Transformers for frame sequence analysis
- **Multimodal Analysis**: Add audio analysis capabilities
- **Web Service**: Deploy as scalable cloud-based service
- **Real-time Optimization**: Optimize for live video stream analysis

## Project Structure

```
deepfake-detection/
├── consolidate_dataset.py      # Dataset preparation
├── train_model.py             # Model training and preprocessing
├── desktop_app.py             # GUI application
├── best_deepfake_model_effnet.keras  # Trained model (generated)
├── requirements.txt           # Python dependencies
├── README.md                 # This file
└── models/                   # Pre-trained face detection models
```

## Contributing

This project was developed as part of an academic research initiative. For contributions or improvements:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Abhishek Singh and Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Please ensure compliance with dataset licenses and terms of use when using this software.

## Citation

If you use this work in your research, please cite:

```
Deepfake Video Classification using the EfficientNetV2B0 Architecture
Authors: Abhishek Singh, Gaurav Gautam, Yatin, Mradul Agrawal, Reenul Sirsat
Institution: VIT Bhopal University
Year: 2025
```

## Acknowledgments

- **VIT Bhopal University** - School of Computer Science and Engineering
- **Project Guide**: Dr. Manorama Chouhan
- **FaceForensics++ Dataset** creators
- **TensorFlow** and **OpenCV** communities

## Contact

For questions, support, or collaboration opportunities:

**Lead Developer**: Abhishek Singh
- Email: [abhisheksingh995639@gmail.com](mailto:abhisheksingh995639@gmail.com)
- LinkedIn: [https://www.linkedin.com/in/abhisheksingh995639](https://www.linkedin.com/in/abhisheksingh995639)

For institutional inquiries, please contact through VIT Bhopal University's official channels.

---

**Note**: This system is designed for research and educational purposes. Always verify results with additional methods when used in critical applications.
