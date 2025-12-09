# Deepfake Detection Frontend

A Streamlit web application for detecting deepfake content in videos using a trained deep learning model.

## Features

- 📁 **Video Upload**: Support for multiple video formats (MP4, AVI, MOV, MKV)
- 📹 **Video Preview**: Preview uploaded videos before analysis
- 🤖 **AI-Powered Detection**: Uses EfficientNet-based deep learning models
- 📊 **Detailed Results**: Shows prediction confidence, probabilities, and analysis metrics
- 🎨 **User-Friendly Interface**: Clean, modern UI with progress tracking
- 📥 **Export Results**: Download analysis results as JSON reports

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster inference)
- Trained deepfake detection model (should be located at `../models/finetuned/finetuned_model.pth`)

## Installation

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install required dependencies:**
   ```bash
   pip install -r ../requirements.txt
   ```

   This will install all necessary packages including:
   - streamlit
   - torch & torchvision
   - opencv-python
   - facenet-pytorch
   - timm
   - PIL/Pillow
   - numpy

## Running the Application

1. **Start the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Open your web browser** and navigate to:
   ```
   http://localhost:8501
   ```

3. **Load the model** using the sidebar button when the app starts

4. **Upload a video** and click "Submit for Analysis"

## How It Works

1. **Face Detection**: The system uses MTCNN to detect faces in video frames
2. **Frame Sampling**: Analyzes every 30th frame to balance accuracy and speed
3. **Deep Learning Analysis**: Each detected face is processed through the trained model
4. **Probability Aggregation**: Results from all frames are combined to give final prediction
5. **Result Display**: Shows verdict (REAL/FAKE) with confidence scores and probabilities

## Model Architecture

The application supports two model architectures:
- **DeepfakeDetector**: EfficientNet-B4 with attention mechanism
- **FastDeepfakeDetector**: EfficientNet-B0 for faster inference

The app automatically detects which model architecture to use based on the saved checkpoint.

## Usage Tips

- **Video Quality**: Higher quality videos with clear facial features work best
- **Video Length**: Longer videos provide more data points for analysis
- **Face Visibility**: Ensure faces are clearly visible and well-lit
- **File Size**: Large files may take longer to process

## Troubleshooting

### Model Loading Issues
- Ensure the model file exists at `../models/finetuned/finetuned_model.pth`
- Check that you have sufficient GPU memory (or the model will run on CPU)
- Verify all dependencies are installed correctly

### Video Processing Issues
- Supported formats: MP4, AVI, MOV, MKV
- If no faces are detected, try with videos containing clearer facial features
- Large video files may require more processing time

### Performance Issues
- Use GPU for faster inference (CUDA required)
- Reduce frame step size in `utils.py` for more detailed analysis (slower)
- Consider using smaller video files for testing

## File Structure

```
frontend/
├── app.py          # Main Streamlit application
├── utils.py        # Model loading and inference utilities
└── README.md       # This file
```

## Configuration

You can modify these settings in `utils.py`:
- `CONFIDENCE_THRESHOLD`: Threshold for fake detection (default: 0.5)
- `IMG_SIZE`: Input image size for the model (default: 224)
- `frame_step`: Frame sampling rate in videos (default: 30)

## API Reference

### Main Functions

- `load_deepfake_model()`: Loads the trained model
- `predict_video_frames(model, video_path, frame_step, progress_callback)`: Analyzes video
- `cleanup_temp_files(file_path)`: Cleans up temporary files

### Result Format

```json
{
    "verdict": "FAKE" | "REAL",
    "avg_fake_probability": float,
    "max_fake_probability": float,
    "frames_analyzed": int,
    "confidence": float,
    "frames_processed": int,
    "total_frames": int
}
```

## Contributing

When modifying the application:
1. Update model paths if needed
2. Test with various video formats
3. Ensure error handling works properly
4. Update documentation for any new features

## Support

If you encounter issues:
1. Check the console output for detailed error messages
2. Verify model file integrity
3. Ensure all dependencies are properly installed
4. Check GPU memory if using CUDA

## License

This project is part of the Final Year Project for deepfake detection research.