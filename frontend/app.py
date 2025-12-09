"""
Streamlit Frontend for Deepfake Detection
Upload a video and get prediction results
"""

import streamlit as st
import tempfile
import os
from pathlib import Path
import time
from utils import load_deepfake_model, predict_video_frames, cleanup_temp_files

# Page configuration
st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .result-container {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid;
        backgtound-color: #f9f9f9;
    }
    
    .fake-result {
        background-color: #ffe6e6;
        border-left-color: #ff4444;
    }
    
    .real-result {
        background-color: #e6ffe6;
        border-left-color: #44ff44;
    }
    
    .stats-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .upload-section {
        border: 2px dashed #cccccc;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.model_loaded = False
    st.session_state.prediction_results = None
    st.session_state.processed_video = None

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🕵️ Deepfake Detection System</h1>
        <p>Upload a video to detect if it contains deepfake content</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.info("""
        This system uses advanced deep learning models to detect deepfake videos by analyzing facial features frame by frame.
        
        **Features:**
        - Video upload with preview
        - Face detection and analysis
        - Probability scoring
        - Real-time processing feedback
        """)
        
        st.header("📋 Instructions")
        st.markdown("""
        1. **Upload** a video file (MP4, AVI, MOV)
        2. **Preview** your uploaded video
        3. **Click Submit** to start analysis
        4. **View Results** with probability scores
        """)
        
        # Model loading section
        st.header("🤖 Model Status")
        if not st.session_state.model_loaded:
            if st.button("Load Model", type="primary"):
                with st.spinner("Loading deepfake detection model..."):
                    try:
                        st.session_state.model = load_deepfake_model()
                        st.session_state.model_loaded = True
                        st.success("✅ Model loaded successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error loading model: {str(e)}")
        else:
            st.success("✅ Model ready")
            if st.button("Reload Model"):
                st.session_state.model = None
                st.session_state.model_loaded = False
                st.rerun()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 Upload Video")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file to analyze for deepfake content"
        )
        
        if uploaded_file is not None:
            # Display video info
            st.markdown("### 📹 Video Preview")
            st.video(uploaded_file)
            
            # Video details
            file_details = {
                "Filename": uploaded_file.name,
                "File Size": f"{uploaded_file.size / (1024*1024):.2f} MB",
                "File Type": uploaded_file.type
            }
            
            st.markdown("### 📊 File Details")
            for key, value in file_details.items():
                st.write(f"**{key}:** {value}")
    
    with col2:
        st.header("🔍 Analysis")
        
        if uploaded_file is not None:
            if not st.session_state.model_loaded:
                st.warning("⚠️ Please load the model first using the sidebar")
                
            elif st.button("🚀 Submit for Analysis", type="primary", disabled=not st.session_state.model_loaded):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    temp_video_path = tmp_file.name
                
                try:
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("🔍 Analyzing video...")
                    progress_bar.progress(25)
                    
                    # Run prediction
                    with st.spinner("Processing video frames..."):
                        results = predict_video_frames(
                            st.session_state.model, 
                            temp_video_path,
                            progress_callback=lambda p: progress_bar.progress(25 + int(p * 0.7))
                        )
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    
                    # Store results
                    st.session_state.prediction_results = results
                    st.session_state.processed_video = uploaded_file.name
                    
                    # Clean up
                    cleanup_temp_files(temp_video_path)
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    cleanup_temp_files(temp_video_path)
                
        else:
            st.info("👆 Please upload a video file to begin analysis")
    
    # Results section
    if st.session_state.prediction_results:
        st.markdown("---")
        display_results(st.session_state.prediction_results, st.session_state.processed_video)

def display_results(results, video_name):
    """Display prediction results in a formatted way"""
    
    st.header("📊 Analysis Results")
    
    if results is None:
        st.error("❌ No faces detected in the video. Please try with a video that contains visible faces.")
        return
    
    # Main verdict
    verdict = results['verdict']
    avg_fake_prob = results['avg_fake_probability']
    max_fake_prob = results['max_fake_probability']
    confidence = results['confidence']
    frames_analyzed = results['frames_analyzed']
    
    # Display main result
    result_class = "fake-result" if verdict == "FAKE" else "real-result"
    verdict_emoji = "🚨" if verdict == "FAKE" else "✅"
    
    st.markdown(f"""
    <div class="result-container {result_class}">
        <h2>{verdict_emoji} Prediction: {verdict}</h2>
        <h3>Confidence: {confidence:.1f}%</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Average Fake Probability",
            value=f"{avg_fake_prob:.1f}%",
            delta=f"{avg_fake_prob - 50:.1f}%" if avg_fake_prob != 50 else None
        )
    
    with col2:
        st.metric(
            label="Maximum Fake Probability", 
            value=f"{max_fake_prob:.1f}%",
            delta=f"{max_fake_prob - 50:.1f}%" if max_fake_prob != 50 else None
        )
    
    with col3:
        st.metric(
            label="Frames Analyzed",
            value=frames_analyzed
        )
    
    # Progress bars for probabilities
    st.markdown("### 📈 Probability Breakdown")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Real Probability:**")
        st.progress(max(0, min(100, 100 - avg_fake_prob)) / 100)
        st.write(f"{100 - avg_fake_prob:.1f}%")
    
    with col2:
        st.write("**Fake Probability:**")
        st.progress(max(0, min(100, avg_fake_prob)) / 100)
        st.write(f"{avg_fake_prob:.1f}%")
    
    # Analysis summary
    st.markdown(f"""
    <div class="stats-container">
        <h4>📋 Analysis Summary</h4>
        <ul>
            <li><strong>Video:</strong> {video_name}</li>
            <li><strong>Frames Analyzed:</strong> {frames_analyzed}</li>
            <li><strong>Detection Method:</strong> Face-based deep learning analysis</li>
            <li><strong>Model:</strong> EfficientNet-based Deepfake Detector</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Interpretation guide
    st.markdown("### 💡 How to Interpret Results")
    
    if avg_fake_prob > 70:
        st.error("**High Risk**: Strong indication of deepfake content")
    elif avg_fake_prob > 50:
        st.warning("**Medium Risk**: Possible deepfake content, manual review recommended")
    else:
        st.success("**Low Risk**: Content appears to be authentic")
    
    # Download results option
    if st.button("📥 Download Results"):
        import json
        results_json = json.dumps(results, indent=2)
        st.download_button(
            label="Download JSON Report",
            data=results_json,
            file_name=f"deepfake_analysis_{video_name}_{int(time.time())}.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()