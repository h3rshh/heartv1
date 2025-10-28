import os
import numpy as np
import librosa
import pickle
import tensorflow as tf
from scipy import signal
import warnings
import tempfile
import base64
from typing import Dict, Any, Union
from io import BytesIO
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

warnings.filterwarnings("ignore")

# FIXED: Use correct parameters matching training
TARGET_SR = 2000
TARGET_DURATION = 10
SEGMENT_LENGTH = 20000
N_FRAMES = 157
N_FFT = 512
HOP_LENGTH = 128  # FIXED: Use 128 like in training, not 256

class PredictionResponse(BaseModel):
    label: str
    confidence: float
    probabilities: Dict[str, float]
    status: str = "success"
    debug_info: Dict[str, Any] = {}

class ErrorResponse(BaseModel):
    error: str
    status: str = "error"

class HeartPredictor:
    def __init__(self):
        model_path = "best_heart_model.keras"
        if not os.path.exists(model_path):
            raise RuntimeError(f"Model file not found: {model_path}")
        
        self.model = tf.keras.models.load_model(model_path, compile=False)
        print("Heart model loaded successfully")
        
        # Load normalization parameters from training
        self.load_normalization_params()
        self.load_class_names()

    def load_normalization_params(self):
        """Load the saved normalization parameters from training"""
        norm_file = "comprehensive_norm_params.pkl"
        if os.path.exists(norm_file):
            with open(norm_file, 'rb') as f:
                self.norm_params = pickle.load(f)
            print("Loaded training normalization parameters")
        else:
            print("Warning: No normalization parameters found, using per-sample normalization")
            self.norm_params = None

    def load_class_names(self):
        class_file = "heart_class_names.pkl"
        if os.path.exists(class_file):
            try:
                with open(class_file, 'rb') as f:
                    self.class_names = pickle.load(f)
                print(f"Loaded class names: {self.class_names}")
            except:
                self.class_names = ["Normal", "Abnormal"]
        else:
            self.class_names = ["Normal", "Abnormal"]

    def denoise_audio(self, audio, sr):
        """Apply same denoising as training"""
        methods = ['adaptive_median', 'bandpass']
        denoised = audio.copy()
        
        for method in methods:
            if method == 'adaptive_median':
                window_size = int(sr * 0.01)
                if window_size % 2 == 0:
                    window_size += 1
                denoised = signal.medfilt(denoised, window_size)
                
            elif method == 'bandpass':
                nyquist = sr / 2
                low = 20 / nyquist
                high = 400 / nyquist
                b, a = signal.butter(4, [low, high], btype='band')
                denoised = signal.filtfilt(b, a, denoised)
        
        return denoised

    def extract_features(self, audio_data, sr):
        """FIXED: Use correct hop_length parameter"""
        n_fft = N_FFT
        hop_length = HOP_LENGTH  # 128, not 256
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data, sr=sr, n_mels=64, n_fft=n_fft, hop_length=hop_length)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13, hop_length=hop_length)
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, hop_length=hop_length)
        
        return {"mel_spec": mel_spec_db, "mfcc": mfcc, "chroma": chroma}

    def pad_or_crop(self, arr, shape):
        """Pad or crop array to target shape"""
        out = np.zeros(shape, dtype=arr.dtype)
        n_feat, n_fr = arr.shape
        out[:min(n_feat, shape[0]), :min(n_fr, shape[1])] = arr[:shape[0], :shape[1]]
        return out

    def prepare_input(self, features):
        """Prepare input exactly as in training"""
        mfcc = self.pad_or_crop(features["mfcc"], (13, N_FRAMES))
        chroma = self.pad_or_crop(features["chroma"], (12, N_FRAMES))
        mspec = self.pad_or_crop(features["mel_spec"], (64, N_FRAMES))
        
        return (
            mfcc[..., np.newaxis][np.newaxis, ...],
            chroma[..., np.newaxis][np.newaxis, ...],
            mspec[..., np.newaxis][np.newaxis, ...]
        )

    def normalize_with_training_stats(self, X, feature_type):
        """FIXED: Use training statistics for normalization"""
        if self.norm_params is None:
            # Fallback to per-sample normalization
            X_flat = X.reshape(X.shape[0], -1)
            mean = X_flat.mean(axis=1, keepdims=True)
            std = X_flat.std(axis=1, keepdims=True) + 1e-8
            X_normalized = (X_flat - mean) / std
            return X_normalized.reshape(X.shape)
        
        # Use saved training statistics
        norm_data = self.norm_params[feature_type]
        train_mean = norm_data['mean']
        train_std = norm_data['std']
        
        X_flat = X.reshape(X.shape[0], -1)
        X_normalized = (X_flat - train_mean) / train_std
        return X_normalized.reshape(X.shape)

    def process_audio_from_bytes(self, audio_bytes: bytes) -> np.ndarray:
        """Process audio from bytes"""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            
            audio, sr = librosa.load(tmp_path, sr=TARGET_SR)
            os.unlink(tmp_path)
            return audio
            
        except Exception as e1:
            try:
                audio_io = BytesIO(audio_bytes)
                audio, sr = sf.read(audio_io)
                
                if sr != TARGET_SR:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
                
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1)
                    
                return audio
                
            except Exception as e2:
                raise ValueError(f"Could not process audio file: {e1}, {e2}")

    def predict(self, audio_input: Union[bytes, np.ndarray], debug=False) -> Dict[str, Any]:
        """FIXED: Use single segment processing like Kaggle version"""
        try:
            if isinstance(audio_input, bytes):
                audio = self.process_audio_from_bytes(audio_input)
            else:
                audio = audio_input

            # FIXED: Process as single segment like Kaggle version
            # Ensure exact segment length
            if len(audio) < SEGMENT_LENGTH:
                audio = np.pad(audio, (0, SEGMENT_LENGTH - len(audio)), 'constant')
            elif len(audio) > SEGMENT_LENGTH:
                audio = audio[:SEGMENT_LENGTH]

            # Process the single segment
            result = self._predict_segment(audio, debug)
            
            # Get prediction probability
            pred_prob = result['raw_output']
            
            # Convert to final prediction
            prob_abnormal = pred_prob
            prob_normal = 1.0 - pred_prob
            
            if prob_abnormal > prob_normal:
                label = self.class_names[1]  # "Abnormal"
                confidence = prob_abnormal
            else:
                label = self.class_names[0]  # "Normal" 
                confidence = prob_normal

            debug_info = {}
            if debug:
                debug_info = {
                    "audio_length": len(audio),
                    "raw_prediction": pred_prob,
                    "normalization_stats": result.get("normalization_check", {}),
                    "audio_stats": result.get("audio_stats", {})
                }

            return {
                "label": label,
                "confidence": confidence,
                "probabilities": {
                    self.class_names[0]: prob_normal,
                    self.class_names[1]: prob_abnormal
                },
                "debug_info": debug_info
            }

        except Exception as e:
            return {
                "label": "Error",
                "confidence": 0.0,
                "probabilities": {self.class_names[0]: 0.0, self.class_names[1]: 0.0},
                "debug_info": {"error": str(e)}
            }
    
    def _predict_segment(self, audio_segment, debug=False):
        """Predict on a single segment using training methodology"""
        # Apply same denoising as training
        denoised_audio = self.denoise_audio(audio_segment, TARGET_SR)
        
        # Extract features exactly as in training
        features = self.extract_features(denoised_audio, TARGET_SR)
        
        # Prepare inputs exactly as in training  
        X_mfcc, X_chroma, X_mspec = self.prepare_input(features)

        # FIXED: Use training normalization approach
        X_mfcc_norm = self.normalize_with_training_stats(X_mfcc, 'mfcc')
        X_chroma_norm = self.normalize_with_training_stats(X_chroma, 'chroma') 
        X_mspec_norm = self.normalize_with_training_stats(X_mspec, 'mspec')

        # Get model prediction
        raw_prediction = self.model.predict([X_mfcc_norm, X_chroma_norm, X_mspec_norm], verbose=0)
        pred_prob = float(raw_prediction[0][0])
        
        result = {
            "raw_output": pred_prob,
            "audio_stats": {
                "mean": float(np.mean(denoised_audio)),
                "std": float(np.std(denoised_audio))
            } if debug else {}
        }
        
        if debug:
            result["normalization_check"] = {
                "mfcc_norm_mean": float(np.mean(X_mfcc_norm)),
                "mfcc_norm_std": float(np.std(X_mfcc_norm)),
                "chroma_norm_mean": float(np.mean(X_chroma_norm)),
                "chroma_norm_std": float(np.std(X_chroma_norm)),
                "mspec_norm_mean": float(np.mean(X_mspec_norm)),
                "mspec_norm_std": float(np.std(X_mspec_norm))
            }
        
        return result

# FastAPI app
app = FastAPI(title="Heart Sound Classifier API", version="1.0.0")
predictor = None

@app.on_event("startup")
async def load_model():
    global predictor
    try:
        predictor = HeartPredictor()
        print("Heart Sound Classifier loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise

@app.get("/")
async def root():
    return {"message": "Heart Sound Classifier API", "status": "healthy", "version": "1.0.0"}

@app.get("/health")
async def health():
    if predictor is None:
        return {"status": "unhealthy", "error": "Model not loaded"}
    return {"status": "healthy", "model_loaded": True}

@app.get("/classes")
async def classes():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"classes": predictor.class_names, "num_classes": len(predictor.class_names)}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), debug: bool = False):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.filename.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
        raise HTTPException(status_code=400, detail="Unsupported audio format")
    
    try:
        audio_bytes = await file.read()
        result = predictor.predict(audio_bytes, debug=debug)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-base64", response_model=PredictionResponse)
async def predict_base64(data: dict):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if "audio" not in data:
        raise HTTPException(status_code=400, detail="Missing 'audio' field")
    
    try:
        debug = data.get("debug", False)
        audio_bytes = base64.b64decode(data["audio"])
        result = predictor.predict(audio_bytes, debug=debug)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/debug-predict")
async def debug_predict(file: UploadFile = File(...)):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        audio_bytes = await file.read()
        result = predictor.predict(audio_bytes, debug=True)
        return result
    except Exception as e:
        return {"error": str(e), "status": "error"}

@app.get("/model-info")
async def model_info():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "target_sampling_rate": TARGET_SR,
        "target_duration": TARGET_DURATION,
        "segment_length": SEGMENT_LENGTH,
        "n_frames": N_FRAMES,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "model_inputs": [list(inp.shape) for inp in predictor.model.inputs],
        "class_names": predictor.class_names
    }

if __name__ == "__main__":
    print("Starting Heart Sound Classifier API...")
    print(f"Target sampling rate: {TARGET_SR} Hz")
    print(f"Target duration: {TARGET_DURATION} seconds") 
    print(f"Expected input length: {SEGMENT_LENGTH} samples")
    uvicorn.run(app, host="0.0.0.0", port=7860)