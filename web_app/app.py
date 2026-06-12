import os
import sys
import io
import base64
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.v2 as T
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Add the src folder to Python search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from configs import config
from models import GreenMultimodalSiamese
from utils.xai import RightEyeWrapper, LeftEyeWrapper, compute_gradcam
from utils.visualization import tensor_to_rgb

# Create static directories if they don't exist
os.makedirs(os.path.join(os.path.dirname(__file__), 'static'), exist_ok=True)

app = FastAPI(title="CIMT Multimodal CVD Predictor")

# Load Model Globals
model = None
target_layer = None

# Optimized calibration parameters (from validation sets)
OPTIMAL_T = 2.0
THRESH_YOUDEN = 0.380  # Balanced threshold
THRESH_SENS85 = 0.250  # Sensitivity prioritized threshold

@app.on_event("startup")
def load_model():
    global model, target_layer
    print("Initializing RETFound-Green Multimodal model...")
    
    # Locate backbone weights
    rt_weights = config.WEIGHTS_PATH
    if not os.path.exists(rt_weights):
        rt_weights = os.path.abspath(os.path.join(os.path.dirname(__file__), '../RETFound_oct_weights.pth'))
        if not os.path.exists(rt_weights):
            rt_weights = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/RETFound_oct_weights.pth'))

    # Locate fine-tuned weights
    ft_weights = os.path.abspath(os.path.join(os.path.dirname(__file__), '../weights/best_cimt_green_multimodal_v2.pth'))
    if not os.path.exists(ft_weights):
        ft_weights = os.path.abspath(os.path.join(os.path.dirname(__file__), '../best_cimt_green_multimodal_v2.pth'))
        if not os.path.exists(ft_weights):
            raise FileNotFoundError(f"Fine-tuned weights not found. Please train the model first.")

    model = GreenMultimodalSiamese(
        weights_path=rt_weights,
        img_size=config.IMG_SIZE,
        clinical_in_dim=3,
        clinical_feat_dim=config.CLINICAL_FEAT_DIM
    ).to(config.DEVICE)
    
    print(f"Loading checkpoint weights from {ft_weights}...")
    ckpt = torch.load(ft_weights, map_location=config.DEVICE)
    if 'model' in ckpt:
        state_dict = ckpt['model']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
        
    model_keys = set(model.state_dict().keys())
    new_state_dict = {}
    for k, v in state_dict.items():
        # Map old-style clinical_mlp to new-style
        if k.startswith("clinical_mlp.") and not k.startswith("clinical_mlp.clinical_mlp."):
            new_key = k.replace("clinical_mlp.", "clinical_mlp.clinical_mlp.", 1)
            if new_key in model_keys:
                new_state_dict[new_key] = v
                continue
        # Map new-style clinical_mlp to old-style
        elif k.startswith("clinical_mlp.clinical_mlp."):
            new_key = k.replace("clinical_mlp.clinical_mlp.", "clinical_mlp.", 1)
            if new_key in model_keys:
                new_state_dict[new_key] = v
                continue
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    # Hook the target layer for CAM calculations
    target_layer = model.backbone.blocks[-1].norm1
    print("Model successfully loaded and hooked.")

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Preprocesses upload image bytes to match dataset base transformations."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    tf = T.Compose([
        T.ToImage(),
        T.Resize((config.IMG_SIZE, config.IMG_SIZE), antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tf(img).unsqueeze(0).to(config.DEVICE)

def array_to_base64(arr: np.ndarray) -> str:
    """Converts a uint8 RGB image array to a base64 encoded source string."""
    img = Image.fromarray(arr)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

@app.post("/predict")
async def predict(
    right_eye: UploadFile = File(...),
    left_eye: UploadFile = File(...),
    age: float = Form(...),
    gender: str = Form(...)  # "male" or "female"
):
    global model, target_layer
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    try:
        # 1. Read files and preprocess inputs
        right_bytes = await right_eye.read()
        left_bytes = await left_eye.read()

        right_tensor = preprocess_image(right_bytes)
        left_tensor = preprocess_image(left_bytes)

        # Normalize age (assuming min-max parameters from train/val set, let's normalize between 40 and 85)
        # Using a general min-max normalization: (age - 40) / 45
        age_norm = (age - 40.0) / 45.0
        age_norm = max(0.0, min(1.0, age_norm))

        gender_female = 1.0 if gender.lower() == "female" else 0.0
        gender_male = 1.0 if gender.lower() == "male" else 0.0

        clinical_tensor = torch.tensor([[
            age_norm,
            gender_female,
            gender_male
        ]], dtype=torch.float32).to(config.DEVICE)

        # 2. Run Inference
        with torch.no_grad():
            logits = model(right_tensor, left_tensor, clinical_tensor)
            
            # Apply Temperature Scaling Calibration
            calibrated_logits = logits / OPTIMAL_T
            calibrated_prob = torch.sigmoid(calibrated_logits).item()

            # Encode eye features for isolating context during GradCAM
            fixed_right_feat = model.encode_image(right_tensor)
            fixed_left_feat = model.encode_image(left_tensor)
            fixed_clin_feat = model.encode_clinical(clinical_tensor)

        # Determine diagnostic categories
        is_thickened_youden = calibrated_prob >= THRESH_YOUDEN
        is_thickened_sens85 = calibrated_prob >= THRESH_SENS85

        # 3. Generate GradCAM++ Heatmaps (Right & Left)
        right_wrapper = RightEyeWrapper(model, fixed_left_feat.detach(), fixed_clin_feat.detach()).to(config.DEVICE)
        left_wrapper = LeftEyeWrapper(model, fixed_right_feat.detach(), fixed_clin_feat.detach()).to(config.DEVICE)

        right_overlay, _ = compute_gradcam(right_wrapper, right_tensor, target_layer)
        left_overlay, _ = compute_gradcam(left_wrapper, left_tensor, target_layer)

        # Convert raw tensors back to normal display images
        right_rgb = (tensor_to_rgb(right_tensor) * 255).astype(np.uint8)
        left_rgb = (tensor_to_rgb(left_tensor) * 255).astype(np.uint8)

        # 4. Package response with Base64 representations
        return {
            "probability": calibrated_prob,
            "prediction_youden": "Thickened" if is_thickened_youden else "Normal",
            "prediction_sens85": "Thickened" if is_thickened_sens85 else "Normal",
            "threshold_youden": THRESH_YOUDEN,
            "threshold_sens85": THRESH_SENS85,
            "right_original": array_to_base64(right_rgb),
            "right_gradcam": array_to_base64(right_overlay),
            "left_original": array_to_base64(left_rgb),
            "left_gradcam": array_to_base64(left_overlay),
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Mount Static Files (placed at end to allow path prioritization)
app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), 'static'), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
