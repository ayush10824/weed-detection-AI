# Save this as `app.py` in the same directory as your `best.pt` file

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import time
import os # Added for font path check

# --- 1. FastAPI App Setup ---
app = FastAPI(
    title="Weed Detection & Classification API",
    description="API for deep learning-based weed detection and classification using YOLO."
)

# --- CORS Configuration ---
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5173",
    "null", # Allows requests from file:// (when you open index.html directly)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. Load YOLO Model and Define Class Information ---
# IMPORTANT: Ensure your 'best.pt' file is in the same directory as this app.py
try:
    model = YOLO('best.pt') # Using your trained model
    print("YOLO model ('best.pt') loaded successfully!")
except Exception as e:
    print(f"Error loading YOLO model ('best.pt'): {e}")
    print("Please ensure 'best.pt' is in the same directory as 'app.py'.")
    # You might want to raise an exception or exit the app if the model cannot be loaded
    # For demonstration, we'll try to continue but expect issues.

# Define class names and a color map for visualization
# ENSURE THESE MATCH THE CLASSES YOUR MODEL WAS TRAINED ON!
class_names = ['carpetweed', 'morningglory', 'palmer_amaranth']
class_color_map = {
    'carpetweed': '#FF0000',  # Red (for drawing, matches Tailwind red-500 if possible)
    'morningglory': '#0000FF', # Blue (for drawing, matches Tailwind blue-500)
    'palmer_amaranth': '#008000' # Green (for drawing, matches Tailwind green-500)
}
default_color = '#FFFF00' # Yellow fallback

# Try to load a font for drawing text on the image
# You might need to adjust this path or provide a font file
font_path = "arial.ttf" # Or "Roboto-Regular.ttf" if you download one
if os.path.exists(font_path):
    try:
        font = ImageFont.truetype(font_path, 18)
    except IOError:
        print(f"Could not load {font_path}, using default PIL font.")
        font = ImageFont.load_default()
else:
    print(f"Font file '{font_path}' not found, using default PIL font. Text rendering might differ.")
    font = ImageFont.load_default()


# --- 3. Helper Function to Draw Bounding Boxes and Encode Image ---
def draw_and_encode_image(image: Image.Image, detections: list, class_names: list, class_color_map: dict) -> str:
    draw = ImageDraw.Draw(image)
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        class_name = det['class_name']
        confidence = det['confidence']
        
        # Get color from the map, fall back to default
        color = class_color_map.get(class_name, default_color)
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label text with class and confidence
        label_text = f"{class_name} ({confidence:.2f})"
        
        # Get text size for background rectangle
        try:
            text_bbox = draw.textbbox((0,0), label_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except AttributeError: # Fallback for older Pillow versions
            text_width, text_height = draw.textsize(label_text, font=font)

        # Ensure label background is positioned correctly (above box, within image bounds)
        text_x = x1
        text_y = max(y1 - text_height - 5, 0)

        draw.rectangle([text_x, text_y, text_x + text_width + 10, text_y + text_height + 5], fill=color)
        draw.text((text_x + 5, text_y + 2), label_text, fill="white", font=font)

    # Convert annotated image to base64 for frontend display
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

# --- 4. Prediction Endpoint ---
@app.post("/detect")
async def detect_weeds(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        start_time = time.time()
        
        # Perform inference using the loaded YOLO model
        # conf=0.25 is a common confidence threshold, adjust if needed
        results = model.predict(source=image, conf=0.25)
        
        end_time = time.time()
        processing_time = f"{((end_time - start_time) * 1000):.2f} ms"

        all_detections = []
        weed_counts = {name: 0 for name in class_names} # Initialize counts for each class
        total_confidence = 0
        
        for r in results: # 'r' is a Results object for a single image
            boxes = r.boxes.xyxy.tolist()
            confs = r.boxes.conf.tolist()
            clss = r.boxes.cls.tolist()
            
            for i in range(len(boxes)):
                class_id = int(clss[i])
                confidence = confs[i]
                
                # Get class name, with a fallback for robustness
                class_name = class_names[class_id] if class_id < len(class_names) else f"unknown_{class_id}"
                
                all_detections.append({
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": boxes[i]
                })
                
                # Increment count for the detected class
                if class_name in weed_counts:
                    weed_counts[class_name] += 1
                
                total_confidence += confidence
        
        total_weeds = len(all_detections) # Total number of detections
        avg_confidence = (total_confidence / total_weeds) if total_weeds > 0 else 0

        # Draw detections on a copy of the original image
        annotated_image_base64 = draw_and_encode_image(image.copy(), all_detections, class_names, class_color_map)

        return JSONResponse(content={
            "success": True,
            "detections": all_detections,
            "total_weeds": total_weeds,
            "weed_counts": weed_counts, # Send classification counts to frontend
            "processing_time": processing_time,
            "model_confidence": avg_confidence,
            "annotated_image": annotated_image_base64
        })

    except Exception as e:
        print(f"Error during detection: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

# To run this app:
# 1. Save this code as `app.py`.
# 2. Place your `best.pt` model file in the same directory.
# 3. Open your terminal in that directory.
# 4. Run: `uvicorn app:app --reload --port 8000`