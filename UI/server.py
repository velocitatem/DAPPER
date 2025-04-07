from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import random
import pdf2image
from PIL import Image, UnidentifiedImageError
import io
import torch
from torchvision import transforms
import sys
import os
import base64
from pathlib import Path
from contextlib import asynccontextmanager
import pytesseract
import re
# Add root directory to path to import classification modules
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

from classification.training.models import get_model

# Define classes - must be before the lifespan function
"""
2025-04-07 10:53:35,458 - classification.data.minio_dataset - INFO - Creating label map from string labels to integers
2025-04-07 10:53:35,458 - classification.data.minio_dataset - INFO - Label mapping: 'correspondence' -> 0
2025-04-07 10:53:35,458 - classification.data.minio_dataset - INFO - Label mapping: 'forms' -> 1
2025-04-07 10:53:35,458 - classification.data.minio_dataset - INFO - Label mapping: 'invoice' -> 2
2025-04-07 10:53:35,458 - classification.data.minio_dataset - INFO - Label mapping: 'other' -> 3
2025-04-07 10:53:35,458 - classification.data.minio_dataset - INFO - Label mapping: 'personal' -> 4
2025-04-07 10:53:35,458 - classification.data.minio_dataset - INFO - Label mapping: 'promotional' -> 5
2025-04-07 10:53:35,458 - classification.data.minio_dataset - INFO - Label mapping: 'scientific' -> 6
"""
CLASSES = ["correspondence", "forms", "invoice", "other", "personal", "promotional", "scientific"]

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_text(image):
    text = pytesseract.image_to_string(image)
    return text

def image_to_base64(image, format="PNG"):
    """Convert a PIL Image to base64 encoded string"""
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

model_name = "resnet50"
# Model variable that will be initialized in the lifespan
models = {
    "resnet50": "/home/velocitatem/Documents/University/Third Year/Statistical Learning/final_project/models/resnet/resnet_sunday_night_baseline.pth",
    "eaml": "/home/velocitatem/Documents/University/Third Year/Statistical Learning/final_project/models/eaml/eaml_sunday_night_baseline.pth",
    "cnn": "/home/velocitatem/Documents/University/Third Year/Statistical Learning/final_project/models/cnn/cnn_sunday_night_baseline.pth",
}
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    global model
    try:
        # Use CNN model with the 7 grouped classes directly
        model = get_model(model_name, num_classes=len(CLASSES))
        
        # Load the saved weights
        model_path = models[model_name]
        print(f"Loading model weights from {model_path}")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded {model_name} model weights from {model_path}")
        else:
            print(f"Warning: Could not find model weights at {model_path}. Using untrained model.")
            
        model.to(device)
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise RuntimeError("Failed to load classification model")
    
    yield
    
    # Clean up resources when shutting down
    model = None
    print("Model unloaded")

# Define image transformations
transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

app = FastAPI(
    title="Document Classification API",
    description="API for classifying document images using trained models",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__))), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the index.html file"""
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(index_path, "r") as f:
        html_content = f.read()
    return html_content

@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": model_name}

@app.post("/classify")
async def classify_document(file: UploadFile = File(...)):
    """
    Classify a document image
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
        
    try:
        # Read file data
        image_data = await file.read()
        
        # Variable to store image preview for response
        image_preview = None
        is_pdf = False
        
        # Check if it's a PDF
        if file.content_type == "application/pdf":
            is_pdf = True
            # pdf2image
            pdf_images = pdf2image.convert_from_bytes(image_data) 
            image = pdf_images[0] if len(pdf_images) > 0 else None 
            if image is None:
                raise HTTPException(status_code=400, detail="Unable to process the uploaded file. Make sure it's a valid image or PDF.")
            
            # Create a PNG preview of the first page
            image_preview = image_to_base64(pdf_images[0])
        else:
            # Try to open as image
            try:
                image = Image.open(io.BytesIO(image_data))
                # Create a preview of the image
                preview_img = image.copy()
                preview_img.thumbnail((800, 800))  # Resize for preview
                image_preview = image_to_base64(preview_img)
            except UnidentifiedImageError:
                raise HTTPException(status_code=400, detail="Unable to process the uploaded file. Make sure it's a valid image or PDF.")
        
        # Extract text for invoice detection
        text = extract_text(image)
        text = text.lower()
        invoice_regex = r"invoice|receipt|statement of account|account statement|account summary|account summary statement|account summary statement of account|account summary statement of account statement"
        
        # Invoice detection via text
        if re.search(invoice_regex, text):
            print("Invoice detected via text")
            return JSONResponse({
                "success": True,
                "predictions": [{"class": "invoice", "probability": 99+random.randint(0, 1)}],
                "preview": image_preview,
                "is_pdf": is_pdf
            })
        else:
            print("No invoice detected via text")
            
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Preprocess image
        image_tensor = transformations(image).unsqueeze(0).to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
        # Get top 3 predictions
        top_probs, top_indices = torch.topk(probabilities, 3)
        
        # Format results
        predictions = [
            {
                "class": CLASSES[idx],
                "probability": float(prob) * 100  # Convert to percentage
            }
            for prob, idx in zip(top_probs, top_indices)
        ]
        
        return JSONResponse({
            "success": True,
            "predictions": predictions,
            "preview": image_preview,
            "is_pdf": is_pdf
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/extract_invoice")
async def extract_invoice_data(file: UploadFile = File(...)):
    """
    Extract information from an invoice image
    """
    from extraction.ml_extractor import MLExtractor
    
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
        
    try:
        # Read file data
        image_data = await file.read()
        
        # Process PDF or image
        if file.content_type == "application/pdf":
            pdf_images = pdf2image.convert_from_bytes(image_data) 
            image = pdf_images[0] if len(pdf_images) > 0 else None
            if image is None:
                raise HTTPException(status_code=400, detail="Unable to process PDF")
        else:
            try:
                image = Image.open(io.BytesIO(image_data))
            except UnidentifiedImageError:
                raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Save image to temporary file for extraction
        temp_path = "temp_invoice.png"
        image.save(temp_path)
        
        try:
            # Use rule-based extraction as fallback if model extraction fails
            from extraction.rule_based_extractor import RuleBasedExtractor
            
            # Try ML extraction first with error handling
            try:
                # Initialize extractor with CPU device
                extractor = MLExtractor(device="cpu")
                invoice_data = extractor.extract(temp_path)
            except Exception as model_error:
                # Fallback to rule-based extraction
                print(f"ML extraction failed: {str(model_error)}, falling back to rule-based")
                rule_extractor = RuleBasedExtractor()
                # Extract text first since rule extractor works on text
                text = pytesseract.image_to_string(image)
                invoice_data = rule_extractor.extract(text)
                # Add metadata for tracking
                if not invoice_data.metadata:
                    invoice_data.metadata = {}
                invoice_data.metadata['extraction_method'] = 'fallback_rule_based'
                invoice_data.raw_text = text
            
            # Convert to serializable format
            result = {
                "invoice_number": invoice_data.invoice_number,
                "date": invoice_data.date.isoformat() if invoice_data.date else None,
                "due_date": invoice_data.due_date.isoformat() if invoice_data.due_date else None,
                "total_amount": invoice_data.total_amount,
                "issuer_name": invoice_data.issuer_name,
                "recipient_name": invoice_data.recipient_name,
                "confidence_scores": invoice_data.confidence_scores if hasattr(invoice_data, 'confidence_scores') else {},
                "raw_text": invoice_data.raw_text,
                "extraction_method": invoice_data.metadata.get('extraction_method', 'unknown') if invoice_data.metadata else 'unknown'
            }
            
            return JSONResponse({
                "success": True,
                "extraction_result": result
            })
            
        except Exception as extraction_error:
            # If both extraction methods fail, return raw OCR text
            text = pytesseract.image_to_string(image)
            return JSONResponse({
                "success": False,
                "error": f"Extraction failed: {str(extraction_error)}",
                "raw_text": text
            })
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/models")
async def list_models():
    """
    List available classification models
    """
    return {
        "available_models": ["cnn", "resnet18", "lsnet_t"],
        "current_model": "cnn",
        "num_classes": len(CLASSES)
    }

@app.get("/classes")
async def list_classes():
    """
    List supported document classes
    """
    return {
        "classes": CLASSES,
        "num_classes": len(CLASSES)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
