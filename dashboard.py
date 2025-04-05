import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import streamlit as st
import pdf2image
import os
import zipfile
import io
import tempfile
import shutil
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Document Analysis, Processing, and Pattern Extraction Repository",
    page_icon="📄",
    layout="wide"
)

# Apply custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
        text-align: center;
    }
    .page-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .prediction-container {
        background-color: white;
        border-radius: 5px;
        padding: 15px;
        margin-top: 10px;
    }
    .progress-bar {
        height: 20px;
        border-radius: 10px;
    }
    .stProgress > div > div > div > div {
        background-color: #1E3A8A;
    }
    .upload-section {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Define class labels
classes = ["letter", "form", "email", "handwritten", "advertisement", "scientific report",
           "scientific publication", "specification", "file folder", "news article", "budget",
           "invoice", "presentation", "questionnaire", "resume", "memo"]

# Define transformations
transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    try:
        model.load_state_dict(torch.load('classification/model.pth', map_location=device), strict=False)
        model = model.to(device)
        model.eval()
        return model
    except FileNotFoundError:
        st.error("Model file 'classification/model.pth' not found. Please ensure the model file exists in the classification directory.")
        return None

model = load_model()

# Function to process and classify an image
def classify_image(image):
    image_tensor = transformations(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        top3_prob, top3_classes = torch.topk(probabilities, 3)
    results = [(classes[top3_classes[i].item()], top3_prob[i].item() * 100) for i in range(3)]
    return results

# Header
st.markdown("<h1 class='main-header'>📄 PDF Document Classification Dashboard</h1>", unsafe_allow_html=True)

# Create tabs for different upload methods
tab1, tab2 = st.tabs(["Single PDF", "Batch Processing (ZIP)"])

with tab1:
    # Single PDF upload section
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"], key="single_pdf")

with tab2:
    # ZIP batch processing section
    st.markdown("### Batch Process Multiple PDFs")
    st.write("Upload a ZIP file containing multiple PDFs. The system will process each PDF, classify all pages, and return a ZIP file with folders organized by document type.")

    uploaded_zip = st.file_uploader("Upload a ZIP file containing PDFs", type=["zip"], key="zip_upload")

# Function to process a PDF and classify its pages
def process_pdf(pdf_path, return_images=False):
    try:
        # Convert PDF to images
        images = pdf2image.convert_from_path(pdf_path)

        if len(images) == 0:
            return None, "No pages found in the PDF."

        # Classify each page
        page_results = []
        for img in images:
            results = classify_image(img)
            page_results.append(results)

        if return_images:
            return page_results, images
        else:
            return page_results, None
    except Exception as e:
        return None, f"Error processing PDF: {str(e)}"

# Function to process a ZIP file containing PDFs
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

def process_zip_file(zip_file, output_zip_path, process_pdf, classes):
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()

    try:
        # Extract zip file contents to temp_dir
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Find all PDFs recursively
        pdf_files = list(Path(temp_dir).glob('**/*.pdf'))
        if not pdf_files:
            return None, "No PDF files found in the ZIP archive."

        # Create category subdirectories inside output_dir
        categories = {cls: Path(output_dir) / cls for cls in classes}
        for cat_dir in categories.values():
            cat_dir.mkdir(exist_ok=True)

        results = {}
        for pdf_path in pdf_files:
            pdf_name = pdf_path.name
            page_results, _ = process_pdf(str(pdf_path))

            if page_results:
                results[pdf_name] = page_results
                majority_class = max(
                    set([res[0][0] for res in page_results]),
                    key=[res[0][0] for res in page_results].count
                )
                shutil.copy(pdf_path, categories[majority_class] / pdf_name)

        # Create ZIP on disk with categorized results
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_dir)
                    zf.write(file_path, arcname)

        return output_zip_path, results

    except Exception as e:
        return None, f"Error processing ZIP file: {str(e)}"

    finally:
        # Clean up temporary folders
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)

# Handle Single PDF Upload
if uploaded_file is not None and model:
    try:
        # Save uploaded PDF
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Processing PDF..."):
            page_results, images = process_pdf("temp.pdf", return_images=True)

        if not page_results:
            st.error("No pages found in the PDF.")
        else:
            st.success(f"Successfully extracted {len(images)} pages from the PDF.")

            # Add page selection using tabs
            if len(images) > 1:
                st.markdown("### Select a page to view:")
                tabs = st.tabs([f"Page {i+1}" for i in range(len(images))])

                for idx, tab in enumerate(tabs):
                    with tab:
                        col1, col2 = st.columns([3, 2])

                        with col2:
                            st.image(images[idx], width=500)

                        with col1:
                            st.markdown("<div class='prediction-container'>", unsafe_allow_html=True)
                            st.subheader(f"Classification Results")

                            results = classify_image(images[idx])

                            top_class = results[0][0]
                            top_prob = results[0][1]

                            st.markdown(f"**Top prediction:** {top_class}")
                            st.markdown(f"**Confidence:** {top_prob:.2f}%")

                            st.markdown("### Top 3 Predictions:")
                            for label, prob in results:
                                st.markdown(f"**{label}**")
                                st.progress(prob/100)
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")


# Handle ZIP Upload
if uploaded_zip is not None and model:
    with st.spinner("Processing ZIP file..."):
        result_file, results = process_zip_file(uploaded_zip, "classified_documents.zip", process_pdf, classes)

        if result_file:
            st.success("Successfully processed all PDFs in the ZIP file!")

            # Display summary of results
            st.markdown("### Processing Summary")
            for pdf_name, page_results in results.items():
                st.markdown(f"**{pdf_name}**")
                for i, results in enumerate(page_results):
                    top_class = results[0][0]
                    top_prob = results[0][1]
                    st.markdown(f"- Page {i+1}: {top_class} ({top_prob:.2f}%)")

            # Provide download button for the processed ZIP
            st.download_button(
                label="Download Processed ZIP",
                data=result_file,
                file_name="classified_documents.zip",
                mime="application/zip"
            )
        else:
            st.error("Error processing ZIP file. Please check the file contents and try again.")

# Clean up temporary files
if os.path.exists("temp.pdf"):
    os.remove("temp.pdf")
