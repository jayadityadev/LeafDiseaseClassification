"""
Leaf Condition Classifier - Web Interface
Simple Flask app for uploading leaf images and getting predictions
"""

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import sys
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import time
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib import colors

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src' / 'inference'))
from predict_leaf import predict_with_localization

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'uploads'
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

# Flag to track if warmup has been done
_MODEL_WARMED_UP = False

# Model configuration
PROJECT_ROOT = Path(__file__).parent

EXPECTED_MODEL_PATH = PROJECT_ROOT / "models/leaf/current/densenet121/leaf_densenet121_final.keras"

def resolve_model_path(expected_path: Path) -> Path:
    if expected_path.exists():
        return expected_path
    model_dir = expected_path.parent
    if model_dir.exists() and model_dir.is_dir():
        candidates = sorted(model_dir.glob('*.keras'), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            return candidates[0]
    return expected_path

MODEL_PATH = resolve_model_path(EXPECTED_MODEL_PATH)
MODEL_TYPE = "densenet121"

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _create_prediction_report_pdf(prediction_payload: dict, pdf_path: Path) -> None:
    """Generate a field-friendly PDF report for a prediction."""
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    width, height = A4

    margin = 25 * mm
    y = height - margin

    def draw_header():
        nonlocal y
        bar_height = 18
        c.setFillColor(colors.HexColor("#283593"))
        c.rect(0, height - bar_height - 10, width, bar_height + 10, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 13)
        c.drawString(margin, height - bar_height, "Leaf Condition AI Report")
        c.setFont("Helvetica", 9)
        c.drawRightString(width - margin, height - bar_height, "For field support only · Not a lab diagnosis")
        y = height - bar_height - 20

    def section_title(text):
        nonlocal y
        if y < 80 * mm:
            c.showPage()
            draw_header()
        c.setFont("Helvetica-Bold", 12)
        c.setFillColor(colors.HexColor("#37474F"))
        c.drawString(margin, y, text)
        y -= 6
        c.setStrokeColor(colors.HexColor("#B0BEC5"))
        c.setLineWidth(0.5)
        c.line(margin, y, width - margin, y)
        y -= 10

    def body(text, size=10):
        nonlocal y
        c.setFont("Helvetica", size)
        c.setFillColor(colors.black)
        for line in text.split("\n"):
            if y < 50 * mm:
                c.showPage()
                draw_header()
            c.drawString(margin, y, line)
            y -= 13

    def bullet(text, size=10):
        nonlocal y
        c.setFont("Helvetica", size)
        c.setFillColor(colors.black)
        if y < 50 * mm:
            c.showPage()
            draw_header()
        c.drawString(margin, y, u"• " + text)
        y -= 13

    def spacer(space=6):
        nonlocal y
        y -= space

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    draw_header()

    # Patient / study info banner
    c.setFillColor(colors.HexColor("#ECEFF1"))
    info_height = 32
    c.roundRect(margin - 5, y - info_height + 4, width - 2 * margin + 10, info_height, 6, stroke=0, fill=1)
    c.setFillColor(colors.HexColor("#455A64"))
    c.setFont("Helvetica-Bold", 10)
    c.drawString(margin, y + 14, "Study details")
    c.setFont("Helvetica", 9)
    c.drawString(margin, y, f"Report generated: {now}")
    c.drawRightString(width - margin, y, "AI model: DenseNet121 (research use only)")
    y -= info_height + 6

    section_title("1. AI Summary")
    pred_label = prediction_payload.get("prediction", "-")
    confidence = float(prediction_payload.get("confidence", 0.0))
    is_uncertain = bool(prediction_payload.get("is_uncertain", False))
    conf_threshold = float(prediction_payload.get("confidence_threshold", 0.0))

    if is_uncertain:
        c.setFillColor(colors.HexColor("#FFF3E0"))
        box_h = 40
        c.roundRect(margin - 5, y - box_h + 8, width - 2 * margin + 10, box_h, 6, stroke=0, fill=1)
        c.setFillColor(colors.HexColor("#E65100"))
        c.setFont("Helvetica-Bold", 10)
        c.drawString(margin, y + 20, "AI status: UNCERTAIN")
        c.setFont("Helvetica", 9)
        c.setFillColor(colors.black)
        c.drawString(margin, y + 6, f"Confidence {confidence:.1f}% (threshold {conf_threshold:.0f}%) · Top suggestion: {pred_label}")
        y -= box_h + 4
    else:
        c.setFillColor(colors.HexColor("#E8F5E9"))
        box_h = 36
        c.roundRect(margin - 5, y - box_h + 8, width - 2 * margin + 10, box_h, 6, stroke=0, fill=1)
        c.setFillColor(colors.HexColor("#1B5E20"))
        c.setFont("Helvetica-Bold", 10)
        c.drawString(margin, y + 16, "AI status: CONFIDENT")
        c.setFont("Helvetica", 9)
        c.setFillColor(colors.black)
        c.drawString(margin, y + 2, f"Predicted condition: {pred_label} · Confidence: {confidence:.1f}%")
        y -= box_h + 4

    highlighted = float(prediction_payload.get("highlighted_area", 0.0))
    spacer(4)
    body(f"Estimated highlighted region: {highlighted:.1f}% of the image area.")

    warning_message = prediction_payload.get("warning_message")
    if is_uncertain and warning_message:
        spacer(8)
        section_title("Field note")
        body(warning_message)

    spacer(8)
    section_title("2. Leaf condition overview")

    body("This AI system can distinguish between six leaf conditions. The notes "
         "below are general explanations and do not replace lab confirmation.")
    spacer(4)
    bullet("Healthy Leaf – normal foliage with no visible disease symptoms.")
    bullet("Healthy Nut – normal nuts without discoloration or lesions.")
    bullet("Mahali Koleroga – combined condition category in the dataset.")
    bullet("Yellow Leaf – yellowing patterns consistent with stress or disease.")
    bullet("Ring Spot – ring-like lesions or spot patterns on leaf surfaces.")
    bullet("Bud Rot – rot-related damage visible on buds or stems.")

    # 2a. Field guidance (non-prescriptive, general)
    spacer(6)
    section_title("3. Field guidance")

    body("The points below are general good-practice measures often used in crop "
         "scouting workflows. They are not treatment instructions:")
    spacer(2)
    bullet("Inspect multiple leaves and plants before making decisions.")
    bullet("Compare with known healthy samples from the same field.")
    bullet("Record lighting conditions and camera distance for repeatability.")
    bullet("Consult a local agronomist for lab confirmation when needed.")

    spacer(8)
    section_title("4. Class probabilities")
    probs = prediction_payload.get("probabilities", {}) or {}
    if probs:
        for name, val in sorted(probs.items(), key=lambda kv: kv[1], reverse=True):
            body(f"{name}: {float(val):.1f}%")
    else:
        body("No probability breakdown available.")

    spacer(8)
    section_title("5. Important disclaimer")
    disclaimer = (
        "This system is a research tool for classifying six labeled leaf "
        "conditions. It is NOT designed to diagnose unseen diseases.\n\n"
        "This report must not be used as a standalone basis for agricultural "
        "decisions. Final interpretation should consider field context and "
        "expert guidance."
    )
    body(disclaimer)

    c.showPage()
    c.save()


@app.route('/')
def index():
    """Main page with upload form."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction."""
    
    # Warmup model on first prediction to avoid slow initial inference
    global _MODEL_WARMED_UP
    if not _MODEL_WARMED_UP:
        try:
            print("🔥 Warming up model (first prediction)...")
            warmup_start = time.time()
            
            # Create a dummy 224x224 RGB image
            dummy_img = np.random.rand(224, 224, 3) * 255
            dummy_path = app.config['UPLOAD_FOLDER'] / '_warmup_dummy.png'
            from PIL import Image as PILImage
            PILImage.fromarray(dummy_img.astype('uint8')).save(dummy_path)
            
            # Run a dummy prediction to compile graph and initialize GPU
            try:
                _ = predict_with_localization(
                    image_path=str(dummy_path),
                    model_path=MODEL_PATH,
                    confidence_threshold=0.70
                )
            finally:
                # Clean up dummy files
                if dummy_path.exists():
                    dummy_path.unlink()
            
            warmup_time = time.time() - warmup_start
            print(f"✅ Model warmed up in {warmup_time:.2f}s (subsequent predictions will be faster)")
            _MODEL_WARMED_UP = True
        except Exception as e:
            print(f"⚠️  Warmup failed (non-fatal): {e}")
            # Don't block the actual prediction even if warmup fails
            _MODEL_WARMED_UP = True
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload an image (PNG, JPG, etc.)'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = app.config['UPLOAD_FOLDER'] / filename
        file.save(filepath)
        
        # Run prediction
        results = predict_with_localization(
            image_path=str(filepath),
            model_path=MODEL_PATH,
            confidence_threshold=0.70
        )
        
        # Clean up uploaded file
        filepath.unlink()

        # Convert all values to native Python types (fix JSON serialization)
        payload = {
            'success': True,
            'prediction': str(results['predicted_class']),
            'confidence': float(results['confidence']),
            'is_uncertain': bool(results['is_uncertain']),
            'entropy': float(results['entropy']),
            'confidence_threshold': float(results['confidence_threshold']),
            'warning_message': results['warning_message'],
            'probabilities': {k: float(v) for k, v in results['probabilities'].items()},
            'highlighted_area': 0.0
        }

        return jsonify(payload)
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/about')
def about():
    """About page with model information."""
    return render_template('about.html')


@app.route('/metrics')
def metrics():
    """Metrics dashboard page."""
    return render_template('metrics.html')


@app.route('/api/metrics')
def api_metrics():
    """API endpoint for model and dataset metrics."""
    try:
        # Check GPU without importing full TensorFlow
        gpu_available = False
        try:
            import tensorflow as tf
            gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
        except:
            pass
        
        reported_accuracy = None
        reports_dir = PROJECT_ROOT / "outputs/leaf/reports"
        if reports_dir.exists():
            candidates = sorted(reports_dir.glob("classification_report_*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if candidates:
                try:
                    lines = candidates[0].read_text().splitlines()
                    for line in lines:
                        if line.strip().startswith("accuracy"):
                            parts = line.split()
                            if len(parts) >= 2:
                                reported_accuracy = float(parts[1]) * 100
                                break
                except Exception:
                    reported_accuracy = None

        metrics_data = {
            'model': {
                'name': MODEL_TYPE,
                'path': str(MODEL_PATH.name),
                'architecture': 'DenseNet121 (Transfer Learning)',
                'input_shape': '224x224x3 (RGB)',
                'status': 'loaded' if MODEL_PATH.exists() else 'not_found'
            },
            'dataset': {
                'train_samples': 0,
                'test_samples': 0,
                'total_samples': 0,
                'classes': {},
                'sources': {}
            },
            'performance': {
                'reported_accuracy': f"{reported_accuracy:.2f}%" if reported_accuracy is not None else '--',
                'avg_inference_time': '--',
                'gpu_accelerated': gpu_available
            }
        }
        
        # Load dataset statistics (optional if pandas is unavailable)
        train_csv = PROJECT_ROOT / "data/leaf_dataset/train_split.csv"
        test_csv = PROJECT_ROOT / "data/leaf_dataset/test_split.csv"

        try:
            import pandas as pd
        except Exception:
            pd = None

        if pd is not None:
            if train_csv.exists():
                train_df = pd.read_csv(train_csv)
                metrics_data['dataset']['train_samples'] = len(train_df)

            if test_csv.exists():
                test_df = pd.read_csv(test_csv)
                metrics_data['dataset']['test_samples'] = len(test_df)

            if train_csv.exists() and test_csv.exists():
                combined_df = pd.concat([train_df, test_df])
                metrics_data['dataset']['total_samples'] = len(combined_df)

                # Class distribution
                if 'class_name' in combined_df.columns:
                    class_counts = combined_df['class_name'].value_counts().to_dict()
                    metrics_data['dataset']['classes'] = class_counts
                # Source distribution (optional)
                if 'source' in combined_df.columns:
                    source_counts = {}
                    for raw_value in combined_df['source'].dropna().astype(str).tolist():
                        normalized = raw_value.replace('\\', '/').strip()
                        if not normalized:
                            key = 'unknown'
                        else:
                            key = normalized.split('/')[0]
                        source_counts[key] = source_counts.get(key, 0) + 1
                    metrics_data['dataset']['sources'] = source_counts
        
        # Add model metadata without loading the full model (too slow/memory intensive)
        if MODEL_PATH.exists():
            metrics_data['model']['total_parameters'] = 7_319_107  # DenseNet121 params
            metrics_data['model']['file_size_mb'] = round(MODEL_PATH.stat().st_size / (1024*1024), 2)
        
        return jsonify(metrics_data)
        
    except Exception as e:
        import traceback
        error_details = {
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        print(f"Error in /api/metrics: {error_details}")
        return jsonify({'error': str(e)}), 500


@app.route('/report', methods=['POST'])
def generate_report():
    """Render a rich HTML report for the latest prediction; browser handles PDF printing."""
    try:
        data = request.get_json(force=True, silent=False)
        if not data or not isinstance(data, dict):
            return jsonify({'error': 'Invalid request payload'}), 400

        prediction = data.get('prediction', '-')
        confidence = float(data.get('confidence', 0.0))
        is_uncertain = bool(data.get('is_uncertain', False))
        entropy = float(data.get('entropy', 0.0))
        confidence_threshold = float(data.get('confidence_threshold', 0.0))
        highlighted_area = float(data.get('highlighted_area', 0.0))
        warning_message = data.get('warning_message')
        probs = data.get('probabilities', {}) or {}

        probabilities_sorted = [
            {'name': k, 'value': float(v)} for k, v in sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        ]

        viz_data_url = None
        if 'visualization' in data:
            viz_data_url = f"data:image/png;base64,{data['visualization']}"

        generated_at = datetime.now().strftime('%Y-%m-%d %H:%M')

        # Render HTML report; client will open in a new tab or replace view
        html = render_template(
            'report.html',
            generated_at=generated_at,
            prediction=prediction,
            confidence=confidence,
            is_uncertain=is_uncertain,
            entropy=entropy,
            confidence_threshold=confidence_threshold,
            highlighted_area=highlighted_area,
            warning_message=warning_message,
            probabilities_sorted=probabilities_sorted,
            visualization_data_url=viz_data_url,
        )
        return html
    except Exception as e:
        return jsonify({'error': f'Failed to generate report: {e}'}), 500


if __name__ == '__main__':
    print("="*80)
    print("🍃 LEAF CONDITION CLASSIFIER - WEB INTERFACE")
    print("="*80)
    # Try to report model accuracy from training_summary.csv (if present)
    reported_accuracy = None
    try:
        summary_csv = PROJECT_ROOT / 'models' / 'current' / 'training_summary.csv'
        if summary_csv.exists():
            import pandas as _pd
            df = _pd.read_csv(summary_csv)
            matched = df[df['model'].str.lower() == 'densenet121'] if 'model' in df.columns else _pd.DataFrame()
            if not matched.empty:
                reported_accuracy = float(matched.iloc[-1]['accuracy'])
    except Exception:
        reported_accuracy = None

    print(f"\n📁 Project Root: {PROJECT_ROOT}")
    print(f"🤖 Model path: {MODEL_PATH}")
    if reported_accuracy is not None:
        print(f"🤖 Reported accuracy (training_summary.csv): {reported_accuracy*100:.2f}%")
    else:
        print(f"🤖 Model: {MODEL_TYPE} (leaf dataset)")
    print(f"📂 Upload Folder: {app.config['UPLOAD_FOLDER']}")
    print("\n🌐 Starting server...")
    print("   Access at: http://localhost:5000")
    print("\n⚠️  Press Ctrl+C to stop the server")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
