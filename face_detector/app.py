import os
import time
import base64
import uuid
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
import cv2
import numpy as np

# Initialize Flask application
app = Flask(__name__)

# Configuration
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
CASCADE_FOLDER = os.path.join(BASE_DIR, 'cascades')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload limit

# Ensure required directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CASCADE_FOLDER, exist_ok=True)

# -----------------------------------------------------------------------------
# HAAR CASCADE CLASSIFIER INITIALIZATION (OFFLINE COMPATIBLE)
# -----------------------------------------------------------------------------
# Haar Cascades are machine learning object detection algorithms used to identify
# objects in images or videos based on Haar features (edge, line, and center-surround).
# -----------------------------------------------------------------------------

def get_cascade_path(filename):
    """
    Locates Haar Cascade XML file across local directories and OpenCV package paths.
    Ensures 100% offline functionality for BCA lab evaluations.
    """
    possible_paths = [
        os.path.join(CASCADE_FOLDER, filename),
        os.path.join(BASE_DIR, filename),
        os.path.join(getattr(cv2, 'data', None) and cv2.data.haarcascades or '', filename)
    ]
    for path in possible_paths:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(f"Haar cascade XML file '{filename}' not found in paths: {possible_paths}")

FACE_CASCADE_PATH = get_cascade_path('haarcascade_frontalface_default.xml')
EYE_CASCADE_PATH = get_cascade_path('haarcascade_eye.xml')

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)

if face_cascade.empty():
    print(f"[WARNING] Failed to load face cascade from {FACE_CASCADE_PATH}")
else:
    print(f"[SUCCESS] Loaded Face Cascade: {FACE_CASCADE_PATH}")

if eye_cascade.empty():
    print(f"[WARNING] Failed to load eye cascade from {EYE_CASCADE_PATH}")
else:
    print(f"[SUCCESS] Loaded Eye Cascade: {EYE_CASCADE_PATH}")


# -----------------------------------------------------------------------------
# CORE OPENCV IMAGE PROCESSING & PRIVACY ROUTINES
# -----------------------------------------------------------------------------

def apply_privacy_effect(image, x, y, w, h, mode, blur_ksize=31, pixel_scale=0.08):
    """
    Applies privacy filter (Blur, Pixelate, or Blackout) to a specified Region of Interest (ROI).
    
    Parameters:
        image: Original BGR NumPy image array
        x, y, w, h: Bounding box coordinates of target area
        mode: 'blur', 'pixelate', 'blackout'
        blur_ksize: Kernel size for Gaussian blur (must be odd integer)
        pixel_scale: Scaling factor for pixelation effect
    """
    roi = image[y:y+h, x:x+w]
    if roi.size == 0:
        return

    if mode == 'blur':
        # Ensure ksize is odd and >= 3
        k = max(3, blur_ksize | 1)
        blurred_roi = cv2.GaussianBlur(roi, (k, k), 30)
        image[y:y+h, x:x+w] = blurred_roi

    elif mode == 'pixelate':
        # Downscale region then upscale using nearest-neighbor interpolation
        small_w = max(1, int(w * pixel_scale))
        small_h = max(1, int(h * pixel_scale))
        small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        image[y:y+h, x:x+w] = pixelated

    elif mode == 'blackout':
        # Draw a solid black rectangle over the region
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), -1)


def draw_styled_bounding_box(image, x, y, w, h, label, color, corner_length=15):
    """
    Draws a modern HUD-style bounding box with corner accents and text badge.
    """
    thickness = 2
    # Base rectangle
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    
    # Corner Accents
    # Top-Left
    cv2.line(image, (x, y), (x + corner_length, y), color, thickness + 1)
    cv2.line(image, (x, y), (x, y + corner_length), color, thickness + 1)
    # Top-Right
    cv2.line(image, (x + w, y), (x + w - corner_length, y), color, thickness + 1)
    cv2.line(image, (x + w, y), (x + w, y + corner_length), color, thickness + 1)
    # Bottom-Left
    cv2.line(image, (x, y + h), (x + corner_length, y + h), color, thickness + 1)
    cv2.line(image, (x, y + h), (x, y + h - corner_length), color, thickness + 1)
    # Bottom-Right
    cv2.line(image, (x + w, y + h), (x + w - corner_length, y + h), color, thickness + 1)
    cv2.line(image, (x + w, y + h), (x + w, y + h - corner_length), color, thickness + 1)

    # Label Badge Background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    font_thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    
    badge_y1 = max(0, y - text_h - 8)
    badge_y2 = y
    cv2.rectangle(image, (x, badge_y1), (x + text_w + 10, badge_y2), color, -1)
    cv2.putText(image, label, (x + 5, y - 4), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)


# -----------------------------------------------------------------------------
# FLASK ROUTES
# -----------------------------------------------------------------------------

@app.route('/')
def index():
    """Renders the main web interface for detection and analytics."""
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def process_detection():
    """
    Main detection endpoint accepting image data via multipart form or JSON base64.
    Performs Haar cascade detection, applies selected visual mode, tracks latency,
    and returns detailed coordinate tables.
    """
    start_total_time = time.perf_counter()
    
    # 1. Parse Parameters
    detect_target = request.form.get('target', 'both')  # 'face', 'eye', 'both'
    mode = request.form.get('mode', 'highlight')         # 'highlight', 'blur', 'pixelate', 'blackout'
    scale_factor = float(request.form.get('scaleFactor', 1.1))
    min_neighbors = int(request.form.get('minNeighbors', 5))
    min_size_val = int(request.form.get('minSize', 30))

    # Clamp parameters to valid ranges
    scale_factor = max(1.01, min(scale_factor, 2.0))
    min_neighbors = max(1, min(min_neighbors, 20))
    min_size = (min_size_val, min_size_val)

    img_bytes = None

    # Handle File Upload or Base64 Data
    if 'image' in request.files and request.files['image'].filename != '':
        file = request.files['image']
        img_bytes = file.read()
    elif 'base64_image' in request.form:
        b64_data = request.form['base64_image']
        if ',' in b64_data:
            b64_data = b64_data.split(',')[1]
        img_bytes = base64.b64decode(b64_data)
    else:
        return jsonify({'error': 'No image provided. Please upload an image or capture from webcam.'}), 400

    # 2. Decode Image to OpenCV BGR Format
    load_start = time.perf_counter()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Invalid image file or format could not be decoded by OpenCV.'}), 400

    img_height, img_width = img.shape[:2]
    load_time_ms = round((time.perf_counter() - load_start) * 1000, 2)

    # 3. Preprocessing (Grayscale Conversion)
    # Haar Cascades require single-channel 8-bit grayscale images for integral image calculation.
    det_start = time.perf_counter()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Histogram equalization improves contrast for detection

    detected_faces = []
    detected_eyes = []
    coords_table = []

    # Colors (BGR)
    FACE_COLOR = (255, 165, 0)   # Vivid Cyan/Blue
    EYE_COLOR = (255, 0, 255)    # Magenta / Neon Purple

    # 4. Face Detection
    if detect_target in ['face', 'both']:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        )

        for i, (fx, fy, fw, fh) in enumerate(faces):
            face_id = f"Face #{i+1}"
            center_x = int(fx + fw / 2)
            center_y = int(fy + fh / 2)
            area = int(fw * fh)

            coords_table.append({
                'type': 'Face',
                'id': face_id,
                'x': int(fx),
                'y': int(fy),
                'w': int(fw),
                'h': int(fh),
                'center_x': center_x,
                'center_y': center_y,
                'area': area
            })

            if mode == 'highlight':
                draw_styled_bounding_box(img, fx, fy, fw, fh, face_id, FACE_COLOR)
            else:
                apply_privacy_effect(img, fx, fy, fw, fh, mode)

            # 5. Nested Eye Detection within Face ROI (Region of Interest)
            if detect_target in ['eye', 'both']:
                roi_gray = gray[fy:fy+fh, fx:fx+fw]
                roi_color = img[fy:fy+fh, fx:fx+fw]

                # Detect eyes inside the face box to minimize false positives
                eyes = eye_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.1,
                    minNeighbors=max(3, min_neighbors - 1),
                    minSize=(max(10, int(fw * 0.1)), max(10, int(fh * 0.1))),
                    maxSize=(int(fw * 0.5), int(fh * 0.5))
                )

                for j, (ex, ey, ew, eh) in enumerate(eyes):
                    # Convert ROI relative coordinates to global image coordinates
                    abs_ex = fx + ex
                    abs_ey = fy + ey
                    eye_id = f"Eye #{i+1}.{j+1}"
                    e_center_x = int(abs_ex + ew / 2)
                    e_center_y = int(abs_ey + eh / 2)
                    e_area = int(ew * eh)

                    coords_table.append({
                        'type': 'Eye',
                        'id': eye_id,
                        'x': int(abs_ex),
                        'y': int(abs_ey),
                        'w': int(ew),
                        'h': int(eh),
                        'center_x': e_center_x,
                        'center_y': e_center_y,
                        'area': e_area
                    })

                    if mode == 'highlight':
                        draw_styled_bounding_box(img, abs_ex, abs_ey, ew, eh, eye_id, EYE_COLOR, corner_length=8)
                    else:
                        apply_privacy_effect(img, abs_ex, abs_ey, ew, eh, mode)

    elif detect_target == 'eye':
        # Direct full-image eye detection if selected without face constraint
        eyes = eye_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(20, 20)
        )
        for j, (ex, ey, ew, eh) in enumerate(eyes):
            eye_id = f"Eye #{j+1}"
            e_center_x = int(ex + ew / 2)
            e_center_y = int(ey + eh / 2)
            e_area = int(ew * eh)

            coords_table.append({
                'type': 'Eye',
                'id': eye_id,
                'x': int(ex),
                'y': int(ey),
                'w': int(ew),
                'h': int(eh),
                'center_x': e_center_x,
                'center_y': e_center_y,
                'area': e_area
            })

            if mode == 'highlight':
                draw_styled_bounding_box(img, ex, ey, ew, eh, eye_id, EYE_COLOR, corner_length=8)
            else:
                apply_privacy_effect(img, ex, ey, ew, eh, mode)

    detection_time_ms = round((time.perf_counter() - det_start) * 1000, 2)

    # 6. Save processed output image to static/uploads/
    render_start = time.perf_counter()
    unique_filename = f"processed_{uuid.uuid4().hex[:10]}.jpg"
    output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    cv2.imwrite(output_filepath, img)

    # Also convert processed image to base64 data URI for instant client rendering
    _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    processed_data_uri = f"data:image/jpeg;base64,{img_b64}"

    rendering_time_ms = round((time.perf_counter() - render_start) * 1000, 2)
    total_latency_ms = round((time.perf_counter() - start_total_time) * 1000, 2)

    face_count = sum(1 for c in coords_table if c['type'] == 'Face')
    eye_count = sum(1 for c in coords_table if c['type'] == 'Eye')

    return jsonify({
        'status': 'success',
        'processed_image_url': url_for('static', filename=f'uploads/{unique_filename}'),
        'processed_image_b64': processed_data_uri,
        'summary': {
            'faces_detected': face_count,
            'eyes_detected': eye_count,
            'total_objects': len(coords_table),
            'image_width': img_width,
            'image_height': img_height,
            'mode_applied': mode,
            'target': detect_target
        },
        'latency': {
            'image_load_ms': load_time_ms,
            'detection_ms': detection_time_ms,
            'rendering_ms': rendering_time_ms,
            'total_latency_ms': total_latency_ms,
            'fps_estimate': round(1000.0 / total_latency_ms, 1) if total_latency_ms > 0 else 0
        },
        'coordinates': coords_table
    })


@app.route('/sample', methods=['GET'])
def generate_sample_image():
    """
    Generates a synthetic face test image programmatically using OpenCV shapes
    so users can test detection immediately without having to upload a file.
    """
    img = np.ones((400, 500, 3), dtype=np.uint8) * 240 # Light gray background
    
    # Draw simple face representation
    # Face 1
    cv2.ellipse(img, (180, 200), (90, 120), 0, 0, 360, (210, 180, 140), -1) # Face oval
    cv2.circle(img, (145, 175), 16, (255, 255, 255), -1) # Left eye white
    cv2.circle(img, (145, 175), 7, (80, 50, 20), -1)    # Left pupil
    cv2.circle(img, (215, 175), 16, (255, 255, 255), -1) # Right eye white
    cv2.circle(img, (215, 175), 7, (80, 50, 20), -1)    # Right pupil
    cv2.ellipse(img, (180, 240), (30, 15), 0, 0, 180, (50, 50, 180), 3) # Mouth

    # Face 2
    cv2.ellipse(img, (370, 180), (70, 95), 0, 0, 360, (200, 170, 130), -1)
    cv2.circle(img, (345, 160), 12, (255, 255, 255), -1)
    cv2.circle(img, (345, 160), 5, (20, 20, 20), -1)
    cv2.circle(img, (395, 160), 12, (255, 255, 255), -1)
    cv2.circle(img, (395, 160), 5, (20, 20, 20), -1)
    cv2.ellipse(img, (370, 215), (20, 10), 0, 0, 180, (40, 40, 160), 3)

    _, buffer = cv2.imencode('.jpg', img)
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    return jsonify({
        'status': 'success',
        'image_b64': f"data:image/jpeg;base64,{img_b64}"
    })


if __name__ == '__main__':
    print("=================================================================")
    print("  BCA Lab Submission: Offline OpenCV Haar Cascade Detection Engine")
    print("  Server running on http://127.0.0.1:5000")
    print("=================================================================")
    app.run(host='0.0.0.0', port=5000, debug=True)
