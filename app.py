import os
import base64
import json
import numpy as np
import face_recognition
from flask import Flask, jsonify, request, render_template, send_from_directory
import cv2
from lib.FaceAntiSpoofing import AntiSpoof
from lib.face_detector import YOLOv5

app = Flask(__name__)
# Directories
MEDIA_DIR = os.path.join(os.getcwd(), "media")
FACES_JSON_PATH = os.path.join(os.getcwd(), "faces.json")

face_detector = YOLOv5("models/yolov5s-face.onnx")  # Adjust path to your YOLOv5 model
anti_spoof = AntiSpoof("models/AntiSpoofing_bin_1.5_128.onnx")  # Adjust path to your Anti-Spoof mod

#face_detector = YOLOv5("models/facenet.tflite")  # Adjust path to your YOLOv5 model
#anti_spoof = AntiSpoof("models/AntiSpoofing.tflite")  # Adjust path to your Anti-Spoof model


# Ensure media directory exists
os.makedirs(MEDIA_DIR, exist_ok=True)

# Initialize faces.json if it doesn't exist
if not os.path.exists(FACES_JSON_PATH):
    with open(FACES_JSON_PATH, "w") as f:
        json.dump([], f)


COLOR_REAL = (0, 255, 0)
COLOR_FAKE = (0, 0, 255)
COLOR_UNKNOWN = (127, 127, 127)

def increased_crop(img, bbox: tuple, bbox_inc: float = 1.5):
    # Crop face based on its bounding box
    real_h, real_w = img.shape[:2]
    x, y, w, h = bbox
    w, h = w - x, h - y
    l = max(w, h)
    xc, yc = x + w/2, y + h/2
    x, y = int(xc - l*bbox_inc/2), int(yc - l*bbox_inc/2)
    x1 = 0 if x < 0 else x 
    y1 = 0 if y < 0 else y
    x2 = real_w if x + l*bbox_inc > real_w else x + int(l*bbox_inc)
    y2 = real_h if y + l*bbox_inc > real_h else y + int(l*bbox_inc)
    
    img = img[y1:y2, x1:x2, :]
    img = cv2.copyMakeBorder(img, 
                             y1-y, int(l*bbox_inc-y2+y), 
                             x1-x, int(l*bbox_inc)-x2+x, 
                             cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img

def make_prediction(img, face_detector, anti_spoof):
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bbox = face_detector([img])[0]
    if bbox.shape[0] > 0:
        bbox = bbox.flatten()[:4].astype(int)
    else:
        return None

    pred = anti_spoof([increased_crop(img, bbox, bbox_inc=1.5)])[0]
    score = pred[0][0]
    label = np.argmax(pred)   
    return bbox, label, score


def start_camera():
    global camera
    camera = cv2.VideoCapture(0)  # Open the camera
    if not camera.isOpened():
        return False  # Camera couldn't be opened
    return True

def stop_camera():
    global camera
    if camera:
        camera.release()  # Release the camera
        camera = None


def generate_frames():
    global camera
    while camera:
        success, frame = camera.read()
        if not success:
            break
        else:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')



def load_known_faces(directory):
    """ Load face encodings from image files (.jpg, .png, .jpeg) in the directory """
    known_face_encodings = []
    known_face_names = []

    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return known_face_encodings, known_face_names

    for file_name in os.listdir(directory):
        if file_name.endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(directory, file_name)
            image = face_recognition.load_image_file(img_path)

            try:
                # Get face encoding from the image
                face_encoding = face_recognition.face_encodings(image)[0]
                known_face_encodings.append(face_encoding)
                known_face_names.append(os.path.splitext(file_name)[0])
            except IndexError:
                print(f"No face found in: {file_name}")

    return known_face_encodings, known_face_names


@app.route("/start-recognition", methods=["POST"])
def start_recognition():
    try:
        data = request.json
        frame_data = data.get("frame")
        if not frame_data:
            return jsonify({'error': 'No image data received'}), 400

        # Decode the base64 image
        img_data = base64.b64decode(frame_data.split(',')[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Load known faces
        known_face_encodings, known_face_names = load_known_faces(MEDIA_DIR)

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_small_frame)
        print(f"Detected {len(face_locations)} faces")
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        detected_faces = []
        for face_encoding, face_location in zip(face_encodings, face_locations):
            print(f"checking face at location: {face_location}")
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            print(f"Matches found: {matches}")  # Log the matches

            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                print(f"Recognized face: {name}")

                with open(FACES_JSON_PATH, "r") as f:
                    faces_data = json.load(f)
                    for face_data in faces_data:
                        if face_data['name'] == name:
                            hash_value = face_data['hash']
                            break
            else:
                hash_value = "Unknown"

            # Run liveness detection for each detected face
            bbox, label, score = make_prediction(frame, face_detector, anti_spoof)
            liveness_result = "Unknown"
            color = COLOR_UNKNOWN

            if label == 0:  # Real face detected
                if score > 0.70:
                    liveness_result = f"REAL {score:.2f}"
                    color = COLOR_REAL
                else:
                    liveness_result = "Unknown"
            else:  # Fake face detected
                liveness_result = f"FAKE {score:.2f}"
                color = COLOR_FAKE
            print(liveness_result)
            detected_faces.append({
                "name": name,
                "status": "True" if name != "Unknown" else "False",
                "liveness": liveness_result,
                "color": color,
                "hash": hash_value if name != "Unknown" else "Unknown"
            })

        return jsonify({"faces": detected_faces})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500



@app.route("/delete-image", methods=["POST"])
def delete_image():
    data = request.get_json()
    filename = data.get("filename")
    if not filename:
        return jsonify({"success": False, "message": "Filename is missing"}), 400

    file_path = os.path.join(MEDIA_DIR, filename)

    if os.path.exists(file_path):
        os.remove(file_path)
        return jsonify({"success": True, "message": "Image deleted successfully!"})
    else:
        return jsonify({"success": False, "message": "File not found!"}), 404


@app.route('/save-face', methods=['POST'])
def save_face():
    data = request.get_json()
    image_data = data.get('image')
    name = data.get('name')
    hash_value = data.get("hash")

    if not image_data or not name or not hash_value:
        return jsonify({"message": "Missing data"}), 400
    
    # Decode base64 image
    img_data = base64.b64decode(image_data.split(',')[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"message": "Invalid image data"}), 400

    # Convert to grayscale and detect face
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return jsonify({"message": "No face detected"}), 400
    
    # Get face encoding for the captured image
    #face_encoding = face_recognition.face_encodings(image)[0]

    encodings = face_recognition.face_encodings(image)

    if not encodings:
        return jsonify({"message": "No face found"}), 400

    face_encoding = encodings[0]
    # Load known faces from directory
    known_face_encodings, known_face_names = load_known_faces(MEDIA_DIR)

    # Check if the captured face matches any known faces
    for known_face_encoding in known_face_encodings:
        matches = face_recognition.compare_faces([known_face_encoding], face_encoding)
        if True in matches:
            return jsonify({"message": "This face already exists in the directory."}), 400



    # If face does not exist, save it
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        file_path = os.path.join("media", f"{name}.jpg")
        cv2.imwrite(file_path, face)

    # Save face data (name, hash, path)
    with open(FACES_JSON_PATH, "r+") as f:
        data = json.load(f)
        data.append({"name": name, "hash": hash_value, "path": file_path})
        f.seek(0)
        json.dump(data, f, indent=4)

    return jsonify({"message": "Face saved successfully!", "hash": hash_value})


@app.route("/media/<filename>")
def serve_image(filename):
    return send_from_directory(MEDIA_DIR, filename)


@app.route('/stop-camera')
def stop_camera_route():
    stop_camera()
    return jsonify({"message": "Camera stopped."})


@app.route("/face-library")
def face_library():
    images = [f for f in os.listdir(MEDIA_DIR) if f.endswith((".png", ".jpg", ".jpeg"))]
    return render_template("library.html", images=images)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/capture")
def capture_page():
    """ Display the capture page where camera is used for capturing face """
    return render_template('capture.html')


@app.route('/capture-face', methods=["POST"])
def capture_face():
    """ Capture face from the webcam """
    if camera is None or not camera.isOpened():
        if not start_camera():
            return jsonify({"error": "Failed to open camera"}), 500

    success, frame = camera.read()
    if not success:
        return jsonify({"error": "Failed to capture frame"}), 500

    img_path = os.path.join(MEDIA_DIR, "captured_face.jpg")
    cv2.imwrite(img_path, frame)
    return jsonify({"message": "Face captured successfully!", "image_path": img_path})


@app.route("/recognize")
def recognize_page():
    return render_template("rec.html")


if __name__ == "__main__":
    app.run(debug=True)
