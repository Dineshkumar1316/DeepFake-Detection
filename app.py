import os
import gc
import tempfile

import cv2
import numpy as np
import mysql.connector

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash
)
from flask_session import Session
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input
from mtcnn import MTCNN


# ============================================================
# Flask configuration
# ============================================================

app = Flask(__name__)

SECRET_KEY = os.getenv("FLASK_SECRET_KEY")

if not SECRET_KEY:
    raise RuntimeError(
        "FLASK_SECRET_KEY is not set. "
        "Set it before starting the application."
    )

app.secret_key = SECRET_KEY

app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_USE_SIGNER"] = True
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = False

# Maximum upload size accepted by Flask.
# The application does NOT save the uploaded file to disk.
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024

Session(app)


# ============================================================
# Database configuration
# ============================================================

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME", "deep_fake_detection"),
}

if not DB_CONFIG["password"]:
    raise RuntimeError(
        "DB_PASSWORD is not set. "
        "Set it before starting the application."
    )


def get_db_connection():
    """Create a new MySQL database connection."""
    return mysql.connector.connect(**DB_CONFIG)


# ============================================================
# GPU configuration
# ============================================================

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        print("GPU detected:", gpus)

    except RuntimeError as error:
        print("GPU configuration warning:", error)

else:
    print("Using CPU")


# ============================================================
# Model configuration
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "checkpoints",
    "xception_deepfake_image_5o.h5"
)

IMAGE_SIZE = (224, 224)

# Preserve the original project's inference behavior.
FRAME_SAMPLE_RATE = 10
MAX_FACES = 20
BATCH_SIZE = 32

ALLOWED_EXTENSIONS = {
    "mp4",
    "avi",
    "mov",
    "mkv",
    "webm"
}


# ============================================================
# Load model
# ============================================================

print("Loading Xception deepfake detection model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

model = load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully.")
print("Model input:", model.input_shape)
print("Model output:", model.output_shape)


# ============================================================
# Face detector
# ============================================================

print("Loading MTCNN face detector...")

face_detector = MTCNN()

print("MTCNN face detector loaded successfully.")


# ============================================================
# Utility functions
# ============================================================

def allowed_file(filename):
    """Check whether the uploaded file has an allowed extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


def clamp(value, minimum, maximum):
    """Keep a value inside a specified range."""
    return max(minimum, min(value, maximum))


# ============================================================
# Video frame streaming
# ============================================================

def stream_video_frames(file_storage):
    """
    Temporarily store the uploaded video so OpenCV can read it.
    The temporary file is deleted after processing.
    """

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".mp4",
            delete=False
        ) as temp_file:

            temp_path = temp_file.name

            while True:
                chunk = file_storage.stream.read(1024 * 1024)

                if not chunk:
                    break

                temp_file.write(chunk)

        cap = cv2.VideoCapture(temp_path)

        if not cap.isOpened():
            raise ValueError(
                "The uploaded video could not be opened."
            )

        frame_number = 0

        while True:
            ok, frame = cap.read()

            if not ok:
                break

            if frame_number % FRAME_SAMPLE_RATE == 0:
                yield frame

            frame_number += 1

        cap.release()

    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass

# ============================================================
# Face detection
# ============================================================

def detect_faces(frame):
    """
    Detect faces in one frame and return cropped face images.
    """

    try:
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        detections = face_detector.detect_faces(
            rgb_frame
        )

    except Exception as error:
        print("Face detection error:", error)
        return []

    faces = []

    height, width = frame.shape[:2]

    for detection in detections:
        x, y, w, h = detection.get(
            "box",
            (0, 0, 0, 0)
        )

        x = clamp(x, 0, width)
        y = clamp(y, 0, height)

        w = max(0, w)
        h = max(0, h)

        x2 = clamp(x + w, 0, width)
        y2 = clamp(y + h, 0, height)
        if x2 <= x or y2 <= y:
            continue

        face = frame[y:y2, x:x2]

        if face.size == 0:
            continue

        faces.append(face)

    return faces

def preprocess_face(face):
    """
    Resize a face and apply the same Xception preprocessing
    used by the original project.
    """

    face = cv2.resize(
        face,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    face = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )

    face = face.astype(
        np.float32
    )

    face = preprocess_input(
        face
    )

    return face


# ============================================================
# Prediction
# ============================================================

def predict_video(file_storage):
    """
    Perform FAKE/REAL prediction directly from the uploaded video.

    Processing pipeline:

        Browser upload
              |
        Flask request stream
              |
             OpenCV
              |
        Sample every 10th frame
              |
          MTCNN faces
              |
       Xception preprocessing
              |
       Xception prediction
              |
       Average probabilities
              |
          FAKE / REAL

    No uploaded video, face image, graph, processed video,
    or output file is saved by the application.
    """

    face_batches = []
    total_faces = 0
    sampled_frames = 0

    for frame in stream_video_frames(file_storage):

        sampled_frames += 1

        faces = detect_faces(frame)

        for face in faces:

            if total_faces >= MAX_FACES:
                break

            processed_face = preprocess_face(
                face
            )

            face_batches.append(
                processed_face
            )

            total_faces += 1

        if total_faces >= MAX_FACES:
            break

    if not face_batches:
        gc.collect()

        raise ValueError(
            "No detectable face was found in the uploaded video."
        )

    predictions = []

    for start in range(
        0,
        len(face_batches),
        BATCH_SIZE
    ):

        batch = np.asarray(
            face_batches[
                start:start + BATCH_SIZE
            ],
            dtype=np.float32
        )

        batch_predictions = model.predict(
            batch,
            verbose=0
        )

        predictions.extend(
            np.asarray(
                batch_predictions
            ).reshape(-1)
        )

        del batch
        gc.collect()

    if not predictions:
        gc.collect()

        raise ValueError(
            "The model could not produce a prediction."
        )

    fake_probability = float(
        np.mean(predictions)
    )

    fake_probability = max(
        0.0,
        min(
            fake_probability,
            1.0
        )
    )

    if fake_probability >= 0.5:

        result = "FAKE"
        confidence = fake_probability * 100

    else:

        result = "REAL"
        confidence = (
            1.0 - fake_probability
        ) * 100

    print(
        f"Prediction: {result} | "
        f"Confidence: {confidence:.2f}% | "
        f"Faces: {total_faces} | "
        f"Sampled frames: {sampled_frames}"
    )

    del face_batches
    del predictions

    gc.collect()

    return result, confidence


# ============================================================
# Authentication
# ============================================================

@app.before_request
def require_login():
    """
    Protect application pages that require authentication.
    """

    public_endpoints = {
        "home",
        "login",
        "signup",
        "static"
    }

    if request.endpoint in public_endpoints:
        return None

    if "username" not in session:
        return redirect(
            url_for("login")
        )

    return None


# ============================================================
# Home
# ============================================================

@app.route("/")
def home():

    return render_template(
        "home.html",
        username=session.get("username")
    )


# ============================================================
# Signup
# ============================================================

@app.route(
    "/signup",
    methods=["GET", "POST"]
)
def signup():

    if request.method == "POST":

        name = request.form.get(
            "name",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip()

        phone = request.form.get(
            "phone",
            ""
        ).strip()

        date_of_birth = request.form.get(
            "date_of_birth",
            ""
        ).strip()

        gender = request.form.get(
            "gender",
            ""
        ).strip()

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        if not all([
            name,
            email,
            username,
            password
        ]):

            flash(
                "Please fill in all required fields.",
                "error"
            )

            return redirect(
                url_for("signup")
            )

        password_hash = generate_password_hash(
            password
        )

        connection = None
        cursor = None

        try:

            connection = get_db_connection()
            cursor = connection.cursor()

            cursor.execute(
                """
                SELECT id
                FROM users
                WHERE username = %s
                """,
                (username,)
            )

            existing_user = cursor.fetchone()

            if existing_user:

                flash(
                    "Username already exists.",
                    "error"
                )

                return redirect(
                    url_for("signup")
                )

            cursor.execute(
                """
                INSERT INTO users
                (
                    name,
                    email,
                    phone,
                    date_of_birth,
                    gender,
                    username,
                    password
                )
                VALUES
                (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s
                )
                """,
                (
                    name,
                    email,
                    phone,
                    date_of_birth or None,
                    gender,
                    username,
                    password_hash
                )
            )

            connection.commit()

            flash(
                "Account created successfully. Please log in.",
                "success"
            )

            return redirect(
                url_for("login")
            )

        except mysql.connector.Error as error:

            print(
                "Database error during signup:",
                error
            )

            flash(
                "Unable to create account.",
                "error"
            )

        finally:

            if cursor:
                cursor.close()

            if connection:
                connection.close()

    return render_template(
        "signup.html"
    )


# ============================================================
# Login
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        if not username or not password:

            flash(
                "Please enter username and password.",
                "error"
            )

            return redirect(
                url_for("login")
            )

        connection = None
        cursor = None

        try:

            connection = get_db_connection()
            cursor = connection.cursor()

            cursor.execute(
                """
                SELECT
                    id,
                    name,
                    email,
                    phone,
                    date_of_birth,
                    gender,
                    username,
                    password
                FROM users
                WHERE username = %s
                """,
                (username,)
            )

            user = cursor.fetchone()

            if not user:

                flash(
                    "Invalid username or password.",
                    "error"
                )

                return redirect(
                    url_for("login")
                )

            stored_password = user[7]

            password_valid = False

            # New accounts use Werkzeug password hashes.
            try:

                password_valid = check_password_hash(
                    stored_password,
                    password
                )

            except (ValueError, TypeError):

                password_valid = False

            # Compatibility with old plaintext passwords.
            # If an old account successfully logs in,
            # immediately replace its plaintext password
            # with a secure hash.
            if not password_valid and stored_password == password:

                password_valid = True

                new_hash = generate_password_hash(
                    password
                )

                cursor.execute(
                    """
                    UPDATE users
                    SET password = %s
                    WHERE id = %s
                    """,
                    (
                        new_hash,
                        user[0]
                    )
                )

                connection.commit()

            if not password_valid:

                flash(
                    "Invalid username or password.",
                    "error"
                )

                return redirect(
                    url_for("login")
                )

            session.clear()

            session["user_id"] = user[0]
            session["username"] = user[6]

            return redirect(
                url_for(
                    "upload_page",
                    username=user[6]
                )
            )

        except mysql.connector.Error as error:

            print(
                "Database error during login:",
                error
            )

            flash(
                "Unable to log in.",
                "error"
            )

            return redirect(
                url_for("login")
            )

        finally:

            if cursor:
                cursor.close()

            if connection:
                connection.close()

    return render_template(
        "login.html"
    )


# ============================================================
# Upload page
# ============================================================

@app.route(
    "/upload/<username>"
)
def upload_page(username):

    if session.get("username") != username:

        flash(
            "Unauthorized access.",
            "error"
        )

        return redirect(
            url_for("login")
        )

    return render_template(
        "upload.html",
        username=username
    )


# ============================================================
# Video upload + prediction
# ============================================================

@app.route(
    "/upload_video/<username>",
    methods=["POST"]
)
def upload_video(username):

    if session.get("username") != username:

        flash(
            "Unauthorized access.",
            "error"
        )

        return redirect(
            url_for("login")
        )

    if "file" not in request.files:

        flash(
            "No video was uploaded.",
            "error"
        )

        return redirect(
            url_for(
                "upload_page",
                username=username
            )
        )

    video = request.files["file"]

    if not video or not video.filename:

        flash(
            "Please select a video.",
            "error"
        )

        return redirect(
            url_for(
                "upload_page",
                username=username
            )
        )

    filename = secure_filename(
        video.filename
    )

    if not allowed_file(filename):

        flash(
            "Unsupported video format. "
            "Allowed formats: MP4, AVI, MOV, MKV, WEBM.",
            "error"
        )

        return redirect(
            url_for(
                "upload_page",
                username=username
            )
        )

    try:

        result, confidence = predict_video(
            video
        )

        return render_template(
            "result.html",
            username=username,
            result=result,
            confidence=round(
                confidence,
                2
            )
        )

    except ValueError as error:

        print(
            "Prediction error:",
            error
        )

        flash(
            str(error),
            "error"
        )

        return redirect(
            url_for(
                "upload_page",
                username=username
            )
        )

    except Exception as error:

        print(
            "Unexpected prediction error:",
            error
        )

        flash(
            "An error occurred while processing the video.",
            "error"
        )

        return redirect(
            url_for(
                "upload_page",
                username=username
            )
        )

    finally:

        gc.collect()


# ============================================================
# Logout
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    flash(
        "You have been logged out.",
        "success"
    )

    return redirect(
        url_for("login")
    )


# ============================================================
# Error handlers
# ============================================================

@app.errorhandler(413)
def request_too_large(error):

    flash(
        "The uploaded video is too large. "
        "Maximum size is 512 MB.",
        "error"
    )

    return redirect(
        url_for("login")
    )


@app.errorhandler(500)
def internal_server_error(error):

    print(
        "Internal server error:",
        error
    )

    return (
        "Internal server error.",
        500
    )


# ============================================================
# Application entry point
# ============================================================

if __name__ == "__main__":

    debug_mode = (
        os.getenv(
            "FLASK_DEBUG",
            "false"
        ).lower()
        == "true"
    )

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=debug_mode
    )









