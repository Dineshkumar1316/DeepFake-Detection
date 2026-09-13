# DeepFake Detection

A Flask-based web application for detecting whether an uploaded video is **REAL** or **FAKE** using a pretrained Xception-based deepfake detection model.

The application provides user authentication, video upload, face detection, frame sampling, model prediction, and a simple web interface for viewing detection results.

## Features

* User registration and login
* Password hashing
* Session-based authentication
* Video upload
* Support for common video formats
* Face detection using MTCNN
* Frame sampling for video analysis
* Xception-based deepfake classification
* Batch prediction using TensorFlow
* REAL / FAKE classification with confidence score
* MySQL database integration
* Responsive web interface

## Technology Stack

* Python
* Flask
* TensorFlow / Keras
* Xception
* MTCNN
* OpenCV
* NumPy
* MySQL
* HTML / CSS

## Project Structure

```text
deep_fake_detection_website/
├── database/
│   └── setup.sql
├── static/
│   ├── css/
│   ├── images/
│   ├── d2.jpg
│   └── hero-background.jpg
├── templates/
│   ├── header.html
│   ├── home.html
│   ├── login.html
│   ├── result.html
│   ├── signup.html
│   └── upload.html
├── .gitignore
├── app.py
├── README.md
└── requirements.txt
```

## How It Works

1. A user creates an account or logs in.
2. The user uploads a supported video file.
3. The application temporarily stores the uploaded video for processing.
4. OpenCV samples frames from the video.
5. MTCNN detects faces in the sampled frames.
6. Detected faces are resized and preprocessed for the Xception-based model.
7. TensorFlow performs batch predictions on the detected faces.
8. The individual face predictions are averaged.
9. The application classifies the video as **REAL** or **FAKE**.
10. The result and confidence score are displayed to the user.

## Supported Video Formats

The application supports:

* `.mp4`
* `.avi`
* `.mov`
* `.mkv`
* `.webm`

## Installation

### 1. Clone the Repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd deep_fake_detection_website
```

### 2. Create and Activate a Virtual Environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

## Database Setup

The application uses MySQL.

1. Make sure MySQL Server is installed and running.
2. Open MySQL Workbench or the MySQL command-line client.
3. Run:

```sql
SOURCE database/setup.sql;
```

The script creates the required `deep_fake_detection` database and `users` table.

## Environment Variables

The application requires the following environment variables:

```text
FLASK_SECRET_KEY=your-secret-key
DB_PASSWORD=your-mysql-password
```

Optional database settings:

```text
DB_HOST=localhost
DB_USER=root
DB_NAME=deep_fake_detection
```

Do not commit real passwords, secret keys, or other sensitive values to GitHub.

## Model Setup

The pretrained Xception-based model is intentionally **not included in this repository**.

The model file is excluded through `.gitignore` because it is larger than GitHub's standard file-size limit and its redistribution rights have not been established for this repository.

Place the required model file at:

```text
checkpoints/
└── xception_deepfake_image_5o.h5
```

The application expects the model at this exact path.

> The model must be obtained from a source that permits its use and redistribution where applicable. Verify the source's licensing and terms before downloading or sharing the model.

## Running the Application

Set the required environment variables and start the Flask application:

```powershell
python app.py
```

Then open the local address shown by Flask in your browser.

## Detection Process

The detection pipeline consists of:

```text
Uploaded Video
      ↓
Frame Sampling
      ↓
Face Detection (MTCNN)
      ↓
Face Preprocessing
      ↓
Xception-based Model
      ↓
Batch Predictions
      ↓
Average Fake Probability
      ↓
REAL / FAKE Result
```

The application samples video frames, detects faces, processes the detected faces in batches, and combines the predictions to produce the final classification.

## Security Considerations

* Passwords are stored using password hashing.
* Session cookies are configured with HTTP-only and SameSite settings.
* Secret keys and database passwords are loaded through environment variables.
* Uploaded videos are temporarily stored during processing.
* Temporary processing files are removed after processing.

For production deployment, additional security measures should be implemented, including HTTPS, secure cookie settings, production-grade session storage, stronger database configuration, request validation, and appropriate server configuration.

## Limitations

* Detection accuracy depends on the quality and characteristics of the input video.
* Face detection may fail on heavily distorted, low-resolution, or partially visible faces.
* The application currently performs video processing synchronously.
* Processing time depends on video length, number of detected faces, and available hardware.
* A binary **REAL / FAKE** classification does not identify the specific type or source of manipulation.
* The model is a pretrained model and was not trained as part of this repository.

## Disclaimer

This project is intended for educational and research purposes. Deepfake detection models can produce false positives and false negatives. Detection results should not be treated as definitive proof of authenticity or manipulation.

## License

A license for this project has not yet been selected.

If this project is published for reuse, add an appropriate open-source license after confirming the licensing requirements of the source code, model, datasets, and other third-party components used by the project.
