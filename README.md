\# DeepFake Detection



A Flask-based web application for detecting whether an uploaded video is \*\*REAL\*\* or \*\*FAKE\*\* using a pretrained Xception-based deepfake detection model.



The application provides user authentication, video upload, face detection, frame sampling, model prediction, and a simple web interface for viewing detection results.



\## Features



\- User registration and login

\- Password hashing

\- Session-based authentication

\- Video upload

\- Support for common video formats

\- Face detection using MTCNN

\- Frame sampling for video analysis

\- Xception-based deepfake classification

\- Batch prediction using TensorFlow

\- REAL / FAKE classification with confidence score

\- MySQL database integration

\- Responsive web interface



\## Technology Stack



\- Python

\- Flask

\- TensorFlow / Keras

\- Xception

\- MTCNN

\- OpenCV

\- NumPy

\- MySQL

\- HTML / CSS



\## Project Structure



```text

deep\_fake\_detection\_website/

├── checkpoints/

│   └── xception\_deepfake\_image\_5o.h5

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

