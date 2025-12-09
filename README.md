# IdentityGuard: Intelligent KYC and Identity Verification API

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Modern-green.svg)
![Computer Vision](https://img.shields.io/badge/Computer_Vision-OpenCV-red.svg)
![Java](https://img.shields.io/badge/Integration-Java_Client-orange.svg)
![Docker](https://img.shields.io/badge/Containerization-Docker-informational)

## Project Overview
IdentityGuard is a full-stack identity verification pipeline built for modern digital onboarding. It automates KYC processes by combining deep learning, computer vision, and cross-language integration to deliver accurate, real-time identity checks.

This project showcases practical machine learning engineering, including:
* Automated document processing through OCR.
* Biometric face verification using feature embeddings.
* Real-time anti-spoofing checks through motion analysis.
* API-driven microservice architecture with Java integration.

---

## Key Features

### 1. Automated ID Data Extraction (OCR)
**Tech:** Tesseract OCR, OpenCV, Python  
Extracts structured fields such as Name and National ID Number from ID card images using text detection, thresholding, and noise-reduction pre-processing.

### 2. Biometric Face Verification
**Tech:** DeepFace, TensorFlow/Keras  
Generates 128-dimensional embeddings for both the selfie and the ID photo. Uses Euclidean distance to determine identity similarity.

### 3. Liveness Detection (Anti-Spoofing)
**Tech:** OpenCV  
Analyzes facial movement over multiple frames to confirm the user is physically present. Prevents spoofing attempts involving static images.

### 4. Java Integration Layer
**Tech:** Java 11 HttpClient  
Demonstrates enterprise-grade interoperability by consuming the Python API from a Java client, mirroring typical financial sector environments.

---

## System Architecture
IdentityGuard follows a modular architecture:
* FastAPI manages endpoints and routing.
* CV and ML models run as isolated processing modules.
* Verification scores are combined into a single decision output.
* Java client communicates with the backend using REST.

This design keeps the system scalable, testable, and deployable.

---

## Tech Stack
* **Backend:** FastAPI (Python)
* **Computer Vision:** OpenCV
* **Deep Learning:** DeepFace, TensorFlow
* **OCR:** Tesseract
* **Integration:** Java 11 HttpClient
* **Deployment:** Docker-ready

---

## Setup and Usage

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/IdentityGuard-KYC.git
cd IdentityGuard-KYC
```

### Prerequisites
1. **Python 3.9+**
2. **Tesseract OCR**: 
   - Download and install [Tesseract for Windows](https://github.com/UB-Mannheim/tesseract/wiki).
   - Ensure the installation path matches the configuration in `app/core/ocr_utils.py` or add it to your System PATH.

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

Make sure Tesseract OCR is installed on your machine and added to your system PATH.

### 3. Start the API Server
```bash
uvicorn app.main:app --reload
```

The interactive API docs will be available at:
http://127.0.0.1:8000/docs

### API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/verify-identity/` | Upload ID image and selfie for face matching |
| `POST` | `/check-liveness/` | Upload a video to confirm natural movement |
| `POST` | `/extract-id-data/` | Extract text fields from an ID image using OCR |

### Java Client Integration

A lightweight Java example is available in the java-client/ folder.

### Run the Java Client

Ensure the Python server is running, then:
```bash
cd java-client
javac IdentityClient.java
java IdentityClient
```

The client will send a request to the Python API and print the OCR results in the terminal.

## Future Improvements
Deploy to AWS using Docker containers.
Add a database layer (PostgreSQL or MongoDB) for audit trails.
Build a React Native mobile onboarding app.
Optimize models for faster inference and lower latency.