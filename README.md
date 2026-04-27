# AI-Based Cloud Resource Optimization System

This repository contains a full-stack system for predicting cloud machine CPU usage and making scaling decisions.

## Structure

- `frontend/` — ReactJS dashboard
- `backend/` — Spring Boot REST API
- `ml-service/` — FastAPI ML microservice
- `archive/` — Sample dataset

## Setup Steps

### 1. ML Service (FastAPI)
- Go to `ml-service/`
- Create a Python virtual environment
- Install requirements: `pip install -r requirements.txt`
- Run: `uvicorn main:app --reload`

### 2. Backend (Spring Boot)
- Go to `backend/`
- Build with Maven/Gradle
- Run: `./mvnw spring-boot:run` or `./gradlew bootRun`

### 3. Frontend (React)
- Go to `frontend/`
- Run: `npm install && npm start`

### 4. Dataset
- Sample data in `archive/borg_traces_data.csv`

### 5. Deployment
- Deploy backend and ML service to AWS EC2
- Store dataset in S3

---

## Features
- Predicts CPU usage (time-series, RandomForest)
- Scaling decisions (SCALE UP/DOWN/STABLE)
- Dashboard with charts and controls

---

## Authors
- Your Name

---

## License
See LICENSE
