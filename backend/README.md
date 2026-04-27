# Backend - Spring Boot

This is the backend REST API for the AI-Based Cloud Resource Optimization System.

## Structure
- src/main/java/com/cloudopt/
- src/main/resources/
- pom.xml

## How to Run
- Build: `./mvnw clean install`
- Run: `./mvnw spring-boot:run`

## Endpoints
- `GET /metrics` — Current + predicted CPU
- `POST /predict` — Send data to ML service
- `GET /decision` — Scaling decision

## Integrates with FastAPI ML service.
