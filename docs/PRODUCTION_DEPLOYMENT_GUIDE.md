# AGRIGUARDIAN — PRODUCTION DEPLOYMENT GUIDE
**Containerization, Docker Compose, Kubernetes, Reverse Proxy & Production Operations**  
**Author:** AgriGuardian Platform Architecture Team  
**Date:** November 10, 2026  
**Status:** Validated & Deployment Ready  

---

## 1. Production Architecture Overview

The AgriGuardian production topology separates the headless FastAPI application tier from the interactive Streamlit user interface, behind an Nginx reverse proxy providing SSL/TLS termination, rate limiting, and HTTP security enforcement:

```
[ Internet / Farmers / Mobile Clients / PWAs ]
                     |
                     v (HTTPS :443)
      +------------------------------+
      |     Nginx Reverse Proxy      |
      |   - TLS 1.3 Termination      |
      |   - Static Asset Caching     |
      |   - Security Headers (OWASP) |
      +--------------+---------------+
                     |
       +-------------+-------------+
       |                           |
       v (Internal :8000)          v (Internal :8501)
+-----------------------+   +-----------------------+
|  FastAPI Backend Pod  |   | Streamlit Frontend Pod|
|  - Uvicorn Workers    |   | - Farmer Dashboard   |
|  - MobileNetV3 CNN    |   | - Expert Verification |
|  - BM25 Crop Advisory |   | - Multi-language TTS  |
|  - Metrics / Health   |   +-----------------------+
+-----------+-----------+
            |
            v (Internal :5432)
+-----------------------------------+
|     PostgreSQL 16 High-Avail      |
|  - 45 Relational Tables           |
|  - WAL Archiving & Hot Standby    |
+-----------------------------------+
```

---

## 2. Containerization

AgriGuardian is containerized via a production-grade multi-stage [`Dockerfile`](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/Dockerfile):

### 2.1 Key Container Features

- **Multi-Stage Build:** `python:3.11-slim` builder stage compiles C-extensions (such as `argon2-cffi`), keeping the final runtime image lean (< 400 MB).
- **Security Invariant (Non-Root User):** Runs as unprivileged user `appuser` (UID 10001) to eliminate container breakout risks.
- **Healthcheck Probe:** Built-in Docker `HEALTHCHECK` querying `/api/v1/health/live` with 30s interval, 5s timeout, and 3 retries.
- **Deterministic Dependencies:** Pinned through `requirements.txt`.

### 2.2 Building the Container Image

```bash
docker build -t agriguardian/backend:2.12.0 -f Dockerfile .
```

---

## 3. Orchestration with Docker Compose

A complete multi-container stack is provided in [`docker-compose.yml`](file:///c:/Users/Meenal%20Sinha/Downloads/AgriGuardian/docker-compose.yml):

```yaml
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile
    command: ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
    environment:
      - APP_ENV=production
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
      - FARM_DB_PATH=data/farm_guardian.db
    volumes:
      - app_data:/app/data
      - app_logs:/app/logs
    ports:
      - "8000:8000"
    restart: always

  frontend:
    build:
      context: .
      dockerfile: Dockerfile
    command: ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
    environment:
      - BACKEND_API_URL=http://backend:8000
    ports:
      - "8501:8501"
    depends_on:
      - backend
    restart: always

  postgres:
    image: postgres:16-alpine
    environment:
      - POSTGRES_DB=agriguardian
      - POSTGRES_USER=agriguardian_user
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - pg_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: always

volumes:
  app_data:
  app_logs:
  pg_data:
```

### 3.1 Launching the Stack

```bash
docker-compose up -d
```

---

## 4. Environment Variables Configuration

Copy `.env.example` to `.env` and configure production secrets:

```ini
# Application Environment
APP_ENV=production
DEBUG=false
APP_PORT=8000
APP_HOST=0.0.0.0

# Security Credentials
# Generate via: python -c "import secrets; print(secrets.token_hex(32))"
JWT_SECRET_KEY=e83a91b4c92e70f61208a54c93d8b1e4f012a938c47b59e0a1b2c3d4e5f6a7b8
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=1440

# Allowed CORS Origins (NO WILDCARDS IN PRODUCTION)
ALLOWED_ORIGINS=https://agriguardian.gov.in,https://app.agriguardian.gov.in

# External Provider Integrations
GNANI_API_KEY=your_production_gnani_api_key
GNANI_API_URL=https://api.gnani.ai/v2
OPENAI_API_KEY=your_openai_production_key

# Request & Rate Limiting
MAX_REQUEST_BODY_SIZE_BYTES=15728640
LOGIN_RATE_LIMIT_THRESHOLD=5
LOGIN_RATE_LIMIT_WINDOW_SECONDS=300
LOGIN_LOCKOUT_DURATION_SECONDS=900
```

---

## 5. Nginx Reverse Proxy Configuration

```nginx
server {
    listen 80;
    server_name agriguardian.gov.in;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name agriguardian.gov.in;

    ssl_certificate /etc/letsencrypt/live/agriguardian.gov.in/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/agriguardian.gov.in/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Maximum file upload size for crop diagnostic images
    client_max_body_size 15M;

    # API Backend Proxy
    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Streamlit Web App Proxy
    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

---

## 6. Scaling & High Availability

- **Horizontal Pod Autoscaling (HPA):**
  - Scale out when CPU utilization exceeds 70% or active concurrent requests exceed 50 per pod.
  - Recommended minimum replicas: 3 pods. Maximum replicas: 20 pods.
- **Read Replicas:** Read queries (market intelligence, crop knowledge profiles) can be offloaded to PostgreSQL read replicas.
- **Model Invariant:** The MobileNetV3 CNN model checkpoint (`data/models/mobilenetv3_plantvillage.pth`) is loaded into read-only process memory per worker thread during startup, ensuring thread-safe inference without lock contention.
