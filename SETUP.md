# 🧠 Mental Health Risk Prediction Dashboard - Setup Guide

## What's Included in This Package

This project contains:
- ✅ **Source Code**: All Python scripts (dags/, dashboard/, etc.)
- ✅ **Configuration Files**: Dockerfile, docker-compose.yml, requirements.txt
- ✅ **Data Files**: Sample CSV data files for dashboard and ML pipeline
- ✅ **Trained Models**: Pre-trained ML models (.pkl files) to run predictions immediately
- ❌ **NOT Included**: Virtual environment (.venv) - you must create your own

---

## Setup Instructions for Others

### **Prerequisites**
- Docker Desktop (Windows/Mac) or Docker + Docker Compose (Linux)
- **Python 3.12.3** (REQUIRED - newer packages have specific Python version requirements)
  - ⚠️ Python 3.8, 3.9, 3.10, or 3.11 may cause Streamlit compatibility issues
  - Check your version: `python3 --version`
- ~2GB disk space
- A terminal/command prompt

### **Step 1: Unzip and Navigate**
```bash
unzip IS3107_Mental_Health_Project.zip
cd IS3107_Mental_Health_Project
```

### **Step 2: Start Docker Containers (Airflow + PostgreSQL)**
This is **required** for the dashboard to work, even if they don't run the entire pipeline.

```bash
docker compose up -d
```

**Wait 30-60 seconds** for containers to fully initialize.

✅ **Verify success:**
```bash
docker compose ps
```

You should see:
- `postgres` - Healthy
- `airflow-scheduler` - Running  
- `airflow-webserver` - Running (or shortly will be)

**View Airflow UI**: http://localhost:8080 , or http://localhost:8081 (if 8080 is used)
username: admin
password: admin

---

### **Step 3: Create Python Virtual Environment**
open another terminal

**IMPORTANT: Ensure you're using Python 3.12+**
```bash
python3 --version  # Should show Python 3.12.x or higher
```

If your Python version is lower:
- **Using pyenv (Linux/macOS):** A `.python-version` file is included. Run `pyenv install 3.12.3` then `pyenv local 3.12.3`
- **Windows:** Download Python 3.12 from [python.org](https://www.python.org/downloads/)
- **Linux/macOS Homebrew:** `brew install python@3.12` then use `python3.12 -m venv .venv`

#### **Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### **Windows (CMD/PowerShell):**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**You'll know it worked when your prompt shows `(.venv)` at the start.**

---

### **Step 4: Install Python Dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**⚠️ Permission Note (Linux/macOS ONLY):**
If you get permission errors with data/ or models/, run:
```bash
chmod 755 data/
chmod 755 models/
```

*(Windows users: this is not needed. File permissions work differently in Windows.)*

---

### **Step 5: Launch the Dashboard**

```bash
cd dashboard
streamlit run app.py
```

The dashboard will open at: **http://localhost:8501**

---

## Platform-Specific Notes

### **Windows**
- ✅ No `chmod` commands needed
- ✅ Docker Desktop handles all permissions automatically
- ✅ Path separators: Use `\` or `/` both work
- ⚠️ First Docker startup may take 5-15 minutes on first run
- File permissions are inherited from NTFS, no manual fixes needed

### **macOS**
- ✅ Same as Linux setup
- ⚠️ May need `chmod 755` if copied from Linux
- ✅ Docker Desktop available
- M1/M2 chips: Fully supported

### **Linux (Ubuntu/Debian)**
- ✅ Standard setup as shown above
- ⚠️ May need `chmod 755` for data/ and models/ folders if permission denied errors occur
- ✅ Ensure Docker daemon is running: `sudo systemctl start docker`

---

## Troubleshooting

### **Port Already in Use (8080 or 8501)**

If you see: `Address already in use`

**Option 1 - Kill the process:**
```bash
# macOS/Linux
lsof -i :8080  # Find what's using it
kill -9 <PID>

# Windows (PowerShell as Admin)
netstat -ano | findstr :8080
taskkill /PID <PID> /F
```

**Option 2 - Use different ports:**
Edit `docker-compose.yml`:
```yaml
ports:
  - "8081:8080"  # Use 8081 instead
```

Then `docker compose up -d` again.

### **Permission Denied on data/ or models/**

**Linux/macOS only:**
```bash
chmod 755 data/
chmod 755 models/
```

Do NOT use `chmod 777` unless necessary - `755` is safer.

### **Docker containers won't start**

```bash
# Check logs
docker compose logs

# Restart everything
docker compose down
docker compose up -d
```

### **Streamlit "Module not found" errors**

```bash
# Make sure .venv is activated (you should see (.venv) in your prompt)
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Reinstall requirements
pip install -r requirements.txt
```

### **Streamlit not launching or crashes immediately**

This is usually due to **Python version incompatibility**. The project requires **Python 3.12.3**.

**Check your Python version:**
```bash
python3 --version
```

**If it's not 3.12.x:**

- **Linux/macOS (using Homebrew):**
  ```bash
  brew install python@3.12
  rm -rf .venv  # Delete old virtual environment
  python3.12 -m venv .venv  # Create new with Python 3.12
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

- **Linux (apt):**
  ```bash
  sudo apt install python3.12 python3.12-venv
  rm -rf .venv
  python3.12 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

- **Windows:**
  - Uninstall current Python from Settings > Apps
  - Download Python 3.12 from [python.org](https://www.python.org/downloads/)
  - **Check "Add Python 3.12 to PATH"** during installation
  - Delete `.venv` folder
  - Run: `python -m venv .venv` then `.venv\Scripts\activate`
  - Run: `pip install --upgrade pip` then `pip install -r requirements.txt`

### **Database connection errors (PostgreSQL)**

```bash
# Ensure containers are running
docker compose ps

# If postgres is down, restart it
docker compose restart postgres

# Check postgres logs
docker compose logs postgres
```

---

## File Structure

```
IS3107_Mental_Health_Project/
├── docker-compose.yml          # Docker orchestration
├── Dockerfile                   # Container configuration
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── README.md                    # (Optional) Project overview
│
├── dags/                        # Airflow DAGs
│   ├── mental_health_pipeline_dag.py
│   └── scripts/                 # Helper scripts
│
├── dashboard/                   # Streamlit application
│   └── app.py
│
├── data/                        # CSV data files
│   ├── raw_mental_health_risk_data.csv
│   ├── ml_ready_data.csv
│   ├── clean_mental_health_data.csv
│   ├── dashboard_ready_data.csv
│   └── final_predictions_with_recs.csv
│
└── models/                      # Trained ML models
    ├── champion_model.pkl       # Primary model
    ├── xgboost.pkl
    ├── lightgbm.pkl
    ├── random_forest.pkl
    └── logistic_regression.pkl
```

