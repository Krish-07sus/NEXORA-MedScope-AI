# MedScope-AI: AI-Powered Multi-Disease Blood Report Analyzer

## 📌 Problem Statement
Blood reports often contain early warning signs of chronic diseases, but many people struggle to interpret biomarkers quickly. In remote or high-volume healthcare settings, delays in screening can lead to late detection of preventable conditions.

## 💡 Solution Overview
**MedScope-AI** is an intelligent preventive healthcare platform that analyzes blood reports using OCR, machine learning, and rule-based clinical logic.

### Key Capabilities
- **Multi-Disease Screening:** Detects risk patterns related to Liver, Diabetes, Kidney, Heart, Anemia, and Infection.
- **Universal Report Upload:** Supports image reports and PDF reports.
- **Dynamic Biomarker Extraction:** Automatically reads available biomarkers such as ALT, AST, Glucose, Creatinine, HbA1c, Cholesterol, LDL, HDL, TSH, Vitamin D, and more.
- **AI Clinical Summary:** Generates structured summaries, key issues, next steps, and lifestyle guidance.
- **User Dashboard:** Secure login, scan history, and previous report review.
- **Edge Ready:** Can run locally with no constant cloud dependency.

## 🧠 Tech Stack
- **Backend:** Flask (Python)
- **Machine Learning:** Scikit-learn (.pkl models)
- **OCR:** Tesseract OCR + OpenCV preprocessing
- **PDF Parsing:** pdfplumber
- **Database:** SQLite
- **Frontend:** HTML, CSS, Jinja2

## ⚙️ Setup Instructions
1. **Clone the Project**
   ```bash
   git clone https://github.com/Krish-07sus/NEXORA-MedScope-AI
   cd MedScope-AI
   ```
2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run Application**
   ```bash
   python3 app.py
   ```
4. **Open in Browser**
   ```text
   http://127.0.0.1:8080
   ```

## 🔗 Live Demo Link
https://ammonia-rocker-wired.ngrok-free.dev/

## ⚠️ Disclaimer
MedScope-AI is designed for educational screening support and early risk awareness. It is **not** a substitute for professional medical diagnosis or treatment.
# 🚀 MedScope-AI
### AI-Powered Multi-Disease Blood Report Analyzer for Preventive Healthcare

MedScope-AI transforms raw blood reports into understandable health insights using **OCR, Machine Learning, and Clinical Logic**. It helps users detect potential health risks early and understand which biomarkers need attention.

---

## 📌 Problem Statement
Millions of people receive blood test reports but cannot interpret biomarker values quickly. Delays in understanding abnormal results can lead to late diagnosis of preventable diseases such as diabetes, kidney disease, liver dysfunction, anemia, and heart risk.

---

## 💡 Our Solution
MedScope-AI allows users to upload a blood report image or PDF and instantly receive:

- Extracted biomarkers from the report
- Multi-disease risk screening
- AI-generated health summary
- Personalized health suggestions
- Secure scan history dashboard

---

## ✨ Key Features

### 🩺 Smart Disease Screening
Detects risk patterns related to:
- Liver Disorders
- Diabetes
n- Kidney Dysfunction
- Heart / Lipid Risk
- Anemia
- Infection / Inflammation

### 📄 Universal Report Upload
- Upload Images (JPG / PNG)
- Upload PDFs
- OCR-based text extraction

### 🧪 Dynamic Biomarker Engine
Automatically reads available biomarkers such as:
ALT, AST, Bilirubin, Creatinine, Urea, Glucose, HbA1c, Cholesterol, LDL, HDL, Platelets, Hemoglobin, WBC, TSH, Vitamin D and more.

### 🤖 AI Clinical Summary
Generates:
- Summary of findings
- Key risk flags
- Lifestyle guidance
- Suggested next steps

### 👤 User Dashboard
- Login / Signup
- Report history
- Previous scan access

### 🌐 Deployable Anywhere
Runs locally or via ngrok/cloud for remote access.

---

## ⚙️ How It Works
```text
Upload Report → OCR / PDF Parsing → Biomarker Extraction → ML Risk Engine → AI Summary → Dashboard
```

---

## 🧠 Tech Stack
- **Backend:** Flask (Python)
- **Machine Learning:** Scikit-learn Models (.pkl)
- **OCR:** Tesseract OCR + OpenCV
- **PDF Parsing:** pdfplumber
- **Database:** SQLite
- **Frontend:** HTML, CSS, Jinja2
- **Hosting:** ngrok / Cloud compatible

---

## 🌍 UN Sustainable Development Goals
- **SDG 3:** Good Health & Well-being
- **SDG 9:** Industry, Innovation & Infrastructure
- **SDG 12:** Responsible Healthcare Awareness

---

## ⚙️ Setup Instructions
### 1️⃣ Clone Repository
```bash
git clone https://github.com/Krish-07sus/NEXORA-MedScope-AI
cd MedScope-AI
```

### 2️⃣ Install Requirements
```bash
pip install -r requirements.txt
```

### 3️⃣ Run App
```bash
python3 app.py
```

### 4️⃣ Open in Browser
```text
http://127.0.0.1:8080
```

---

## 🔗 Live Demo
Temporary Public Tunnel:
https://ammonia-rocker-wired.ngrok-free.dev/

---

## 📌 Why MedScope-AI Stands Out
- Preventive healthcare focused
- Multi-disease screening in one platform
- Converts confusing reports into clear insights
- Combines OCR + ML + Explainable Output
- Useful for remote and underserved communities

---

## ⚠️ Disclaimer
MedScope-AI is built for **educational screening support and early awareness only**. It is **not a substitute for professional medical diagnosis, emergency care, or treatment advice**.

---

## 👨‍💻 Team / Project
Built during hackathon innovation sprint to simplify healthcare diagnostics using AI.