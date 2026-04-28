from flask import Flask, render_template, request, redirect, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
from datetime import datetime
import numpy as np
import joblib
import pytesseract
from PIL import Image
import re
import requests
import json
import socket
import cv2
import pdfplumber
import io
def preprocess_report_image(file):
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return gray


def extract_text_from_report(file):
    filename = getattr(file, "filename", "").lower()

    if filename.endswith(".pdf"):
        try:
            file.seek(0)
            text = ""
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            file.seek(0)
            if text.strip():
                return text
        except:
            file.seek(0)

    file.seek(0)
    processed = preprocess_report_image(file)
    config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(processed, config=config)
    return text


app = Flask(__name__)
app.secret_key = "MedScope-AI-hackathon-secret-key"

rf = joblib.load("rf.pkl")
gb = joblib.load("gb.pkl")
lr = joblib.load("lr.pkl")
scaler = joblib.load("scaler.pkl")

try:
    heart_model = joblib.load("heart_model.pkl")
except:
    heart_model = None

try:
    kidney_model = joblib.load("kidney_model.pkl")
except:
    kidney_model = None

try:
    mimic_model = joblib.load("mimic_model.pkl")
except:
    mimic_model = None


def get_db_connection():
    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS history(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            date TEXT,
            alt REAL,
            ast REAL,
            bilirubin REAL,
            albumin REAL,
            platelets REAL,
            risk REAL,
            level TEXT,
            color TEXT
        )
    """)

    cols = [
        "reasons TEXT",
        "abnormal_count INTEGER",
        "conditions TEXT",
        "explanation TEXT",
        "symptoms TEXT",
        "report_type TEXT"
    ]

    for col in cols:
        try:
            conn.execute(f"ALTER TABLE history ADD COLUMN {col}")
        except:
            pass

    conn.commit()
    conn.close()


init_db()


def format_to_lines(text):
    lines = [x.strip() for x in text.split("\n") if x.strip()]
    return "\n".join([f"• {x}" if not x.startswith("•") else x for x in lines])


def call_ollama(prompt):
    try:
        res = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        return res.json().get("response", "")
    except:
        return "AI explanation unavailable."



def smart_extract(text, patterns):
    if isinstance(patterns, str):
        patterns = [patterns]

    for pattern in patterns:
        m = re.search(pattern, text, re.I)
        if m:
            try:
                return float(m.group(1))
            except:
                pass
    return 0

# ======================================================
# DYNAMIC BIOMARKER EXTRACTION
# ======================================================
def extract_dynamic_biomarkers(text):
    known = {
        "ALT": [r"ALT.*?(\d+\.?\d*)", r"SGPT.*?(\d+\.?\d*)"],
        "AST": [r"AST.*?(\d+\.?\d*)", r"SGOT.*?(\d+\.?\d*)"],
        "Bilirubin": [r"Bilirubin.*?(\d+\.?\d*)"],
        "Albumin": [r"Albumin.*?(\d+\.?\d*)"],
        "Platelets": [r"Platelets.*?(\d+\.?\d*)"],
        "Hemoglobin": [r"Hemoglobin.*?(\d+\.?\d*)", r"\bHb\b.*?(\d+\.?\d*)"],
        "WBC": [r"WBC.*?(\d+\.?\d*)", r"TLC.*?(\d+\.?\d*)"],
        "Creatinine": [r"Creatinine.*?(\d+\.?\d*)"],
        "Urea": [r"Blood Urea.*?(\d+\.?\d*)", r"Urea.*?(\d+\.?\d*)"],
        "Glucose": [r"Glucose.*?(\d+\.?\d*)", r"FBS.*?(\d+\.?\d*)", r"PPBS.*?(\d+\.?\d*)"],
        "HbA1c": [r"HbA1c.*?(\d+\.?\d*)"],
        "Cholesterol": [r"Total Cholesterol.*?(\d+\.?\d*)", r"Cholesterol.*?(\d+\.?\d*)"],
        "HDL": [r"HDL.*?(\d+\.?\d*)"],
        "LDL": [r"LDL.*?(\d+\.?\d*)"],
        "Triglycerides": [r"Triglycerides.*?(\d+\.?\d*)"],
        "TSH": [r"TSH.*?(\d+\.?\d*)"],
        "Vitamin D": [r"Vitamin\s*D.*?(\d+\.?\d*)"]
    }

    values = {}
    for name, patterns in known.items():
        val = smart_extract(text, patterns)
        if val > 0:
            values[name] = val
    return values


def detect_report_type(text):
    t = text.lower()

    diabetes_words = ["hba1c", "fbs", "ppbs", "blood sugar", "glucose fasting"]
    liver_words = ["alt", "ast", "bilirubin", "sgpt", "sgot", "lft"]
    kidney_words = ["creatinine", "urea", "bun", "rft", "egfr"]
    heart_words = ["cholesterol", "hdl", "ldl", "triglycerides", "lipid"]

    if any(word in t for word in diabetes_words):
        return "diabetes"

    if any(word in t for word in liver_words):
        return "liver"

    if any(word in t for word in kidney_words):
        return "kidney"

    if any(word in t for word in heart_words):
        return "heart"

    return "general"


def liver_model_score(values):
    try:
        features = np.array([[
            values["ALT"],
            values["AST"],
            values["Bilirubin"],
            values["Albumin"],
            values["Platelets"]
        ]])

        features = scaler.transform(features)

        probs = [
            rf.predict_proba(features)[0][1],
            gb.predict_proba(features)[0][1],
            lr.predict_proba(features)[0][1]
        ]

        return round(sum(probs) / 3 * 100, 2)

    except:
        return 0


def heart_model_score(values):
    if not heart_model:
        return 0

    try:
        X = np.array([[
            50,
            1,
            0,
            130,
            values["Cholesterol"],
            1 if values["Glucose"] > 120 else 0,
            1,
            150,
            0,
            1.0,
            1,
            0,
            2
        ]])

        p = heart_model.predict_proba(X)[0][1]
        return round(p * 100, 2)
    except:
        return 0


def kidney_model_score(values):
    if not kidney_model:
        return 0

    try:
        X = np.array([[
            130,
            1.020,
            0,
            0,
            1,
            values["Urea"],
            values["Creatinine"],
            140,
            4.5,
            values["Hemoglobin"],
            values["WBC"],
            5.0,
            0
        ]])

        p = kidney_model.predict_proba(X)[0][1]
        return round(p * 100, 2)
    except:
        return 0


def mimic_model_score(values):
    if not mimic_model:
        return 0

    try:
        X = np.array([[
            values["Bilirubin"],
            values["Creatinine"],
            values["Glucose"],
            values["Hemoglobin"]
        ]])

        probs = []

        for est in mimic_model.estimators_:
            probs.append(est.predict_proba(X)[0][1])

        return round(sum(probs) / len(probs) * 100, 2)
    except:
        return 0


def generate_ai_report(values, symptoms, disease_scores, risk, level, report_type):
    prompt = f"""
You are a preventive healthcare assistant. Use cautious, evidence-based language.
Do NOT diagnose disease with certainty.
Do NOT prescribe medicines.
Give educational guidance only.

Report Type: {report_type}
Biomarkers: {values}
Symptoms: {symptoms}
Disease Scores: {disease_scores}
Overall Risk: {risk}% ({level})

Return ONLY in this format:
Summary:
- 2 to 3 bullet points

Key Issues:
- abnormal markers and possible meaning

Recommended Next Steps:
- tests / doctor follow-up / monitoring

Diet Advice:
- food and lifestyle suggestions

Safety Note:
- Educational guidance only. Consult a licensed doctor.
"""
    return format_to_lines(call_ollama(prompt))


def predict_all(values, symptoms, report_type="general"):

    reasons = []
    conditions = []

    liver = 0
    diabetes = 0
    kidney = 0
    anemia = 0
    heart = 0
    infection = 0

    s = symptoms.lower()

    if report_type in ["liver", "general"]:
        liver = liver_model_score(values)

    if report_type in ["diabetes", "general"]:
        glucose = values["Glucose"]
        hba1c = values.get("HbA1c", 0)
        rule_score = 0

        if glucose >= 250:
            rule_score = 95
        elif glucose >= 200:
            rule_score = 90
        elif glucose >= 126:
            rule_score = 75
        elif glucose >= 100:
            rule_score = 45
        elif glucose > 0:
            rule_score = 15

        if hba1c >= 8:
            rule_score = max(rule_score, 95)
        elif hba1c >= 6.5:
            rule_score = max(rule_score, 80)
        elif hba1c >= 5.7:
            rule_score = max(rule_score, 45)

        diabetes = rule_score

    if report_type in ["kidney", "general"]:
        kidney = kidney_model_score(values)

        if kidney == 0:
            creatinine = values["Creatinine"]
            urea = values["Urea"]
            rule_score = 0

            if creatinine > 2 or urea > 60:
                rule_score = 95
            elif creatinine > 1.5 or urea > 45:
                rule_score = 80
            elif creatinine > 1.2 or urea > 35:
                rule_score = 55
            elif creatinine > 0 or urea > 0:
                rule_score = 15

            kidney = max(kidney, rule_score)

    hb = values["Hemoglobin"]

    if hb > 0:
        if hb < 8:
            anemia = 95
        elif hb < 11:
            anemia = 75
        elif hb < 13:
            anemia = 40
        else:
            anemia = 10

    if report_type in ["heart", "general"]:
        heart = heart_model_score(values)

        if heart == 0:
            chol = values["Cholesterol"]
            ldl = values.get("LDL", 0)
            hdl = values.get("HDL", 0)
            tg = values.get("Triglycerides", 0)
            rule_score = 0

            if chol > 240 or ldl > 160 or tg > 250:
                rule_score = 85
            elif chol > 200 or ldl > 130 or tg > 150:
                rule_score = 60
            elif chol > 0 or ldl > 0:
                rule_score = 20

            if hdl > 0 and hdl < 40:
                rule_score = max(rule_score, 65)

            heart = max(heart, rule_score)

    wbc = values["WBC"]

    if wbc >= 15000:
        infection = 95
    elif wbc > 12000:
        infection = 85
    elif wbc > 10000:
        infection = 60
    elif wbc > 0:
        infection = 10

    mimic = mimic_model_score(values)

    if mimic > 0:
        if report_type == "general":
            liver = max(liver, mimic)
            kidney = max(kidney, mimic)

    if report_type == "diabetes":
        diabetes = min(100, diabetes + 10)
    if report_type == "kidney":
        kidney = min(100, kidney + 10)
    if report_type == "heart":
        heart = min(100, heart + 10)
    if report_type == "liver":
        liver = min(100, liver + 10)

    if "thirst" in s:
        diabetes += 10

    if "urination" in s:
        diabetes += 10

    if "fatigue" in s:
        anemia += 10

    if "weakness" in s:
        anemia += 10

    if "fever" in s:
        infection += 10

    if "cough" in s:
        infection += 10

    if "chest pain" in s:
        heart += 10

    if "swelling" in s:
        kidney += 10

    if "yellow" in s or "jaundice" in s:
        liver += 10

    liver = min(liver, 100)
    diabetes = min(diabetes, 100)
    kidney = min(kidney, 100)
    anemia = min(anemia, 100)
    heart = min(heart, 100)
    infection = min(infection, 100)

    disease_scores = {
        "Liver": round(liver, 2),
        "Diabetes": round(diabetes, 2),
        "Kidney": round(kidney, 2),
        "Anemia": round(anemia, 2),
        "Heart": round(heart, 2),
        "Infection": round(infection, 2),
        "Hospital AI": round(mimic, 2)
    }

    active = [v for v in disease_scores.values() if v > 0]
    risk = round(max(active) * 0.65 + (sum(active) / len(active)) * 0.35, 2) if active else 10

    if risk < 30:
        level = "Low"
        color = "green"
    elif risk < 70:
        level = "Medium"
        color = "orange"
    else:
        level = "High"
        color = "red"

    if liver >= 70:
        conditions.append("Possible Liver Disorder")

    if diabetes >= 70:
        conditions.append("Possible Diabetes Risk")

    if kidney >= 70:
        conditions.append("Possible Kidney Dysfunction")

    if anemia >= 70:
        conditions.append("Possible Anemia")

    if heart >= 70:
        conditions.append("Possible Heart Risk")

    if infection >= 70:
        conditions.append("Possible Infection / Inflammation")

    if values["ALT"] > 50:
        reasons.append("High ALT")

    if values["AST"] > 50:
        reasons.append("High AST")

    if values["Glucose"] > 125:
        reasons.append("High Glucose")

    if values["Creatinine"] > 1.2:
        reasons.append("High Creatinine")

    if values["Urea"] > 45:
        reasons.append("High Urea")

    if values["Hemoglobin"] < 11 and values["Hemoglobin"] > 0:
        reasons.append("Low Hemoglobin")

    if values["Cholesterol"] > 200:
        reasons.append("High Cholesterol")

    if values["WBC"] > 10000:
        reasons.append("High WBC")

    abnormal_count = len(reasons)

    explanation = generate_ai_report(
        values,
        symptoms,
        disease_scores,
        risk,
        level,
        report_type
    )

    disease_scores["Confidence"] = min(
        96,
        40 + len([v for v in values.values() if v > 0]) * 5
    )

    return (
        risk,
        level,
        color,
        reasons,
        abnormal_count,
        conditions,
        explanation,
        disease_scores
    )


def save_to_history(values, result, symptoms, report_type):
    if "user_id" not in session:
        return

    risk, level, color, reasons, abnormal_count, conditions, explanation, _ = result

    conn = get_db_connection()

    conn.execute("""
        INSERT INTO history(
            user_id,date,alt,ast,bilirubin,albumin,platelets,
            risk,level,color,reasons,abnormal_count,
            conditions,explanation,symptoms,report_type
        )
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        session["user_id"],
        datetime.now().strftime("%b %d, %Y %H:%M"),
        values["ALT"],
        values["AST"],
        values["Bilirubin"],
        values["Albumin"],
        values["Platelets"],
        risk,
        level,
        color,
        json.dumps(reasons),
        abnormal_count,
        json.dumps(conditions),
        explanation,
        symptoms,
        report_type
    ))

    conn.commit()
    conn.close()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":
        action = request.form["action"]
        username = request.form["username"].strip()
        password = request.form["password"].strip()

        conn = get_db_connection()

        if action == "register":
            try:
                hashed = generate_password_hash(password)
                conn.execute(
                    "INSERT INTO users(username,password) VALUES(?,?)",
                    (username, hashed)
                )
                conn.commit()
                flash("Registration successful")
            except:
                flash("Username exists")

        if action == "login":
            user = conn.execute(
                "SELECT * FROM users WHERE username=?",
                (username,)
            ).fetchone()

            if user and check_password_hash(user["password"], password):
                session["user_id"] = user["id"]
                session["username"] = user["username"]
                conn.close()
                return redirect("/")

            flash("Invalid login")

        conn.close()

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")



@app.route("/history")
def history():

    if "user_id" not in session:
        return redirect("/login")

    conn = get_db_connection()

    records = conn.execute(
        "SELECT * FROM history WHERE user_id=? ORDER BY id DESC",
        (session["user_id"],)
    ).fetchall()

    records = [dict(row) for row in records]

    conn.close()

    return render_template("history.html", records=records)


# ======================================================
# VIEW INDIVIDUAL REPORT
# ======================================================
@app.route("/view/<int:report_id>")
def view_report(report_id):

    if "user_id" not in session:
        return redirect("/login")

    conn = get_db_connection()

    row = conn.execute(
        "SELECT * FROM history WHERE id=? AND user_id=?",
        (report_id, session["user_id"])
    ).fetchone()

    conn.close()

    if not row:
        return "Report not found", 404

    values = {
        "ALT": row["alt"] or 0,
        "AST": row["ast"] or 0,
        "Bilirubin": row["bilirubin"] or 0,
        "Albumin": row["albumin"] or 0,
        "Platelets": row["platelets"] or 0
    }

    clean_values = {k: v for k, v in values.items() if v and float(v) > 0}

    return render_template(
        "result.html",
        risk=row["risk"],
        level=row["level"],
        color=row["color"],
        reasons=json.loads(row["reasons"]) if row["reasons"] else [],
        abnormal_count=row["abnormal_count"] or 0,
        conditions=json.loads(row["conditions"]) if row["conditions"] else [],
        explanation=row["explanation"] or "No AI summary available.",
        answer=None,
        disease_scores={},
        values=clean_values,
        report_type=row["report_type"] or "history"
    )


@app.route("/predict", methods=["POST"])
def predict():

    def num(name):
        val = request.form.get(name, "").strip()
        return float(val) if val else 0

    values = {
        "ALT": num("ALT"),
        "AST": num("AST"),
        "Bilirubin": num("Bilirubin"),
        "Albumin": num("Albumin"),
        "Platelets": num("Platelets"),
        "Hemoglobin": num("Hemoglobin"),
        "WBC": num("WBC"),
        "Creatinine": num("Creatinine"),
        "Glucose": num("Glucose"),
        "Cholesterol": num("Cholesterol"),
        "Urea": num("Urea")
    }

    symptoms = request.form.get("symptoms", "")
    report_type = "general"

    result = predict_all(values, symptoms, report_type)

    session["values"] = values
    session["result"] = result
    session["symptoms"] = symptoms
    session["report_type"] = report_type
    session.pop("answer", None)

    save_to_history(values, result, symptoms, report_type)

    return redirect("/result")


@app.route("/upload", methods=["POST"])
def upload():

    file = request.files["file"]
    if file.filename == "":
        return redirect("/")

    text = extract_text_from_report(file)
    if not text.strip():
        flash("Could not read report clearly. Try clearer file or PDF.")
        return redirect("/")
    report_type = detect_report_type(text)
    dynamic_values = extract_dynamic_biomarkers(text)

    values = {
        "ALT":0,"AST":0,"Bilirubin":0,"Albumin":0,
        "Platelets":0,"Hemoglobin":0,"WBC":0,
        "Creatinine":0,"Glucose":0,"Cholesterol":0,"Urea":0
    }
    values.update(dynamic_values)

    if "HbA1c" in dynamic_values and values.get("Glucose",0) == 0:
        values["Glucose"] = dynamic_values["HbA1c"] * 25

    symptoms = request.form.get("symptoms", "")
    result = predict_all(values, symptoms, report_type)

    session["values"] = values
    session["result"] = result
    session["symptoms"] = symptoms
    session["report_type"] = report_type
    session.pop("answer", None)

    save_to_history(values, result, symptoms, report_type)
    return redirect("/result")


@app.route("/ask", methods=["POST"])
def ask():

    question = request.form["question"]

    values = session["values"]
    risk, level, *_ = session["result"]
    symptoms = session.get("symptoms", "")
    report_type = session.get("report_type", "general")

    prompt = f"""
You are a healthcare assistant. Be concise and safe.
Do not prescribe medicines.
Do not claim certainty.

Patient report type: {report_type}
Symptoms: {symptoms}
Biomarkers: {values}
Overall Risk: {risk}% ({level})

Question: {question}

Answer in 3 short bullet points with practical lifestyle or follow-up advice.
End with: Consult a doctor for diagnosis.
"""

    answer = format_to_lines(call_ollama(prompt))
    session["answer"] = answer

    return redirect("/result")


@app.route("/result")
def result():

    if "result" not in session:
        return redirect("/")

    risk, level, color, reasons, abnormal_count, conditions, explanation, disease_scores = session["result"]
    answer = session.get("answer")

    clean_values = {}
    for k, v in session.get("values", {}).items():
        try:
            if float(v) > 0:
                clean_values[k] = v
        except:
            pass

    return render_template(
        "result.html",
        risk=risk,
        level=level,
        color=color,
        reasons=reasons,
        abnormal_count=abnormal_count,
        conditions=conditions,
        explanation=explanation,
        answer=answer,
        disease_scores=disease_scores,
        values=clean_values,
        report_type=session.get("report_type", "general")
    )


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)