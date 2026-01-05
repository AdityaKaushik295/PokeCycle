import os
import sqlite3
import random
import uuid
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, render_template, request, redirect, session
from datetime import date, timedelta
from transformers import pipeline  # <-- NEW
import io  # <-- NEW

# ==========================
# Flask App Config
# ==========================
app = Flask(__name__)
app.secret_key = "pokemon-trash-secret"

UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

DB_PATH = "database.db"

RARITY_WEIGHTS = {
    "Common": 60,
    "Uncommon": 25,
    "Rare": 10,
    "Legendary": 5
}

SHINY_CHANCE = 0.05 

# ==========================
# Database Setup
# ==========================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_pokemon (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        pokemon_name TEXT,
        waste_type TEXT,
        caught_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS recycling_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        date TEXT
    )
    """)

    conn.commit()
    conn.close()

init_db()

def get_db():
    return sqlite3.connect(DB_PATH)

# ==========================
# Load Pokémon CSV
# ==========================
pokemon_df = pd.read_csv("pokemon.csv")

# ==========================
# Waste → Pokémon Type Map
# ==========================
WASTE_TO_TYPES = {
    "cardboard": ["Grass", "Normal"],
    "paper": ["Grass", "Normal"],
    "plastic": ["Water", "Poison"],
    "glass": ["Ice", "Psychic"],
    "metal": ["Steel", "Electric"]
}

# ==========================
# Pokémon Helpers
# ==========================
def get_rarity():
    r = random.randint(1, 100)
    s = 0
    for rarity, w in RARITY_WEIGHTS.items():
        s += w
        if r <= s:
            return rarity
    return "Common"

def is_shiny():
    return random.random() < SHINY_CHANCE

def get_random_pokemon_by_types(types):
    candidates = pokemon_df[
        pokemon_df["Type1"].isin(types) |
        pokemon_df["Type2"].isin(types)
    ]
    if candidates.empty:
        return None

    p = candidates.sample(1).iloc[0]
    name = p["Name"].lower()

    type2 = p["Type2"]
    if pd.isna(type2):
        type2 = None

    return {
        "name": p["Name"],
        "type1": p["Type1"],
        "type2": type2,
        "image": f"static/images/{name}.png",
        "rarity": get_rarity(),
        "shiny": is_shiny()
    }

# ==========================
# Waste Classification Model
# ==========================
MODEL_PATH = "models/waste_mobilenetv3_6class.pt"
LABELS_PATH = "models/labels.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open(LABELS_PATH) as f:
    WASTE_LABELS = [l.strip() for l in f.readlines()]

waste_model = models.mobilenet_v3_small(pretrained=False)
waste_model.classifier[3] = nn.Linear(
    waste_model.classifier[3].in_features,
    len(WASTE_LABELS)
)
waste_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
waste_model.to(DEVICE)
waste_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_waste(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(waste_model(tensor), dim=1)[0]
    idx = torch.argmax(probs).item()
    return WASTE_LABELS[idx], probs[idx].item()

"""
# ==========================
# Pokémon Appearance Classifier (NEW)
# ==========================
LOCAL_POKEMON_MODEL_PATH = "models/pokemon_appearance"

print("Loading Pokémon appearance classifier from local files...")
try:
    pokemon_pipe = pipeline(
        "image-classification",
        model=LOCAL_POKEMON_MODEL_PATH,
        tokenizer=LOCAL_POKEMON_MODEL_PATH,  # not used for vision, but safe
        feature_extractor=LOCAL_POKEMON_MODEL_PATH,
        device=0 if torch.cuda.is_available() else -1
    )
    print("✅ Local Pokémon model loaded!")
except Exception as e:
    print(f"⚠️ Failed to load local Pokémon model: {e}")
    pokemon_pipe = None

"""

# ==========================
# Recycling & Streak Logic
# ==========================
def log_recycling(user_id):
    today = date.today().isoformat()
    db = get_db()
    cur = db.cursor()
    cur.execute(
        "SELECT 1 FROM recycling_log WHERE user_id=? AND date=?",
        (user_id, today)
    )
    if not cur.fetchone():
        db.execute(
            "INSERT INTO recycling_log (user_id, date) VALUES (?,?)",
            (user_id, today)
        )
        db.commit()
    db.close()

def calculate_streak(user_id):
    db = get_db()
    cur = db.cursor()
    cur.execute(
        "SELECT DISTINCT date FROM recycling_log WHERE user_id=? ORDER BY date DESC",
        (user_id,)
    )
    dates = [date.fromisoformat(d[0]) for d in cur.fetchall()]
    db.close()

    streak = 0
    prev = None
    for d in dates:
        if prev is None or prev - d == timedelta(days=1):
            streak += 1
            prev = d
        else:
            break
    return streak

# ==========================
# Routes
# ==========================
@app.route("/", methods=["GET", "POST"])
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = request.form["username"]
        p = request.form["password"]
        db = get_db()
        cur = db.cursor()
        cur.execute(
            "SELECT id FROM users WHERE username=? AND password=?",
            (u, p)
        )
        row = cur.fetchone()
        db.close()
        if row:
            session["user_id"] = row[0]
            session["username"] = u
            return redirect("/dashboard")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        u = request.form["username"]
        p = request.form["password"]
        try:
            db = get_db()
            db.execute(
                "INSERT INTO users (username, password) VALUES (?,?)",
                (u, p)
            )
            db.commit()
            db.close()
            return redirect("/login")
        except:
            return "Username already exists"
    return render_template("register.html")

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect("/login")

    db = get_db()
    cur = db.cursor()

    cur.execute(
        "SELECT pokemon_name, COALESCE(waste_type,'unknown') FROM user_pokemon WHERE user_id=?",
        (session["user_id"],)
    )
    rows = cur.fetchall()

    pokemons = []
    for name, waste in rows:
        row = pokemon_df[pokemon_df["Name"] == name].iloc[0]
        type2 = row["Type2"]
        if pd.isna(type2):
            type2 = None

        pokemons.append({
            "name": name,
            "image": f"static/images/{name.lower()}.png",
            "type1": row["Type1"],
            "type2": type2,
            "waste": waste
        })

    total_recycles = len(rows)

    cur.execute("""
        SELECT COALESCE(waste_type,'unknown'), COUNT(*)
        FROM user_pokemon
        WHERE user_id=?
        GROUP BY COALESCE(waste_type,'unknown')
    """, (session["user_id"],))
    waste_stats = {w: c for w, c in cur.fetchall()}

    streak = calculate_streak(session["user_id"])
    db.close()

    return render_template(
        "dashboard.html",
        pokemons=pokemons,
        total_recycles=total_recycles,
        streak=streak,
        waste_stats=waste_stats
    )

@app.route("/catch", methods=["GET", "POST"])
def catch():
    if "user_id" not in session:
        return redirect("/login")

    pokemon = waste = confidence = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            fname = f"{uuid.uuid4()}_{file.filename}"
            path = os.path.join(UPLOAD_DIR, fname)
            file.save(path)

            waste, confidence = predict_waste(path)

            if waste != "trash":
                pokemon = get_random_pokemon_by_types(WASTE_TO_TYPES[waste])
                if pokemon:
                    db = get_db()
                    db.execute(
                        "INSERT INTO user_pokemon (user_id, pokemon_name, waste_type) VALUES (?,?,?)",
                        (session["user_id"], pokemon["name"], waste)
                    )
                    db.commit()
                    db.close()
                    log_recycling(session["user_id"])

    return render_template(
        "catch.html",
        pokemon=pokemon,
        waste=waste,
        confidence=confidence
    )

# ==========================
# NEW: Catch by Appearance Route
# ==========================
from gradio_client import Client, handle_file
import os
import uuid

HF_SPACE_ID = "AdityaK007/pokeCycle"

@app.route("/catch_appearance", methods=["GET", "POST"])
def catch_appearance():
    if "user_id" not in session:
        return redirect("/login")

    pokemon = None
    predicted_name = None
    confidence = 0.0

    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename != '':
            fname = f"{uuid.uuid4()}_{file.filename}"
            path = os.path.join(UPLOAD_DIR, fname)
            file.save(path)

            try:
                client = Client(HF_SPACE_ID)
                result = client.predict(
                    img=handle_file(path),
                    api_name="/classify_pokemon"
                )

                # --- NEW: Handle your Space's ACTUAL output format ---
                if isinstance(result, dict):
                    if "confidences" in result and len(result["confidences"]) > 0:
                        top_pred = result["confidences"][0]
                        predicted_name = str(top_pred["label"]).strip().lower()
                        confidence = float(top_pred["confidence"])
                    elif "label" in result:
                        # Fallback to top-level 'label' if 'confidences' missing
                        predicted_name = str(result["label"]).strip()
                        confidence = 1.0
                    else:
                        predicted_name = "No label found"
                        confidence = 0.0
                else:
                    predicted_name = f"Unexpected format: {type(result)}"
                    confidence = 0.0

                # --- Now match against your dataset ---
                if confidence > 0:
                    normalized_name = predicted_name
                    if normalized_name in pokemon_df["Name"].values:
                        row = pokemon_df[pokemon_df["Name"] == normalized_name].iloc[0]
                        type2 = row["Type2"] if not pd.isna(row["Type2"]) else None

                        pokemon = {
                            "name": normalized_name,
                            "type1": row["Type1"],
                            "type2": type2,
                            "image": f"static/images/{normalized_name.lower()}.png",
                            "rarity": get_rarity(),
                            "shiny": is_shiny()
                        }

                        # Save to DB
                        db = get_db()
                        db.execute(
                            "INSERT INTO user_pokemon (user_id, pokemon_name, waste_type) VALUES (?,?,?)",
                            (session["user_id"], normalized_name, "appearance")
                        )
                        db.commit()
                        db.close()

            except Exception as e:
                print(f"Gradio client error: {e}")
                predicted_name = "Classification failed"
                confidence = 0.0
            finally:
                if os.path.exists(path):
                    os.remove(path)

    return render_template(
        "catch_appearance.html",
        pokemon=pokemon,
        predicted_name=predicted_name,
        confidence=confidence
    )

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

# ==========================
# Run App
# ==========================
if __name__ == "__main__":
    app.run(debug=True)