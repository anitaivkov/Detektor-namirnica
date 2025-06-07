import os
os.environ["PYTORCH_JIT"] = "0"

import streamlit as st

from app.model_utils import load_model
from app.ui_elements import (
    init_session_state,
    render_login,
    render_camera_url,
    show_detected_items,
)
from app.detection import detect_from_camera
from database.users import UserDB
from database.food import FoodDB
from database.lists import ListDB

# === Streamlit config ===
st.set_page_config(page_title="Detektor Namirnica", layout="wide")
st.title("🧠📷 Detektor Namirnica")

# === Inicijalizacija stanja i baza ===
init_session_state()
user_db = UserDB()
food_db = FoodDB()
list_db = ListDB()

# === Prijava korisnika ===
username = render_login(user_db)
if not username or not st.session_state.user_id:
    st.stop()

# === Učitaj model ===
model = load_model()

# === URL kamere ===
camera_url = render_camera_url()

# === Threshold slideri ===
foods = food_db.get_all_foods()
thresholds = {
    food: st.slider(f"Prag za: {food}", min_value=0.0, max_value=1.0, value=0.25)
    for food in foods
}

# === Detekcija: Start / Stop + spremanje ===
if "run_detection" not in st.session_state:
    st.session_state.run_detection = False

col1, col2 = st.columns(2)
with col1:
    if st.button("▶ Pokreni detekciju"):
        st.session_state.run_detection = True
        st.session_state.detected_items = detect_from_camera(
            model, camera_url, thresholds, st, st.session_state.user_id, list_db
        )

with col2:
    if st.button("⏹ Zaustavi detekciju") and st.session_state.run_detection:
        st.session_state.run_detection = False

        if st.session_state.detected_items:
            list_db.save_list(st.session_state.user_id, st.session_state.detected_items)
            st.success("Detekcija zaustavljena i popis spremljen!")
        else:
            st.warning("Nema detektiranih stavki za spremiti.")

# === Prikaz detekcija uživo ===
show_detected_items()

# === Prikaz spremljenih popisa ===
st.subheader("📋 Vaši spremljeni popisi")
user_lists = list_db.get_lists(st.session_state.user_id)

if user_lists:
    for entry in user_lists:
        count = entry[0]
        confidence = float(entry[1])
        timestamp = entry[2]
        naziv = entry[3]
        st.write(f"- **{naziv}** x{count} (točnost: {confidence:.2f}, zadnje ažuriranje: {timestamp})")
else:
    st.write("Još nema spremljenih popisa.")


