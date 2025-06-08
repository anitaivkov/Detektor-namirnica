import os
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

os.environ["PYTORCH_JIT"] = "0"
st.set_page_config(page_title="Detektor Namirnica", layout="wide")
st.title("🧠📷 Detektor Namirnica")

init_session_state() #inicijalizira sve potrebne session_state varijable

if 'stframe' not in st.session_state:
    st.session_state.stframe = st.empty()

user_db = UserDB()
food_db = FoodDB()
list_db = ListDB(food_db)

# === Red 1: Prijava korisnika ===
row1_left, row1_right = st.columns([1, 2])
with row1_left:
    username = render_login(user_db)
with row1_right:
    pass

# === Red 2: URL kamere i dugmad / Kamera ===
row2_left, row2_right = st.columns([1, 2])

with row2_left:
    camera_url = render_camera_url()

    # Dugmad za kontrolu detekcije
    col_start_stop_left, col_start_stop_right = st.columns(2)
    
    with col_start_stop_left:
        if st.button("▶ Pokreni detekciju", key="start_detection_btn"):
            if not st.session_state.get("run_detection", False): # Spriječi višestruko pokretanje
                st.session_state.run_detection = True
                st.info("Detekcija pokrenuta. Pričekajte da se stream učita...")
                
                with st.spinner("Učitavam model..."):
                    model = load_model()
                st.success("Model uspješno učitan.")
                
                # Pozovi funkciju za detekciju
                detect_from_camera(
                    model, camera_url, st.session_state.thresholds,
                    st.session_state.stframe,
                    st.session_state.user_id, list_db
                )
                # Nakon što detect_from_camera završi (npr. petlja se prekine), resetiraj stanje
                st.session_state.run_detection = False 
                st.info("Detekcija završena ili zaustavljena.")
    
    with col_start_stop_right:
        if st.button("◼ Zaustavi detekciju", key="stop_detection_btn"):
            if st.session_state.get("run_detection", False): # Provjeri je li detekcija aktivna
                st.session_state.run_detection = False
                st.info("Zahtjev za zaustavljanje detekcije poslan. Pričekajte...")
            else:
                st.info("Detekcija već nije aktivna.")

# --- Prikaz video streama u desnom stupcu (gdje je stframe inicijaliziran) ---
with row2_right:
    pass 
    
if not username or not st.session_state.user_id:
    st.stop()

# === Red 3: Popisi i detektirane namirnice ===
row3_left, row3_mid, row3_right = st.columns(3)

# -- Lijevo: Popisi --
with row3_left:
    st.markdown("#### 📋 Prethodni popisi:")
    user_timestamps = list_db.get_unique_timestamps(st.session_state.user_id)
    if user_timestamps:
        for idx, timestamp in enumerate(user_timestamps[:5], start=1):
            display_name = f"Popis: {timestamp.strftime('%d.%m.%Y.; %H:%M')}"
            if st.button(f"🗒️ {display_name}", key=f"list_{idx}"):
                st.session_state.selected_timestamp = timestamp
    else:
        st.write("Još nema spremljenih popisa.")

# -- Sredina: Sadržaj odabranog popisa --
with row3_mid:
    if "selected_timestamp" in st.session_state and st.session_state.selected_timestamp:
        st.markdown("#### 📦 Sadržaj odabranog popisa:")
        selected_time = st.session_state.selected_timestamp
        st.markdown(f"🗒️ Popis: {selected_time.strftime('%d.%m.%Y.; %H:%M')}")
        list_items = list_db.get_list_items_by_timestamp(
            st.session_state.user_id, selected_time
        )
        for item in list_items:
            st.write(f"- **{item[0]}** x{item[1]} (točnost: {item[2]:.2f})")
    else:
        st.write("Odaberite popis s lijeve strane.")

# -- Desno: Detektirane namirnice --
with row3_right:
    st.markdown("#### 🔍 Detektirano:")
    if st.session_state.get("detected_items"):
        for name, data in st.session_state.detected_items.items():
            st.write(f"- {name} x{data['count']} (točnost: {data['confidence']:.2f})")
    else:
        st.write("Nema novih detekcija.")

# === Red 5: Podešavanje pragova ===
st.divider()
with st.expander("⚙️ Izmijeni pragove detekcije"):
    foods = food_db.get_all_foods()
    if "thresholds" not in st.session_state:
        st.session_state.thresholds = {food: 0.25 for food in foods}
    for food in foods:
        st.session_state.thresholds[food] = st.slider()
        f"Prag za: {food}", 0.0, 1.0,
        value=st.session_state.thresholds.get(food, 0.25), step=0.01
