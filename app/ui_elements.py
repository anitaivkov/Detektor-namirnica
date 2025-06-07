import streamlit as st
from collections import defaultdict


def init_session_state():
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'detected_items' not in st.session_state:
        st.session_state.detected_items = defaultdict(lambda: {"count": 0, "confidence": 0})
    if 'detecting' not in st.session_state:
        st.session_state.detecting = False


def show_detected_items():
    if st.session_state.detected_items:
        st.markdown("#### 🧾 Detektirane namirnice:")
        for item, data in st.session_state.detected_items.items():
            st.write(f"- {item} x{data['count']} (najveća točnost: {data['confidence']:.2f})")


def reset_detected_items():
    st.session_state.detected_items = defaultdict(lambda: {"count": 0, "confidence": 0})


def render_login(user_db):
    username = st.text_input("Korisničko ime:")
    if st.button("Prijavi se"):
        if username:
            user_id = user_db.get_or_create_user(username)
            st.session_state.user_id = user_id
            st.success(f"Korisnik '{username}' prijavljen. ID: {user_id}")
        else:
            st.error("Unesi korisničko ime.")
    return username


def render_camera_url():
    return st.text_input("URL kamere \n(npr. http://192.168.1.133:8080/video)")

def show_threshold_sliders(food_list):
    st.subheader("🎚️ Pragovi detekcije po namirnicama")
    thresholds = {}
    for food in food_list:
        thresholds[food] = st.slider(f"{food}", 0.0, 1.0, 0.25)
    return thresholds

