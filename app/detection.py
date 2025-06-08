import cv2
import time
import streamlit as st
from collections import defaultdict

def detect_from_camera(model, camera_url, thresholds, stframe, user_id, list_db):
    # Očisti prethodni sadržaj stframe-a i prikaži poruku o pokušaju spajanja
    stframe.empty()
    stframe.info(f"Pokušavam se spojiti na kameru: {camera_url}...")

    # Provjeri je li URL ispravnog formata
    if not (camera_url.startswith('http://') or camera_url.startswith('https://')):
        stframe.error("URL kamere mora početi s 'http://' ili 'https://'.")
        return

    # Pokušaj otvaranja kamere s rukovanjem iznimkama
    try:
        cap = cv2.VideoCapture(camera_url)
    except Exception as e:
        stframe.error(f"Greška pri otvaranju video stream-a: {e}. Provjerite URL i dostupnost kamere.")
        return

    # Provjeri je li kamera uspješno otvorena
    if not cap.isOpened():
        stframe.error(f"Ne mogu otvoriti video stream s URL-a '{camera_url}'. Molimo provjerite: \n"
                      f"1. Je li URL ispravan i dostupan s Streamlit Cloud poslužitelja (javni IP ili port forwarding)?\n"
                      f"2. Je li IP Webcam aplikacija na vašem telefonu aktivna i stream-a video?\n"
                      f"3. Postoji li firewall koji blokira promet?\n"
                      f"Trenutni status veze: {cap.isOpened()}")
        return

    # Inicijaliziraj ili resetiraj detektirane stavke
    if "detected_items" not in st.session_state or not st.session_state.get("run_detection", False):
        st.session_state.detected_items = defaultdict(lambda: {"count": 0, "confidence": 0.0}) 

    presence_start = {}  # Vrijeme kada je objekt prvi put detektiran u TRENUTNOM NEPREKIDNOM pojavljivanju
    last_seen = {}       # Vrijeme kada je objekt zadnji put viđen u kadru
    MIN_VISIBLE_DURATION = 3  # sekundi, koliko dugo mora biti vidljiv da bi se "detektirao"
    DISAPPEAR_TOLERANCE = 1.5 # sekundi, koliko dugo može nestati pa da se i dalje tretira kao ista instanca
    RESET_THRESHOLD = 2.5 # sekundi, koliko dugo mora nestati da se brojač resetira za NOVO brojanje

    # status za praćenje brojača i prisutnosti pojedinog objekta
    item_status = defaultdict(lambda: {"is_present": False, "counted_in_current_presence": False, "last_disappeared": 0})

    stframe.success("Video stream uspješno otvoren. Počinjem s detekcijom...")

    try:
        # Glavna petlja za obradu video streama
        while cap.isOpened() and st.session_state.get("run_detection", False):
            ret, frame = cap.read()
            if not ret:
                stframe.warning("Nema frame-a iz kamere. Provjerite vezu kamere ili URL.")
                time.sleep(0.1) # Mala pauza prije ponovnog pokušaja
                continue # Pokušaj ponovno dohvatiti frame

            results = model.predict(frame, imgsz=640, conf=0.25, verbose=False)[0]
            current_time = time.time()
            current_frame_detections = set() # Što je detektirano u ovom konkretnom frameu

            for box in results.boxes:
                conf = box.conf.item()
                class_name = model.names[int(box.cls)].lower()

                if conf > thresholds.get(class_name, 0.25): 
                    current_frame_detections.add(class_name)

                    # Ažuriraj kada je zadnji put viđen
                    last_seen[class_name] = current_time

                    # Ako je nova pojava (ili se tek pojavljuje nakon što je bila odsutna neko vrijeme)
                    if class_name not in presence_start:
                        presence_start[class_name] = current_time

                    # Provera za "potvrđenu prisutnost"
                    if (current_time - presence_start[class_name] >= MIN_VISIBLE_DURATION):
                        item_status[class_name]["is_present"] = True
                        # Ažuriraj samopouzdanje u session_state
                        st.session_state.detected_items[class_name]["confidence"] = max(st.session_state.detected_items[class_name]["confidence"], conf)

                        # Ako je namirnica bila odsutna dovoljno dugo da se brojač resetira, ili ako je ovo prva detekcija
                        if (item_status[class_name]["counted_in_current_presence"] == False and \
                            (current_time - item_status[class_name]["last_disappeared"] > RESET_THRESHOLD or item_status[class_name]["last_disappeared"] == 0)):

                            st.session_state.detected_items[class_name]["count"] += 1
                            item_status[class_name]["counted_in_current_presence"] = True # Označi da je brojač povećan za ovu prisutnost
                            item_status[class_name]["last_disappeared"] = 0 # Resetiraj za trenutnu prisutnost
                            print(f"DEBUG: Broj za {class_name} povećan na {st.session_state.detected_items[class_name]['count']}")


            #===== Logika za nestajanje namirnica
            # Prođi kroz sve namirnice koje su prije bile viđene
            for name in list(item_status.keys()):
                if name not in current_frame_detections:
                    # Namirnica nije u trenutnom frameu
                    if item_status[name]["is_present"]:
                        # Ako je namirnica bila "prisutna" (potvrđena), ali je sada nestala
                        if current_time - last_seen.get(name, 0) > DISAPPEAR_TOLERANCE:
                            # Dovoljno dugo je nestala da je više ne smatramo istom instancom unutar trenutnog brojanja
                            item_status[name]["is_present"] = False
                            item_status[name]["counted_in_current_presence"] = False # Resetiraj da se može ponovno brojati kada se pojavi
                            item_status[name]["last_disappeared"] = current_time # Zabilježi vrijeme nestanka
                            if name in presence_start: # Potrebno je osigurati da ključ postoji prije brisanja
                                del presence_start[name] # Resetiraj timer za prisutnost
                    else: # Ako nije bila "is_present" (tj. već je "nestala" prije), provjeri je li stvarno otišla zauvijek
                        if current_time - item_status[name]["last_disappeared"] > RESET_THRESHOLD * 1.5: # Neko veće vrijeme za brisanje iz cachea
                            # Brišemo iz item_status samo ako je dovoljno dugo nema
                            if name in presence_start:
                                del presence_start[name]
                            if name in last_seen:
                                del last_seen[name]
                            # Samo brisanje iz item_status rječnika bi trebalo biti dovoljno da se resetira
                            del item_status[name]

            # Prikaz frame-a u Streamlit-u
            annotated = results.plot()
            stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
            time.sleep(0.01) # Mala pauza kako bi UI bio odzivan i ne preopteretio CPU

    except Exception as e:
        # Uhvati sve neočekivane greške tijekom detekcije i ispiši ih
        stframe.error(f"Neočekivana greška tijekom detekcije: {e}")
    finally:
        # Oslobodi resurse kamere kada se petlja završi (ili se dogodi greška)
        cap.release()
        # Očisti video prozor u Streamlitu nakon zaustavljanja
        stframe.empty() 
        st.info("Detekcija zaustavljena ili prekinuta.")
        print("FINAL DETECTED (from session state):", st.session_state.detected_items)
