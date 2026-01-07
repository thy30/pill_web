import streamlit as st
from roboflow import Roboflow
from PIL import Image
import requests

# --- CONFIGURATION ---
API_KEY = st.secrets["ROBOFLOW_API_KEY"]
FULL_MODEL_ID = "lab-e3lrr/3"  # Project/Version
# Use the standard Hosted API URL for visualization
SERVERLESS_URL = "https://detect.roboflow.com"

if not API_KEY:
    st.error("Missing API Key. Please set the ROBOFLOW_API_KEY environment variable.")
    st.stop()

# --- KNOWLEDGE BASE ---
PILL_INFO = {
    "dynapharm ibufen": {
        "name": "Dynapharm Ibufen",
        "type": "Ibuprofen (NSAID)",
        "description": "Used to relieve pain, inflammation, and fever.",
        "timing": "Take every 4-6 hours as needed.",
        "advice": "⚠️ Take with food or milk to prevent stomach upset."
    },
    "t mefenamic acid": {
        "name": "T Mefenamic Acid",
        "type": "NSAID",
        "description": "Used for short-term treatment of mild to moderate pain.",
        "timing": "Usually taken 3 times daily.",
        "advice": "⚠️ Take with a full glass of water and food."
    },
    "loramide": {
        "name": "Loramide (Loperamide)",
        "type": "Antidiarrheal",
        "description": "Used to treat sudden diarrhea.",
        "timing": "Take after each loose stool.",
        "advice": "Drink plenty of water/electrolytes."
    },
    "deltacarbon": {
        "name": "Deltacarbon (Activated Charcoal)",
        "type": "Adsorbent",
        "description": "Used for indigestion or gas.",
        "timing": "Take as needed.",
        "advice": "⚫ Take 2 hours apart from other medicines."
    },
    "paracetamol": {
        "name": "Paracetamol",
        "type": "Analgesic / Antipyretic",
        "description": "Common painkiller for aches and fever.",
        "timing": "Take every 4-6 hours.",
        "advice": "🔴 Do not take more than four doses in 24 hours."
    }
}

# --- APP SETUP ---
st.set_page_config(page_title="PillScout", page_icon="💊")
st.title("💊 PillScout: AI Medication Assistant")

# --- LOAD MODEL ---
@st.cache_resource
def load_roboflow_model():
    project_id = FULL_MODEL_ID.split("/")[0]
    version_number = int(FULL_MODEL_ID.split("/")[1])
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace().project(project_id)
    model = project.version(version_number).model
    model.api_url = SERVERLESS_URL # Set the detection URL
    return model

model = load_roboflow_model()

# --- INPUT METHOD ---
tab1, tab2 = st.tabs(["📷 Camera", "📂 Upload"])
image_data = None

with tab1:
    camera_img = st.camera_input("Take a picture")
    if camera_img: image_data = camera_img
with tab2:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file: image_data = uploaded_file

# --- INFERENCE & DISPLAY LOGIC ---
if image_data is not None:
    # Save temporary file for the API
    temp_filename = "temp_pill.jpg"
    img = Image.open(image_data)
    img.save(temp_filename)

    st.divider()
    
    with st.spinner('Scanning for pills...'):
        try:
            # 1. GET DATA: Run prediction to get JSON coordinates
            prediction = model.predict(temp_filename, confidence=70, overlap=30)
            result_json = prediction.json()
            
            # 2. GET VISUAL: Fetch the image with boxes from the Hosted URL
            project_id, version = FULL_MODEL_ID.split("/")
            viz_url = f"{SERVERLESS_URL}/{project_id}/{version}?api_key={API_KEY}&format=image&labels=on&stroke=2&confidence=70"
            
            # Send the image to Roboflow to get the 'annotated' version back
            with open(temp_filename, "rb") as f:
                response = requests.post(viz_url, files={"file": f})

            if response.status_code == 200:
                # Show the image with bounding boxes
                st.image(response.content, caption="Detections Found", use_column_width=True)
            else:
                st.warning("Could not load annotated image, showing original instead.")
                st.image(img, use_column_width=True)

            # 3. PROCESS RESULTS (Deduplication)
            unique_pills = {p['class']: p for p in result_json['predictions']}.values()

            if not unique_pills:
                st.warning("No pills detected. Please ensure the lighting is good.")
            else:
                st.subheader("💊 Medication Details")
                for i, pill in enumerate(unique_pills):
                    detected_class = pill['class']
                    confidence = pill['confidence']
                    
                    # Matching Logic
                    key_to_search = detected_class.lower()
                    matched_info = PILL_INFO.get(key_to_search)

                    # Fuzzy match if direct fails
                    if not matched_info:
                        for key, info in PILL_INFO.items():
                            if key in key_to_search or key_to_search in key:
                                matched_info = info
                                break

                    st.divider()
                    if matched_info:
                        st.success(f"**{matched_info['name']}** ({confidence:.1%} confidence)")
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown(f"**📖 Info:** {matched_info['description']}")
                            st.markdown(f"**⏰ Use:** {matched_info['timing']}")
                        with c2:
                            st.info(f"**💡 Note:** {matched_info['advice']}")
                    else:
                        st.error(f"Unknown Pill: **{detected_class}**")

        except Exception as e:
            st.error(f"Error: {e}")