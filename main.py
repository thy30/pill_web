import streamlit as st
from roboflow import Roboflow
from PIL import Image
import io

# --- CONFIGURATION ---
API_KEY = "qPnO1IYxFW6y0ZnMw3U9"
FULL_MODEL_ID = "lab-e3lrr/3"  # Your Project ID + Version

# --- CUSTOM URL SETUP ---
# We define the custom serverless URL here as requested
SERVERLESS_URL = "https://serverless.roboflow.com"

# --- KNOWLEDGE BASE ---
PILL_INFO = {
    "dynapharm ibufen": {
        "name": "Dynapharm Ibufen",
        "type": "Ibuprofen (NSAID)",
        "description": "Used to relieve pain, inflammation, and fever.",
        "timing": "Take every 4-6 hours as needed.",
        "advice": "⚠️ Take with food or milk to prevent stomach upset. Do not take on an empty stomach."
    },
    "t mefenamic acid": {
        "name": "T Mefenamic Acid",
        "type": "NSAID",
        "description": "Used for short-term treatment of mild to moderate pain and menstrual cramps.",
        "timing": "Usually taken 3 times daily.",
        "advice": "⚠️ Take with a full glass of water and food. Do not lie down for 10 minutes after taking."
    },
    "loramide": {
        "name": "Loramide (Loperamide)",
        "type": "Antidiarrheal",
        "description": "Used to treat sudden diarrhea by slowing down gut movement.",
        "timing": "Take after each loose stool, up to the daily maximum.",
        "advice": "Drink plenty of water/electrolytes to stay hydrated."
    },
    "deltacarbon": {
        "name": "Deltacarbon (Activated Charcoal)",
        "type": "Adsorbent",
        "description": "Used to treat indigestion, gas, or accidental poisoning.",
        "timing": "Take as needed for gas/indigestion.",
        "advice": "⚫ Takes 2 hours apart from other medicines, as it can stop them from working."
    },
    "paracetamol": {
        "name": "Paracetamol",
        "type": "Analgesic / Antipyretic",
        "description": "Common painkiller for aches and fever.",
        "timing": "Take every 4-6 hours. Do not exceed 8 tablets in 24 hours.",
        "advice": "✅ Can be taken with or without food. Safe for most people if dosage is respected."
    }
}

# --- APP SETUP ---
st.set_page_config(page_title="PillScout", page_icon="💊")
st.title("💊 PillScout: AI Medication Assistant")
st.markdown("Take a photo or upload an image of your pill to get dosage advice.")

# --- SIDEBAR ---
st.sidebar.header("About")
st.sidebar.info("This app uses a custom Computer Vision model trained on Roboflow.")
st.sidebar.warning("**Disclaimer:** This is an AI prototype. Always consult a doctor.")

# --- LOAD MODEL ---
@st.cache_resource
def load_roboflow_model():
    # 1. Parse the Project ID and Version from "lab-e3lrr/3"
    try:
        project_id = FULL_MODEL_ID.split("/")[0]
        version_number = int(FULL_MODEL_ID.split("/")[1])
    except IndexError:
        st.error("Error: MODEL_ID must be in format 'project/version' (e.g., lab-e3lrr/3)")
        st.stop()

    # 2. Initialize Roboflow
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace().project(project_id)
    model = project.version(version_number).model
    
    # 3. SET THE CUSTOM SERVERLESS URL
    # This overrides the default inference endpoint
    model.api_url = SERVERLESS_URL
    
    return model

try:
    model = load_roboflow_model()
    st.sidebar.success(f"Model Loaded: {FULL_MODEL_ID}")
    st.sidebar.text(f"URL: {SERVERLESS_URL}")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- INPUT METHOD ---
tab1, tab2 = st.tabs(["📷 Camera", "📂 Upload"])
image_data = None

with tab1:
    camera_img = st.camera_input("Take a picture")
    if camera_img: image_data = camera_img

with tab2:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file: image_data = uploaded_file

# --- INFERENCE LOGIC ---
if image_data is not None:
    image = Image.open(image_data)
    temp_filename = "temp_pill.jpg"
    image.save(temp_filename)

    st.divider()
    st.subheader("Analysis Results")

    with st.spinner('Analyzing pill...'):
        try:
            # Run inference
            prediction = model.predict(temp_filename, confidence=40, overlap=30).json()
            
            if not prediction['predictions']:
                st.warning("No pills detected. Please try moving closer.")
            else:
                top_prediction = prediction['predictions'][0]
                detected_class = top_prediction['class']
                confidence = top_prediction['confidence']

                st.image(image, caption=f"Detected: {detected_class}", use_column_width=True)
                
                # --- MATCHING LOGIC ---
                key_to_search = detected_class.lower()
                matched_info = None
                
                if key_to_search in PILL_INFO:
                    matched_info = PILL_INFO[key_to_search]
                else:
                    for key, info in PILL_INFO.items():
                        if key in key_to_search or key_to_search in key:
                            matched_info = info
                            break
                
                if matched_info:
                    st.success(f"Detected: **{matched_info['name']}** ({confidence:.1%} confidence)")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**📖 Description:**\n{matched_info['description']}")
                        st.markdown(f"**⏰ When to take:**\n{matched_info['timing']}")
                    with col2:
                        st.info(f"**💡 Suggestion:**\n{matched_info['advice']}")
                else:
                    st.error(f"Detected **{detected_class}**, but details are missing from the database.")
                    st.json(prediction)

        except Exception as e:
            st.error(f"An error occurred: {e}")