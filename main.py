import streamlit as st
from roboflow import Roboflow
from PIL import Image
import os
import requests
import io

# --- CONFIGURATION ---
API_KEY = os.getenv("ROBOFLOW_API_KEY")
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
        "advice": "✅ Safe for most people if dosage is respected."
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
            prediction = model.predict(temp_filename, confidence=40, overlap=30)
            result_json = prediction.json()
            
            # 2. GET VISUAL: Fetch the image with boxes from the Hosted URL
            project_id, version = FULL_MODEL_ID.split("/")
            viz_url = f"{SERVERLESS_URL}/{project_id}/{version}?api_key={API_KEY}&format=image&labels=on&stroke=2"
            
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

# import streamlit as st
# from roboflow import Roboflow
# from PIL import Image
# import os

# # --- CONFIGURATION ---
# API_KEY = os.getenv("ROBOFLOW_API_KEY")
# FULL_MODEL_ID = "lab-e3lrr/3"  # Your Project ID + Version

# if not API_KEY:
#     raise ValueError("Roboflow API key not found. Set environment variable ROBOFLOW_API_KEY")

# # --- CUSTOM URL SETUP ---
# # We define the custom serverless URL here as requested
# SERVERLESS_URL = "https://serverless.roboflow.com"

# # --- KNOWLEDGE BASE ---
# PILL_INFO = {
#     "dynapharm ibufen": {
#         "name": "Dynapharm Ibufen",
#         "type": "Ibuprofen (NSAID)",
#         "description": "Used to relieve pain, inflammation, and fever.",
#         "timing": "Take every 4-6 hours as needed.",
#         "advice": "⚠️ Take with food or milk to prevent stomach upset. Do not take on an empty stomach."
#     },
#     "t mefenamic acid": {
#         "name": "T Mefenamic Acid",
#         "type": "NSAID",
#         "description": "Used for short-term treatment of mild to moderate pain and menstrual cramps.",
#         "timing": "Usually taken 3 times daily.",
#         "advice": "⚠️ Take with a full glass of water and food. Do not lie down for 10 minutes after taking."
#     },
#     "loramide": {
#         "name": "Loramide (Loperamide)",
#         "type": "Antidiarrheal",
#         "description": "Used to treat sudden diarrhea by slowing down gut movement.",
#         "timing": "Take after each loose stool, up to the daily maximum.",
#         "advice": "Drink plenty of water/electrolytes to stay hydrated."
#     },
#     "deltacarbon": {
#         "name": "Deltacarbon (Activated Charcoal)",
#         "type": "Adsorbent",
#         "description": "Used to treat indigestion, gas, or accidental poisoning.",
#         "timing": "Take as needed for gas/indigestion.",
#         "advice": "⚫ Takes 2 hours apart from other medicines, as it can stop them from working."
#     },
#     "paracetamol": {
#         "name": "Paracetamol",
#         "type": "Analgesic / Antipyretic",
#         "description": "Common painkiller for aches and fever.",
#         "timing": "Take every 4-6 hours. Do not exceed 8 tablets in 24 hours.",
#         "advice": "✅ Can be taken with or without food. Safe for most people if dosage is respected."
#     }
# }

# # --- APP SETUP ---
# st.set_page_config(page_title="PillScout", page_icon="💊")
# st.title("💊 PillScout: AI Medication Assistant")
# st.markdown("Take a photo or upload an image of your pill to get dosage advice.")

# # --- SIDEBAR ---
# st.sidebar.header("About")
# st.sidebar.info("This app uses a custom Computer Vision model trained on Roboflow.")
# st.sidebar.warning("**Disclaimer:** This is an AI prototype. Always consult a doctor.")

# # --- LOAD MODEL ---
# @st.cache_resource
# def load_roboflow_model():
#     # 1. Parse the Project ID and Version from "lab-e3lrr/3"
#     try:
#         project_id = FULL_MODEL_ID.split("/")[0]
#         version_number = int(FULL_MODEL_ID.split("/")[1])
#     except IndexError:
#         st.error("Error: MODEL_ID must be in format 'project/version' (e.g., lab-e3lrr/3)")
#         st.stop()

#     # 2. Initialize Roboflow
#     rf = Roboflow(api_key=API_KEY)
#     project = rf.workspace().project(project_id)
#     model = project.version(version_number).model
    
#     # 3. SET THE CUSTOM SERVERLESS URL
#     # This overrides the default inference endpoint
#     model.api_url = SERVERLESS_URL
    
#     return model

# try:
#     model = load_roboflow_model()
#     st.sidebar.success(f"Model Loaded: {FULL_MODEL_ID}")
#     st.sidebar.text(f"URL: {SERVERLESS_URL}")
# except Exception as e:
#     st.error(f"Error loading model: {e}")
#     st.stop()

# # --- INPUT METHOD ---
# tab1, tab2 = st.tabs(["📷 Camera", "📂 Upload"])
# image_data = None

# with tab1:
#     camera_img = st.camera_input("Take a picture")
#     if camera_img: image_data = camera_img

# with tab2:
#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
#     if uploaded_file: image_data = uploaded_file

# # --- INFERENCE LOGIC ---
# if image_data is not None:
#     image = Image.open(image_data)
#     temp_filename = "temp_pill.jpg"
#     image.save(temp_filename)

#     st.divider()
#     st.subheader("Analysis Results")

#     with st.spinner('Analyzing pill...'):
#         try:
#             # 1. Run inference
#             result_json = model.predict(temp_filename, confidence=40, overlap=30).json()
            
#             # 2. Extract unique pills (Deduplication logic)
#             # This creates a dictionary where the 'class' is the key, keeping only one entry per pill type
#             unique_pills = {p['class']: p for p in result_json['predictions']}.values()
            
#             if not unique_pills:
#                 st.warning("No pills detected. Please try moving closer.")
#             else:
#                 st.image(image, caption="Analysis Complete", use_column_width=True)
                
#                 # --- LOOP THROUGH UNIQUE DETECTED PILLS ---
#                 for i, pill in enumerate(unique_pills):
#                     detected_class = pill['class']
#                     confidence = pill['confidence']
                    
#                     st.divider()
#                     st.markdown(f"### Pill #{i+1}: {detected_class}")

#                     # --- MATCHING LOGIC ---
#                     key_to_search = detected_class.lower()
#                     matched_info = None
                    
#                     # Search for the pill in your PILL_INFO database
#                     if key_to_search in PILL_INFO:
#                         matched_info = PILL_INFO[key_to_search]
#                     else:
#                         for key, info in PILL_INFO.items():
#                             if key in key_to_search or key_to_search in key:
#                                 matched_info = info
#                                 break
                    
#                     # --- DISPLAY RESULTS ---
#                     if matched_info:
#                         st.success(f"Identified as: **{matched_info['name']}** ({confidence:.1%} confidence)")
#                         col1, col2 = st.columns(2)
#                         with col1:
#                             st.markdown(f"**📖 Description:**\n{matched_info['description']}")
#                             st.markdown(f"**⏰ When to take:**\n{matched_info['timing']}")
#                         with col2:
#                             st.info(f"**💡 Suggestion:**\n{matched_info['advice']}")
#                     else:
#                         st.error(f"Detected **{detected_class}**, but this pill is not in our database.")
        
#         except Exception as e:
#             st.error(f"An error occurred: {e}")