import streamlit as st
from io import BytesIO
from PIL import Image
import requests
import io
import base64
import os
from ultralytics import YOLO
import torch
import json
import time
import shutil

# Direct download link
url = "https://drive.google.com/uc?export=download&id=135InVrHL21pLTU9vbLIt9RgIN8cRpZco"
response = requests.get(url)
image = Image.open(BytesIO(response.content))

# Set Streamlit page configuration
st.set_page_config(
    page_title="Crack Detector AI",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="auto",
)

# Sidebar navigation
page = st.sidebar.selectbox("Choose Page", ["Crack Detection & Repair Tips", "About"])
st.sidebar.image(image, caption="Built with ❤️ by Our Team", use_container_width=True)

if page == "Crack Detection & Repair Tips":

    # Get Gemini API key and URL from secrets
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
    GEMINI_API_URL = st.secrets.get("GEMINI_API_URL")

    if not GEMINI_API_KEY or not GEMINI_API_URL:
        st.warning("Gemini API Key or URL not found in Streamlit secrets. Please add them to your .streamlit/secrets.toml file.")
        st.stop()

    # Load YOLOv8 model
    @st.cache_resource
    def load_model():
        return YOLO("models/best.pt")

    model = load_model()

    # Crack detection
    def detect_crack(image_path):
        results = model.predict(source=image_path, save=True, conf=0.5)
        boxes = results[0].boxes if results and results[0].boxes else None
        crack_detected = False
        if boxes is not None:
            detected_classes = results[0].names
            class_ids = boxes.cls.tolist()
            crack_detected = any(detected_classes[int(c)] == 'crack' for c in class_ids)
        return crack_detected, results

    # Ask Gemini API for repair advice
    @st.cache_data(show_spinner=False)
    def ask_gemini(prompt_text):
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 2048
            }
        }

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {GEMINI_API_KEY}"
        }

        retries = 0
        max_retries = 5
        base_delay = 1

        while retries < max_retries:
            try:
                response = requests.post(
                    GEMINI_API_URL,
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                result = response.json()

                if (
                    result.get("candidates") and
                    result["candidates"][0].get("content") and
                    result["candidates"][0]["content"].get("parts") and
                    result["candidates"][0]["content"]["parts"][0].get("text")
                ):
                    return result["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    finish_reason = result["candidates"][0].get("finishReason", "Unknown")
                    st.error(f"Unexpected response. Finish reason: {finish_reason}")
                    return "Sorry, I couldn't generate a response."

            except requests.exceptions.RequestException as e:
                retries += 1
                st.error(f"Gemini API error: {e}")
                if retries < max_retries:
                    delay = base_delay * (2 ** (retries - 1))
                    st.warning(f"Retrying in {delay} seconds... ({retries}/{max_retries})")
                    time.sleep(delay)
                else:
                    st.error("Max retries reached. Could not connect to Gemini API.")
                    return "Unable to provide advice at this time."

            except json.JSONDecodeError:
                st.error(f"Failed to parse JSON response from Gemini API. Raw response:\n{response.text}")
                return "Sorry, the response from the AI could not be processed."

        return "Unable to provide advice after multiple attempts."

    # UI
    st.title("🧱 Crack Detection & Repair Advisor") 
    st.markdown("Upload an image of a road or wall to detect cracks and get repair/maintenance advice.")

    uploaded_file = st.file_uploader("📷 Upload an image (road or wall)", type=["jpg", "jpeg", "png"])
    col1, col2 = st.columns([1, 1])

    if uploaded_file:
        temp_image_path = "temp_uploaded_image.jpg"
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with col1:
            st.image(temp_image_path, caption="Uploaded Image", width=300)

        if st.button("🔍 Detect Crack and Get Advice"):
            with st.spinner("Analyzing image and generating advice..."):
                crack_found, results = detect_crack(temp_image_path)

                result_img_path = None
                if results and hasattr(results[0], 'save_dir'):
                    saved_dir = results[0].save_dir
                    for file in os.listdir(saved_dir):
                        if file.endswith(('.jpg', '.png')):
                            result_img_path = os.path.join(saved_dir, file)
                            break

                if crack_found:
                    st.warning("⚠️ Crack detected!")
                    if result_img_path and os.path.exists(result_img_path):
                        with col2:
                            st.image(result_img_path, caption="Detection Result", width=300)
                    prompt = (
                        "I have detected a crack in a concrete surface. "
                        "Please provide step-by-step repair instructions, materials needed, and estimated cost. "
                        "Be concise and actionable."
                    )
                    response = ask_gemini(prompt)
                    st.success("🛠 Repair Advice from AI Advisor:")
                    st.markdown(response)
                else:
                    st.success("✅ No crack detected.")
                    if result_img_path and os.path.exists(result_img_path):
                        with col2:
                            st.image(result_img_path, caption="Detection Result", width=300)
                    prompt = (
                        "No crack was detected in the structure. "
                        "Please provide maintenance advice to keep it intact and prevent future damage. "
                        "Be concise and actionable."
                    )
                    response = ask_gemini(prompt)
                    st.info("🧼 Maintenance Advice from AI Advisor:")
                    st.markdown(response)

            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            if results and hasattr(results[0], 'save_dir'):
                try:
                    shutil.rmtree(results[0].save_dir)
                except Exception as e:
                    st.error(f"Error cleaning up YOLO run directory: {e}")

    st.markdown("""---""")
    st.markdown("© 2025 Crack Detector | Built with ❤️ by Asmaa Elkashef | Beshoy Osama | Omar Mohamed | Omnia Eldeeb")

elif page == "About":
    st.title("📄 About This App")
    st.markdown("""
    Welcome to **Crack Detection & Repair Advisor** — an AI-powered tool designed to help identify and suggest solutions for surface cracks in structures.

    ### 🧠 Powered By:
    - **YOLOv8** for real-time crack detection using deep learning
    - **Gemini AI** for generating step-by-step repair and maintenance advice
    - **Streamlit** for building this interactive web interface

    ### 🔍 What It Does:
    - Detects visible cracks in uploaded concrete surface images
    - Provides repair advice (if a crack is found)
    - Offers preventive maintenance suggestions (if no crack is found)

    ### 🛠 Use Cases:
    - Civil engineering inspection
    - Home maintenance
    - Structural monitoring and documentation

    ### 👨‍💻 Our Team:
    - **Asmaa Elkashef**  
    - **Beshoy Osama**
    - **Omar Mohamed**   
    - **Omnia Eldeeb**   
    """)
    st.markdown("""---""")
    st.markdown("© 2025 Crack Detector | Built with ❤️ by Asmaa Elkashef | Beshoy Osama | Omar Mohamed | Omnia Eldeeb")
