import streamlit as st
from PIL import Image
import requests
import io
import base64
import os
from ultralytics import YOLO
import torch
import json # Import json for handling API payloads
import time # Import time for exponential backoff
import shutil # Import shutil for directory removal
image = r"C:\Users\MF\Downloads\Gemini_Generated_Image_c1cex1c1cex1c1ce.png"
# Sidebar navigation
st.set_page_config(
        page_title="Crack Detector AI",         # 🔤 Title of the browser tab
        page_icon="🛠️",                         # 🔧 Icon (can be emoji or image path)
        layout="wide",                      # Optional: "centered" or "wide"
        initial_sidebar_state="auto",           # Optional: "auto", "expanded", "collapsed"
    )
page = st.sidebar.selectbox("Choose Page", ["Crack Detection & Repair Tips", "About"])

# Simulated page navigation
if page == "Crack Detection & Repair Tips":


    # --- Configuration for Gemini API ---
    # Attempt to get API key from Streamlit secrets.
    # In a deployed Streamlit app, you would configure this in your .streamlit/secrets.toml file:
    # gemini_api_key = "YOUR_GEMINI_API_KEY_HERE"
    # If running locally, ensure you have a .streamlit/secrets.toml file with your key.
    GEMINI_API_KEY = st.secrets.get("gemini_api_key") 

    # Check if the API key is available
    GEMINI_API_KEY = st.secrets.get("gemini_api_key") 

# Check if the API key is available
    if not GEMINI_API_KEY:
            st.warning("Gemini API Key not found in Streamlit secrets. Please add it to your .streamlit/secrets.toml file or configure it in your deployment environment.")
            st.stop() # Stop the app execution if the key is missing

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"

    # --- Load YOLOv8 model ---
    @st.cache_resource
    def load_model():
        # Ensure 'best.pt' is in your project directory or provide a full path
        return YOLO("models/best.pt")
    model = load_model()

    # --- Crack detection function ---
    def detect_crack(image_path):
        # YOLOv8 expects the image path or a PIL Image object
        results = model.predict(source=image_path, save=True, conf=0.5) # Added confidence threshold for better results
        
        # Ensure results[0] exists and has boxes
        boxes = results[0].boxes if results and results[0].boxes else None
        
        crack_detected = False
        if boxes is not None:
            detected_classes = results[0].names
            class_ids = boxes.cls.tolist()
            crack_detected = any(detected_classes[int(c)] == 'crack' for c in class_ids)
        
        return crack_detected, results


    # --- Gemini API interaction function ---
    @st.cache_data(show_spinner=False)
    def ask_gemini(prompt_text):
        """
        Calls the Gemini API to generate a response based on the given prompt.
        Uses exponential backoff on failure and caches responses.
        """

        # Load credentials from Streamlit secrets
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
        GEMINI_API_URL = st.secrets["GEMINI_API_URL"]

        # Construct payload
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 2048
            }
        }

        headers = {
            'Content-Type': 'application/json'
        }

        retries = 0
        max_retries = 5
        base_delay = 1  # seconds

        while retries < max_retries:
            try:
                response = requests.post(
                    f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
                    headers=headers,
                    data=json.dumps(payload)
                )
                response.raise_for_status()
                result = response.json()

                # Extract and return generated text
                if (
                    result.get("candidates") and
                    result["candidates"][0].get("content") and
                    result["candidates"][0]["content"].get("parts") and
                    result["candidates"][0]["content"]["parts"][0].get("text")
                ):
                    return result["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    finish_reason = result["candidates"][0].get("finishReason", "Unknown")
                    st.error(f"Unexpected response or empty result. Finish reason: {finish_reason}")
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
                    return "Unable to provide advice at this time due to a connection issue."

            except json.JSONDecodeError:
                st.error(f"Failed to parse JSON response from Gemini API. Raw response:\n{response.text}")
                return "Sorry, the response from the AI could not be processed."

        return "Unable to provide advice after multiple attempts."


    # --- Streamlit UI ---
    #st.image(r"C:\Users\MF\Downloads\Gemini_Generated_Image_c1cex1c1cex1c1ce.png", width=100)  # Add your image file here

    st.title("🧱 Crack Detection & Repair Advisor") 
    st.markdown("Upload an image of a road or wall to detect cracks and get repair/maintenance advice.")

    uploaded_file = st.file_uploader("📷 Upload an image (road or wall)", type=["jpg", "jpeg", "png"])
    col1, col2= st.columns([1, 1])  # 3 columns, center one wider
    if uploaded_file:
        # Save the uploaded file temporarily
        temp_image_path = "temp_uploaded_image.jpg"
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        with col1:

            st.image(temp_image_path, caption="Uploaded Image",  width=300) # Changed to use_container_width

        if st.button("🔍 Detect Crack and Get Advice"):
            with st.spinner("Analyzing image and generating advice..."):
                crack_found, results = detect_crack(temp_image_path)

                # --- Get the path to the actual result image from YOLOv8 output ---
                result_img_path = None
                if results and hasattr(results[0], 'save_dir'):
                    saved_dir = results[0].save_dir  # Path like runs/detect/predictX/
                    for file in os.listdir(saved_dir):
                        if file.endswith(('.jpg', '.png')):
                            result_img_path = os.path.join(saved_dir, file)
                            break

                if crack_found:
                    st.warning("⚠️ Crack detected!")
                    if result_img_path and os.path.exists(result_img_path):
                        with col2 :
                            st.image(result_img_path, caption="Detection Result", width=300)
                    else:
                        st.error("Could not find the detection result image.")

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
                        with col2 :
                            st.image(result_img_path, caption="Detection Result", width=300)
                    else:
                        st.info("No detection image generated as no crack was found.")

                    prompt = (
                        "No crack was detected in the structure. "
                        "Please provide maintenance advice to keep it intact and prevent future damage. "
                        "Be concise and actionable."
                    )
                    response = ask_gemini(prompt)
                    st.info("🧼 Maintenance Advice from AI Advisor:")
                    st.markdown(response)

            # Clean up the temporary uploaded file
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)

            # Clean up YOLO output folder
            if results and hasattr(results[0], 'save_dir'):
                try:
                    shutil.rmtree(results[0].save_dir)
                except Exception as e:
                    st.error(f"Error cleaning up YOLO run directory: {e}")

    # Add footer
    st.markdown("""---""")
    st.markdown("© 2025 Crack Detector | Built with ❤️ by Asmaa Elkashef | Beshoy Osama | Omar Mohamed | Omnia Eldeeb  using YOLO and Gemini API.")

elif  page == "About":
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

    # Add footer
    st.markdown("""---""")
    st.markdown("© 2025 Crack Detector | Built with ❤️ by Asmaa Elkashef | Beshoy Osama | Omar Mohamed | Omnia Eldeeb  using YOLO and Gemini API.")

