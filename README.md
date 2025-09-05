import streamlit as st
from PIL import Image
import os
from dotenv import load_dotenv
from src.helper import voice_input, llm_model_object, text_to_speech, store_pdf_in_db

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found in environment variables. Please set it and try again.")
    st.stop()

# Set custom CSS to match Car Damage Identifier
def set_custom_style():
    st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
        font-family: 'Arial', sans-serif;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1 {
        color: #1a73e8;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 10px;
    }
    h3 {
        color: #333;
        font-size: 1.5em;
    }
    .stButton>button {
        background-color: #1a73e8;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #1557b0;
    }
    .stTextInput>div>div>input, .stTextArea textarea {
        background-color: white;
        color: black;
        border-radius: 8px;
        border: 1px solid #ccc;
        padding: 10px;
    }
    .stFileUploader {
        background-color: white;
        border-radius: 8px;
        border: 1px solid #ccc;
        padding: 10px;
    }
    .result-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .download-link {
        color: #1a73e8;
        font-weight: bold;
        text-decoration: none;
    }
    .download-link:hover {
        text-decoration: underline;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    set_custom_style()

    # Load logo (use relative path or uploaded file for portability)
    try:
        logo = Image.open("digit-logo.png")  # Adjust path as needed
        st.image(logo, width=100)
    except FileNotFoundError:
        st.warning("Logo file (digit-logo.png) not found. Please ensure it exists in the project directory.")

    st.markdown("<h1 style='color:#1a73e8;'>Multilingual AI Assistant</h1>", unsafe_allow_html=True)
    st.write("Upload Your PDF File or Provide Input Below")

    # File uploader for PDF
    uploaded_file = st.file_uploader("Choose File", type="pdf")

    if uploaded_file:
        with open("uploaded_file.pdf", "wb") as f:
            f.write(uploaded_file.read())
        try:
            store_pdf_in_db("uploaded_file.pdf")
            st.success("PDF successfully uploaded and stored!")
        except Exception as e:
            st.error(f"Error storing PDF: {str(e)}")

    # Input options
    st.write("Choose how you'd like to provide your input:")
    user_text_input = st.text_input("Enter your query here:", placeholder="Type your question...")
    voice_button = st.button("Ask Me Anything (Voice)", key="voice_btn")

    # Test case for Hindi input (uncomment to test)
    # user_text = "हिंदी टाइपिंग के बेहतरीन ऐप्स"  # Hindi text for testing

    user_text = user_text_input if user_text_input else None

    if voice_button and not user_text_input:
        with st.spinner("Listening..."):
            try:
                user_text = voice_input()
                if user_text:
                    st.write(f"Voice Input: {user_text}")
                else:
                    st.error("No audio detected. Please try again.")
            except Exception as e:
                st.error(f"Voice input error: {str(e)}")

    if user_text:
        with st.spinner("Processing your query..."):
            try:
                hindi_response, english_response = llm_model_object(user_text)
                text_to_speech(hindi_response, english_response)

                # Display responses in styled containers
                st.subheader("Hindi Response:")
                st.markdown('<div class="result-container">', unsafe_allow_html=True)
                st.text_area("Hindi:", value=hindi_response, height=150, key="hindi_response")
                st.markdown('</div>', unsafe_allow_html=True)

                st.subheader("English Response:")
                st.markdown('<div class="result-container">', unsafe_allow_html=True)
                st.text_area("English:", value=english_response, height=150, key="english_response")
                st.markdown('</div>', unsafe_allow_html=True)

                # Audio playback and download
                try:
                    with open("response_hindi.mp3", "rb") as f_hindi, open("response_english.mp3", "rb") as f_english:
                        hindi_audio = f_hindi.read()
                        english_audio = f_english.read()

                    col1, col2 = st.columns(2)
                    with col1:
                        st.audio(hindi_audio, format="audio/mp3")
                        st.download_button("Download Hindi Speech", hindi_audio, "response_hindi.mp3", "audio/mp3", key="download_hindi")
                    with col2:
                        st.audio(english_audio, format="audio/mp3")
                        st.download_button("Download English Speech", english_audio, "response_english.mp3", "audio/mp3", key="download_english")
                except FileNotFoundError:
                    st.error("Audio files not found. Ensure text-to-speech processing completed successfully.")
                except Exception as e:
                    st.error(f"Error loading audio: {str(e)}")

            except Exception as e:
                st.error(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()

