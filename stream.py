import joblib
import streamlit as st

# Load the trained model and vectorizer
model_file_name = "finalized_model2.sav"
vectorizer_file_name = "vectorizer.sav"
loaded_model = joblib.load(model_file_name)
loaded_vectorizer = joblib.load(vectorizer_file_name)

# Streamlit app title
st.title("Stress Analysis")

# Markdown instruction
st.markdown("Share with us what are you feeling these days. We are here for you")

# Removing the Streamlit banner at the bottom
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Text input for user
user_text = st.text_input("Text", key="user_text")

# Access the value and handle prediction
if user_text:
    try:
        # Preprocess the input text
        processed_text = loaded_vectorizer.transform([user_text])  # Vectorize the input text

        # Predict the result
        result = loaded_model.predict(processed_text)  # Make prediction with the vectorized text

        # Display the result
        st.write(f"Prediction: {result[0]}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
