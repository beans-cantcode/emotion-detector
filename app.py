import streamlit as st
from deepface import DeepFace
from PIL import Image
import numpy as np
import pandas as pd
from datetime import datetime
import os

st.title("Emotion Detector")
st.write("Take a photo and I'll detect your emotion and track it over time")

# Load existing data if it exists
if os.path.exists("emotion_history.csv"):
    history = pd.read_csv("emotion_history.csv")
else:
    history = pd.DataFrame(columns=["time", "emotion", "happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"])

# Photo input
photo = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

if photo is None:
    photo = st.camera_input("Or take a photo")

if photo is not None:
    image = Image.open(photo)
    st.image(image, caption="Your photo")
    
    img_array = np.array(image)
    
    result = DeepFace.analyze(img_array, actions=['emotion'], enforce_detection=False)    
    emotion = result[0]['dominant_emotion']
    scores = result[0]['emotion']
    
    st.success(f"Dominant emotion: {emotion}")
    st.bar_chart(scores)

    # Save to history
    new_row = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "emotion": emotion,
        "happy": round(scores['happy'], 1),
        "sad": round(scores['sad'], 1),
        "angry": round(scores['angry'], 1),
        "fear": round(scores['fear'], 1),
        "surprise": round(scores['surprise'], 1),
        "disgust": round(scores['disgust'], 1),
        "neutral": round(scores['neutral'], 1)
    }

    history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)
    history.to_csv("emotion_history.csv", index=False)
    st.success("Saved to history!")

# Show history
if len(history) > 0:
    st.subheader("Your Emotion History")
    st.dataframe(history)
    
    st.subheader("Emotion Trends Over Time")
    st.line_chart(history.set_index("time")[["happy", "sad", "angry", "neutral"]])