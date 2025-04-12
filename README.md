# 🎵 Facial Emotion and Hand Gesture Based Music Recommendation System

## 🔍 Overview

Our project aims to create an intelligent music recommendation system that detects the user's facial emotions and hand gestures in real-time to suggest suitable music tracks. By combining computer vision, machine learning, and multimedia interaction, this system offers a personalized and immersive user experience.

## 💡 Features

- 🎭 **Facial Emotion Detection**  
  Uses a webcam to analyze the user's facial expressions and detect emotions like Happy, Sad, Angry, Neutral, etc.

- ✋ **Hand Gesture Recognition**  
  Recognizes predefined hand gestures (e.g., thumbs up, peace sign) to allow manual control over music suggestions or playback.

- 🎶 **Dynamic Music Recommendation**  
  Based on the detected emotion and gesture, the system recommends a song from a categorized playlist.

- 🧠 **Machine Learning Powered**  
  Utilizes trained models for accurate facial and gesture recognition.

- 📊 **User-Friendly Interface**  
  Simple GUI or console output for emotion/gesture feedback and song suggestions.

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Libraries:** OpenCV, Mediapipe, TensorFlow/Keras 
- **Music API/Playlist:** Local audio files ( ' https://drive.google.com/drive/folders/1-yy2Q6g5-fdsqsbeRP_8XuVCUarxi3cF?usp=sharing ')
- **GUI (optional):** Tkinter

## 🧪 How It Works

1. **Capture Input:** Webcam feed is used to continuously monitor the user's face and hands.
2. **Emotion Detection:** Facial landmarks are analyzed to predict emotional state.
3. **Gesture Detection:** Hand landmarks and finger positions are used to classify gestures.
4. **Music Matching:** Based on detected inputs, the system recommends and plays music from the relevant category.
5. **Real-Time Feedback:** The system provides real-time visual and/or audio feedback to the user.

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   https://github.com/sayan6823/Facial-Emotion-And-Hand-Gesture-Based-Music-Recommendation-System.git
   cd Facial-Emotion-And-Hand-Gesture-Based-Music-Recommendation-System
   ```

2. Run the application:
   ```bash
   python H&F.py
   ```

## 📚 Future Improvements

- Integration with online music platforms like Spotify or YouTube Music  
- Voice command support  
- Enhanced emotion classification (multi-label emotions)  
- Mobile app version
