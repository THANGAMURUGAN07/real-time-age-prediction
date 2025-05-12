# real-time-age-prediction

sample output:
![Screenshot 2025-05-12 215633](https://github.com/user-attachments/assets/a0db708e-0fed-484a-8514-f68a2126b52c)

Real-Time Age Prediction Using Webcam
🎯 Features
1.Detect faces from webcam feed.
2.predict age group in real-time.
3.Streamlit-based user interface.

🧠 Model Details
1.Trained on UTKFace dataset with 10 age categories.
2.CNN built using Keras and TensorFlow backend.

▶️ How to Train the Model
python train_model.py

▶️ Run the Streamlit App
streamlit run app.py

📊 Age Categories
0–2, 3–12, 13–19, 20–30, 31–40, 41–50, 51–60, 61–70, 71–80, 80+

🗂 Folder Structure
📁 age_prediction/
    ├── app.py                 # Streamlit UI for age prediction
    ├── train_model.py         # Model training script
    ├── age_model.h5           # Trained CNN model (generated after training)

📌 Requirements
Install dependencies:
pip install opencv-python mediapipe pyautogui pynput streamlit tensorflow keras


📄 License
This project is open-source and available under the MIT License.
Feel free to use, modify, and distribute it with proper attribution.


