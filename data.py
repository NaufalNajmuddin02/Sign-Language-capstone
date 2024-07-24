from flask import Flask, request, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
import base64
from pymongo import MongoClient
import datetime
import threading
from playsound import playsound
from flask_pymongo import PyMongo
from bson import ObjectId

app = Flask(__name__)

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['gesture_database']
collection = db['gesture_logs']

# MongoDB for gender and age group data
app.config["MONGO_URI"] = "mongodb://localhost:27017/gesture_database"
mongo = PyMongo(app)

# Load the pre-trained model
with open('modelcoba.pkl', 'rb') as f:
    model = pickle.load(f)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Global variables to store gender and age group
selected_gender = None
selected_age_group = None

def play_sound_for_class(class_name):
    sound_file = f'sounds/{class_name}.mp3'
    threading.Thread(target=playsound, args=(sound_file,)).start()

def process_frame(frame):
    global selected_gender, selected_age_group
    image = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        results = holistic.process(image_rgb)
        image_rgb.flags.writeable = True   
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # Draw landmarks
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))

        try:
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            
            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

            row = pose_row + face_row

            X = pd.DataFrame([row])
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]
            
            coords = tuple(np.multiply(
                            np.array(
                                (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                 results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)),
                            [640,480]).astype(int))
            
            cv2.rectangle(image, 
                          (coords[0], coords[1]+5), 
                          (coords[0]+len(body_language_class)*20, coords[1]-30), 
                          (245, 117, 16), -1)
            cv2.putText(image, body_language_class, coords, 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Status box
            cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
            
            # Display Class
            cv2.putText(image, 'CLASS', (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class.split(' ')[0], (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display Prob
            cv2.putText(image, 'PROB', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)], 2)), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Play sound for detected class
            play_sound_for_class(body_language_class.split(' ')[0])

            # Log results to MongoDB
            log_entry = {
                'timestamp': datetime.datetime.now(),
                'day': datetime.datetime.now().strftime('%A'),
                'class': body_language_class,
                'probability': float(body_language_prob[np.argmax(body_language_prob)]),
                'frame_data': base64.b64encode(frame).decode('utf-8'),
                'gender': selected_gender,
                'age_group': selected_age_group
            }
            collection.insert_one(log_entry)
        except Exception as e:
            print(f"Error: {e}")
            pass

        _, buffer = cv2.imencode('.jpg', image)
        return buffer.tobytes()

@app.route('/video_feed', methods=['POST'])
def video_feed():
    frame = base64.b64decode(request.form['frame'])
    processed_frame = process_frame(frame)
    response = {
        'frame': base64.b64encode(processed_frame).decode('utf-8')
    }
    return jsonify(response)

@app.route('/add', methods=['POST'])
def add_data():
    global selected_gender, selected_age_group
    data = request.json
    gender = data.get('gender')
    age_group = data.get('age_group')
    
    if not gender or not age_group:
        return jsonify({"error": "Gender and age group are required"}), 400
    
    selected_gender = gender
    selected_age_group = age_group
    
    record_id = mongo.db.gesture_logs.insert_one({
        "gender": gender,
        "age_group": age_group
    }).inserted_id
    
    return jsonify({"id": str(ObjectId(record_id)), "gender": gender, "age_group": age_group}), 201

if __name__ == '__main__':
    app.run(debug=True, host='192.168.43.249', port=5000)
