import cv2
from fer import FER
import numpy as np
import sqlite3
import time
import os

import pyautogui

def process_video(input_video_path, output_video_path):
    emotion_detector = FER()
    conn = sqlite3.connect('emotions.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS emotions (
        timestamp TEXT,
        emotion TEXT,
        screenshot BLOB
    )
    ''')

    video_capture = cv2.VideoCapture(input_video_path)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(video_capture.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), frame_rate, (frame_width, frame_height))

    start_time = time.time()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        result = emotion_detector.detect_emotions(frame)
        for person in result:
            bounding_box = person["box"]
            emotions = person["emotions"]
            top_emotion = max(emotions, key=emotions.get)
            cv2.rectangle(frame,
                          (bounding_box[0], bounding_box[1]),
                          (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                          (0, 255, 0), 2)
            cv2.putText(frame, top_emotion,
                        (bounding_box[0], bounding_box[1] + bounding_box[3] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        out.write(frame)
        if time.time() - start_time > 10:
            start_time = time.time()
            screenshot = np.array(pyautogui.screenshot())
            screenshot_data = cv2.imencode('.png', cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR))[1].tobytes()
            cursor.execute('''
            INSERT INTO emotions (timestamp, emotion, screenshot)
            VALUES (?, ?, ?)
            ''', (time.ctime(), top_emotion, screenshot_data))
            conn.commit()

    video_capture.release()
    out.release()
    cv2.destroyAllWindows()
    conn.close()

process_video('4.mp4', 'output5.mp4')

