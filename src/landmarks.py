"""
Minimal pipeline:
camera -> Haar face box -> MediaPipe FaceMesh (Tasks API) -> extract 5 keypoints -> draw
Run:
python -m src.landmarks
Keys:
q : quit
"""
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import os

# 5-point indices (FaceMesh)
IDX_LEFT_EYE = 33
IDX_RIGHT_EYE = 263
IDX_NOSE_TIP = 1
IDX_MOUTH_LEFT = 61
IDX_MOUTH_RIGHT = 291

def main():
    # Haar
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face = cv2.CascadeClassifier(cascade_path)
    if face.empty():
        raise RuntimeError(f"Failed to load cascade: {cascade_path}")
    
    # FaceMesh (Tasks API)
    model_path = os.path.join("models", "face_landmarker.task")
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model not found: {model_path}. Please download it first.")

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    
    detector = vision.FaceLandmarker.create_from_options(options)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not opened. Try camera index 0/1/2.")
    
    print("Haar + FaceMesh 5pt (Tasks API). Press 'q' to quit.")
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        
        # draw ALL haar faces (no ranking)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # FaceMesh on full frame (Tasks API)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # Use current time in ms for timestamp
        timestamp_ms = int(time.time() * 1000)
        res = detector.detect_for_video(mp_image, timestamp_ms)
        
        if res.face_landmarks:
            lm = res.face_landmarks[0]
            idxs = [IDX_LEFT_EYE, IDX_RIGHT_EYE, IDX_NOSE_TIP, IDX_MOUTH_LEFT, IDX_MOUTH_RIGHT]
            pts = []
            for i in idxs:
                p = lm[i]
                pts.append([p.x * W, p.y * H])
            
            kps = np.array(pts, dtype=np.float32) # (5,2)
            
            # enforce left/right ordering
            if kps[0, 0] > kps[1, 0]:
                kps[[0, 1]] = kps[[1, 0]]
            if kps[3, 0] > kps[4, 0]:
                kps[[3, 4]] = kps[[4, 3]]
            
            # draw 5 points
            for (px, py) in kps.astype(int):
                cv2.circle(frame, (int(px), int(py)), 4, (0, 255, 0), -1)
            
            cv2.putText(frame, "5pt", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow("5pt Landmarks", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break
    
    detector.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()