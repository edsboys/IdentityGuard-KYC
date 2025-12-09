import cv2
import numpy as np

def check_liveness(video_path: str):
    cap = cv2.VideoCapture(video_path)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    previous_center = None
    movement_detected = False
    max_movement = 0
    frames_read = 0
    faces_found_count = 0
    
    MOVEMENT_THRESHOLD = 15 

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frames_read += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 1. Try detecting face normally
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            # 2. If no face, try rotating 90 degrees (Fix for phone videos)
            if len(faces) == 0:
                gray_rotated = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
                faces = face_cascade.detectMultiScale(gray_rotated, 1.1, 5)
            
            # 3. If STILL no face, try rotating 270 degrees
            if len(faces) == 0:
                gray_rotated_2 = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
                faces = face_cascade.detectMultiScale(gray_rotated_2, 1.1, 5)

            if len(faces) > 0:
                faces_found_count += 1
                # Take the largest face found
                (x, y, w, h) = max(faces, key=lambda b: b[2] * b[3])
                
                current_center = np.array([x + w//2, y + h//2])

                if previous_center is not None:
                    distance = np.linalg.norm(current_center - previous_center)
                    if distance > max_movement:
                        max_movement = distance
                    
                    if distance > MOVEMENT_THRESHOLD:
                        movement_detected = True

                previous_center = current_center

        cap.release()
        
        # DEBUGGING: This message helps us know why it failed
        debug_msg = f"Processed {frames_read} frames. Found faces in {faces_found_count} frames."

        if faces_found_count == 0:
            return {
                "status": "fail",
                "message": "No face detected at all. Video might be too dark, too far, or heavily rotated.",
                "debug": debug_msg
            }

        return {
            "status": "success",
            "is_alive": movement_detected,
            "max_movement_pixels": float(max_movement),
            "message": "Liveness confirmed" if movement_detected else "Face detected, but no movement.",
            "debug": debug_msg
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}