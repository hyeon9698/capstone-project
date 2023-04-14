import cv2
import mediapipe as mp
from PIL import Image
import pytesseract
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Set the tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

# For webcam input:
camera = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    # Initialize FPS calculation variables
    fps_start_time = 0
    fps_frame_count = 0
    while cv2.waitKey(1) & 0xFF != 27:
        ret, frame = camera.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        # Start FPS timer
        if fps_frame_count == 0:
            fps_start_time = time.time()

        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                # x, y = int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0])

                # box_size = 200
                # half_box_size = box_size // 2
                # top_left_corner = (x - half_box_size, y - half_box_size)
                # bottom_right_corner = (x + half_box_size, y + half_box_size)
                # cv2.rectangle(frame, top_left_corner, bottom_right_corner, (0, 255, 0), 2)

                # if (0 <= top_left_corner[0] < frame.shape[1] and 0 <= top_left_corner[1] < frame.shape[0] and
                #         0 <= bottom_right_corner[0] < frame.shape[1] and 0 <= bottom_right_corner[1] < frame.shape[0]):
                #     roi = frame[top_left_corner[1]:bottom_right_corner[1], top_left_corner[0]:bottom_right_corner[0]]

                #     img_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                #     img_blurred = cv2.GaussianBlur(img_gray, ksize=(5, 5), sigmaX=0)

                #     img_blur_thresh = cv2.adaptiveThreshold(
                #         img_blurred,
                #         maxValue=255.0,
                #         adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                #         thresholdType=cv2.THRESH_BINARY_INV,
                #         blockSize=25,
                #         C=9
                #     )

                #     # Perform OCR on the ROI
                #     text = pytesseract.image_to_string(img_blur_thresh, config='--psm 6')
                #     print(text)
        # Calculate FPS and print on frame
        fps_frame_count += 1
        if fps_frame_count >= 15:
            fps_end_time = time.time()
            fps = int(fps_frame_count / (fps_end_time - fps_start_time))
            cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('MediaPipe Hands', frame) #cv2.flip(frame, 1))


camera.release()
cv2.destroyAllWindows()
