import easyocr
import cv2
import time

reader = easyocr.Reader(['en', 'ko'], gpu=False) # this needs to run only once to load the model into memory

# For webcam input:
camera = cv2.VideoCapture(0)

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

    # Calculate FPS and print on frame
    fps_frame_count += 1
    if fps_frame_count >= 15:
        fps_end_time = time.time()
        fps = int(fps_frame_count / (fps_end_time - fps_start_time))
        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('fff', frame) #cv2.flip(frame, 1))

result = reader.readtext(frame, detail=0, paragraph=True)
# result = reader.readtext('test_image2.png', detail=0, paragraph=True)
print(result)