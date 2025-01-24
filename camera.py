import cv2
from detection import AccidentDetectionModel
import numpy as np
import os
import time

model = AccidentDetectionModel("model.json", 'model_weights.keras')
font = cv2.FONT_HERSHEY_SIMPLEX

def startapplication(fps=30):
    video = cv2.VideoCapture('demo.mp4')  # for camera use video = cv2.VideoCapture(0)
    frame_time = 1.0 / fps  # Time each frame should be displayed for

    while True:
        start_time = time.time()  # Start time of the frame processing

        ret, frame = video.read()
        if not ret:
            break  # Exit the loop if no frame is captured

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(gray_frame, (250, 250))

        pred, prob = model.predict_accident(roi[np.newaxis, :, :])
        if pred == "Accident":
            prob = (round(prob[0][0] * 100, 2))

            # to beep when alert:
            # if(prob > 90):
            #     os.system("say beep")

            cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
            cv2.putText(frame, pred + " " + str(prob), (20, 30), font, 1, (255, 255, 0), 2)

        cv2.imshow('Video', frame)

        # Calculate the time taken to process the frame
        processing_time = time.time() - start_time
        wait_time = max(1, int((frame_time - processing_time) * 1000))  # Convert to milliseconds

        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    startapplication(fps=30)  # Set the desired FPS here