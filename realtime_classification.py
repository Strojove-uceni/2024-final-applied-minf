import cv2
from transformers import AutoImageProcessor, AutoModelForImageClassification

import torch

# Path to your MKV file
video_path = '/home/martin/Coding/2024-final-applied-minf/test_data/test_recording.mkv'

# Open the video file
cap = cv2.VideoCapture(video_path)
processor = AutoImageProcessor.from_pretrained("trpakov/vit-face-expression")
model = AutoModelForImageClassification.from_pretrained("trpakov/vit-face-expression")

if not cap.isOpened():
    print("Error: Could not open video file.")
else:
    # Loop to read and display each frame
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Display the frame
            cv2.imshow('Frame', frame)
            i += 1
            if i == 30:
                inputs = processor(frame, return_tensors="pt")
                logits = model(**inputs).logits
                predicted_label = torch.argmax(logits, dim=-1).int()
                print(model.config.id2label[predicted_label.item()])
                i = 0
            # Press 'q' to exit the video display
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()
