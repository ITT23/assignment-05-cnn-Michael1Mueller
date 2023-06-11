import cv2
import keras
import tensorflow as tf
import sys
import numpy as np
from pynput.keyboard import Key, Controller
from time import sleep

video_id = 0
rect_width = 200
rect_height = 250
IMG_SIZE = 64
SIZE = (IMG_SIZE, IMG_SIZE)
COLOR_CHANNELS = 3
label_names = ['like', 'no_gesture', 'dislike', 'stop', 'rock']

model = keras.models.load_model("gesture_recognition")

# for pynput
keyboard = Controller()

predictions_array = []

if len(sys.argv) > 1:
    video_id = int(sys.argv[1])

# https://stackoverflow.com/questions/37799847/python-playing-a-video-with-audio-with-opencv
cap = cv2.VideoCapture(video_id)

video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# for rect on frame
start_point = (int((video_width - rect_width)/2), int((video_height - rect_height)/2))
end_point = (int((video_width + rect_width)/2), int((video_height + rect_height)/2))


def apply_input(set):
    # https://stackoverflow.com/questions/59825/how-to-retrieve-an-element-from-a-set-without-removing-it
    input_condition = list(set)[0]
    if input_condition == "no_gesture":
        print("no_gesture")
    elif input_condition == "like":
        keyboard.press(Key.up)
        keyboard.release(Key.up)
        sleep(0.5)
    elif input_condition == "dislike":
        keyboard.press(Key.down)
        keyboard.release(Key.down)
        sleep(0.5)
    elif input_condition == "stop":
        keyboard.press(Key.space)
        keyboard.release(Key.space)
        sleep(0.5)
    elif input_condition == "rock":
        keyboard.press(Key.right)
        keyboard.release(Key.right)
        sleep(0.5)


def check_for_input(arr):
    global predictions_array
    # https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical#:~:text=if%20len(set(input_list)),input_list%20has%20all%20identical%20elements.
    if len(set(arr)) == 1:
        apply_input(set(arr))
        predictions_array = []
    else:
        predictions_array.pop(0)


def predict_frame(frame):
    # from notebook
    cropped_img = frame[start_point[1]:start_point[1]+rect_height, start_point[0]:start_point[0]+rect_width]
    cv2.imshow('crop', cropped_img)

    resized = cv2.resize(cropped_img, SIZE)
    reshaped = resized.reshape(-1, IMG_SIZE, IMG_SIZE, COLOR_CHANNELS)

    prediction = model.predict(reshaped)
    predicted_label_index = np.argmax(prediction, axis=1).item()
    predicted_label = label_names[predicted_label_index]

    if len(predictions_array) < 10:
        predictions_array.append(predicted_label)
    if len(predictions_array) == 10:
        check_for_input(predictions_array)


# from previous exercise
while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    cv2.rectangle(frame, start_point, end_point, (255, 0, 0), thickness=2)

    predict_frame(frame)
    cv2.imshow('frame', frame)

    # Wait for a key press and check if it's the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
