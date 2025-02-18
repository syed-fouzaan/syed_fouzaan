import cv2
import mediapipe as mp   
import subprocess
import os
import math
import numpy as np
import pyautogui
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from tkinter import messagebox, Tk
import time

# Solution APIs
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Volume Control Library Usage 
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol, maxVol = volRange[0], volRange[1]

# Webcam Setup
wCam, hCam = 640, 480
cam = cv2.VideoCapture(0)
cam.set(3, wCam)
cam.set(4, hCam)

def show_permission_dialog(case_number):
    root = Tk()
    root.withdraw()  # Hide the main window
    response = messagebox.askyesno("Camera Permission", f"Do you want to execute Case {case_number}?")
    root.destroy()
    return response

def launch_dino_game():
    chrome_path = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
    if os.path.exists(chrome_path):
        subprocess.Popen([chrome_path, "chrome://dino"])
    else:
        print("Please install Google Chrome to play the Dino game.")

def count_fingers(hand_landmarks):
    fingers_down = 0
    # Thumb
    if hand_landmarks.landmark[4].y >= hand_landmarks.landmark[3].y:  # Thumb is down
        fingers_down += 1
    # Other fingers
    for i in range(1, 5):  # 1 to 4 for index to pinky
        if hand_landmarks.landmark[i * 4 + 2].y >= hand_landmarks.landmark[i * 4].y:  # Finger is down
            fingers_down += 1
    return fingers_down

def case_1(hands):
    last_y_index = None
    last_y_middle = None
    last_y_right_index = None
    last_y_right_middle = None
    last_x_index = None
    last_y_index_cursor = None
    minimized = False
    
    # Variables for left hand tracking
    last_x_index = None
    last_y_index_cursor = None

    while cam.isOpened():
        success, image = cam.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks and results.multi_handedness:
            lmList = []
            hand_labels = []
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label
                hand_labels.append(hand_label)
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])

            # Volume Control and Mouse Scroll
            if 'Right' in hand_labels:
                right_hand_index = hand_labels.index('Right')
                y_right_index = lmList[right_hand_index * 21 + 8][2]
                y_right_middle = lmList[right_hand_index * 21 + 12][2]

                if last_y_right_index is not None and last_y_right_middle is not None:
                    if y_right_index < last_y_right_index and y_right_middle < last_y_right_middle:
                        current_vol = volume.GetMasterVolumeLevel()
                        volume.SetMasterVolumeLevel(min(current_vol + 1, maxVol), None)
                    elif y_right_index > last_y_right_index and y_right_middle > last_y_right_middle:
                        current_vol = volume.GetMasterVolumeLevel()
                        volume.SetMasterVolumeLevel(max(current_vol - 1, minVol), None)

                last_y_right_index = y_right_index
                last_y_right_middle = y_right_middle

            if 'Left' in hand_labels and 'Right' in hand_labels:
                left_hand_index = hand_labels.index('Left')
                right_hand_index = hand_labels.index('Right')
                left_index_x = lmList[left_hand_index * 21 + 8][1]
                right_index_x = lmList[right_hand_index * 21 + 8][1]
                distance = abs(left_index_x - right_index_x)

                if distance < 50 and not minimized:  # Hands are close, minimize all windows
                    pyautogui.hotkey('win', 'd')  # Minimize all windows
                    minimized = True
                elif distance > 150 and minimized:  # Hands are far, restore all windows
                    pyautogui.hotkey('win', 'd')  # Restore all windows
                    minimized = False

            if 'Left' in hand_labels:
                left_hand_index = hand_labels.index('Left')
                y_index = lmList[left_hand_index * 21 + 8][2]  # Left hand index finger y position
                y_middle = lmList[left_hand_index * 21 + 12][2]  # Left hand middle finger y position
                y_thumb = lmList[left_hand_index * 21 + 4][2]  # Left hand thumb y position
                x_index = lmList[left_hand_index * 21 + 8][1]  # Left hand index finger x position

                # Check for scrolling
                if last_y_index is not None and last_y_middle is not None:
                    if abs(y_index - last_y_index) > 5 and abs(y_middle - last_y_middle) > 5:
                        # Both fingers moved, scroll
                        if y_index < last_y_index and y_middle < last_y_middle:
                            # Fingers moved up, scroll up
                            pyautogui.scroll(100)
                        elif y_index > last_y_index and y_middle > last_y_middle:
                            # Fingers moved down, scroll down
                            pyautogui.scroll(-100)

                # Check for click action
                finger_distance = math.hypot(y_index - y_thumb, 0)
                if finger_distance < 30:
                    # Thumb and index finger are close, register a click
                    pyautogui.click()

                # Move cursor based on index finger position (inverted movement for x-axis)
                if last_x_index is not None and last_y_index_cursor is not None:
                    if abs(x_index - last_x_index) > 5 or abs(y_index - last_y_index_cursor) > 5:
                        screen_width, screen_height = pyautogui.size()
                        x_index_norm = np.interp(x_index, [0, wCam], [screen_width, 0])  # Inverted x movement
                        y_index_norm = np.interp(y_index, [0, hCam], [0, screen_height])  # Normal y movement
                        # Direct cursor movement
                        pyautogui.moveTo(x_index_norm, y_index_norm)

                last_x_index = x_index
                last_y_index_cursor = y_index
                last_y_index = y_index
                last_y_middle = y_middle

        cv2.imshow('Volume Control - Case 1', image)
        if cv2.waitKey(5) & 0xFF == 27:  # Exit on 'ESC' key
            break

def case_2(hands):
    launch_dino_game()  # Automatically launch the Dino game
    time.sleep(2)  # Wait for the game to load

    last_index_y = None  # Track the last y position of the index finger

    while cam.isOpened():
        success, image = cam.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                index_y = hand_landmarks.landmark[8].y  # Get the y position of the index finger

                if last_index_y is not None:
                    if index_y < last_index_y - 0.05:  # If the index finger moves up
                        pyautogui.press('space')  # Simulate pressing the spacebar to jump

                last_index_y = index_y  # Update the last index finger position

        cv2.imshow('Dino Game Control - Case 2', image)
        if cv2.waitKey(5) & 0xFF == 27:  # Exit on 'ESC' key
            break

def is_fist(hand_landmarks):
    """Check if the hand is in a fist position."""
    for i in range(1, 5):  # Check index, middle, ring, pinky
        if hand_landmarks.landmark[i * 4 + 2].y < hand_landmarks.landmark[i * 4].y:  # Finger is not curled
            return False
    if hand_landmarks.landmark[4].y > hand_landmarks.landmark[3].y:  # Thumb is not curled
        return False
    return True

def case_3(hands):
    last_x = None
    swipe_threshold = 0.1  # Adjusted threshold for swipe detection
    permission_granted = False  # Track if permission is granted

    while cam.isOpened():
        success, image = cam.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:  # Only one hand
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            wrist_x = hand_landmarks.landmark[0].x
            index_x = hand_landmarks.landmark[8].x  # Index finger x position

            if last_x is not None:
                # Detect swipe direction
                if index_x - last_x > swipe_threshold:  # Swipe right
                    pyautogui.press('right')  # Move to next slide
                elif index_x - last_x < -swipe_threshold:  # Swipe left
                    pyautogui.press('left')  # Move to previous slide

            last_x = index_x  # Update last index position

            # Check for fist gesture
            if is_fist(hand_landmarks):
                if not permission_granted:  # If permission hasn't been granted yet
                    if show_permission_dialog(3):  # Show permission dialog for ESC
                        permission_granted = True  # Set permission granted
                else:
                    pyautogui.press('esc')  # Simulate pressing the ESC key

        cv2.imshow('PowerPoint Control - Case 3', image)
        if cv2.waitKey(5) & 0xFF == 27:  # Exit on 'ESC' key
            break

def main():
    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
        while cam.isOpened():
            success, image = cam.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) <= 2:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    finger_count = count_fingers(hand_landmarks)

                    cv2.putText(image, f'Fingers: {finger_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

                    if finger_count in [1, 2, 3]:
                        if finger_count == 1:
                            if show_permission_dialog(1):
                                case_1(hands)
                        elif finger_count == 2:
                            if show_permission_dialog(2):
                                case_2(hands)
                        elif finger_count == 3:
                            cv2.putText(image, "Case 3 is now active.", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            case_3(hands)  # Call case 3

            cv2.imshow('Hand Gesture Recognition', image)
            if cv2.waitKey(5) & 0xFF == 27:  # Exit on 'ESC' key
                break

if __name__ == "__main__":
    main()