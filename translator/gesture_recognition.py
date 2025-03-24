import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

def detect_gestures(image_path):
    """ Detect hand gestures in an image and return the recognized sign + processed image path. """
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    gesture_text = "No gesture detected"  # Default text

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract key hand points
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            # Convert landmarks to numpy array for calculations
            thumb = np.array([thumb_tip.x, thumb_tip.y])
            index = np.array([index_tip.x, index_tip.y])
            middle = np.array([middle_tip.x, middle_tip.y])
            ring = np.array([ring_tip.x, ring_tip.y])
            pinky = np.array([pinky_tip.x, pinky_tip.y])
            wrist_pos = np.array([wrist.x, wrist.y])

            # ✅ **Thumbs Up**
            if thumb[1] < wrist_pos[1] and index[1] > thumb[1] and middle[1] > thumb[1]:
                gesture_text = "👍 Thumbs Up"

            # ✅ **OK Sign**
            elif np.linalg.norm(thumb - index) < 0.05:
                gesture_text = "👌 OK Sign"

            # ✅ **Peace Sign**
            elif index[1] < wrist_pos[1] and middle[1] < wrist_pos[1] and ring[1] > wrist_pos[1] and pinky[1] > wrist_pos[1]:
                gesture_text = "✌️ Peace Sign"

            # ✅ **Fist (Closed Hand)**
            elif (np.linalg.norm(thumb - wrist_pos) < 0.1 and
                  np.linalg.norm(index - wrist_pos) < 0.1 and
                  np.linalg.norm(middle - wrist_pos) < 0.1 and
                  np.linalg.norm(ring - wrist_pos) < 0.1 and
                  np.linalg.norm(pinky - wrist_pos) < 0.1):
                gesture_text = "✊ Fist"

            # ✅ **Open Palm**
            elif (thumb[1] < wrist_pos[1] and
                  index[1] < wrist_pos[1] and
                  middle[1] < wrist_pos[1] and
                  ring[1] < wrist_pos[1] and
                  pinky[1] < wrist_pos[1]):
                gesture_text = "🖐️ Open Palm"

    # Save the processed image
    processed_image_path = image_path.replace("sign_language_inputs", "processed_images")
    cv2.imwrite(processed_image_path, image)

    return processed_image_path, gesture_text  # Return both image path & detected gesture
