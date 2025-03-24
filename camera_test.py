import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Gesture recognition logic (Example: Thumb up)
import numpy as np

def recognize_gesture(hand_landmarks):
 if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        wrist = hand_landmarks.landmark[0]  # WRIST
        thumb_tip = hand_landmarks.landmark[4]  # THUMB_TIP
        index_tip = hand_landmarks.landmark[8]  # INDEX_TIP
        middle_tip = hand_landmarks.landmark[12]  # MIDDLE_TIP
        ring_tip = hand_landmarks.landmark[16]  # RING_TIP
        pinky_tip = hand_landmarks.landmark[20]  # PINKY_TIP

    # ✊ **Fist** (All fingertips below their base points)
    if (
        index_tip.y > hand_landmarks.landmark[6].y and  # INDEX_FINGER_MCP
        middle_tip.y > hand_landmarks.landmark[10].y and  # MIDDLE_FINGER_MCP
        ring_tip.y > hand_landmarks.landmark[14].y and  # RING_FINGER_MCP
        pinky_tip.y > hand_landmarks.landmark[18].y  # PINKY_FINGER_MCP
    ):
        return "✊ Fist"

    # ✋ **Open Palm** (All fingers extended)
    if (
        index_tip.y < hand_landmarks.landmark[6].y and
        middle_tip.y < hand_landmarks.landmark[10].y and
        ring_tip.y < hand_landmarks.landmark[14].y and
        pinky_tip.y < hand_landmarks.landmark[18].y
    ):
        return "✋ Open Palm"

    # 👍 **Thumbs Up** (Thumb up, other fingers down)
    if thumb_tip.y < wrist.y and index_tip.y > thumb_tip.y and middle_tip.y > thumb_tip.y:
        return "👍 Thumbs Up"

    # 👌 **OK Sign** (Thumb & Index touching)
    if np.linalg.norm([thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y]) < 0.05:
        return "👌 OK Sign"

    # ✌️ **Victory/Peace Sign** (Index & Middle up, others down)
    if (
        index_tip.y < hand_landmarks.landmark[6].y and
        middle_tip.y < hand_landmarks.landmark[10].y and
        ring_tip.y > hand_landmarks.landmark[14].y and
        pinky_tip.y > hand_landmarks.landmark[18].y
    ):
        return "✌️ Peace/Victory Sign"

    return "🤷 Unknown Gesture"  # Default case


# Start capturing video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    gesture_text = "No Gesture Detected"

    # Draw landmarks and recognize gestures
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Convert landmarks to a list
            landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            gesture_text = recognize_gesture(landmarks)

    # Display gesture text on screen
    cv2.putText(frame, gesture_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Hand Gesture Detection", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
