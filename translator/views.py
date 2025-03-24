from django.shortcuts import render
from django.core.files import File
from django.conf import settings
from .models import SignLanguageInput, ProcessedImage
from PIL import Image, ImageOps
import os
from gtts import gTTS
from django.http import StreamingHttpResponse
import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

@csrf_exempt
def process_gesture(request):
    if request.method == "POST" and request.FILES.get("image"):
        image_file = request.FILES["image"]
        image_array = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Convert the image to RGB (for MediaPipe)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        detected_gesture = "No gesture detected"
        audio_file_url = None  # Initialize audio file URL

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract key landmarks
                wrist = hand_landmarks.landmark[0]
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]
                middle_tip = hand_landmarks.landmark[12]
                ring_tip = hand_landmarks.landmark[16]
                pinky_tip = hand_landmarks.landmark[20]

                # Gesture Detection Logic
                if (
                    index_tip.y > hand_landmarks.landmark[6].y and
                    middle_tip.y > hand_landmarks.landmark[10].y and
                    ring_tip.y > hand_landmarks.landmark[14].y and
                    pinky_tip.y > hand_landmarks.landmark[18].y
                ):
                    detected_gesture = "✊ Fist"

                elif (
                    index_tip.y < hand_landmarks.landmark[6].y and
                    middle_tip.y < hand_landmarks.landmark[10].y and
                    ring_tip.y < hand_landmarks.landmark[14].y and
                    pinky_tip.y < hand_landmarks.landmark[18].y
                ):
                    detected_gesture = "✋ Open Palm"

                elif thumb_tip.y < wrist.y and index_tip.y > thumb_tip.y and middle_tip.y > thumb_tip.y:
                    detected_gesture = "👍 Thumbs Up"

                elif np.linalg.norm([thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y]) < 0.05:
                    detected_gesture = "👌 OK Sign"

                elif (
                    index_tip.y < hand_landmarks.landmark[6].y and
                    middle_tip.y < hand_landmarks.landmark[10].y and
                    ring_tip.y > hand_landmarks.landmark[14].y and
                    pinky_tip.y > hand_landmarks.landmark[18].y
                ):
                    detected_gesture = "✌️ Peace/Victory Sign"

                # 🎤 Convert Gesture to Audio
                audio_file_url = convert_text_to_speech(detected_gesture, "gesture_audio.mp3")

        #return JsonResponse({"gesture": detected_gesture, "audio": audio_file_url})
    return JsonResponse({
    "gesture": detected_gesture,
    "gesture_audio": audio_file_url  # Make sure gesture_audio_url is correctly set
})


    return JsonResponse({"error": "Invalid request"}, status=400)



import time  # Import time for unique filenames

def convert_text_to_speech(text, filename="gesture_audio.mp3"):
    """Convert text to speech and save as an MP3 file."""
    
    # 🆕 Create a separate folder for gesture audios
    audio_folder = os.path.join(settings.MEDIA_ROOT, "gesture_audio")
    if not os.path.exists(audio_folder):
        os.makedirs(audio_folder)

    audio_file_path = os.path.join(audio_folder, filename)

    if text.strip():  # Ensure text is not empty
        tts = gTTS(text=text, lang="en")
        tts.save(audio_file_path)
        
        # 🆕 Return the correct media URL
        return "/media/gesture_audio/" + filename
    
    return None









from .gesture_recognition import detect_gestures

def generate_frames():
    cap = cv2.VideoCapture(0)  # Open webcam
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def camera_feed(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def process_image_and_save(image_obj):
    try:
        image_path = os.path.join(settings.MEDIA_ROOT, image_obj.image.name)
        processed_image_path = os.path.join(settings.MEDIA_ROOT, "processed_images", "processed_" + os.path.basename(image_obj.image.name))

        # Open image and apply thresholding
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, thresh_img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)

        # Save the processed image
        cv2.imwrite(processed_image_path, thresh_img)

        # Perform OCR
        extracted_text = pytesseract.image_to_string(Image.open(processed_image_path))
        print("Extracted Text:", extracted_text)  # Debugging output

        return processed_image_path, extracted_text  # Return processed image path & text

    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def extract_text_from_image(image_path):
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
        return text.strip()
    except Exception as e:
        print(f"Error extracting text: {e}")
        return None




def text_to_speech(text):
    """Convert extracted text to speech and save as an MP3 file in 'audio_text' folder."""
    audio_folder = os.path.join(settings.MEDIA_ROOT, "audio_text")  # New folder
    if not os.path.exists(audio_folder):
        os.makedirs(audio_folder)

    filename = f"text_audio_{int(time.time())}.mp3"
    audio_file_path = os.path.join(audio_folder, filename)

    if text.strip():
        tts = gTTS(text=text, lang="en")
        tts.save(audio_file_path)
        return "/media/audio_text/" + filename  # Updated URL
    
    return None



def home(request):
    output = None
    latest_image = SignLanguageInput.objects.last()
    processed_image = None
    extracted_text = None
    audio_file_url = None
    detected_gesture = None  # Store detected gesture
    gesture_audio_url = None  # 🆕 Store gesture audio separately

    if request.method == "POST":
        user_input = request.POST.get("user_input")
        uploaded_image = request.FILES.get("uploaded_image")

        if uploaded_image:
            new_entry = SignLanguageInput.objects.create(image=uploaded_image)
            latest_image = new_entry  # Update latest uploaded image

            # Process image and extract text
            processed_image_path, extracted_text = process_image_and_save(new_entry)

            if processed_image_path:
                with open(processed_image_path, "rb") as f:
                    processed_entry = ProcessedImage.objects.create(original_image=new_entry)
                    processed_entry.processed_image.save(os.path.basename(processed_image_path), File(f))
                    processed_image = processed_entry.processed_image.url  # Use .url for display

                    # **NEW: Detect Gesture**
                    processed_image_path, detected_gesture = detect_gestures(processed_image_path)

                    # 🎤 Extract text and generate speech audio
                    audio_file_url = text_to_speech(extracted_text) if extracted_text else None

                    # 🎤 Generate gesture speech audio separately
                    if detected_gesture and detected_gesture != "No gesture detected":
                        gesture_audio_url = convert_text_to_speech(detected_gesture, "gesture_audio.mp3")

    return render(request, 'translator/home.html', {
        "output": output,
        "latest_image": latest_image,
        "processed_image": processed_image,
        "extracted_text": extracted_text,
        "audio_file": audio_file_url,  # Text-to-speech audio
        "detected_gesture": detected_gesture,  # Detected gesture
        "gesture_audio": gesture_audio_url,  # 🆕 Gesture speech audio
    })




def about(request):
    return render(request, 'translator/about.html')

def contact(request):
    return render(request, 'translator/contact.html')
