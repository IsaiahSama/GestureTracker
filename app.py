import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

base_options = python.BaseOptions(model_asset_path="gesture_recognizer.task")
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

webcam = cv2.VideoCapture(0)

gestures = {
    "Closed_fist": "👊🏾",
    "Open_Palm": "🤲🏾",
    "Pointing_Up": "☝🏾",
    "Thumb_Down": "👎🏾",
    "Thumb_Up": "👍🏾",
    "Victory": "✌🏾",
    "ILoveYou": "🥰"
}

while webcam.isOpened():
    success, img = webcam.read()

    img = cv2.flip(img, 1)
    # img_rgb = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB) # Converting to RGB
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

    result = recognizer.recognize(mp_img)
    if result.gestures:
        img = mp_img.numpy_view().copy()
        top_gesture = result.gestures[0][0]
        hand_landmarks = result.hand_landmarks
        
        if top_gesture.category_name in gestures and hand_landmarks:
            for hand_landmark in hand_landmarks:
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmark
                ])

                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks_proto,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

    cv2.imshow("Live Gesture Viewer", img)

    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

webcam.release()
cv2.destroyAllWindows()