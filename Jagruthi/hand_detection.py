import cv2
import mediapipe as mp
# Webcam connection
cap = cv2.VideoCapture(0)

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

while True:
    success, frame = cap.read()

    if not success:
        print("Failed to capture frame")
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe
    results = hands.process(rgb_frame)

    # If hand detected
    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            row = []

            for landmark in hand_landmarks.landmark:
                row.append(landmark.x)
                row.append(landmark.y)
                row.append(landmark.z)

            print(len(row))

    cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(1) == 27 or cv2.waitKey(1) == ord('q'):
        break


