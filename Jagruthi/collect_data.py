import cv2
import mediapipe as mp
import csv

# Gesture label
gesture_name = input("Enter gesture name: ")

# CSV file
file = open("dataset.csv", "a", newline="")
writer = csv.writer(file)

# Webcam
cap = cv2.VideoCapture(0)

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Drawing utility
mp_draw = mp.solutions.drawing_utils

while True:

    success, frame = cap.read()

    if not success:
        print("Failed to capture frame")
        break

    # Convert BGR -> RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = hands.process(rgb_frame)

    # If hand detected
    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            row = []

            # Extract x,y,z
            for landmark in hand_landmarks.landmark:

                row.append(landmark.x)
                row.append(landmark.y)
                row.append(landmark.z)

            # Add label
            row.append(gesture_name)

            # Save row to CSV
            writer.writerow(row)

    # Show webcam
    cv2.imshow("Collect Data", frame)

    # Exit on ESC or q
    key = cv2.waitKey(1)

    if key == 27 or key == ord('q'):
        break

# Cleanup
file.close()
cap.release()
cv2.destroyAllWindows()
