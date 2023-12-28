import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get X-coordinate of specific landmarks for left and right hands
            thumb_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
            pinky_x = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x

            if thumb_x < pinky_x:
                print("Right hand detected.")
            else:
                print("Left hand detected.")

    cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
