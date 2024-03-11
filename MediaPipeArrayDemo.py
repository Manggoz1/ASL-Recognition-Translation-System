import cv2
import mediapipe as mp
import csv

# Initialize MediaPipe hand tracking
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize video capture (replace 0 with your camera index or video file path)
cap = cv2.VideoCapture(0)

# List to store landmarks for each frame
landmarks_data = []

while cap.isOpened():
    # Read a frame from the video capture
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Process the frame with MediaPipe hand tracking
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # If hands are detected, get the landmarks and store them
    if results.multi_hand_landmarks:
        landmarks_for_frame = []
        for landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Append landmarks to the list for this frame
            for landmark in landmarks.landmark:
                landmarks_for_frame.append([landmark.x, landmark.y, landmark.z ])

        # Append the landmarks for this frame to the overall list
        landmarks_data.append(landmarks_for_frame)

    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()

# Export the landmarks data to a CSV file
csv_filename = 'hand_landmarks_data.csv'
with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    
    # Write header
    csv_writer.writerow(['Frame', 'Landmark Index', 'X', 'Y', 'Z'])
    
    # Write data
    for frame_idx, landmarks_for_frame in enumerate(landmarks_data):
        for landmark_idx, landmarks in enumerate(landmarks_for_frame):
            csv_writer.writerow([frame_idx, landmark_idx, *landmarks])

print(f'Landmarks data exported to {csv_filename}')
