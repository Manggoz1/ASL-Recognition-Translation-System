import cv2
import mediapipe as mp
import csv


def pre_process_landmark(landmarks_for_frame):
    temp_landmark_data = []

    base_x, base_y = 0,0
    i = 0
    base_x = landmarks_for_frame[0]
    base_y = landmarks_for_frame[1]
    base_z = landmarks_for_frame[2]
    while(i<len(landmarks_for_frame)-2):
            temp_landmark_data.append(landmarks_for_frame[i] - base_x)
            i+=1
            temp_landmark_data.append(landmarks_for_frame[i] - base_y)
            i+=1 
            i+=1

    return temp_landmark_data
    



# Initialize MediaPipe hand tracking
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize video capture (replace 0 with your camera index or video file path)
cap = cv2.VideoCapture(0)

# List to store landmarks for each frame
landmarks_data = []

# Counter to keep track of the number of frames processed
frame_count = 0

#Letter Being Trained 
Letter="Y"

#Counter for Training Purposes
LetterNumber = 1

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
                landmarks_for_frame.extend([landmark.x, landmark.y, landmark.z])

        # Append the landmarks for this frame to the overall list
            landmarks_data.extend(pre_process_landmark(landmarks_for_frame))
          

    # Increment frame count
    frame_count += 1

    
    
    # Check if 30 frames have been processed
    if frame_count == 30:
        landmarks_data_truncated = landmarks_data[:1260]
        
        # Export the landmarks data to a CSV file
        csv_filename = f'hand_landmarks.csv'
        if(len(landmarks_data_truncated)==1260):
            with open(csv_filename, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                # Write data
                csv_writer.writerow(landmarks_data_truncated)
                
            
       
        print('Landmarks of Letter',Letter,f'data exported to {csv_filename}',"\n Number:",LetterNumber,"Frames:",len(landmarks_data_truncated))
        LetterNumber+=1
        
        csv_fileletter=f'hand_landmarks_letter.csv'
        with open(csv_fileletter,'a', newline='')as csvfile:
            csv_writer = csv.writer(csvfile)

            # Write Letter
            csv_writer.writerow(Letter)

        # # Reset frame count and landmarks data
        frame_count = 0
        landmarks_data = []
        

    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
