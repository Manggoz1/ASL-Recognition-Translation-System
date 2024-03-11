import cv2
import mediapipe as mp
import csv

def process_image(image_path):
    # Initialize MediaPipe hand tracking
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    # Read the image
    frame = cv2.imread(image_path)

    # Process the image with MediaPipe hand tracking
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # List to store finger positions for each hand
    finger_positions_for_frame = []

    # If hands are detected, get the finger positions
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            hand_landmarks = [(landmark.x, landmark.y, landmark.z) for landmark in landmarks.landmark]

            # Extract finger positions
            thumb = hand_landmarks[1:5]
            index = hand_landmarks[5:9]
            middle = hand_landmarks[9:13]
            ring = hand_landmarks[13:17]
            pinky = hand_landmarks[17:21]

            finger_positions_for_frame.append({
                'thumb': thumb,
                'index': index,
                'middle': middle,
                'ring': ring,
                'pinky': pinky,
            })

    return finger_positions_for_frame

def export_to_csv(data, csv_filename):
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write header
        header = ['Finger', 'Landmark', 'X', 'Y', 'Z']
        csv_writer.writerow(header)

        # Write data
        for hand_idx, finger_positions in enumerate(data):
            for finger, landmarks in finger_positions.items():
                for landmark_idx, landmark_position in enumerate(landmarks):
                    row = [f'Hand_{hand_idx + 1}', finger, landmark_idx, *landmark_position]
                    csv_writer.writerow(row)

if __name__ == "__main__":
    image_path = "letter_A.jpg"  # Replace with the path to your image
    output_csv_filename = 'Alphabet/LetterA.csv'

    # Process the image and get finger positions
    finger_positions_data = process_image(image_path)

    # Export the finger positions data to a CSV file
    export_to_csv(finger_positions_data, output_csv_filename)

    print(f'Finger positions data exported to {output_csv_filename}')
