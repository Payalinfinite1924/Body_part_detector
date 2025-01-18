import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize the webcam (0 for default camera)
cap = cv2.VideoCapture(0)

# STEP 2: Create a PoseLandmarker object.
base_options = python.BaseOptions(
    model_asset_path='C:/Users/Asus/OneDrive/Desktop/Bodypartdetection/pose_landmarker_heavy.task'
)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False  # We only need landmarks here
)
detector = vision.PoseLandmarker.create_from_options(options)

# Define indices for specific landmarks
landmark_indices = {
    "Nose": 0,
    "Left Eye": 2,
    "Right Eye": 5,
    "Left Ear": 7,
    "Right Ear": 8,
    "Left Hand": 15,
    "Right Hand": 16,
    "Left Leg": 27,
    "Right Leg": 28,
}

print("Press 'q' to stop...")

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert frame to RGB as MediaPipe requires RGB images
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create a MediaPipe Image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Detect pose landmarks
    detection_result = detector.detect(mp_image)

    # Overlay selected landmarks and label boxes on the frame
    if detection_result.pose_landmarks:
        for label, index in landmark_indices.items():
            landmark = detection_result.pose_landmarks[0][index]

            # Convert normalized coordinates to pixel values
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])

            # Determine if landmark is left or right side based on x coordinate
            if x < frame.shape[1] // 2:
                # Left side
                label_position = (x + 10, y - 10)  # Slightly to the right
                box_position = (x + 5, y + 5)       # Box slightly to the right
            else:
                # Right side
                label_position = (x - 100, y - 10)  # Slightly to the left
                box_position = (x - 100, y + 5)     # Box slightly to the left

            # Draw a circle at each landmark
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

          

            # Put the label text near the landmark
            cv2.putText(frame, label, label_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Display the video frame
    cv2.imshow('Pose Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
