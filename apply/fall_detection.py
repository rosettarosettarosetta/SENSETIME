import cv2
from STESDK import BodyDetector, BodyAligner

# Create the detector
body_detector = BodyDetector()
body_aligner = BodyAligner()

# Open the input video
cap = cv2.VideoCapture(3)

while cap.isOpened():
    # Read the frame
    ret, frame = cap.read()
    if not ret:
        break

    # Detect bodies
    rects = body_detector.detect(frame)

    # Detect keypoints
    for rect in rects:
        points = body_aligner.align(frame, rect)
        frame = body_aligner.render(frame, points)

        # Draw rectangle around detected bodies
        x, y, w, h = rect # Assuming rect returns a tuple (x,y,width,height)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # Green color for rectangle, thickness 2

    # Display the resulting frame
    cv2.imshow('output', frame)

    # If 'q' is pressed, break from the loop
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
