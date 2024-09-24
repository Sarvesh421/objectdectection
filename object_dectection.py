import cv2

# Load the video and Haar cascade
cap = cv2.VideoCapture(r'C:\Users\sarve\OneDrive\Documents\python\python\holo1.mp4')
human_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        break  # Exit loop if no more frames

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect humans
    humans = human_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected humans
    for (x, y, w, h) in humans:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with detections
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

