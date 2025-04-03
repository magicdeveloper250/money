import cv2
import numpy as np
import easyocr

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Define valid Rwandan Franc (RWF) notes
valid_notes = {'500', '1000', '2000', '5000'}

def extract_rw_notes(frame):
    """Detect Rwandan Franc notes in a given video frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rwf_notes = []

    for contour in contours:
        # Approximate the contour
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

        if len(approx) == 4:  # If the shape is rectangular
            x, y, w, h = cv2.boundingRect(approx)
            roi = gray[y:y+h, x:x+w]  # Extract region of interest

            # Use EasyOCR to extract text
            result = reader.readtext(roi)
            for (bbox, text, prob) in result:
                text = text.replace(',', '')  # Remove comma separators
                if text in valid_notes:
                    rwf_notes.append(int(text))
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle
                    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame, rwf_notes


# Start webcam capture
cap = cv2.VideoCapture(0)  # Change index if you have multiple cameras

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame, detected_notes = extract_rw_notes(frame)

    # Display the frame
    cv2.imshow("RWF Note Detector", processed_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
