import cv2
import numpy as np
import easyocr
import threading

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True)  # Use GPU if available for better performance

# Define valid Rwandan Franc (RWF) notes
VALID_NOTES = {'500', '1000', '2000', '5000'}

def extract_rw_notes(frame):
    """Detect Rwandan Franc notes in a given video frame."""
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding for better contrast in varying lighting conditions
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rwf_notes = []

    for contour in contours:
        if cv2.contourArea(contour) < 500:  # Ignore small contours
            continue

        # Approximate the contour
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

        if len(approx) == 4:  # If the shape is rectangular
            x, y, w, h = cv2.boundingRect(approx)
            roi = gray[y:y+h, x:x+w]  # Extract region of interest

            # Use EasyOCR in a separate thread
            ocr_thread = threading.Thread(target=process_ocr, args=(roi, rwf_notes, frame, x, y, w, h))
            ocr_thread.start()
            ocr_thread.join()

    return frame, rwf_notes

def process_ocr(roi, rwf_notes, frame, x, y, w, h):
    """Process OCR in a separate thread for efficiency."""
    result = reader.readtext(roi)
    for (_, text, _) in result:
        text = text.replace(',', '').strip()  # Remove commas and whitespace
        if text in VALID_NOTES:
            rwf_notes.append(int(text))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Start webcam capture
cap = cv2.VideoCapture(0)  # Change index if multiple cameras are present

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame, detected_notes = extract_rw_notes(frame)
    print("Detected Notes:", detected_notes)

    # Display the frame
    cv2.imshow("RWF Note Detector", processed_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
