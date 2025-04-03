import cv2
import numpy as np
import easyocr

# Initialize EasyOCR reader once (outside the function)
reader = easyocr.Reader(['en'], gpu=True)  # Enable GPU if available

# Define valid Rwandan Franc (RWF) notes as integers for faster comparison
VALID_NOTES = {500, 1000, 2000, 5000}
VALID_NOTES_STR = {'500', '1000', '2000', '5000'}

# Predefined constants to avoid recreating them in each frame
THRESHOLD_TYPE = cv2.THRESH_BINARY_INV
CONTOUR_MODE = cv2.RETR_EXTERNAL
CONTOUR_METHOD = cv2.CHAIN_APPROX_SIMPLE
RECTANGLE_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 255, 0)

def extract_rw_notes(frame):
    """Optimized function to detect Rwandan Franc notes in a video frame."""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding for better performance in varying light
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  THRESHOLD_TYPE, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, CONTOUR_MODE, CONTOUR_METHOD)
    
    rwf_notes = []
    processed_rois = set()  # To avoid processing the same note multiple times
    
    for contour in contours:
        # Skip small contours that can't be notes
        if cv2.contourArea(contour) < 1000:
            continue
            
        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 4:  # If the shape is rectangular
            x, y, w, h = cv2.boundingRect(approx)
            
            # Skip very small or very large regions
            if w < 50 or h < 50 or w > frame.shape[1]//2 or h > frame.shape[0]//2:
                continue
                
            # Create a unique identifier for this ROI to avoid duplicate processing
            roi_id = f"{x}_{y}_{w}_{h}"
            if roi_id in processed_rois:
                continue
            processed_rois.add(roi_id)
            
            # Extract region of interest with padding
            padding = 5
            y1, y2 = max(0, y-padding), min(frame.shape[0], y+h+padding)
            x1, x2 = max(0, x-padding), min(frame.shape[1], x+w+padding)
            roi = gray[y1:y2, x1:x2]
            
            # Use EasyOCR to extract text
            results = reader.readtext(roi, detail=1, paragraph=False)
            
            for (bbox, text, prob) in results:
                # Only process high confidence results
                if prob < 0.7:
                    continue
                    
                text = text.replace(',', '').replace('.', '').strip()
                
                if text in VALID_NOTES_STR:
                    note_value = int(text)
                    rwf_notes.append(note_value)
                    
                    # Draw rectangle and text
                    cv2.rectangle(frame, (x, y), (x + w, y + h), RECTANGLE_COLOR, 2)
                    cv2.putText(frame, text, (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, TEXT_COLOR, 2)
    
    return frame, rwf_notes

def main():
    # Start webcam capture with optimized settings
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Lower resolution for faster processing
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for less processing load
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame, detected_notes = extract_rw_notes(frame)
            
            if detected_notes:
                print(f"Detected notes: {detected_notes}")
            
            # Display the frame
            cv2.imshow("RWF Note Detector", processed_frame)
            
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()