import cv2
import numpy as np
import easyocr

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def extract_rw_notes(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply threshold to detect rectangles
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rwf_notes = []
    valid_notes = {'500', '1000', '2000', '5000'}

    for contour in contours:
        # Approximate the contour
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

        if len(approx) == 4:  # Check if the shape is rectangular
            x, y, w, h = cv2.boundingRect(approx)
            roi = gray[y:y+h, x:x+w]  # Extract the region of interest

            # Use EasyOCR to extract text
            result = reader.readtext(roi)
            for (bbox, text, prob) in result:
                text = text.replace(',', '')  # Remove potential comma separators
                if text in valid_notes:
                    rwf_notes.append(int(text))
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle
                    cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show detected notes
    cv2.imshow("Detected Notes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return rwf_notes

# Run the function on an image
image_path = "notes.jpg"  # Change to your image path
detected_notes = extract_rw_notes(image_path)
print("Extracted RWF Notes:", detected_notes)
