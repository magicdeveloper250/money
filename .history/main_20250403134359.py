import cv2
import numpy as np
import easyocr

def extract_rw_fnote_number(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    # Approximate the contour to get a quadrilateral
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) == 4:
        # Arrange points for perspective transformation
        pts = approx.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # Define the transformed image size
        (tl, tr, br, bl) = rect
        width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (width, height))

        # Convert to grayscale for OCR
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        # OCR using EasyOCR
        reader = easyocr.Reader(['en'])
        results = reader.readtext(warped_gray)

        # Filter the extracted text for RWF serial number (usually a mix of letters and digits)
        serial_numbers = [text[1] for text in results if len(text[1]) >= 6]

        return serial_numbers, warped

    return None, None

# Example usage
image_path = "rwf_banknote.jpg"
serial_numbers, processed_image = extract_rw_fnote_number(image_path)

if serial_numbers:
    print("Extracted Serial Numbers:", serial_numbers)
else:
    print("No serial numbers detected.")
