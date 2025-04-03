from flask import Flask, request, jsonify
import cv2
import numpy as np
import easyocr
import io

app = Flask(__name__)

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
    processed_frame = frame.copy()

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
                    cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(processed_frame, text, (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return processed_frame, rwf_notes

@app.route('/api/detect', methods=['POST'])
def detect_notes():
    if 'file' not in request.files and request.content_type == 'application/octet-stream':
        # Handle raw binary data from ESP32
        frame_data = request.data
        if not frame_data:
            return jsonify({"error": "No frame data received"}), 400
        
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return jsonify({"error": "Failed to decode image"}), 400
                
            # Process the frame
            processed_frame, detected_notes = extract_rw_notes(frame)
            
            # Determine response
            if detected_notes:
                response = {
                    "status": "positive",
                    "notes": detected_notes,
                    "message": f"Detected RWF notes: {', '.join(map(str, detected_notes))}"
                }
            else:
                response = {
                    "status": "negative",
                    "notes": [],
                    "message": "No valid RWF notes detected"
                }
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "Unsupported content type"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)