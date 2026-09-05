from flask import Flask, request, jsonify
import cv2
import numpy as np

# This imports your exact script! 
import intelliplay_ED 

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_frame():
    # 1. Check if the Flutter app actually sent an image
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    
    # 2. Convert the incoming web image into an OpenCV format
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # 3. Send the image to your Master Attention function!
    # We use _, _ to ignore the mesh and debug data since the app only needs the label
    emotion_label, _, _ = intelliplay_ED.analyze_face(frame)

    # 4. Send the result back to Flutter as a JSON response
    return jsonify({
        "status": "success",
        "emotion": emotion_label
    })
@app.route('/reset_calibration', methods=['POST'])
def reset_calibration():
    intelliplay_ED.calibration.reset()
    intelliplay_ED.smile_buffer.clear()
    intelliplay_ED.gaze_buffer.clear()
    return jsonify({"status": "reset"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)