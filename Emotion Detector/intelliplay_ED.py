import os
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
import numpy as np
from collections import deque

# ==========================================
# FACE LANDMARKER SETUP (MediaPipe Tasks API)
# ==========================================
# NOTE: As of mediapipe 0.10.30, Google removed the old `mp.solutions` API
# (FaceMesh, drawing_utils, etc.) entirely. This now uses the replacement
# Tasks API instead. Functionally equivalent for our purposes:
#   - The face_landmarker.task model returns 478 landmarks per face
#     (468 base mesh points + 10 iris points), same as the old
#     refine_landmarks=True did. All landmark INDICES below are unchanged.
#   - RunningMode.IMAGE is used (treats each frame independently) because
#     this same function is called both from a live webcam loop AND from
#     stateless one-off HTTP requests in server.py, which can't guarantee
#     the strictly increasing timestamps that VIDEO/LIVE_STREAM mode needs.
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Missing model file: {MODEL_PATH}\n"
        "Download it with:\n"
        "  wget -O face_landmarker.task "
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    )

_landmarker_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_faces=1,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
)
face_landmarker = FaceLandmarker.create_from_options(_landmarker_options)

# Smoothing buffers - stores last 5 values to reduce single-frame noise
smile_buffer = deque(maxlen=5)
gaze_buffer = deque(maxlen=5)


class Calibration:
    """
    Captures a short neutral baseline (gaze ratio + smile indicator) for the
    current child before classifying anything. This matters because:
      - "Center gaze" is not exactly 0.5 for every eye shape/camera angle.
      - "Neutral mouth" is not exactly 0.0 for every face.
    Classifying against a fixed absolute number is what caused both bugs
    you ran into. Classifying against THIS child's own baseline fixes it.
    """
    def __init__(self, frames_needed=30):
        self.frames_needed = frames_needed
        self.gaze_samples = []
        self.smile_samples = []
        self.done = False
        self.baseline_gaze = 0.5
        self.baseline_smile = 0.0

    def reset(self):
        self.gaze_samples.clear()
        self.smile_samples.clear()
        self.done = False

    def add_sample(self, gaze_ratio, smile_indicator):
        if self.done:
            return
        self.gaze_samples.append(gaze_ratio)
        self.smile_samples.append(smile_indicator)
        if len(self.gaze_samples) >= self.frames_needed:
            # median, not mean - resists getting thrown off by a single
            # blink or jitter frame during the calibration window
            self.baseline_gaze = float(np.median(self.gaze_samples))
            self.baseline_smile = float(np.median(self.smile_samples))
            self.done = True


calibration = Calibration()

# --- Tunable thresholds ---
# These are starting points, not final answers. Run the demo, watch the
# debug overlay (gaze/smile deviation numbers), and adjust these until the
# label flips right around where it actually should.
GAZE_AWAY_DEVIATION = 0.12      # how far the iris ratio can drift from YOUR baseline before "looking away"
SMILE_HAPPY_DEVIATION = 0.0035  # face-width-normalized, so it holds regardless of distance from camera
FRUSTRATED_DEVIATION = -0.012   # must be clearly below baseline - widened from -0.006 since normal
                                # neutral-face jitter was tripping this too easily. Watch the on-screen
                                # "smile dev" debug number while relaxed vs. deliberately frowning, and
                                # set this just below your own relaxed-face dev value.
EYE_HEIGHT_NORM_THRESHOLD = 0.045  # normalized by face width, replaces the old fixed 0.012


def analyze_face(image):
    """
    Analyzes a single frame for distance, focus, gaze, and emotion.
    """
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    result = face_landmarker.detect(mp_image)

    if not result.face_landmarks:
        return "NO FACE DETECTED", None, None

    for landmarks in result.face_landmarks:
        # `landmarks` is already a flat list of landmark points (each with
        # .x/.y/.z) - equivalent to the old `face_landmarks.landmark`.

        # --- 1. CALCULATE ALL RAW METRICS FIRST ---

        # Distance
        left_side_x = landmarks[234].x
        right_side_x = landmarks[454].x
        face_width = max(abs(right_side_x - left_side_x), 0.0001)

        # Head Pose
        nose_x = landmarks[1].x

        # Eye Height
        left_eye_top_y = landmarks[159].y
        left_eye_bottom_y = landmarks[145].y
        right_eye_top_y = landmarks[386].y
        right_eye_bottom_y = landmarks[374].y
        average_eye_height = ((left_eye_bottom_y - left_eye_top_y) + (right_eye_bottom_y - right_eye_top_y)) / 2.0
        normalized_eye_height = average_eye_height / face_width

        # Gaze
        eye_inner_x = landmarks[133].x
        eye_outer_x = landmarks[33].x
        iris_x = landmarks[468].x
        eye_width = max(abs(eye_inner_x - eye_outer_x), 0.0001)
        raw_gaze_ratio = abs(iris_x - eye_outer_x) / eye_width
        gaze_buffer.append(raw_gaze_ratio)
        smoothed_gaze_ratio = sum(gaze_buffer) / len(gaze_buffer)

        # Smile
        mouth_top = landmarks[13].y
        mouth_bottom = landmarks[14].y
        mouth_left_y = landmarks[61].y
        mouth_right_y = landmarks[291].y
        mouth_center_y = (mouth_top + mouth_bottom) / 2
        raw_smile_indicator = (mouth_center_y - ((mouth_left_y + mouth_right_y) / 2)) / face_width
        smile_buffer.append(raw_smile_indicator)
        smoothed_smile_indicator = sum(smile_buffer) / len(smile_buffer)

        # --- 2. CALIBRATION GATE ---
        if not calibration.done:
            calibration.add_sample(smoothed_gaze_ratio, smoothed_smile_indicator)
            debug = {
                "gaze_ratio": smoothed_gaze_ratio,
                "baseline_gaze": calibration.baseline_gaze,
                "smile_indicator": smoothed_smile_indicator,
                "baseline_smile": calibration.baseline_smile,
                "eye_height_norm": normalized_eye_height,
            }
            return "CALIBRATING - LOOK AT SCREEN", result.face_landmarks, debug

        # --- 3. CALCULATE DEVIATIONS ---
        gaze_deviation = smoothed_gaze_ratio - calibration.baseline_gaze
        smile_deviation = smoothed_smile_indicator - calibration.baseline_smile

        is_smiling = smile_deviation > SMILE_HAPPY_DEVIATION

        debug = {
            "gaze_ratio": smoothed_gaze_ratio,
            "baseline_gaze": calibration.baseline_gaze,
            "gaze_deviation": gaze_deviation,
            "smile_indicator": smoothed_smile_indicator,
            "baseline_smile": calibration.baseline_smile,
            "smile_deviation": smile_deviation,
            "eye_height_norm": normalized_eye_height,
        }

        # --- 4. DECISION TREE (The Waterfall) ---

        # A. Screen Distance Check
        if face_width < 0.15:
            return "DISTRACTED - AWAY FROM SCREEN", result.face_landmarks, debug

        # B. Head Pose Check
        if nose_x < left_side_x + 0.05 or nose_x > right_side_x - 0.05:
            return "DISTRACTED - HEAD TURNED", result.face_landmarks, debug

        # C. Eye Drooping Check (With Smiling Forgiveness!)
        # If they are smiling, we lower the threshold so we don't punish them for squinting
        dynamic_eye_threshold = EYE_HEIGHT_NORM_THRESHOLD
        if is_smiling:
            dynamic_eye_threshold = 0.020  # Much more forgiving! (Normal is 0.045)

        if normalized_eye_height < dynamic_eye_threshold:
            return "UNFOCUSED - EYES DROOPING", result.face_landmarks, debug

        # D. Gaze Check
        if abs(gaze_deviation) > GAZE_AWAY_DEVIATION:
            return "DISTRACTED - LOOKING AWAY", result.face_landmarks, debug

        # E. Emotion Check
        if is_smiling:
            label = "FOCUSED - HAPPY"
        elif smile_deviation < FRUSTRATED_DEVIATION:
            label = "FOCUSED - FRUSTRATED"
        else:
            label = "FOCUSED - NEUTRAL"

        return label, result.face_landmarks, debug

    return "NO FACE DETECTED", None, None


def draw_face_points(frame, mesh_data, color=(0, 255, 0)):
    """
    Lightweight stand-in for the old mp.solutions.drawing_utils.draw_landmarks.
    The Tasks API dropped the drawing utilities along with the rest of
    `mp.solutions`, so this just plots each landmark as a small dot -
    enough to visually confirm tracking is working during local debugging.
    """
    h, w = frame.shape[:2]
    for landmarks in mesh_data:
        for point in landmarks:
            x, y = int(point.x * w), int(point.y * h)
            cv2.circle(frame, (x, y), 1, color, -1)


# ==========================================
# LOCAL TESTING WEBCAM LOOP
# ==========================================
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        # Run our Master Attention function
        emotion_label, mesh_data, debug = analyze_face(frame)

        # Visual feedback colors (Green = Good, Red = Distracted/Frustrated, Yellow = Calibrating)
        if "CALIBRATING" in emotion_label:
            color = (0, 255, 255)
        elif "DISTRACTED" in emotion_label or "UNFOCUSED" in emotion_label or "FRUSTRATED" in emotion_label:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        if mesh_data:
            draw_face_points(frame, mesh_data, color=(0, 255, 0))

        cv2.putText(frame, emotion_label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

        # Live numbers so you can SEE how close you are to each threshold
        # instead of guessing why a label didn't flip - watch these while
        # you look hard left/right or exaggerate a frown to find good values.
        if debug:
            line1 = f"gaze: {debug.get('gaze_ratio', 0):.3f}  base: {debug.get('baseline_gaze', 0):.3f}  dev: {debug.get('gaze_deviation', 0):.3f}"
            line2 = f"smile: {debug.get('smile_indicator', 0):.4f}  base: {debug.get('baseline_smile', 0):.4f}  dev: {debug.get('smile_deviation', 0):.4f}"
            cv2.putText(frame, line1, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(frame, line2, (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        cv2.putText(frame, "Press 'c' to recalibrate", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow('IntelliPlay MediaPipe Monitor', frame)
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            calibration.reset()
            smile_buffer.clear()
            gaze_buffer.clear()

    cap.release()
    cv2.destroyAllWindows()