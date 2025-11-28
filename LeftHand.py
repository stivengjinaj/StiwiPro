import mediapipe as mp

class LeftHand:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = None
        self.landmarks = None
        self.gestures = {
            # Gestures
        }

    def set_landmarks(self, landmarks):
        """Assign Mediapipe hand landmarks (or None)."""
        self.landmarks = landmarks

    def detect_gestures(self):
        detected = []
        if self.landmarks:
            for name, func in self.gestures.items():
                if func():
                    detected.append(name)
        return detected

    def is_pinch_middle_thumb(self):
        return False