import mediapipe as mp
import numpy as np


class RightHand:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = None
        self.landmarks = None
        self.gestures = {
            # gestures
        }
        self.is_currently_pinching = False

    def set_landmarks(self, landmarks):
        """Assign Mediapipe hand landmarks (or None)."""
        self.landmarks = landmarks
        self.is_currently_pinching = self.is_pinch_thumb_index()

    def get_pinch_position(self):
        thumb = self._get_landmark(4)
        index = self._get_landmark(8)
        if thumb is None or index is None:
            return None
        # Midpoint
        return (thumb[0] + index[0]) / 2, (thumb[1] + index[1]) / 2

    def _get_landmark(self, index):
        """Get a landmark by index; returns (x, y, z) or None if not available."""
        if self.landmarks is None:
            return None
        try:
            lm = self.landmarks[index]
            return lm.x, lm.y, lm.z
        except (IndexError, AttributeError):
            return None

    def _distance(self, lm1, lm2):
        """Compute Euclidean distance between two landmarks."""
        if lm1 is None or lm2 is None:
            return float('inf')
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(lm1, lm2)))

    def _is_pinch(self, thumb_idx, finger_idx, threshold=0.05):
        """Check if thumb and finger are pinched together (distance < threshold)."""
        thumb = self._get_landmark(thumb_idx)
        finger = self._get_landmark(finger_idx)
        distance = self._distance(thumb, finger)
        return distance < threshold

    def is_pinch_thumb_index(self):
        """Detect pinch between thumb and index finger."""
        return self._is_pinch(4, 8, threshold=0.05)

    def detect_gestures(self):
        detected = []
        if self.landmarks:
            for name, func in self.gestures.items():
                if func():
                    detected.append(name)
        return detected

    def is_pinch_index_thumb(self):
        return False