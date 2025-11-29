import mediapipe as mp
import numpy as np

class LeftHand:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = None
        self.landmarks = None
        self.gestures = {
            'pinch_thumb_index': self.is_pinch_thumb_index,
            'pinch_middle_thumb': self.is_pinch_middle_thumb,
        }
        # Track previous pinch state for edge detection (rising edge = start pinch)
        self.was_pinching = False
        self.is_currently_pinching = False

    def set_landmarks(self, landmarks):
        """Assign Mediapipe hand landmarks (or None)."""
        self.landmarks = landmarks
        self.is_currently_pinching = self.is_pinch_thumb_index()

    def _get_landmark(self, index):
        """Get a landmark by index; returns (x, y, z) or None if not available."""
        if self.landmarks is None:
            return None
        try:
            lm = self.landmarks[index]
            return lm.x, lm.y, lm.z
        except (IndexError, AttributeError):
            return None

    def get_index_tip_position(self):
        """
        Get the position of the index fingertip (landmark 8).
        Returns (x, y) in normalized coordinates [0, 1], or None if unavailable.
        """
        index_tip = self.landmarks[8]
        return index_tip.x, index_tip.y

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
        return self._is_pinch(4, 8, threshold=0.07)

    def is_pinch_middle_thumb(self):
        """Detect pinch between thumb and middle finger."""
        return self._is_pinch(4, 12, threshold=0.07)

    def is_pinch_started(self):
        """Detect rising edge: pinch just started (was not pinching, now is)."""
        result = not self.was_pinching and self.is_currently_pinching
        self.was_pinching = self.is_currently_pinching
        return result

    def is_pinch_released(self):
        """Detect falling edge: pinch just ended (was pinching, now is not)."""
        result = self.was_pinching and not self.is_currently_pinching
        self.was_pinching = self.is_currently_pinching
        return result

    def get_hand_position(self):
        """
        Get the position of the hand (using middle fingertip as representative point).
        Returns (x, y) in normalized coordinates [0, 1], or None if landmarks unavailable.
        """
        if self.landmarks is None:
            return None
        try:
            middle_tip = self.landmarks[12]
            return middle_tip.x, middle_tip.y
        except (IndexError, AttributeError):
            return None

    def get_pinch_position(self):
        """
        Get the position of the pinch point (midpoint between thumb and index).
        Returns (x, y) in normalized coordinates [0, 1], or None if landmarks unavailable.
        """
        thumb = self._get_landmark(4)
        index = self._get_landmark(8)
        if thumb is None or index is None:
            return None
        # Midpoint
        return (thumb[0] + index[0]) / 2, (thumb[1] + index[1]) / 2

    def detect_gestures(self):
        detected = []
        if self.landmarks:
            for name, func in self.gestures.items():
                if func():
                    detected.append(name)
        return detected
