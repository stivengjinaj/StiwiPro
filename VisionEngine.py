import cv2
import mediapipe as mp

from AudioEngine import AudioEngine
from LeftHand import LeftHand
from RightHand import RightHand


class VisionEngine:
    def __init__(self, audio_engine, ui, song_list):
        self.cap = cv2.VideoCapture(0)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands_processor = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.left_hand = LeftHand()
        self.right_hand = RightHand()
        self.audio_engine = audio_engine
        self.running = True
        self.ui = ui
        self.song_list = song_list
        self.deck1_current_song = None
        self.deck2_current_song = None
        self.deck1_current_path = None
        self.deck2_current_path = None
        # UI overlay mode: 'blend', 'opaque' (show only UI), 'hidden' (show only camera)
        self.ui_mode = 'blend'

    def process(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands_processor.process(frame_rgb)

            self.left_hand.set_landmarks(None)
            self.right_hand.set_landmarks(None)

            if results.multi_handedness and results.multi_hand_landmarks:
                for idx, hand_info in enumerate(results.multi_handedness):
                    label = hand_info.classification[0].label
                    landmarks = results.multi_hand_landmarks[idx]
                    if label == 'Left':
                        self.left_hand.set_landmarks(landmarks)
                    elif label == 'Right':
                        self.right_hand.set_landmarks(landmarks)

            left_gestures = self.left_hand.detect_gestures()
            right_gestures = self.right_hand.detect_gestures()

            if self.left_hand.landmarks:
                self.mp_drawing.draw_landmarks(frame, self.left_hand.landmarks, self.mp_hands.HAND_CONNECTIONS)
            if self.right_hand.landmarks:
                self.mp_drawing.draw_landmarks(frame, self.right_hand.landmarks, self.mp_hands.HAND_CONNECTIONS)

            img = self.ui.draw(
                self.ui.deck1_songs,
                self.ui.deck2_songs,
                self.deck1_current_song,
                self.deck2_current_song
            )
            frame_height, frame_width = frame.shape[:2]
            ui_resized = cv2.resize(img, (frame_width, frame_height))

            # Compose final image depending on ui_mode
            if self.ui_mode == 'blend':
                # blend UI over frame
                final = cv2.addWeighted(frame, 0.4, ui_resized, 0.6, 0)
            elif self.ui_mode == 'opaque':
                # show UI only (opaque) — useful for testing visibility
                final = ui_resized.copy()
            else:  # 'hidden'
                final = frame

            cv2.putText(final, f'UI mode: {self.ui_mode}  (press u to toggle)', (10, final.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Stiwi Pro", final)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                self.running = False
            elif key == ord('u'):
                # Cycle UI mode
                if self.ui_mode == 'blend':
                    self.ui_mode = 'opaque'
                elif self.ui_mode == 'opaque':
                    self.ui_mode = 'hidden'
                else:
                    self.ui_mode = 'blend'

        self.cap.release()
        cv2.destroyAllWindows()

    def load_song_to_audio(self, song_path, deck=1):
        if self.audio_engine:
            self.audio_engine.stop()
        self.audio_engine = AudioEngine(song_path)
        self.audio_engine.start()

    def handle_gestures(self, left_gestures, right_gestures):
        if 'pinch_index_thumb' in right_gestures:
            self.audio_engine.set_pitch(0.8)
        else:
            self.audio_engine.set_pitch(0.5)

        if 'pinch_middle_thumb' in left_gestures:
            self.audio_engine.set_reverb(0.5)
        else:
            self.audio_engine.set_reverb(0.0)

        drop_result = self.ui.drop_song("deck1")
        if drop_result:
            song_name, from_deck, target_deck = drop_result
            song_info = next((s for s in self.song_list if s['name'] == song_name), None)
            if song_info:
                if target_deck == "deck1":
                    self.deck1_current_song = song_name
                    self.deck1_current_path = song_info['path']
                elif target_deck == "deck2":
                    self.deck2_current_song = song_name
                    self.deck2_current_path = song_info['path']