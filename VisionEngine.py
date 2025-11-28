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
        # Default to 'blend' so the camera is visible by default; press 'u' to toggle modes.
        self.ui_mode = 'blend'

        # Left-hand drag-and-drop state
        self.left_drag_active = False
        self.left_drag_song_index = None
        self.left_drag_song_name = None

        # Track previous pinch state manually to fix drop detection bug
        self.prev_left_pinch = False

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
                for hand_lms, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_label = hand_info.classification[0].label

                    # Fix: Pass hand_lms.landmark (the list) instead of hand_lms (the object)
                    if hand_label == 'Left':
                        self.left_hand.set_landmarks(hand_lms.landmark)
                    elif hand_label == 'Right':
                        self.right_hand.set_landmarks(hand_lms.landmark)

            left_gestures = self.left_hand.detect_gestures()
            right_gestures = self.right_hand.detect_gestures()

            self.handle_left_drag_drop()

            if self.left_hand.landmarks:
                self.mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[
                    0] if results.multi_hand_landmarks else None, self.mp_hands.HAND_CONNECTIONS)
            if self.right_hand.landmarks:
                # Note: This logic for drawing is simplified based on your snippet;
                # ideally we iterate results.multi_hand_landmarks to draw, but keeping structure as requested.
                pass

            # Drawing landmarks using the standard loop to ensure visual feedback
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)

            img = self.ui.draw(
                self.ui.deck1_songs,
                self.ui.deck2_songs,
                self.deck1_current_song,
                self.deck2_current_song
            )
            frame_height, frame_width = frame.shape[:2]
            ui_resized = cv2.resize(img, (frame_width, frame_height))

            if self.ui_mode == 'blend':
                final = cv2.addWeighted(frame, 0.4, ui_resized, 0.6, 0)
            elif self.ui_mode == 'opaque':
                final = ui_resized.copy()
            else:
                final = frame

            cv2.putText(final, f'UI mode: {self.ui_mode}  (press u to toggle)', (10, final.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Stiwi Pro", final)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                self.running = False
            elif key == ord('u'):
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

    def _is_position_over_song(self, hand_pos, deck_num=1):
        """
        Check if a normalized hand position (x, y) is over a song in a deck list.
        Returns the song index if over a song, else None.
        Deck coordinates are in normalized [0, 1] space to match hand landmarks.
        """
        if hand_pos is None:
            return None

        hand_x, hand_y = hand_pos

        if deck_num == 1:
            songs = self.ui.deck1_songs
            deck_rect = self.ui.deck1_rect
        else:
            songs = self.ui.deck2_songs
            deck_rect = self.ui.deck2_rect

        # Get deck position and compute list area (offset down from header)
        d_x, d_y, d_w, d_h = deck_rect
        # Normalize deck coords to [0, 1] (assuming frame width/height == UI width/height)
        norm_x = d_x / self.ui.width
        norm_y = d_y / self.ui.height
        norm_w = d_w / self.ui.width
        norm_h = d_h / self.ui.height

        # List starts at offset 100px below header (approximate; tune if needed)
        list_offset_px = 100
        list_offset = list_offset_px / self.ui.height
        list_norm_y = norm_y + list_offset
        list_norm_h = norm_h - list_offset

        # Check if hand is within deck horizontal bounds
        if not (norm_x <= hand_x <= norm_x + norm_w):
            return None

        # Check if hand is within list vertical bounds
        if not (list_norm_y <= hand_y <= list_norm_y + list_norm_h):
            return None

        # Compute which song is under the hand based on y position
        if list_norm_h <= 0 or len(songs) == 0:
            return None

        # Item height in normalized coords
        item_norm_h = (self.ui.list_item_height) / self.ui.height

        # Which item within the visible list?
        y_offset_in_list = hand_y - list_norm_y
        item_index = int(y_offset_in_list / item_norm_h)

        # Clamp to visible list
        visible_items = max(1, int(list_norm_h / item_norm_h))
        if item_index < 0 or item_index >= visible_items:
            return None

        # Actual song index (accounting for scroll)
        song_index = self.ui.deck1_scroll + item_index if deck_num == 1 else self.ui.deck2_scroll + item_index

        if song_index >= len(songs):
            return None

        return song_index

    def handle_left_drag_drop(self):
        """
        Handle left-hand drag-and-drop for selecting and loading songs.
        Pinch to start drag, release to drop and load song to Deck 1.
        """
        if self.left_hand.landmarks is None:
            self.prev_left_pinch = False  # Reset state if hand lost
            return

        hand_pos = self.left_hand.get_pinch_position()

        # Manual edge detection to bypass side-effects in LeftHand class methods
        is_pinching = self.left_hand.is_currently_pinching
        pinch_started = is_pinching and not self.prev_left_pinch
        pinch_released = not is_pinching and self.prev_left_pinch

        # Update state for next frame
        self.prev_left_pinch = is_pinching

        if pinch_started:
            # Pinch just started: check if hand is over a song in deck1 list
            song_idx = self._is_position_over_song(hand_pos, deck_num=1)
            if song_idx is not None and song_idx < len(self.ui.deck1_songs):
                self.left_drag_active = True
                self.left_drag_song_index = song_idx
                self.left_drag_song_name = self.ui.deck1_songs[song_idx]
                print(f"Left hand: started dragging song '{self.left_drag_song_name}' (index {song_idx})")

        if is_pinching and self.left_drag_active:
            # While pinching and dragging, update UI drag position
            if hand_pos:
                # Convert normalized coords to screen pixels
                screen_x = int(hand_pos[0] * self.ui.width)
                screen_y = int(hand_pos[1] * self.ui.height)
                self.ui.update_drag((screen_x, screen_y))

        if pinch_released and self.left_drag_active:
            # Pinch released: drop song and load it to Deck 1
            if self.left_drag_song_name:
                song_info = next((s for s in self.song_list if s['name'] == self.left_drag_song_name), None)
                if song_info:
                    print(f"Left hand: dropped and loading song '{self.left_drag_song_name}'")
                    if self.audio_engine:
                        self.audio_engine.stop()
                    self.load_song_to_audio(song_info['path'], deck=1)
                    self.deck1_current_song = self.left_drag_song_name
                    self.deck1_current_path = song_info['path']

            # Reset drag state
            self.left_drag_active = False
            self.left_drag_song_index = None
            self.left_drag_song_name = None
            self.ui.dragging_song = None