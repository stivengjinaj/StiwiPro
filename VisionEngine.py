import cv2
import mediapipe as mp

from AudioEngine import AudioEngine
from LeftHand import LeftHand
from RightHand import RightHand
from vision_helpers import is_position_over_song, is_position_over_play_button


class VisionEngine:
    def __init__(self, audio_engine_left, audio_engine_right, ui, song_list):
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
        self.audio_engine_left = audio_engine_left
        self.audio_engine_right = audio_engine_right
        self.running = True
        self.ui = ui
        self.song_list = song_list
        self.deck1_current_song = None
        self.deck2_current_song = None
        self.deck1_current_path = None
        self.deck2_current_path = None

        self.left_drag_active = False
        self.right_drag_active = False
        self.left_drag_song_index = None
        self.right_drag_song_index = None
        self.left_drag_song_name = None
        self.right_drag_song_name = None

        self.prev_left_pinch = False
        self.prev_right_pinch = False
        self.prev_left_pinch_for_play = False
        self.prev_right_pinch_for_play = False
        self.is_playing_left = False
        self.is_playing_right = False

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

                    if hand_label == 'Left':
                        self.left_hand.set_landmarks(hand_lms.landmark)
                    elif hand_label == 'Right':
                        self.right_hand.set_landmarks(hand_lms.landmark)

            left_gestures = self.left_hand.detect_gestures()
            right_gestures = self.right_hand.detect_gestures()

            self.handle_play_pause()
            self.handle_left_hover()
            self.handle_right_hover()
            self.handle_drag_drop(self.left_hand, 1, 'left')
            self.handle_drag_drop(self.right_hand, 2, 'right')

            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)

            if self.audio_engine_left:
                self.is_playing_left = not self.audio_engine_left.is_paused
            else:
                self.is_playing_left = False

            if self.audio_engine_right:
                self.is_playing_right = not self.audio_engine_right.is_paused
            else:
                self.is_playing_right = False

            img = self.ui.draw(
                self.ui.deck1_songs,
                self.ui.deck2_songs,
                self.deck1_current_song,
                self.deck2_current_song,
                self.is_playing_left,
                self.is_playing_right
            )
            frame_height, frame_width = frame.shape[:2]
            ui_resized = cv2.resize(img, (frame_width, frame_height))

            final = cv2.addWeighted(frame, 0.4, ui_resized, 0.6, 0)
            cv2.imshow("Stiwi Pro", final)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                self.running = False

        self.cap.release()
        cv2.destroyAllWindows()

    def load_song_to_audio(self, song_path, deck=1):
        if deck == 1:
            if self.audio_engine_left:
                self.audio_engine_left.stop()
            self.audio_engine_left = AudioEngine(song_path)
            self.audio_engine_left.start()
        else:
            if self.audio_engine_right:
                self.audio_engine_right.stop()
            self.audio_engine_right = AudioEngine(song_path)
            self.audio_engine_right.start()

    def handle_left_hover(self):
        if self.left_hand.landmarks is None:
            return

        hover_pos = self.left_hand.get_index_tip_position()
        if hover_pos is None:
            return

        song_idx = is_position_over_song(hover_pos, self.ui, deck_num=1)

        if song_idx is not None and song_idx < len(self.ui.deck1_songs):
            self.ui.selected_song_deck1 = song_idx

    def handle_right_hover(self):
        if self.right_hand.landmarks is None:
            return

        hover_pos = self.right_hand.get_index_tip_position()
        if hover_pos is None:
            return

        song_idx = is_position_over_song(hover_pos, self.ui, deck_num=2)

        if song_idx is not None and song_idx < len(self.ui.deck2_songs):
            self.ui.selected_song_deck2 = song_idx

    def handle_play_pause(self):
        right_pinch_pos = self.right_hand.get_pinch_position() if self.right_hand.landmarks else None
        left_pinch_pos = self.left_hand.get_pinch_position() if self.left_hand.landmarks else None

        is_pinching_right = self.right_hand.is_currently_pinching if self.right_hand.landmarks else False
        is_pinching_left = self.left_hand.is_currently_pinching if self.left_hand.landmarks else False

        pinch_started_right = is_pinching_right and not self.prev_right_pinch_for_play
        pinch_started_left = is_pinching_left and not self.prev_left_pinch_for_play

        self.prev_right_pinch_for_play = is_pinching_right
        self.prev_left_pinch_for_play = is_pinching_left

        button_right = is_position_over_play_button(right_pinch_pos, self.ui)
        button_left = is_position_over_play_button(left_pinch_pos, self.ui)

        if pinch_started_right and not self.right_drag_active:
            if button_right == 'left' and self.audio_engine_left:
                self.audio_engine_left.toggle_playback()
                self.is_playing_left = not self.audio_engine_left.is_paused
            elif button_right == 'right' and self.audio_engine_right:
                self.audio_engine_right.toggle_playback()
                self.is_playing_right = not self.audio_engine_right.is_paused

        if pinch_started_left and not self.left_drag_active:
            if button_left == 'left' and self.audio_engine_left:
                self.audio_engine_left.toggle_playback()
                self.is_playing_left = not self.audio_engine_left.is_paused
            elif button_left == 'right' and self.audio_engine_right:
                self.audio_engine_right.toggle_playback()
                self.is_playing_right = not self.audio_engine_right.is_paused

    def handle_drag_drop(self, hand, deck_num, hand_name):
        if hand_name == 'left':
            drag_active = self.left_drag_active
            drag_song_name = self.left_drag_song_name
            drag_song_index = self.left_drag_song_index
            prev_pinch = self.prev_left_pinch
        else:
            drag_active = self.right_drag_active
            drag_song_name = self.right_drag_song_name
            drag_song_index = self.right_drag_song_index
            prev_pinch = self.prev_right_pinch

        if hand.landmarks is None:
            if hand_name == 'left':
                self.prev_left_pinch = False
                self.left_drag_active = False
                self.left_drag_song_index = None
                self.left_drag_song_name = None
            else:
                self.prev_right_pinch = False
                self.right_drag_active = False
                self.right_drag_song_index = None
                self.right_drag_song_name = None
            self.ui.dragging_song = None
            self.ui.dragging_from_deck = None
            return

        hand_pos = hand.get_pinch_position()

        is_pinching = hand.is_currently_pinching
        pinch_started = is_pinching and not prev_pinch
        pinch_released = not is_pinching and prev_pinch

        if hand_name == 'left':
            self.prev_left_pinch = is_pinching
        else:
            self.prev_right_pinch = is_pinching

        if pinch_started:
            over_play_button = is_position_over_play_button(hand_pos, self.ui)
            if over_play_button:
                return

            song_idx = is_position_over_song(hand_pos, self.ui, deck_num=deck_num)
            songs = self.ui.deck1_songs if deck_num == 1 else self.ui.deck2_songs
            if song_idx is not None and song_idx < len(songs):
                if hand_name == 'left':
                    self.left_drag_active = True
                    self.left_drag_song_index = song_idx
                    self.left_drag_song_name = songs[song_idx]
                    self.ui.selected_song_deck1 = song_idx
                else:
                    self.right_drag_active = True
                    self.right_drag_song_index = song_idx
                    self.right_drag_song_name = songs[song_idx]
                    self.ui.selected_song_deck2 = song_idx

                if hand_pos:
                    screen_x = int(hand_pos[0] * self.ui.width)
                    screen_y = int(hand_pos[1] * self.ui.height)
                    self.ui.drag_song(deck_num, song_idx, (screen_x, screen_y))

        if is_pinching and drag_active:
            if hand_pos:
                screen_x = int(hand_pos[0] * self.ui.width)
                screen_y = int(hand_pos[1] * self.ui.height)
                self.ui.update_drag((screen_x, screen_y))
                if not self.ui.dragging_song:
                    drag_name = self.left_drag_song_name if hand_name == 'left' else self.right_drag_song_name
                    if drag_name:
                        self.ui.dragging_song = drag_name

        if pinch_released and drag_active:
            if drag_song_name:
                cx, cy, cw, ch = self.ui.center_decks_rect
                drop_x, drop_y = self.ui.dragging_position

                in_center = (cx <= drop_x <= cx + cw) and (cy <= drop_y <= cy + ch)

                if in_center:
                    song_info = next(
                        (s for s in self.song_list if s['name'] == drag_song_name),
                        None
                    )
                    if song_info:
                        self.load_song_to_audio(song_info['path'], deck=deck_num)
                        if deck_num == 1:
                            self.deck1_current_song = drag_song_name
                            self.deck1_current_path = song_info['path']
                        else:
                            self.deck2_current_song = drag_song_name
                            self.deck2_current_path = song_info['path']

            if hand_name == 'left':
                self.left_drag_active = False
                self.left_drag_song_index = None
                self.left_drag_song_name = None
            else:
                self.right_drag_active = False
                self.right_drag_song_index = None
                self.right_drag_song_name = None
            self.ui.dragging_song = None
            self.ui.dragging_from_deck = None
            self.ui.dragging_position = (0, 0)

