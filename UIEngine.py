import cv2
import numpy as np

from ui_helpers import draw_play_button, draw_deck, draw_scrollable_list


class UIEngine:
    def __init__(self, window_name="Stiwi Pro", width=1280, height=720):
        self.window_name = window_name
        self.width = width
        self.height = height

        self.bg_color = (28, 28, 28)
        self.text_color = (245, 245, 245)
        self.highlight_color = (0, 200, 255)
        self.deck_bg_color = (60, 60, 60)
        self.selection_color = (0, 150, 255)
        self.scrollbar_color = (100, 100, 100)
        self.scrollbar_handle_color = (200, 200, 200)
        self.playing_color = (0, 200, 120)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        self.margin = 20
        self.deck_width = int(width * 0.3)
        self.deck_height = int(height * 0.4)
        self.center_width = int(width * 0.35)
        self.list_item_height = 30
        self.scrollbar_width = 15

        self.deck1_rect = (self.margin, self.margin, self.deck_width, self.deck_height)
        self.deck2_rect = (width - self.deck_width - self.margin, self.margin, self.deck_width, self.deck_height)
        self.center_decks_rect = (self.deck1_rect[0] + self.deck_width + self.margin,
                                  self.margin, self.center_width, self.deck_height)

        self.deck1_songs = []
        self.deck2_songs = []
        self.deck1_scroll = 0
        self.deck2_scroll = 0

        self.selected_song_deck1 = None
        self.selected_song_deck2 = None

        self.dragging_song = None
        self.dragging_from_deck = None
        self.dragging_position = (0, 0)

    def set_song_list(self, deck, songs):
        if deck == 1:
            self.deck1_songs = songs
            self.deck1_scroll = 0
            self.selected_song_deck1 = None
        elif deck == 2:
            self.deck2_songs = songs
            self.deck2_scroll = 0
            self.selected_song_deck2 = None

    def draw_center_decks(self, img, rect, deck1_current=None, deck2_current=None):
        x, y, w, h = rect
        cv2.rectangle(img, (x, y), (x + w, y + h), (45, 45, 45), -1)
        cv2.putText(img, "Deck Controls Area", (x + 10, y + 30), self.font, 0.8, self.text_color, 1, cv2.LINE_AA)

        if deck1_current:
            cv2.putText(img, f"Deck 1: {deck1_current}", (x+5, y+70), self.font, 0.7, self.highlight_color, 2, cv2.LINE_AA)
        if deck2_current:
            cv2.putText(img, f"Deck 2: {deck2_current}", (x+5, y+100), self.font, 0.7, self.highlight_color, 2, cv2.LINE_AA)

    def draw(self, deck1_song_list, deck2_song_list, deck1_current=None, deck2_current=None, is_playing_left=False,
             is_playing_right=False):
        img = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)

        try:
            playing_idx1 = deck1_song_list.index(deck1_current) if deck1_current in deck1_song_list else None
        except Exception:
            playing_idx1 = None
        try:
            playing_idx2 = deck2_song_list.index(deck2_current) if deck2_current in deck2_song_list else None
        except Exception:
            playing_idx2 = None

        draw_deck(img, self.deck1_rect, "Deck 1", self.deck_bg_color, self.font, self.text_color, self.highlight_color,
                  deck1_current)
        draw_deck(img, self.deck2_rect, "Deck 2", self.deck_bg_color, self.font, self.text_color, self.highlight_color,
                  deck2_current)

        list_offset = 100
        d1x, d1y, d1w, d1h = self.deck1_rect
        d2x, d2y, d2w, d2h = self.deck2_rect
        list_rect1 = (d1x, d1y + list_offset, d1w, max(0, d1h - list_offset))
        list_rect2 = (d2x, d2y + list_offset, d2w, max(0, d2h - list_offset))

        self.deck1_scroll = draw_scrollable_list(img, list_rect1, deck1_song_list, self.deck1_scroll,
                                                 self.deck_bg_color,
                                                 self.list_item_height, self.scrollbar_width, self.playing_color,
                                                 self.selection_color, self.scrollbar_color,
                                                 self.scrollbar_handle_color,
                                                 self.font, self.text_color, self.selected_song_deck1,
                                                 playing_index=playing_idx1)
        self.deck2_scroll = draw_scrollable_list(img, list_rect2, deck2_song_list, self.deck2_scroll,
                                                 self.deck_bg_color,
                                                 self.list_item_height, self.scrollbar_width, self.playing_color,
                                                 self.selection_color, self.scrollbar_color,
                                                 self.scrollbar_handle_color,
                                                 self.font, self.text_color, self.selected_song_deck2,
                                                 playing_index=playing_idx2)

        self.draw_center_decks(img, self.center_decks_rect, deck1_current, deck2_current)

        draw_play_button(img, 640, 500, self.deck_bg_color, self.highlight_color, radius=40,
                         is_playing_left=is_playing_left, is_playing_right=is_playing_right)

        if self.dragging_song:
            pos = self.dragging_position
            cv2.putText(img, self.dragging_song, (pos[0], pos[1]), self.font, 0.8, self.highlight_color, 2, cv2.LINE_AA)
            cv2.circle(img, pos, 15, self.highlight_color, 2)

        cv2.imshow(self.window_name, img)
        return img

    def scroll_list(self, deck, direction):
        if deck == 1:
            self.deck1_scroll = max(0, self.deck1_scroll + direction)
        elif deck == 2:
            self.deck2_scroll = max(0, self.deck2_scroll + direction)

    def select_song(self, deck, index):
        if deck == 1 and 0 <= index < len(self.deck1_songs):
            self.selected_song_deck1 = index
        elif deck == 2 and 0 <= index < len(self.deck2_songs):
            self.selected_song_deck2 = index

    def drag_song(self, deck, index, cursor_pos):
        if deck == 1 and 0 <= index < len(self.deck1_songs):
            self.dragging_song = self.deck1_songs[index]
            self.dragging_from_deck = 1
            self.dragging_position = cursor_pos
            self.selected_song_deck1 = index
        elif deck == 2 and 0 <= index < len(self.deck2_songs):
            self.dragging_song = self.deck2_songs[index]
            self.dragging_from_deck = 2
            self.dragging_position = cursor_pos
            self.selected_song_deck2 = index

    def update_drag(self, cursor_pos):
        self.dragging_position = cursor_pos

    def drop_song(self, drop_zone):
        if self.dragging_song is not None:
            song = self.dragging_song
            from_deck = self.dragging_from_deck
            self.dragging_song = None
            self.dragging_from_deck = None
            self.dragging_position = (0, 0)
            return song, from_deck, drop_zone
        return None
