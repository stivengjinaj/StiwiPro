import cv2
import numpy as np


class UIEngine:
    def __init__(self, window_name="Stiwi Pro", width=1280, height=720):
        self.window_name = window_name
        self.width = width
        self.height = height

        # Colors and fonts
        self.bg_color = (28, 28, 28)  # darker background
        self.text_color = (245, 245, 245)  # brighter text for readability
        self.highlight_color = (0, 200, 255)  # brighter cyan for highlights
        self.deck_bg_color = (60, 60, 60)  # lighter deck backgrounds
        self.selection_color = (0, 150, 255)  # stronger selection color
        self.scrollbar_color = (100, 100, 100)
        self.scrollbar_handle_color = (200, 200, 200)

        self.playing_color = (0, 200, 120)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # UI layout parameters
        self.margin = 20
        self.deck_width = int(width * 0.3)
        self.deck_height = int(height * 0.4)
        self.center_width = int(width * 0.35)
        self.list_item_height = 30
        self.scrollbar_width = 15

        # Decks location
        self.deck1_rect = (self.margin, self.margin, self.deck_width, self.deck_height)
        self.deck2_rect = (width - self.deck_width - self.margin, self.margin, self.deck_width, self.deck_height)
        self.center_decks_rect = (self.deck1_rect[0] + self.deck_width + self.margin,
                                  self.margin, self.center_width, self.deck_height)

        # Song lists and scroll states
        self.deck1_songs = []
        self.deck2_songs = []
        self.deck1_scroll = 0
        self.deck2_scroll = 0

        self.selected_song_deck1 = None
        self.selected_song_deck2 = None

        # Drag and drop state
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

    def draw_scrollable_list(self, img, rect, songs, scroll, selected_index=None, playing_index=None):
        x, y, w, h = rect
        cv2.rectangle(img, (x, y), (x + w, y + h), self.deck_bg_color, -1)

        visible_items = h // self.list_item_height
        total_items = len(songs)
        max_scroll = max(0, total_items - visible_items)

        scroll = max(0, min(scroll, max_scroll))

        start_idx = scroll
        end_idx = min(total_items, scroll + visible_items)

        text_x = x + 10
        text_y = y + self.list_item_height - 8

        for i in range(start_idx, end_idx):
            item_y = y + (i - start_idx) * self.list_item_height
            if playing_index is not None and i == playing_index:
                cv2.rectangle(img, (x, item_y), (x + w - self.scrollbar_width, item_y + self.list_item_height),
                              self.playing_color, -1)
            elif i == selected_index:
                cv2.rectangle(img, (x, item_y), (x + w - self.scrollbar_width, item_y + self.list_item_height),
                              self.selection_color, -1)
            cv2.putText(img, songs[i], (text_x, item_y + self.list_item_height - 10), self.font, 0.6, self.text_color,
                        1, cv2.LINE_AA)

        if max_scroll > 0:
            scrollbar_x = x + w - self.scrollbar_width
            cv2.rectangle(img, (scrollbar_x, y), (scrollbar_x + self.scrollbar_width, y + h), self.scrollbar_color, -1)
            handle_height = max(20, int(h * (visible_items / total_items)))
            handle_y = int(y + (scroll / max_scroll) * (h - handle_height))
            cv2.rectangle(img, (scrollbar_x, handle_y), (scrollbar_x + self.scrollbar_width, handle_y + handle_height),
                          self.scrollbar_handle_color, -1)

        return scroll

    def draw_deck(self, img, rect, title, current_song=None):
        x, y, w, h = rect
        cv2.rectangle(img, (x, y), (x + w, y + h), self.deck_bg_color, -1)
        cv2.putText(img, title, (x + 10, y + 25), self.font, 0.8, self.text_color, 2, cv2.LINE_AA)

        song_box_y = y + 40
        song_box_h = 50
        cv2.rectangle(img, (x + 10, song_box_y), (x + w - 10, song_box_y + song_box_h), (70, 70, 70), -1)
        if current_song:
            cv2.putText(img, current_song, (x + 15, song_box_y + 35), self.font, 0.7, self.highlight_color, 2,
                        cv2.LINE_AA)
        else:
            cv2.putText(img, "No song loaded", (x + 15, song_box_y + 35), self.font, 0.7, (120, 120, 120), 1,
                        cv2.LINE_AA)

    def draw_center_decks(self, img, rect):
        x, y, w, h = rect
        # Rectangle for decks area
        cv2.rectangle(img, (x, y), (x + w, y + h), (45, 45, 45), -1)
        cv2.putText(img, "Deck Controls Area", (x + 10, y + 30), self.font, 0.8, self.text_color, 1,
                    cv2.LINE_AA)

    def draw(self, deck1_song_list, deck2_song_list, deck1_current=None, deck2_current=None):
        img = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)

        status_text = f"Deck1: {len(deck1_song_list)} songs  |  Deck2: {len(deck2_song_list)} songs"
        cv2.putText(img, status_text, (self.margin, 18), self.font, 0.6, self.text_color, 2, cv2.LINE_AA)

        try:
            playing_idx1 = deck1_song_list.index(deck1_current) if deck1_current in deck1_song_list else None
        except Exception:
            playing_idx1 = None
        try:
            playing_idx2 = deck2_song_list.index(deck2_current) if deck2_current in deck2_song_list else None
        except Exception:
            playing_idx2 = None

        # Draw decks (header + current song box) first so they don't overwrite the lists
        self.draw_deck(img, self.deck1_rect, "Deck 1", deck1_current)
        self.draw_deck(img, self.deck2_rect, "Deck 2", deck2_current)

        # Compute list rects placed below the deck header/current-song box so items remain visible
        # Using same deck rect but offset down by 100px to leave space for the title and current-song box
        list_offset = 100
        d1x, d1y, d1w, d1h = self.deck1_rect
        d2x, d2y, d2w, d2h = self.deck2_rect
        list_rect1 = (d1x, d1y + list_offset, d1w, max(0, d1h - list_offset))
        list_rect2 = (d2x, d2y + list_offset, d2w, max(0, d2h - list_offset))

        # Draw song lists, passing both selected and playing indices
        self.deck1_scroll = self.draw_scrollable_list(img, list_rect1, deck1_song_list, self.deck1_scroll,
                                                      self.selected_song_deck1, playing_index=playing_idx1)
        self.deck2_scroll = self.draw_scrollable_list(img, list_rect2, deck2_song_list, self.deck2_scroll,
                                                      self.selected_song_deck2, playing_index=playing_idx2)


        # Draw center decks area
        self.draw_center_decks(img, self.center_decks_rect)

        # Draw dragging song if any
        if self.dragging_song:
            pos = self.dragging_position
            cv2.putText(img, self.dragging_song, (pos[0], pos[1]), self.font, 0.8, self.highlight_color, 2, cv2.LINE_AA)
            cv2.circle(img, pos, 15, self.highlight_color, 2)

        cv2.imshow(self.window_name, img)
        return img

    # Scroll management
    def scroll_list(self, deck, direction):
        if deck == 1:
            self.deck1_scroll = max(0, self.deck1_scroll + direction)
        elif deck == 2:
            self.deck2_scroll = max(0, self.deck2_scroll + direction)

    # Select song by index
    def select_song(self, deck, index):
        if deck == 1 and 0 <= index < len(self.deck1_songs):
            self.selected_song_deck1 = index
        elif deck == 2 and 0 <= index < len(self.deck2_songs):
            self.selected_song_deck2 = index

    # Start dragging song
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

    # Update dragging position
    def update_drag(self, cursor_pos):
        self.dragging_position = cursor_pos

    # End dragging and return where dropped, None if no drop zone
    def drop_song(self, drop_zone):
        if self.dragging_song is not None:
            song = self.dragging_song
            from_deck = self.dragging_from_deck
            self.dragging_song = None
            self.dragging_from_deck = None
            self.dragging_position = (0, 0)
            return song, from_deck, drop_zone
        return None
