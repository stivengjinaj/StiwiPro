import cv2
import numpy as np


def draw_scrollable_list(
        img,
        rect,
        songs,
        scroll,
        deck_bg_color,
        list_item_height,
        scrollbar_width,
        playing_color,
        selection_color,
        scrollbar_color,
        scrollbar_handle_color,
        font,
        text_color,
        selected_index=None,
        playing_index=None
):
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), deck_bg_color, -1)

    visible_items = h // list_item_height
    total_items = len(songs)
    max_scroll = max(0, total_items - visible_items)

    scroll = max(0, min(scroll, max_scroll))

    start_idx = scroll
    end_idx = min(total_items, scroll + visible_items)

    text_x = x + 10
    # text_y = y + self.list_item_height - 8

    for i in range(start_idx, end_idx):
        item_y = y + (i - start_idx) * list_item_height
        if playing_index is not None and i == playing_index:
            cv2.rectangle(img, (x, item_y), (x + w - scrollbar_width, item_y + list_item_height),
                          playing_color, -1)
        elif i == selected_index:
            cv2.rectangle(img, (x, item_y), (x + w - scrollbar_width, item_y + list_item_height),
                          selection_color, -1)
        cv2.putText(img, songs[i], (text_x, item_y + list_item_height - 10), font, 0.6, text_color,
                    1, cv2.LINE_AA)

    if max_scroll > 0:
        scrollbar_x = x + w - scrollbar_width
        cv2.rectangle(img, (scrollbar_x, y), (scrollbar_x + scrollbar_width, y + h), scrollbar_color, -1)
        handle_height = max(20, int(h * (visible_items / total_items)))
        handle_y = int(y + (scroll / max_scroll) * (h - handle_height))
        cv2.rectangle(img, (scrollbar_x, handle_y), (scrollbar_x + scrollbar_width, handle_y + handle_height),
                      scrollbar_handle_color, -1)

    return scroll

def draw_deck(img, rect, title, deck_bg_color, font, text_color, highlight_color, current_song=None):
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), deck_bg_color, -1)
    cv2.putText(img, title, (x + 10, y + 25), font, 0.8, text_color, 2, cv2.LINE_AA)

    song_box_y = y + 40
    song_box_h = 50
    cv2.rectangle(img, (x + 10, song_box_y), (x + w - 10, song_box_y + song_box_h), (70, 70, 70), -1)
    if current_song:
        cv2.putText(img, current_song, (x + 15, song_box_y + 35), font, 0.7, highlight_color, 2,
                    cv2.LINE_AA)
    else:
        cv2.putText(img, "No song loaded", (x + 15, song_box_y + 35), font, 0.7, (120, 120, 120), 1,
                    cv2.LINE_AA)


def draw_play_button(img, center_x, center_y, deck_bg_color, highlight_color, radius=30, is_playing_left=False, is_playing_right=False):
    center_x_left = center_x // 2 - 100
    center_x_right = center_x + 400
    center_y_both = center_y - 100

    cv2.circle(img, (center_x_left, center_y_both), radius, deck_bg_color, -1)
    cv2.circle(img, (center_x_left, center_y_both), radius, highlight_color, 2)
    cv2.circle(img, (center_x_right, center_y_both), radius, deck_bg_color, -1)
    cv2.circle(img, (center_x_right, center_y_both), radius, highlight_color, 2)

    if is_playing_left:
        bar_width = int(radius * 0.25)
        bar_height = int(radius * 0.8)
        bar_spacing = int(radius * 0.3)

        left_x = center_x_left - bar_spacing
        cv2.rectangle(img,
                      (left_x - bar_width // 2, center_y_both - bar_height // 2),
                      (left_x + bar_width // 2, center_y_both + bar_height // 2),
                      highlight_color, -1)

        right_x = center_x_left + bar_spacing
        cv2.rectangle(img,
                      (right_x - bar_width // 2, center_y_both - bar_height // 2),
                      (right_x + bar_width // 2, center_y_both + bar_height // 2),
                      highlight_color, -1)
    else:
        triangle_height = int(radius * 0.8)
        triangle_width = int(radius * 0.7)
        offset = int(radius * 0.1)

        pt1 = (center_x_left - triangle_width // 2 + offset, center_y_both - triangle_height // 2)
        pt2 = (center_x_left - triangle_width // 2 + offset, center_y_both + triangle_height // 2)
        pt3 = (center_x_left + triangle_width // 2 + offset, center_y_both)

        triangle = np.array([pt1, pt2, pt3], np.int32)
        cv2.fillPoly(img, [triangle], highlight_color)

    if is_playing_right:
        bar_width = int(radius * 0.25)
        bar_height = int(radius * 0.8)
        bar_spacing = int(radius * 0.3)

        left_x = center_x_right - bar_spacing
        cv2.rectangle(img,
                      (left_x - bar_width // 2, center_y_both - bar_height // 2),
                      (left_x + bar_width // 2, center_y_both + bar_height // 2),
                      highlight_color, -1)

        right_x = center_x_right + bar_spacing
        cv2.rectangle(img,
                      (right_x - bar_width // 2, center_y_both - bar_height // 2),
                      (right_x + bar_width // 2, center_y_both + bar_height // 2),
                      highlight_color, -1)
    else:
        triangle_height = int(radius * 0.8)
        triangle_width = int(radius * 0.7)
        offset = int(radius * 0.1)

        pt1 = (center_x_right - triangle_width // 2 + offset, center_y_both - triangle_height // 2)
        pt2 = (center_x_right - triangle_width // 2 + offset, center_y_both + triangle_height // 2)
        pt3 = (center_x_right + triangle_width // 2 + offset, center_y_both)

        triangle = np.array([pt1, pt2, pt3], np.int32)
        cv2.fillPoly(img, [triangle], highlight_color)

def draw_master_slider(img, center_x, center_y, slider_position=0.0):
    cv2.rectangle(img, (center_x - 150, center_y - 100), (center_x + 150, center_y - 100), (42,42,210), thickness=30)
    knob_x = center_x + (slider_position * 150)
    if knob_x <= (center_x - 150):
        knob_x = center_x - 150
    elif knob_x >= (center_x + 150):
        knob_x = center_x + 150
    cv2.circle(img, (knob_x.__int__(), center_y-100), 25, (255, 255, 255), -1)