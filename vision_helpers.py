def is_position_over_song(hand_pos, ui, deck_num=1):
    if hand_pos is None:
        return None

    hand_x, hand_y = hand_pos

    if deck_num == 1:
        songs = ui.deck1_songs
        deck_rect = ui.deck1_rect
    else:
        songs = ui.deck2_songs
        deck_rect = ui.deck2_rect

    d_x, d_y, d_w, d_h = deck_rect
    norm_x = d_x / ui.width
    norm_y = d_y / ui.height
    norm_w = d_w / ui.width
    norm_h = d_h / ui.height

    list_offset_px = 100
    list_offset = list_offset_px / ui.height
    list_norm_y = norm_y + list_offset
    list_norm_h = norm_h - list_offset

    if not (norm_x <= hand_x <= norm_x + norm_w):
        return None

    if not (list_norm_y <= hand_y <= list_norm_y + list_norm_h):
        return None

    if list_norm_h <= 0 or len(songs) == 0:
        return None

    item_norm_h = ui.list_item_height / ui.height
    y_offset_in_list = hand_y - list_norm_y
    item_index = int(y_offset_in_list / item_norm_h)

    visible_items = max(1, int(list_norm_h / item_norm_h))
    if item_index < 0 or item_index >= visible_items:
        return None

    song_index = ui.deck1_scroll + item_index if deck_num == 1 else ui.deck2_scroll + item_index

    if song_index >= len(songs):
        return None

    return song_index


def is_position_over_play_button(hand_pos, ui, width=1280, height=720, button_radius=40):
    if hand_pos is None:
        return None

    center_x = 640
    center_y = 500

    left_button_x = center_x // 2 - 100
    right_button_x = center_x + 400
    button_y = center_y - 100

    hand_pixel_x = hand_pos[0] * ui.width
    hand_pixel_y = hand_pos[1] * ui.height

    dist_x_left = hand_pixel_x - left_button_x
    dist_x_right = hand_pixel_x - right_button_x
    dist_y = hand_pixel_y - button_y
    distance_left = (dist_x_left ** 2 + dist_y ** 2) ** 0.5
    distance_right = (dist_x_right ** 2 + dist_y ** 2) ** 0.5

    if distance_left <= button_radius:
        return 'left'
    elif distance_right <= button_radius:
        return 'right'
    else:
        return None

def is_position_over_master_slider(hand_pos, ui, slider_position=0.0, button_radius=25):
    if hand_pos is None:
        return None
    center_x = 640
    center_y = 500

    knob_x = center_x + (slider_position * 150)
    knob_y = center_y - 100

    hand_pixel_x = hand_pos[0] * ui.width
    hand_pixel_y = hand_pos[1] * ui.height

    dist_x = hand_pixel_x - knob_x
    dist_y = hand_pixel_y - knob_y

    distance = (dist_x ** 2 + dist_y ** 2) ** 0.5

    return distance <= button_radius