import os
import glob

import cv2

from AudioEngine import AudioEngine
from AvatarEngine import AvatarEngine
from UIEngine import UIEngine
from VisionEngine import VisionEngine


def load_songs_from_directory(directory_path):
    """Load all audio files from directory into a song list."""
    supported_formats = ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg']
    songs = []

    for pattern in supported_formats:
        files = glob.glob(os.path.join(directory_path, pattern)) + \
                glob.glob(os.path.join(directory_path, pattern.upper()))
        songs.extend(files)

    valid_songs = []
    for full_path in songs:
        if os.path.exists(full_path):
            filename = os.path.basename(full_path)
            valid_songs.append({
                'path': full_path,
                'name': os.path.splitext(filename)[0]
            })

    return sorted(valid_songs, key=lambda x: x['name'])


def main():
    music_directory = "music"
    if not os.path.exists(music_directory):
        print(f"Creating music directory: {music_directory}")
        os.makedirs(music_directory)
        print("Please add .wav, .mp3, .flac, .m4a, or .ogg files to the 'music' folder.")
        return

    song_list = load_songs_from_directory(music_directory)
    if not song_list:
        print("No audio files found in music directory. Add some songs!")
        return

    ui = UIEngine()
    ui.set_song_list(1, [song['name'] for song in song_list])
    ui.set_song_list(2, [song['name'] for song in song_list])

    audio_engine_left = None
    audio_engine_right = None

    vision = VisionEngine(audio_engine_left, audio_engine_right, ui, song_list)
    avatar = None

    running = True
    while running:
        if not vision.avatar_mode:
            running = vision.process()
            if not running:
                break
        else:
            if avatar is None:
                avatar = AvatarEngine(vision.audio_engine_left, vision.audio_engine_right)
                avatar.load_model("MaleTron_Lowpoly.obj")
            else:
                avatar.audio_engine_left = vision.audio_engine_left
                avatar.audio_engine_right = vision.audio_engine_right

            vision.cap.release()
            cv2.destroyAllWindows()

            avatar.run()

            vision.cap = cv2.VideoCapture(0)

            vision.avatar_mode = False

    if vision.audio_engine_left:
        vision.audio_engine_left.stop()
    if vision.audio_engine_right:
        vision.audio_engine_right.stop()


if __name__ == "__main__":
    main()
