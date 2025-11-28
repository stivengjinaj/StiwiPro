import os
import sys
import numpy as np
import soundfile as sf
import sounddevice as sd
from pedalboard import Pedalboard, Gain, Compressor, Reverb, Delay, Chorus, Phaser, \
    LowpassFilter, HighpassFilter, PitchShift, Distortion, Limiter, Bitcrush
from pedalboard.io import AudioFile


class AdvancedAudioEngine:
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            print(f"Error: Could not find '{file_path}'.")
            sys.exit(1)

        # Load audio file
        self.data, self.samplerate = sf.read(file_path, dtype='float32')
        if self.data.ndim == 1:
            self.data = np.column_stack((self.data, self.data))

        self.position = 0.0
        self.is_playing = True

        self.board = Pedalboard([
            Gain(gain_db=0),  # Master volume
            Compressor(threshold_db=-24, ratio=4, attack_ms=5, release_ms=100),
            Reverb(room_size=0.5, wet_level=0.33),
            Delay(delay_seconds=0.25, feedback=0.45),
            Chorus(),
            Phaser(),
            LowpassFilter(cutoff_frequency_hz=20000),
            HighpassFilter(cutoff_frequency_hz=20),
            PitchShift(semitones=0),
            Distortion(drive_db=0),
            Bitcrush(bits=16),
            Limiter(threshold_db=-9, release_ms=10)
        ])

        self.controls = {
            'volume': 1.0,
            'pitch': 0.0,  # -12 to +12 semitones
            'reverb': 0.0,
            'delay': 0.0,
            'chorus': 0.0,
            'phaser': 0.0,
            'lowpass': 1.0,  # 1.0 = full open
            'highpass': 0.0,  # 0.0 = full open
            'distortion': 0.0,
            'bitcrush': 0.0,  # 0=16bit, 1=1bit
            'compressor': 1.0
        }

        self.update_pedalboard()

    def update_pedalboard(self):
        """Update all plugin parameters based on current controls."""
        self.board[0].gain_db = np.clip(self.controls['volume'] * 12 - 6, -60, 12)  # -6 to +6dB

        # Compressor
        comp_ratio = np.interp(self.controls['compressor'], [0, 1], [2, 20])
        self.board[1].ratio = comp_ratio
        self.board[1].threshold_db = np.interp(self.controls['compressor'], [0, 1], [-30, -12])

        # Reverb
        self.board[2].wet_level = self.controls['reverb'] * 0.8
        self.board[2].room_size = 0.3 + self.controls['reverb'] * 0.4

        # Delay
        self.board[3].mix = self.controls['delay']
        self.board[3].feedback = 0.3 + self.controls['delay'] * 0.4

        # Chorus
        self.board[4].mix = self.controls['chorus']
        self.board[4].rate_hz = 0.5 + self.controls['chorus'] * 2.0

        # Phaser
        self.board[5].mix = self.controls['phaser']
        self.board[5].rate_hz = 0.3 + self.controls['phaser'] * 1.5

        # Filters
        self.board[6].cutoff_frequency_hz = np.interp(self.controls['lowpass'], [0, 1], [2000, 20000])
        self.board[7].cutoff_frequency_hz = np.interp(self.controls['highpass'], [0, 1], [20, 1000])

        # Pitch
        self.board[8].semitones = np.interp(self.controls['pitch'], [0, 1], [-12, 12])

        # Distortion
        self.board[9].drive_db = self.controls['distortion'] * 24

        # Bitcrush
        bits = np.interp(self.controls['bitcrush'], [0, 1], [16, 4])
        self.board[10].bits = int(max(4, bits))

    def callback(self, outdata, frames, time, status):
        if status:
            print(status)

        if not self.is_playing:
            outdata.fill(0)
            return

        read_len = frames
        current_pos_int = int(self.position)

        if current_pos_int + read_len >= len(self.data):
            self.position = 0
            current_pos_int = 0

        chunk = self.data[current_pos_int: current_pos_int + read_len]

        effected = self.board(chunk, self.samplerate, reset=False)

        outdata[:] = effected
        self.position += read_len

    def start(self):
        self.stream = sd.OutputStream(
            samplerate=self.samplerate,
            channels=2,
            callback=self.callback,
            blocksize=1024
        )
        self.stream.start()

    def stop(self):
        self.is_playing = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()

    # Gesture control functions (0.0 to 1.0 normalized)
    def set_volume(self, value):
        self.controls['volume'] = np.clip(value, 0.0, 1.0)
        self.update_pedalboard()

    def set_pitch(self, value):
        self.controls['pitch'] = np.clip(value, 0.0, 1.0)
        self.update_pedalboard()

    def set_reverb(self, value):
        self.controls['reverb'] = np.clip(value, 0.0, 1.0)
        self.update_pedalboard()

    def set_delay(self, value):
        self.controls['delay'] = np.clip(value, 0.0, 1.0)
        self.update_pedalboard()

    def set_chorus(self, value):
        self.controls['chorus'] = np.clip(value, 0.0, 1.0)
        self.update_pedalboard()

    def set_phaser(self, value):
        self.controls['phaser'] = np.clip(value, 0.0, 1.0)
        self.update_pedalboard()

    def set_lowpass(self, value):
        self.controls['lowpass'] = np.clip(value, 0.0, 1.0)
        self.update_pedalboard()

    def set_highpass(self, value):
        self.controls['highpass'] = np.clip(value, 0.0, 1.0)
        self.update_pedalboard()

    def set_distortion(self, value):
        self.controls['distortion'] = np.clip(value, 0.0, 1.0)
        self.update_pedalboard()

    def set_bitcrush(self, value):
        self.controls['bitcrush'] = np.clip(value, 0.0, 1.0)
        self.update_pedalboard()

    def toggle_playback(self):
        self.is_playing = not self.is_playing

    def get_status(self):
        return {
            'position': self.position / len(self.data),
            'playing': self.is_playing,
            **{k: round(v, 3) for k, v in self.controls.items()}
        }