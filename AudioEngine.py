import os
import sys
import numpy as np
import soundfile as sf
import sounddevice as sd


class AudioEngine:
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            print(f"Error: Could not find '{file_path}'.")
            sys.exit(1)

        self.data, self.samplerate = sf.read(file_path, dtype='float32')

        if self.data.ndim == 1:
            self.data = np.column_stack((self.data, self.data))

        self.position = 0.0
        self.is_playing = True

        self.target_volume = 1.0
        self.current_volume = 1.0

        self.target_pitch = 1.0
        self.current_pitch = 1.0

        self.target_pan = 0.5
        self.current_pan = 0.5

        self.target_echo = 0.0
        self.current_echo = 0.0

        self.smooth_vol = 0.1
        self.smooth_pitch = 0.05
        self.smooth_pan = 0.15
        self.smooth_echo = 0.1

        self.max_delay_samples = int(self.samplerate * 2.0)
        self.echo_buffer = np.zeros((self.max_delay_samples, 2), dtype='float32')
        self.echo_head = 0

    def callback(self, outdata, frames, time, status):
        """
        Real-time audio processing loop.
        """
        if status:
            print(status)

        if not self.is_playing:
            outdata.fill(0)
            return

        self.current_volume += (self.target_volume - self.current_volume) * self.smooth_vol
        self.current_pitch += (self.target_pitch - self.current_pitch) * self.smooth_pitch
        self.current_pan += (self.target_pan - self.current_pan) * self.smooth_pan
        self.current_echo += (self.target_echo - self.current_echo) * self.smooth_echo

        safe_pitch = max(0.25, min(3.0, self.current_pitch))

        read_len = int(frames * safe_pitch)

        current_pos_int = int(self.position)

        if current_pos_int + read_len <= len(self.data):
            raw_chunk = self.data[current_pos_int: current_pos_int + read_len]
        else:
            part1 = self.data[current_pos_int:]
            need = read_len - len(part1)
            part2 = self.data[0:need]

            if len(part2) < need:
                part2 = np.pad(part2, ((0, need - len(part2)), (0, 0)), 'wrap')

            if len(part1) == 0:
                raw_chunk = part2
            else:
                raw_chunk = np.vstack((part1, part2))

        self.position += read_len
        if self.position >= len(self.data):
            self.position %= len(self.data)

        if read_len != frames:
            x_old = np.linspace(0, read_len - 1, read_len)
            x_new = np.linspace(0, read_len - 1, frames)

            l_res = np.interp(x_new, x_old, raw_chunk[:, 0])
            r_res = np.interp(x_new, x_old, raw_chunk[:, 1])
            chunk = np.column_stack((l_res, r_res))
        else:
            chunk = raw_chunk.copy()

        left_gain = 1.0
        right_gain = 1.0

        pan_intensity = 1.5
        min_vol_floor = 0.1  # Quietest side never drops below 10%

        if self.current_pan < 0.5:
            # Pan Left -> Reduce Right
            delta = (0.5 - self.current_pan) * 2.0  # 0.0 to 1.0
            right_gain = max(min_vol_floor, 1.0 - (delta * pan_intensity))
        else:
            # Pan Right -> Reduce Left
            delta = (self.current_pan - 0.5) * 2.0
            left_gain = max(min_vol_floor, 1.0 - (delta * pan_intensity))

        # Apply Volume & Pan
        chunk[:, 0] *= (left_gain * self.current_volume)
        chunk[:, 1] *= (right_gain * self.current_volume)

        if self.current_echo > 0.05:
            delay_seconds = 0.1 + (self.current_echo * 0.4)
            delay_samples = int(self.samplerate * delay_seconds)

            feedback = 0.3 + (self.current_echo * 0.3)

            wet_mix = self.current_echo * 0.8

            buffer_len = self.max_delay_samples
            n_frames = len(chunk)

            read_pos = (self.echo_head - delay_samples) % buffer_len

            indices = (np.arange(n_frames) + read_pos) % buffer_len
            delayed_signal = self.echo_buffer[indices]

            chunk += delayed_signal * wet_mix

            write_indices = (np.arange(n_frames) + self.echo_head) % buffer_len
            self.echo_buffer[write_indices] = chunk * feedback

            self.echo_head = (self.echo_head + n_frames) % buffer_len
        else:
            buffer_len = self.max_delay_samples
            n_frames = len(chunk)
            write_indices = (np.arange(n_frames) + self.echo_head) % buffer_len
            self.echo_buffer[write_indices] = chunk * 0.0  # Clear buffer slowly
            self.echo_head = (self.echo_head + n_frames) % buffer_len

        outdata[:] = chunk

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


    def set_pitch(self, val):
        self.target_pitch = float(val)

    def pitch_control(self, val):
        self.set_pitch(val)

    def set_reverb(self, val):
        self.target_echo = float(val)

    def echo_control(self, val):
        self.target_echo = float(val)

    def set_volume(self, val):
        self.target_volume = float(val)

    def volume_control(self, val):
        self.set_volume(val)

    def set_pan(self, val):
        self.target_pan = float(val)