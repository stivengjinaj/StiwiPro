import cv2
import mediapipe as mp
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import pygame
from pygame.locals import *


class AvatarEngine:
    def __init__(self, audio_engine_left=None, audio_engine_right=None):
        self.audio_engine_left = audio_engine_left
        self.audio_engine_right = audio_engine_right

        self.running = True
        self.exit_gesture_counter = 0

        self.cap = cv2.VideoCapture(0)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        self.mp_drawing = mp.solutions.drawing_utils

        self.landmarks = None

        self.screen_width = 1920
        self.screen_height = 1080

        pygame.init()
        self.display = pygame.display.set_mode(
            (self.screen_width, self.screen_height),
            DOUBLEBUF | OPENGL | FULLSCREEN
        )
        pygame.display.set_caption("Avatar Mode")

        self.setup_opengl()
        self.model_data = None

    def setup_opengl(self):
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (self.screen_width / self.screen_height), 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -5)

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        glLight(GL_LIGHT0, GL_POSITION, (5, 5, 5, 1))
        glLight(GL_LIGHT0, GL_AMBIENT, (0.3, 0.3, 0.3, 1))
        glLight(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1))

    def load_model(self, model_path):
        try:
            from pywavefront import Wavefront
            self.model_data = Wavefront(model_path, collect_faces=True, create_materials=True, parse=True, strict=False)
        except Exception as e:
            try:
                self.model_data = Wavefront(model_path, collect_faces=True, parse=False)
            except Exception as e2:
                print(f"Failed again: {e2}")
                self.model_data = None

    def process_pose(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if results.pose_landmarks:
            self.landmarks = results.pose_landmarks.landmark
            return True
        else:
            self.landmarks = None
            return False

    def detect_exit_gesture(self):
        if not self.landmarks:
            self.exit_gesture_counter = 0
            return False

        left_wrist = self.landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = self.landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        left_shoulder = self.landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = self.landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        threshold_x = 0.07
        threshold_y = 0.5

        wrists_close_x = abs(left_wrist.x - right_wrist.x) <= threshold_x
        wrists_close_y = abs(left_wrist.y - right_wrist.y) <= threshold_y

        wrists_above_shoulders = (left_wrist.y < left_shoulder.y and
                                  right_wrist.y < right_shoulder.y)

        if wrists_close_x and wrists_close_y and wrists_above_shoulders:
            self.exit_gesture_counter += 1
            if self.exit_gesture_counter >= 15:
                return True
        else:
            self.exit_gesture_counter = 0

        return False

    def get_joint_rotation(self, landmark_a, landmark_b):
        dx = landmark_b.x - landmark_a.x
        dy = landmark_b.y - landmark_a.y
        dz = landmark_b.z - landmark_a.z

        angle_y = np.degrees(np.arctan2(dx, dz))
        angle_x = np.degrees(np.arctan2(dy, np.sqrt(dx ** 2 + dz ** 2)))

        return angle_x, angle_y

    def draw_skeleton(self):
        if not self.landmarks:
            return

        glDisable(GL_LIGHTING)
        glLineWidth(3)

        connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (24, 26), (26, 28)
        ]

        glBegin(GL_LINES)
        glColor3f(0.0, 1.0, 1.0)
        for connection in connections:
            start = self.landmarks[connection[0]]
            end = self.landmarks[connection[1]]

            glVertex3f((start.x - 0.5) * 4, -(start.y - 0.5) * 4, -start.z * 2)
            glVertex3f((end.x - 0.5) * 4, -(end.y - 0.5) * 4, -end.z * 2)
        glEnd()

        glPointSize(8)
        glBegin(GL_POINTS)
        glColor3f(1.0, 0.0, 1.0)
        for landmark in self.landmarks:
            glVertex3f((landmark.x - 0.5) * 4, -(landmark.y - 0.5) * 4, -landmark.z * 2)
        glEnd()

        glEnable(GL_LIGHTING)

    def render_model(self):
        if not self.model_data or not self.landmarks:
            print("No model or landmarks")
            return

        glPushMatrix()

        hip_center = self.landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]

        glTranslatef(0, 0, 0)
        glScale(0.5, 0.5, 0.5)

        glColor3f(1.0, 1.0, 1.0)

        for mesh in self.model_data.mesh_list:
            glBegin(GL_TRIANGLES)
            for face in mesh.faces:
                for vertex_i in face:
                    if vertex_i < len(self.model_data.vertices):
                        glVertex3fv(self.model_data.vertices[vertex_i])
            glEnd()

        glPopMatrix()
        print("Model rendered")

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glPushMatrix()

        if self.model_data:
            self.render_model()
        else:
            self.draw_skeleton()

        glPopMatrix()

        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()

        while self.running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        self.running = False

            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)

            self.process_pose(frame)

            if self.detect_exit_gesture():
                print("Exit gesture detected - returning to DJ mode")
                self.running = False

            self.render()

            clock.tick(60)

        self.cleanup()

    def cleanup(self):
        self.cap.release()
        pygame.quit()
