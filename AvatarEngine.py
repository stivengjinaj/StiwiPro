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

        self.cap = None
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2
        )
        self.hand_landmarks = []
        self.mp_drawing = mp.solutions.drawing_utils

        self.landmarks = None

        self.screen_width = 1920
        self.screen_height = 1080

        self.display = None
        self.model_data = None

    def setup_window(self):
        pygame.init()
        self.display = pygame.display.set_mode(
            (self.screen_width, self.screen_height),
            DOUBLEBUF | OPENGL
        )
        pygame.display.set_caption("Avatar Mode - Press ESC to exit")
        self.setup_opengl()

    def setup_opengl(self):
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.2, 0.2, 0.2, 1.0)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (self.screen_width / self.screen_height), 0.1, 100.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0, 5, 15, 0, 5, 0, 0, 1, 0)

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        glLight(GL_LIGHT0, GL_POSITION, (10, 10, 10, 1))
        glLight(GL_LIGHT0, GL_AMBIENT, (0.5, 0.5, 0.5, 1))
        glLight(GL_LIGHT0, GL_DIFFUSE, (1.0, 1.0, 1.0, 1))

    def load_model(self, model_path):
        try:
            from pywavefront import Wavefront
            self.model_data = Wavefront(model_path, collect_faces=True)
        except Exception as e:
            print(f"Error loading model: {e}")
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
        left_elbow = self.landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = self.landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]

        # Check if arms are horizontally extended (T-pose)
        # Wrists should be at similar Y level as shoulders
        left_arm_horizontal = abs(left_wrist.y - left_shoulder.y) < 0.15
        right_arm_horizontal = abs(right_wrist.y - right_shoulder.y) < 0.15

        if not left_arm_horizontal or not right_arm_horizontal:
            self.exit_gesture_counter = 0
            return False

        # Check if arms are extended outward
        left_arm_extended = left_wrist.x < left_shoulder.x
        right_arm_extended = right_wrist.x > right_shoulder.x

        if not left_arm_extended or not right_arm_extended:
            self.exit_gesture_counter = 0
            return False

        # All conditions met
        self.exit_gesture_counter += 1

        if self.exit_gesture_counter >= 20:
            print("T-Pose detected - exiting avatar mode")
            return True

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
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Body connections
        body_connections = [
            (11, 12),  # Shoulders
            (11, 23), (12, 24),  # Shoulder to hip
            (23, 24),  # Hips
            (11, 13), (13, 15),  # Left arm
            (12, 14), (14, 16),  # Right arm
            (23, 25), (25, 27), (27, 29), (29, 31),  # Left leg
            (24, 26), (26, 28), (28, 30), (30, 32),  # Right leg
        ]

        # Face connections
        face_connections = [
            (0, 1), (1, 2), (2, 3),  # Left eye area
            (0, 4), (4, 5), (5, 6),  # Right eye area
            (9, 10),  # Mouth
            (0, 7), (0, 8),  # Ears
        ]

        # Hand connections (wrist to fingers)
        left_hand_connections = [
            (15, 17), (15, 19), (15, 21),  # Left wrist to hand landmarks
        ]

        right_hand_connections = [
            (16, 18), (16, 20), (16, 22),  # Right wrist to hand landmarks
        ]

        # Draw body (cyan - thick)
        self.draw_limb_group(body_connections, (0.0, 1.0, 1.0), 12)

        # Draw face (yellow - medium)
        self.draw_limb_group(face_connections, (1.0, 1.0, 0.0), 8)

        # Draw hands (magenta - thin)
        self.draw_limb_group(left_hand_connections, (1.0, 0.0, 1.0), 6)
        self.draw_limb_group(right_hand_connections, (1.0, 0.0, 1.0), 6)

        # Draw small points at all landmarks to see tracking
        glPointSize(6)
        glBegin(GL_POINTS)
        for i, landmark in enumerate(self.landmarks):
            x = (landmark.x - 0.5) * 10
            y = -(landmark.y - 0.5) * 10
            z = -landmark.z * 5 - 10

            # Color code different body parts
            if i <= 10:  # Face
                glColor4f(1.0, 1.0, 0.0, 1.0)  # Yellow
            elif i <= 16:  # Upper body
                glColor4f(0.0, 1.0, 1.0, 1.0)  # Cyan
            elif i <= 22:  # Hands
                glColor4f(1.0, 0.0, 1.0, 1.0)  # Magenta
            else:  # Legs/feet
                glColor4f(0.2, 0.5, 1.0, 1.0)  # Blue

            glVertex3f(x, y, z)
        glEnd()

        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)

    def draw_limb_group(self, connections, color, line_width):
        # Outer glow
        glLineWidth(line_width + 4)
        glBegin(GL_LINES)
        for connection in connections:
            if connection[0] >= len(self.landmarks) or connection[1] >= len(self.landmarks):
                continue

            start = self.landmarks[connection[0]]
            end = self.landmarks[connection[1]]

            start_x = (start.x - 0.5) * 10
            start_y = -(start.y - 0.5) * 10
            start_z = -start.z * 5 - 10

            end_x = (end.x - 0.5) * 10
            end_y = -(end.y - 0.5) * 10
            end_z = -end.z * 5 - 10

            glColor4f(color[0], color[1], color[2], 0.3)
            glVertex3f(start_x, start_y, start_z)
            glVertex3f(end_x, end_y, end_z)
        glEnd()

        # Main line
        glLineWidth(line_width)
        glBegin(GL_LINES)
        for connection in connections:
            if connection[0] >= len(self.landmarks) or connection[1] >= len(self.landmarks):
                continue

            start = self.landmarks[connection[0]]
            end = self.landmarks[connection[1]]

            start_x = (start.x - 0.5) * 10
            start_y = -(start.y - 0.5) * 10
            start_z = -start.z * 5 - 10

            end_x = (end.x - 0.5) * 10
            end_y = -(end.y - 0.5) * 10
            end_z = -end.z * 5 - 10

            glColor4f(color[0], color[1], color[2], 1.0)
            glVertex3f(start_x, start_y, start_z)
            glVertex3f(end_x, end_y, end_z)
        glEnd()

        # Bright core
        glLineWidth(max(2, line_width - 6))
        glBegin(GL_LINES)
        for connection in connections:
            if connection[0] >= len(self.landmarks) or connection[1] >= len(self.landmarks):
                continue

            start = self.landmarks[connection[0]]
            end = self.landmarks[connection[1]]

            start_x = (start.x - 0.5) * 10
            start_y = -(start.y - 0.5) * 10
            start_z = -start.z * 5 - 10

            end_x = (end.x - 0.5) * 10
            end_y = -(end.y - 0.5) * 10
            end_z = -end.z * 5 - 10

            glColor4f(1.0, 1.0, 1.0, 1.0)
            glVertex3f(start_x, start_y, start_z)
            glVertex3f(end_x, end_y, end_z)
        glEnd()

    def draw_hands(self):
        if not self.hand_landmarks or not self.landmarks:
            return

        # Get wrist positions from pose
        left_wrist_pose = self.landmarks[15]  # Left wrist from pose
        right_wrist_pose = self.landmarks[16]  # Right wrist from pose

        glDisable(GL_LIGHTING)

        hand_scale = 0.6  # Adjust this value (0.5 = half size, 0.3 = smaller, etc.)

        for hand_idx, hand_lms in enumerate(self.hand_landmarks):
            # Determine which wrist to attach to (left or right)
            hand_wrist = hand_lms.landmark[0]

            # Check if hand is on left or right side of screen
            if hand_wrist.x < 0.5:  # Left side
                wrist_pose = right_wrist_pose  # Flipped because camera mirrors
            else:  # Right side
                wrist_pose = left_wrist_pose

            # Calculate pose wrist position
            wrist_pose_x = (wrist_pose.x - 0.5) * 10
            wrist_pose_y = -(wrist_pose.y - 0.5) * 10
            wrist_pose_z = -wrist_pose.z * 5 - 10

            # Finger connections
            finger_connections = [
                # Thumb
                (0, 1), (1, 2), (2, 3), (3, 4),
                # Index
                (0, 5), (5, 6), (6, 7), (7, 8),
                # Middle
                (0, 9), (9, 10), (10, 11), (11, 12),
                # Ring
                (0, 13), (13, 14), (14, 15), (15, 16),
                # Pinky
                (0, 17), (17, 18), (18, 19), (19, 20),
            ]

            # Draw glow
            glLineWidth(6)
            glBegin(GL_LINES)
            glColor4f(1.0, 0.0, 1.0, 0.4)  # Magenta glow

            for connection in finger_connections:
                start = hand_lms.landmark[connection[0]]
                end = hand_lms.landmark[connection[1]]

                # Get hand wrist as origin
                hand_wrist_landmark = hand_lms.landmark[0]

                # Calculate relative positions from wrist
                start_rel_x = (start.x - hand_wrist_landmark.x) * hand_scale
                start_rel_y = (start.y - hand_wrist_landmark.y) * hand_scale
                start_rel_z = (start.z - hand_wrist_landmark.z) * hand_scale

                end_rel_x = (end.x - hand_wrist_landmark.x) * hand_scale
                end_rel_y = (end.y - hand_wrist_landmark.y) * hand_scale
                end_rel_z = (end.z - hand_wrist_landmark.z) * hand_scale

                # Apply to pose wrist position
                start_x = start_rel_x * 10 + wrist_pose_x
                start_y = -start_rel_y * 10 + wrist_pose_y
                start_z = -start_rel_z * 5 + wrist_pose_z

                end_x = end_rel_x * 10 + wrist_pose_x
                end_y = -end_rel_y * 10 + wrist_pose_y
                end_z = -end_rel_z * 5 + wrist_pose_z

                glVertex3f(start_x, start_y, start_z)
                glVertex3f(end_x, end_y, end_z)
            glEnd()

            # Draw main lines
            glLineWidth(4)
            glBegin(GL_LINES)
            glColor3f(1.0, 0.0, 1.0)  # Magenta

            for connection in finger_connections:
                start = hand_lms.landmark[connection[0]]
                end = hand_lms.landmark[connection[1]]

                # Get hand wrist as origin
                hand_wrist_landmark = hand_lms.landmark[0]

                # Calculate relative positions from wrist
                start_rel_x = (start.x - hand_wrist_landmark.x) * hand_scale
                start_rel_y = (start.y - hand_wrist_landmark.y) * hand_scale
                start_rel_z = (start.z - hand_wrist_landmark.z) * hand_scale

                end_rel_x = (end.x - hand_wrist_landmark.x) * hand_scale
                end_rel_y = (end.y - hand_wrist_landmark.y) * hand_scale
                end_rel_z = (end.z - hand_wrist_landmark.z) * hand_scale

                # Apply to pose wrist position
                start_x = start_rel_x * 10 + wrist_pose_x
                start_y = -start_rel_y * 10 + wrist_pose_y
                start_z = -start_rel_z * 5 + wrist_pose_z

                end_x = end_rel_x * 10 + wrist_pose_x
                end_y = -end_rel_y * 10 + wrist_pose_y
                end_z = -end_rel_z * 5 + wrist_pose_z

                glVertex3f(start_x, start_y, start_z)
                glVertex3f(end_x, end_y, end_z)
            glEnd()

        glEnable(GL_LIGHTING)

    def render_model(self):
        if not self.model_data or not self.landmarks:
            return

        glPushMatrix()

        # Get torso center (same calculation as skeleton)
        left_shoulder = self.landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = self.landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = self.landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = self.landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]

        torso_center_x = (left_shoulder.x + right_shoulder.x + left_hip.x + right_hip.x) / 4
        torso_center_y = (left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y) / 4
        torso_center_z = (left_shoulder.z + right_shoulder.z + left_hip.z + right_hip.z) / 4

        # Use same coordinate transformation as skeleton
        screen_x = (torso_center_x - 0.5) * 10
        screen_y = -(torso_center_y - 0.5) * 10
        screen_z = -torso_center_z * 5 - 10

        # Body rotation
        shoulder_dx = right_shoulder.x - left_shoulder.x
        shoulder_dy = right_shoulder.y - left_shoulder.y
        body_rotation = np.degrees(np.arctan2(shoulder_dy, shoulder_dx))

        glTranslatef(screen_x, screen_y, screen_z)
        glRotatef(body_rotation, 0, 0, 1)

        glRotatef(180, 0, 1, 0)  # Flip front/back
        glRotatef(90, 0, 0, 1)  # Adjust if upside down
        glRotatef(90, 0, 0, 1)  # Adjust if sideways

        # Adjust scale - try values: 0.5, 1.0, 2.0, 5.0, 10.0
        glScale(1.0, 1.0, 1.0)

        glColor3f(0.8, 0.8, 1.0)

        for mesh in self.model_data.mesh_list:
            glBegin(GL_TRIANGLES)
            for face in mesh.faces:
                for vertex_i in face:
                    glVertex3fv(self.model_data.vertices[vertex_i])
            glEnd()

        glPopMatrix()

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # if self.model_data:
            # self.render_model()
        self.draw_skeleton()
        self.draw_hands()

        pygame.display.flip()

    def run(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("ERROR: Cannot open camera for avatar mode")
            return

        self.setup_window()
        self.running = True

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
                print("ERROR: Cannot read frame")
                continue

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            self.process_pose(frame)

            hands_results = self.hands.process(frame_rgb)
            self.hand_landmarks = []
            if hands_results.multi_hand_landmarks:
                self.hand_landmarks = hands_results.multi_hand_landmarks

            if self.detect_exit_gesture():
                print("Exit gesture detected - returning to DJ mode")
                self.running = False

            self.render()

            clock.tick(60)

        self.cleanup()

    def cleanup(self):
        self.cap.release()
        pygame.quit()
