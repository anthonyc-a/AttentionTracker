import cv2
import dlib
import numpy as np
import pyautogui
import time
from collections import deque

class EyeHeadTracker:
    def __init__(self):
        # Load face detector and landmark predictor
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # Load OpenCV's DNN face detector
        self.net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
        
        self.calibration = {'center': None, 'up': None, 'down': None, 'left': None, 'right': None}
        self.scroll_threshold = 0.15
        self.scroll_cooldown = 0.2
        self.last_scroll_time = 0
        self.direction_buffer = deque(maxlen=5)
        self.pupil_positions = deque(maxlen=10)
        self.head_position = deque(maxlen=10)
        self.is_paused = False
        self.invert_scroll = False
        self.sensitivity = 5
        self.scroll_momentum = 0
        self.scroll_speed = 0
        self.max_scroll_speed = 20
        self.scroll_acceleration = 0.5
        self.scroll_deceleration = 0.3
        self.momentum_decay = 0.8
        self.head_movement_weight = 0.3  # Adjust this to change the influence of head movement

class EyeHeadTracker:
    def __init__(self):
        # Load face detector and landmark predictor
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # Load OpenCV's DNN face detector
        self.net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
        
        self.calibration = {'center': None, 'up': None, 'down': None, 'left': None, 'right': None}
        self.scroll_threshold = 0.15
        self.scroll_cooldown = 0.2
        self.last_scroll_time = 0
        self.direction_buffer = deque(maxlen=5)
        self.pupil_positions = deque(maxlen=10)
        self.head_position = deque(maxlen=10)
        self.is_paused = False
        self.invert_scroll = False
        self.sensitivity = 5
        self.scroll_speed = 0
        self.max_scroll_speed = 20
        self.scroll_acceleration = 0.5
        self.scroll_deceleration = 0.3
        self.head_movement_weight = 0.3  # Adjust this to change the influence of head movement


    def detect_face_and_eyes(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces using dlib
        faces = self.face_detector(gray)
        
        if len(faces) == 0:
            # If dlib fails, try OpenCV's DNN detector
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            self.net.setInput(blob)
            detections = self.net.forward()
            
            if detections.shape[2] > 0:
                i = np.argmax(detections[0, 0, :, 2])
                confidence = detections[0, 0, i, 2]
                
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (startX, startY, endX, endY) = box.astype("int")
                    face = dlib.rectangle(startX, startY, endX, endY)
                    shapes = self.landmark_predictor(gray, face)
                    return self.get_eyes_from_landmarks(shapes)
            
            return None
        
        shapes = self.landmark_predictor(gray, faces[0])
        head_pos = (shapes.part(30).x, shapes.part(30).y)  # Use nose tip as head position
        self.head_position.append(head_pos)
        return self.get_eyes_from_landmarks(shapes), head_pos

    def get_eyes_from_landmarks(self, shapes):
        landmarks = np.array([[p.x, p.y] for p in shapes.parts()])
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        
        left_eye_region = self.get_eye_region(left_eye)
        right_eye_region = self.get_eye_region(right_eye)
        
        return ([left_eye_region, right_eye_region], (shapes.rect.left(), shapes.rect.top(), shapes.rect.width(), shapes.rect.height()))
    def get_eye_region(self, eye):
        min_x = np.min(eye[:, 0])
        max_x = np.max(eye[:, 0])
        min_y = np.min(eye[:, 1])
        max_y = np.max(eye[:, 1])
        return (min_x, min_y, max_x - min_x, max_y - min_y)

    def detect_pupil(self, eye):
        rows, cols = eye.shape
        
        # Improve pupil detection with adaptive thresholding
        eye_adaptive = cv2.adaptiveThreshold(eye, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # Use morphological operations to reduce noise
        kernel = np.ones((3, 3), np.uint8)
        eye_adaptive = cv2.erode(eye_adaptive, kernel, iterations=1)
        eye_adaptive = cv2.dilate(eye_adaptive, kernel, iterations=1)
        
        contours, _ = cv2.findContours(eye_adaptive, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            
            # Filter out non-circular contours
            aspect_ratio = w / h
            if 0.8 <= aspect_ratio <= 1.2:
                cx = x + w // 2
                cy = y + h // 2
                return (cx, cy)
        
        return None

    def analyze_eye_and_head(self, eye_region, face_position, head_pos):
        gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        pupil = self.detect_pupil(gray_eye)
        
        if pupil is None:
            return (0, 0)

        cx, cy = pupil
        rows, cols = gray_eye.shape
        
        relative_x = (cx - cols/2) / (cols/2)
        relative_y = (cy - rows/2) / (rows/2)
        
        self.pupil_positions.append((relative_x, relative_y))
        
        # Calculate average pupil position over last few frames
        avg_x = sum(p[0] for p in self.pupil_positions) / len(self.pupil_positions)
        avg_y = sum(p[1] for p in self.pupil_positions) / len(self.pupil_positions)
        
        # Calculate head movement
        if len(self.head_position) > 1:
            head_dx = self.head_position[-1][0] - self.head_position[0][0]
            head_dy = self.head_position[-1][1] - self.head_position[0][1]
            head_dx /= face_position[2]  # Normalize by face width
            head_dy /= face_position[3]  # Normalize by face height
        else:
            head_dx, head_dy = 0, 0

        # Combine eye and head movement
        combined_x = avg_x * (1 - self.head_movement_weight) + head_dx * self.head_movement_weight
        combined_y = avg_y * (1 - self.head_movement_weight) + head_dy * self.head_movement_weight

        return (combined_x, combined_y)

    def smooth_scroll(self, direction, ratio):
        target_scroll = direction * self.sensitivity * 20  # Increase scroll amount
        self.scroll_momentum += (target_scroll - self.scroll_momentum) * 0.3  # Faster response
        scroll_amount = int(self.scroll_momentum)
        
        if abs(scroll_amount) > 0:
            pyautogui.scroll(scroll_amount)
        
        self.scroll_momentum *= self.momentum_decay

    def draw_omnidirectional_arrow(self, frame, direction, center, size=50):
        x, y = center
        dx, dy = direction
        
        # Normalize the direction vector
        magnitude = np.sqrt(dx**2 + dy**2)
        if magnitude > 0:
            dx /= magnitude
            dy /= magnitude
        
        # Calculate end point of the arrow
        end_x = int(x + dx * size)
        end_y = int(y + dy * size)
        
        # Draw the arrow
        cv2.arrowedLine(frame, (x, y), (end_x, end_y), (0, 255, 0), 2)

    def draw_text(self, frame, text, position, scale):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, text, position, font, scale, (0, 255, 0), 1, cv2.LINE_AA)

    def draw_sensitivity_bar(self, frame):
        bar_width = 220  # Increased to accommodate spacing
        bar_height = 20
        x = frame.shape[1] - bar_width - 10
        y = frame.shape[0] - bar_height - 10
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (0, 255, 0), 1)
        
        chunk_width = 20
        chunk_spacing = 2
        for i in range(10):
            chunk_x = x + i * (chunk_width + chunk_spacing)
            if i < self.sensitivity:
                cv2.rectangle(frame, (chunk_x, y), (chunk_x + chunk_width - 1, y + bar_height), (0, 255, 0), -1)
            else:
                cv2.rectangle(frame, (chunk_x, y), (chunk_x + chunk_width - 1, y + bar_height), (0, 255, 0), 1)
        
        self.draw_text(frame, f"SENSITIVITY: {self.sensitivity}", (x, y - 10), 0.5)

    def calibrate(self, cap):
        positions = ['center', 'up', 'down', 'left', 'right']
        current_position = 0
        calibration_complete = False

        while not calibration_complete:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                return False

            position = positions[current_position]
            self.draw_text(frame, f"LOOK {position.upper()} AND PRESS 'C' TO CAPTURE", (10, 30), 0.5)
            self.draw_text(frame, "PRESS 'Q' TO QUIT CALIBRATION", (10, 60), 0.5)

            cv2.imshow('Calibration', frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                result = self.detect_face_and_eyes(frame)
                if result is not None:
                    eyes, face_pos = result[0]
                    eye_region = frame[eyes[0][1]:eyes[0][1]+eyes[0][3], eyes[0][0]:eyes[0][0]+eyes[0][2]]
                    pupil = self.detect_pupil(cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY))
                    if pupil:
                        cx, cy = pupil
                        rows, cols = eye_region.shape[:2]
                        relative_x = (cx - cols/2) / (cols/2)
                        relative_y = (cy - rows/2) / (rows/2)
                        self.calibration[position] = (relative_x, relative_y)
                        self.draw_text(frame, f"{position.upper()} CALIBRATED", (10, 90), 0.5)
                        cv2.imshow('Calibration', frame)
                        cv2.waitKey(1000)
                        current_position += 1
                        if current_position >= len(positions):
                            calibration_complete = True
            elif key == ord('q'):
                return False

        self.draw_text(frame, "CALIBRATION COMPLETED", (10, 120), 0.5)
        cv2.imshow('Calibration', frame)
        cv2.waitKey(2000)
        return True
    
    
    

    def update_scroll_speed(self, direction):
        target_speed = direction * self.max_scroll_speed * self.sensitivity
        if abs(target_speed) > abs(self.scroll_speed):
            self.scroll_speed += (target_speed - self.scroll_speed) * self.scroll_acceleration
        else:
            self.scroll_speed += (target_speed - self.scroll_speed) * self.scroll_deceleration
        
        # Apply some thresholding to avoid tiny movements
        if abs(self.scroll_speed) < 0.5:
            self.scroll_speed = 0

    def perform_scroll(self):
        if not self.is_paused:
            scroll_amount = int(self.scroll_speed)
            if self.invert_scroll:
                scroll_amount = -scroll_amount
            if abs(scroll_amount) > 0:
                pyautogui.scroll(scroll_amount)

    def run(self):
        cap = cv2.VideoCapture(0)
        
        if not self.calibrate(cap):
            cap.release()
            cv2.destroyAllWindows()
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            result = self.detect_face_and_eyes(frame)
            if result is not None:
                (eyes, face_pos), head_pos = result
                for (ex, ey, ew, eh) in eyes:
                    eye_region = frame[ey:ey+eh, ex:ex+ew]
                    direction = self.analyze_eye_and_head(eye_region, face_pos, head_pos)
                    
                    cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                    
                    pupil = self.detect_pupil(cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY))
                    if pupil:
                        pupil_x, pupil_y = pupil
                        cv2.circle(eye_region, (pupil_x, pupil_y), 3, (255, 0, 0), -1)
                    
                    eye_center = (ex+ew//2, ey+eh//2)
                    self.draw_omnidirectional_arrow(frame, direction, eye_center)
                    
                    self.draw_text(frame, f"DIRECTION: ({direction[0]:.2f}, {direction[1]:.2f})", (10, 30), 0.5)

                    # Update scroll speed based on vertical direction
                    self.update_scroll_speed(direction[1])

            # Perform scrolling
            self.perform_scroll()

            self.draw_text(frame, "PRESS 'P' TO PAUSE/UNPAUSE", (10, frame.shape[0] - 150), 0.5)
            self.draw_text(frame, "PRESS 'I' TO INVERT SCROLL", (10, frame.shape[0] - 120), 0.5)
            self.draw_text(frame, "PRESS 'R' TO RECALIBRATE", (10, frame.shape[0] - 90), 0.5)
            self.draw_text(frame, "PRESS 'Q' TO QUIT", (10, frame.shape[0] - 60), 0.5)
            status = "PAUSED" if self.is_paused else "ACTIVE"
            self.draw_text(frame, f"STATUS: {status}", (10, 120), 0.5)
            self.draw_text(frame, f"SCROLL: {'INVERTED' if self.invert_scroll else 'NORMAL'}", (10, 150), 0.5)
            self.draw_text(frame, f"SCROLL SPEED: {self.scroll_speed:.2f}", (10, 180), 0.5)
            self.draw_sensitivity_bar(frame)

            cv2.imshow('Eye and Head Tracking', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                self.is_paused = not self.is_paused
            elif key == ord('i'):
                self.invert_scroll = not self.invert_scroll
            elif key == ord('r'):
                self.calibrate(cap)
            elif key == ord('w') or key == 82:  # 'w' key or Up arrow
                self.sensitivity = min(10, self.sensitivity + 1)
            elif key == ord('s') or key == 84:  # 's' key or Down arrow
                self.sensitivity = max(1, self.sensitivity - 1)
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = EyeHeadTracker()
    tracker.run()