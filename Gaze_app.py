import cv2
import numpy as np
import mediapipe as mp
from screeninfo import get_monitors
from collections import deque
import logging
import tkinter as tk
from tkinter import filedialog
import os
from sys import exit
import time
import csv
from datetime import datetime


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gaze_tracker.log"),
        logging.StreamHandler()
    ]
)

def get_screen_resolution():
    monitor = next(m for m in get_monitors() if m.is_primary)
    return monitor.width, monitor.height

def initialize_face_mesh():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8
    )
    return face_mesh, mp_face_mesh 

def get_landmark_coords(landmarks, indices, w, h):
    coords = []
    for i in indices:
        x = int(landmarks[i].x * w)
        y = int(landmarks[i].y * h)
        coords.append((x, y))
    return coords


class GazeTracker:
    def __init__(self):
        self.screen_width, self.screen_height = get_screen_resolution()
        self.cap = cv2.VideoCapture(0)
        self.face_mesh, self.mp_face_mesh = initialize_face_mesh()
       
        self.calibration_positions = [
            (20, 20),
            (self.screen_width - 20, 20),
            (20, self.screen_height - 20),
            (self.screen_width - 20, self.screen_height - 20),
            (self.screen_width // 2, self.screen_height // 2)
        ]
        
        self.current = 0
        self.calibrated = False
        self.transform_matrix = None
        self.gaze_points = []
        self.gaze_trail = deque(maxlen=30)
        self.heatmap = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        self.background_img = None
        self.eye_crop_box = None
        self.pupil_buffer = deque(maxlen=3)
        self.cropped_shape = None
        self.image_path = None
        self.all_gaze_points = []
        self.start_time = None
        self.frame_count = 0

        cv2.namedWindow('Dot Cycle', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Dot Cycle', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        self.show_instruction_dialog()

    def show_instruction_dialog(self):
        while True:
            dialog_img = np.zeros((300, 800, 3), dtype=np.uint8)
            instructions = [
                "L: load image",
                "C: start calibration",
                "S: save gaze points",
                "Q or ESC: quit"
            ]
            for i, text in enumerate(instructions):
                cv2.putText(dialog_img, text, (30, 50 + i * 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.imshow("Instructions", dialog_img)
            key = cv2.waitKey(100)
            if key == ord('l'):
                self.load_image()
                logging.info("Image loaded.")
            elif key == ord('c') and self.background_img is not None:
                logging.info("Starting calibration.")
                cv2.destroyWindow("Instructions")
                break
            elif key in [27, ord('q')]:
                cv2.destroyAllWindows()
                exit()

    def load_image(self):
        cv2.destroyWindow("Instructions")
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        path = filedialog.askopenfilename(
            title="Select Background Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        root.destroy()

        if not path or not os.path.exists(path):
            logging.error("Invalid image selected.")
            return

        self.image_path = path
        img = cv2.imread(path)
        if img is None:
            logging.error("Failed to load image.")
            return

        self.background_img = cv2.resize(img, (self.screen_width, self.screen_height))
        cv2.namedWindow("Instructions", cv2.WINDOW_NORMAL)

    def run(self):
        self.start_time = time.time()
        if self.cap.isOpened() is False:
            logging.error("Failed to open webcam.")
            exit()
        while self.cap.isOpened():            
            success, frame = self.cap.read()
            if not success:
                logging.error("Failed to capture frame from webcam.")
                continue
            self.frame_count += 1
            frame = self.preprocess_frame(frame)
            middle_point = self.process_landmarks(frame)

            self.handle_display(middle_point)
            self.handle_keypress(middle_point)

        self.cap.release()
        cv2.destroyAllWindows()

    def preprocess_frame(self, frame):
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (700, 540))
        return frame

    def process_landmarks(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            logging.warning("No face landmarks detected.")
            return None

        h, w, _ = frame.shape
        for face_landmarks in results.multi_face_landmarks:
            left_eye, right_eye = self.get_iris_centers(face_landmarks, w, h)

            for center in [left_eye, right_eye]:
                if center:
                    cx = sum(x for x, _ in center) // len(center)
                    cy = sum(y for _, y in center) // len(center)
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 0), -1)

            if self.eye_crop_box is None:
                crop_ids = [234, 162, 389, 454]
                coords = get_landmark_coords(face_landmarks.landmark, crop_ids, w, h)
                xs = []
                ys = []
                for pt in coords:
                    xs.append(pt[0])
                    ys.append(pt[1])
                self.eye_crop_box = (
                    max(min(xs) - 10, 0), min(max(xs) + 10, w),
                    max(min(ys) - 10, 0), min(max(ys) + 10, h)
                )

            if self.eye_crop_box:
                self.cropped_shape = frame[self.eye_crop_box[2]:self.eye_crop_box[3],
                                           self.eye_crop_box[0]:self.eye_crop_box[1]].shape
                return self.detect_pupil_center(frame)
        return None

    def get_iris_centers(self, face_landmarks, w, h):
        left_indices = [474, 475, 476, 477]
        right_indices = [469, 470, 471, 472]
        eye_coords = []

        for index in left_indices + right_indices:
            x = int(face_landmarks.landmark[index].x * w)
            y = int(face_landmarks.landmark[index].y * h)
            eye_coords.append((index, x, y))
        
        left_eye = []
        right_eye = []

        for idx, x, y in eye_coords:
            if idx in left_indices:
                left_eye.append((x, y))
            elif idx in right_indices:
                right_eye.append((x, y))
        return left_eye, right_eye

    def detect_pupil_center(self, frame):
        min_x, max_x, min_y, max_y = self.eye_crop_box
        cropped = frame[min_y:max_y, min_x:max_x]

        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (cropped.shape[1] * 5, cropped.shape[0] * 5))
        gray = cv2.GaussianBlur(gray, (9, 9), 0)
        _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        centers = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy))

        if centers:
            avg_x = sum(p[0] for p in centers) // len(centers)
            avg_y = sum(p[1] for p in centers) // len(centers)
            self.pupil_buffer.append((avg_x, avg_y))
            if len(self.pupil_buffer) == self.pupil_buffer.maxlen:
                smoothed = tuple(
                    sum(p[i] for p in self.pupil_buffer) // len(self.pupil_buffer) for i in (0, 1)
                )
                return smoothed
            return (avg_x, avg_y)
        return None

    def handle_display(self, middle_point):
        if not self.calibrated and self.current < 5:
            dot_img = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
            cv2.circle(dot_img, self.calibration_positions[self.current], 20, (255, 0, 255), -1)
            cv2.imshow('Dot Cycle', dot_img)
        elif self.calibrated and self.transform_matrix is not None and middle_point and self.background_img is not None:
            pupil_input = np.array([[middle_point]], dtype=np.float32)
            mapped = cv2.transform(pupil_input, self.transform_matrix)
            gaze_x, gaze_y = int(mapped[0][0][0]), int(mapped[0][0][1])
            self.gaze_trail.append((gaze_x, gaze_y))

            if 0 <= gaze_x < self.screen_width and 0 <= gaze_y < self.screen_height:
                self.all_gaze_points.append((gaze_x, gaze_y))  
                cv2.circle(self.heatmap, (gaze_x, gaze_y), 5, (0, 255, 255), -1)

            display_img = self.background_img.copy()
            for i, (x, y) in enumerate(self.gaze_trail):
                alpha = int(255 * (i + 1) / len(self.gaze_trail))
                cv2.circle(display_img, (x, y), 8, (0, alpha, alpha), -1)

            combined = cv2.addWeighted(display_img, 0.7, self.heatmap, 0.3, 0)
            cv2.imshow('Dot Cycle', combined)
    def export_to_csv(self):
        
        with open("gaze_data.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", "timestamp"])
            for x, y in self.all_gaze_points:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                writer.writerow([x, y, ts])
        logging.info("Gaze data exported to gaze_data.csv")

    def calculate_average_fps(self):        
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        print(f"Average FPS: {fps:.2f}")
        logging.info(f"Average FPS: {fps:.2f}")

    def handle_keypress(self, middle_point):
        key = cv2.waitKey(1)
        if key == ord('d') and not self.calibrated:
            if middle_point:
                self.gaze_points.append((middle_point, self.calibration_positions[self.current]))
                self.current += 1
            if self.current >= 5:
                self.map_eye_coordinates()
        elif key == ord('s'):
            np.save("gaze_points.npy", np.array(self.all_gaze_points))
            self.export_to_csv()
            logging.info("All gaze points saved to 'gaze_points.npy'.")
        elif key in [27, ord('q')]:
            self.calculate_average_fps()
            self.cap.release()
            cv2.destroyAllWindows()
            exit()

    def map_eye_coordinates(self):
        self.calibrated = True
        pupil_coords = np.array([pt[0] for pt in self.gaze_points], dtype=np.float32)
        screen_coords = np.array([pt[1] for pt in self.gaze_points], dtype=np.float32)
        self.transform_matrix, _ = cv2.estimateAffine2D(pupil_coords, screen_coords)
        logging.info("Calibration completed.")
        if self.background_img is None:
            logging.error("Background image not loaded.")
            exit()

class GazeReplayer:
    def __init__(self, delay=30):
        self.delay = delay
        self.background_img = None
        self.gaze_points = None

    def load_background_image(self):
        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select background image",
            filetypes=[("Images", "*.png *.jpg *.jpeg")])
        root.destroy()
        if not path or not os.path.exists(path):
            print("No valid image selected.")
            return
        self.background_img = cv2.imread(path)

    def load_gaze_points(self):
        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select gaze points file",
            filetypes=[("NumPy Array", "*.npy")])
        root.destroy()
        if not path or not os.path.exists(path):
            print("No valid file selected.")
            return
        self.gaze_points = np.load(path, allow_pickle=True)

    def show_instruction_dialog(self):
        instr = np.zeros((300, 800, 3), dtype=np.uint8)
        lines = [
            "L: load background image",
            "G: load gaze data (.npy)",
            "R: start replay",
            "Q or ESC: cancel"
        ]
        while True:
            img = instr.copy()
            for i, txt in enumerate(lines):
                cv2.putText(img, txt, (30, 50 + 50*i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow("Replay Instructions", img)
            key = cv2.waitKey(100) & 0xFF

            if key == ord('l'):
                self.load_background_image()
            elif key == ord('g'):
                self.load_gaze_points()
            elif key == ord('r'):
                if self.background_img is not None and self.gaze_points is not None:
                    cv2.destroyWindow("Replay Instructions")
                    return True
            elif key in (27, ord('q')):
                cv2.destroyAllWindows()
                return False

    def replay(self):
        screen_h, screen_w = self.background_img.shape[:2]
        heatmap = np.zeros_like(self.background_img, dtype=np.uint8)
        window_name = "Gaze Replay"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(window_name, self.background_img)

        for i, (x, y) in enumerate(self.gaze_points):
            frame = self.background_img.copy()
            alpha = int(255 * (i+1) / len(self.gaze_points))
            cv2.circle(frame, (x, y), 8, (0, alpha, alpha), -1)
            cv2.circle(heatmap, (x, y), 5, (0, 255, 255), -1)
            combined = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)
            cv2.imshow(window_name, combined)
            if cv2.waitKey(self.delay) in (27, ord('q')):
                cv2.destroyAllWindows()
                return

        final = cv2.addWeighted(self.background_img, 0.7, heatmap, 0.3, 0)
        cv2.putText(final, "Replay finished. Press any key to exit.",
                    (50, screen_h-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 4)
        cv2.imshow(window_name, final)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

class HeatmapVisualizer:
    def __init__(self, blur_radius=30):
        self.background = None
        self.gaze_points = None
        self.blur_radius = blur_radius

    def load_background(self):
        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select background image",
            filetypes=[("Images","*.jpg *.jpeg *.png *.bmp")])
        root.destroy()
        if path and os.path.exists(path):
            self.background = cv2.imread(path)

    def load_gaze_points(self):
        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select gaze .npy file",
            filetypes=[("NumPy Array","*.npy")])
        root.destroy()
        if path and os.path.exists(path):
            self.gaze_points = np.load(path, allow_pickle=True)

    def show_instruction_dialog(self):
        instr = np.zeros((300,800,3),dtype=np.uint8)
        lines = [
            "L: load background image",
            "G: load gaze data (.npy)",
            "H: show heatmap",
            "Q or ESC: cancel"
        ]
        while True:
            img = instr.copy()
            for i,txt in enumerate(lines):
                cv2.putText(img, txt, (30,50+50*i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow("Heatmap Instructions", img)
            key = cv2.waitKey(100) & 0xFF
            if key == ord('l'):
                self.load_background()
            elif key == ord('g'):
                self.load_gaze_points()
            elif key == ord('h'):
                if self.background is not None and self.gaze_points is not None:
                    cv2.destroyWindow("Heatmap Instructions")
                    return True
            elif key in (27, ord('q')):
                cv2.destroyAllWindows()
                return False

    def visualize(self):
        h, w = self.background.shape[:2]
        heat = np.zeros((h,w),dtype=np.float32)
        r = self.blur_radius
        for x,y in self.gaze_points:
            xi, yi = int(x), int(y)
            if 0 <= xi < w and 0 <= yi < h:
                cv2.circle(heat, (xi,yi), r, 1.0, -1)
        heat = cv2.GaussianBlur(heat, (0,0), sigmaX=r/2, sigmaY=r/2)
        heat_u8 = np.clip((heat/heat.max())*255,0,255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(self.background, 0.6, heat_color, 0.4, 0)
        cv2.imshow("Gaze Heatmap", overlay)
        cv2.waitKey(0)
        cv2.destroyWindow("Gaze Heatmap")

class GazeAppLauncher:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Gaze Tracking App")
        self.root.geometry("400x300")
        self.root.configure(bg="#f0f0f0")
        self.build_ui()

    def build_ui(self):
        tk.Label(self.root, text="Choose an option below to proceed:",
                 font=("Helvetica", 14), bg="#f0f0f0").pack(pady=10)

        tk.Button(self.root, text="Start Gaze Tracker",
                  font=("Helvetica", 12), width=25,
                  command=self.start_tracker).pack(pady=5)

        tk.Button(self.root, text="Show Heatmap",
                  font=("Helvetica", 12), width=25,
                  command=self.start_heatmap).pack(pady=5)

        tk.Button(self.root, text="Replay Gaze Data",
                  font=("Helvetica", 12), width=25,
                  command=self.start_replay).pack(pady=5)

        tk.Button(self.root, text="Exit",
                  font=("Helvetica", 12), width=25,
                  command=self.root.quit).pack(pady=5)

    def start_tracker(self):
        self.root.destroy()
        tracker = GazeTracker()
        tracker.run()

    def start_replay(self):
        self.root.destroy()
        replayer = GazeReplayer()
        if replayer.show_instruction_dialog():
            replayer.replay()

    
    def start_heatmap(self):
        self.root.destroy()
        viz = HeatmapVisualizer()
        if viz.show_instruction_dialog():
            viz.visualize()

    def run(self):
        self.root.mainloop()


# ---- Main Entry Point ----
if __name__ == "__main__":
    app = GazeAppLauncher()
    app.run()