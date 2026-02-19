import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time
import subprocess
import platform
from ultralytics import YOLO

# Disable PyAutoGUI fail-safe
pyautogui.FAILSAFE = False

class ObjectRecognitionSystem:
    def __init__(self):
        self.model_loaded = False
        try:
            # Load YOLOv11 model - automatically downloads on first run
            print("🚀 Loading YOLOv11 model...")
            self.model = YOLO('yolo11n.pt')  # 'n' = nano (fastest), 's' = small, 'm' = medium
            self.model_loaded = True
            print("✅ YOLOv11 loaded - 80 object classes supported!")
            print("📊 Model: YOLOv11-nano (optimized for real-time)")
            
        except Exception as e:
            print(f"⚠️  Object detection disabled: {e}")
            print("💡 Install: pip install ultralytics")
            self.model_loaded = False
        
        self.last_detection_time = 0
        self.detection_interval = 0.1  # 10 FPS for object detection
        self.detected_objects = []
        self.confidence_threshold = 0.5  # YOLO confidence threshold
    
    def detect_objects(self, frame):
        """YOLOv11 object detection"""
        if not self.model_loaded:
            return []
        
        current_time = time.time()
        if current_time - self.last_detection_time < self.detection_interval:
            return self.detected_objects
        
        self.last_detection_time = current_time
        
        try:
            # Run YOLOv11 inference
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            
            detected = []
            
            # Process results
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Get confidence and class
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    
                    detected.append({
                        'class': class_name,
                        'confidence': confidence,
                        'box': (x1, y1, x2, y2)
                    })
            
            self.detected_objects = detected
            return detected
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []

class GestureController:
    def __init__(self):
        # Object Recognition System with YOLOv11
        self.object_system = ObjectRecognitionSystem()
        
        # MediaPipe setup - ENHANCED
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Support 2 hands
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Screen dimensions
        self.screen_w, self.screen_h = pyautogui.size()
        
        # Camera dimensions
        self.cam_w, self.cam_h = 640, 480
        
        # Smoothing parameters - OPTIMIZED
        self.smoothing = 4
        self.prev_x, self.prev_y = 0, 0
        
        # Movement buffer
        self.position_buffer = []
        self.buffer_size = 4
        
        # Gesture state tracking - IMPROVED
        self.click_delay = 0.2
        self.last_click_time = 0
        self.scroll_sensitivity = 40
        
        # Finger tip IDs
        self.tip_ids = [4, 8, 12, 16, 20]
        
        # Running flag
        self.running = True
        
        # FPS tracking
        self.prev_time = 0
        self.fps_history = []
        
        # Object detection mode
        self.object_detection_mode = True  # ON by default
        
        # Gesture history for stability
        self.last_gesture = None
        self.gesture_hold_frames = 0
        
    def count_fingers(self, landmarks):
        """Enhanced finger counting"""
        fingers = []
        
        # Thumb - improved detection
        thumb_tip = landmarks[self.tip_ids[0]]
        thumb_ip = landmarks[self.tip_ids[0] - 1]
        thumb_mcp = landmarks[2]
        
        # Better thumb detection using multiple points
        if thumb_tip.x < thumb_ip.x - 0.015:
            fingers.append(1)
        else:
            fingers.append(0)
            
        # Other fingers - improved
        for id in range(1, 5):
            tip = landmarks[self.tip_ids[id]]
            pip = landmarks[self.tip_ids[id] - 2]
            mcp = landmarks[self.tip_ids[id] - 3]
            
            # More reliable detection
            if tip.y < pip.y - 0.008:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers
    
    def get_distance(self, p1, p2):
        """Calculate distance between two points"""
        return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
    
    def smooth_position(self, x, y):
        """Enhanced smoothing"""
        self.position_buffer.append((x, y))
        
        if len(self.position_buffer) > self.buffer_size:
            self.position_buffer.pop(0)
        
        # Weighted average (more weight to recent positions)
        weights = [i+1 for i in range(len(self.position_buffer))]
        total_weight = sum(weights)
        
        avg_x = sum(pos[0] * w for pos, w in zip(self.position_buffer, weights)) / total_weight
        avg_y = sum(pos[1] * w for pos, w in zip(self.position_buffer, weights)) / total_weight
        
        return avg_x, avg_y
    
    def move_mouse(self, index_finger):
        """Smooth mouse movement"""
        x = int(np.interp(index_finger.x, [0.05, 0.95], [0, self.screen_w]))
        y = int(np.interp(index_finger.y, [0.05, 0.95], [0, self.screen_h]))
        
        x, y = self.smooth_position(x, y)
        
        curr_x = self.prev_x + (x - self.prev_x) / self.smoothing
        curr_y = self.prev_y + (y - self.prev_y) / self.smoothing
        
        pyautogui.moveTo(curr_x, curr_y)
        self.prev_x, self.prev_y = curr_x, curr_y
        
        return int(curr_x), int(curr_y)
    
    def process_gestures(self, landmarks, img):
        """Enhanced gesture processing"""
        fingers = self.count_fingers(landmarks)
        finger_count = fingers.count(1)
        
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        current_time = time.time()
        gesture_text = ""
        
        # GESTURE 1: Index finger only - Mouse movement
        if finger_count == 1 and fingers[1] == 1:
            x, y = self.move_mouse(index_tip)
            gesture_text = "🖱️ MOUSE CONTROL"
            cv2.circle(img, (int(index_tip.x * self.cam_w), int(index_tip.y * self.cam_h)), 
                      15, (0, 255, 0), cv2.FILLED)
        
        # GESTURE 2: Thumb + Index Pinch - Left Click
        elif finger_count == 2 and fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 0:
            distance = self.get_distance(thumb_tip, index_tip)
            
            cv2.line(img, 
                    (int(thumb_tip.x * self.cam_w), int(thumb_tip.y * self.cam_h)),
                    (int(index_tip.x * self.cam_w), int(index_tip.y * self.cam_h)),
                    (0, 255, 255), 3)
            
            if distance < 0.05:
                gesture_text = "⬅️ LEFT CLICK!"
                cv2.circle(img, (int(thumb_tip.x * self.cam_w), int(thumb_tip.y * self.cam_h)), 
                          30, (0, 255, 0), cv2.FILLED)
                
                if (current_time - self.last_click_time) > self.click_delay:
                    pyautogui.click()
                    self.last_click_time = current_time
                    print("✓ Left Click")
            else:
                gesture_text = "LEFT CLICK READY"
        
        # GESTURE 3: Closed Fist - Right Click
        elif finger_count == 0 or (finger_count == 1 and fingers[0] == 1):
            gesture_text = "➡️ RIGHT CLICK!"
            
            palm_x = int(landmarks[9].x * self.cam_w)
            palm_y = int(landmarks[9].y * self.cam_h)
            cv2.circle(img, (palm_x, palm_y), 40, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (palm_x, palm_y), 45, (255, 255, 255), 3)
            
            if (current_time - self.last_click_time) > self.click_delay:
                pyautogui.rightClick()
                self.last_click_time = current_time
                print("✓ Right Click")
        
        # GESTURE 4: Four fingers - Screenshot
        elif finger_count == 4:
            if (current_time - self.last_click_time) > 1.5:
                screenshot = pyautogui.screenshot()
                filename = f'screenshot_{int(time.time())}.png'
                screenshot.save(filename)
                self.last_click_time = current_time
                gesture_text = f"📸 SCREENSHOT SAVED!"
                print(f"✓ Screenshot saved: {filename}")
        
        # GESTURE 5: Five fingers - Volume
        elif finger_count == 5:
            hand_height = index_tip.y
            if hand_height < 0.25:
                pyautogui.press('volumeup')
                gesture_text = "🔊 VOLUME UP"
            elif hand_height > 0.75:
                pyautogui.press('volumedown')
                gesture_text = "🔉 VOLUME DOWN"
            else:
                gesture_text = "VOLUME CONTROL (Move hand up/down)"
        
        # GESTURE 6: Index + Middle - Scroll
        elif finger_count == 2 and fingers[1] == 1 and fingers[2] == 1:
            hand_height = (index_tip.y + middle_tip.y) / 2
            
            cv2.circle(img, (int(index_tip.x * self.cam_w), int(index_tip.y * self.cam_h)), 
                      12, (255, 255, 0), cv2.FILLED)
            cv2.circle(img, (int(middle_tip.x * self.cam_w), int(middle_tip.y * self.cam_h)), 
                      12, (255, 255, 0), cv2.FILLED)
            
            if hand_height < 0.35:
                pyautogui.scroll(self.scroll_sensitivity)
                gesture_text = "⬆️ SCROLL UP"
            elif hand_height > 0.65:
                pyautogui.scroll(-self.scroll_sensitivity)
                gesture_text = "⬇️ SCROLL DOWN"
            else:
                gesture_text = "SCROLL READY"
        
        # GESTURE 7: Thumb + Pinky - Play/Pause
        elif finger_count == 2 and fingers[0] == 1 and fingers[4] == 1:
            if (current_time - self.last_click_time) > self.click_delay:
                pyautogui.press('playpause')
                self.last_click_time = current_time
                gesture_text = "⏯️ PLAY/PAUSE"
                print("✓ Play/Pause")
        
        # GESTURE 8: Ring + Pinky - Mute
        elif finger_count == 2 and fingers[3] == 1 and fingers[4] == 1:
            if (current_time - self.last_click_time) > self.click_delay:
                pyautogui.press('volumemute')
                self.last_click_time = current_time
                gesture_text = "🔇 MUTE/UNMUTE"
                print("✓ Mute Toggle")
        
        # GESTURE 9: Index + Ring - Tab Switch
        elif finger_count == 2 and fingers[1] == 1 and fingers[3] == 1:
            if (current_time - self.last_click_time) > self.click_delay:
                pyautogui.hotkey('alt', 'tab')
                self.last_click_time = current_time
                gesture_text = "🔄 WINDOW SWITCH"
                print("✓ Alt+Tab")
        
        # GESTURE 10: Middle + Ring - Open App
        elif finger_count == 2 and fingers[2] == 1 and fingers[3] == 1:
            if (current_time - self.last_click_time) > 1.5:
                self.open_application()
                self.last_click_time = current_time
                gesture_text = "🧮 OPENING CALCULATOR"
                print("✓ Calculator Opened")
        
        # GESTURE 11: Three fingers - Double Click (BONUS)
        elif finger_count == 3 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
            if (current_time - self.last_click_time) > self.click_delay:
                pyautogui.doubleClick()
                self.last_click_time = current_time
                gesture_text = "⏸️ DOUBLE CLICK"
                print("✓ Double Click")
        
        return gesture_text
    
    def open_application(self):
        """Open calculator"""
        system = platform.system()
        try:
            if system == "Windows":
                subprocess.Popen('calc.exe')
            elif system == "Darwin":
                subprocess.Popen(['open', '-a', 'Calculator'])
            elif system == "Linux":
                subprocess.Popen(['gnome-calculator'])
        except Exception as e:
            print(f"Error opening app: {e}")
    
    def calculate_fps(self):
        """Calculate smooth FPS"""
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time) if self.prev_time > 0 else 0
        self.prev_time = current_time
        
        # Smooth FPS display
        self.fps_history.append(fps)
        if len(self.fps_history) > 10:
            self.fps_history.pop(0)
        
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        return int(avg_fps)
    
    def run(self):
        """Main loop - OPTIMIZED WITH YOLOv11"""
        cap = cv2.VideoCapture(0)
        cap.set(3, self.cam_w)
        cap.set(4, self.cam_h)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print("="*70)
        print("🚀 AI GESTURE CONTROL + YOLOv11 OBJECT DETECTION")
        print("="*70)
        print("\n✨ FEATURES:")
        print("  • Ultra-smooth hand gesture recognition")
        print("  • YOLOv11 real-time object detection (80 classes)")
        print("  • 2-hand support")
        print("  • Optimized for real-time performance")
        
        print("\n🎯 GESTURE CONTROLS:")
        print("  1.  ☝️  Index finger          → Mouse movement")
        print("  2.  👌 Thumb + Index pinch   → Left click")
        print("  3.  ✊ Closed fist           → Right click")
        print("  4.  ✌️  Index + Middle       → Scroll up/down")
        print("  5.  🖐️  Five fingers         → Volume control")
        print("  6.  🤟 Thumb + Pinky         → Play/Pause")
        print("  7.  🤘 Ring + Pinky          → Mute/Unmute")
        print("  8.  🤙 Index + Ring          → Alt+Tab")
        print("  9.  ✋ Four fingers          → Screenshot")
        print(" 10.  🤚 Middle + Ring         → Open Calculator")
        print(" 11.  🖖 Three fingers         → Double Click")
        
        print("\n🔍 YOLOv11 OBJECT DETECTION:")
        print("  • Detects: Person, Car, Phone, Laptop, Cup, Book, and 74+ more")
        print("  • Real-time bounding boxes with confidence scores")
        print("  • Much faster and accurate than MobileNet-SSD")
        
        print("\n⌨️  KEYBOARD CONTROLS:")
        print("  • Press 'o' → Toggle Object Detection ON/OFF")
        print("  • Press 's' → Take Screenshot")
        print("  • Press 'q' → Quit")
        print("="*70 + "\n")
        
        while self.running:
            success, img = cap.read()
            if not success:
                continue
                
            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # YOLOv11 Object Detection
            if self.object_detection_mode:
                detected_objects = self.object_system.detect_objects(img)
                
                for obj in detected_objects:
                    x1, y1, x2, y2 = obj['box']
                    label = f"{obj['class'].upper()}"
                    confidence_text = f"{obj['confidence']*100:.0f}%"
                    
                    # Enhanced visual style - Neon green/cyan
                    color = (0, 255, 200)
                    
                    # Draw bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label background
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(img, (x1, y1-30), (x1 + max(label_size[0], 60), y1), color, -1)
                    
                    # Draw text
                    cv2.putText(img, label, (x1 + 3, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    cv2.putText(img, confidence_text, (x1 + 3, y1 - 35),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Process hand gestures
            results = self.hands.process(img_rgb)
            
            gesture_text = "No hand detected"
            fps = self.calculate_fps()
            hands_count = 0
            
            if results.multi_hand_landmarks:
                hands_count = len(results.multi_hand_landmarks)
                
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand with custom style
                    self.mp_draw.draw_landmarks(
                        img, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                        self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=3)
                    )
                    
                    landmarks = hand_landmarks.landmark
                    gesture_text = self.process_gestures(landmarks, img)
            
            # Enhanced UI
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (self.cam_w, 140), (0, 0, 0), -1)
            img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
            
            # Gesture info
            cv2.putText(img, f"Gesture: {gesture_text}", (15, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # FPS and stats
            fps_color = (0, 255, 0) if fps > 20 else (0, 255, 255) if fps > 15 else (0, 0, 255)
            cv2.putText(img, f"FPS: {fps}", (15, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
            
            # Hands count
            cv2.putText(img, f"Hands: {hands_count}", (15, 105), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Object detection status
            obj_status = "YOLO ON" if self.object_detection_mode else "YOLO OFF"
            obj_color = (0, 255, 0) if self.object_detection_mode else (128, 128, 128)
            cv2.putText(img, f"🔍 {obj_status}", (self.cam_w - 160, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, obj_color, 2)
            
            # Controls hint
            cv2.putText(img, "Press: O=YOLO | S=Screenshot | Q=Quit", 
                       (self.cam_w - 400, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            cv2.imshow("AI Gesture + YOLOv11 Detection", img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n👋 Shutting down...")
                self.running = False
                break
            elif key == ord('o'):
                self.object_detection_mode = not self.object_detection_mode
                status = "ON" if self.object_detection_mode else "OFF"
                print(f"\n🔍 YOLOv11 Detection: {status}")
            elif key == ord('s'):
                screenshot = pyautogui.screenshot()
                filename = f'manual_screenshot_{int(time.time())}.png'
                screenshot.save(filename)
                print(f"📸 Screenshot saved: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ System stopped successfully!")

if __name__ == "__main__":
    try:
        controller = GestureController()
        controller.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
