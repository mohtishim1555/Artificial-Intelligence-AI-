# ============================================================================
# AI-Based Vehicle Identification and Parking Management System for Toyota Dealership
# ============================================================================
# Authors: Saad Amjad, Mohtishim Fareed, Agib Seyed
# Description: Automatically identifies Toyota vehicles and grants free entry/parking
#              Charges non-Toyota vehicles standard parking fees
# ============================================================================

import cv2
import numpy as np
import sqlite3
import datetime
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import threading
import os
import json
from collections import deque

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    "camera_index": 0,                   # Default camera
    "frame_width": 640,                  # Reduced for better performance
    "frame_height": 480,                 # Reduced for better performance
    "detection_interval": 2,            # Seconds between auto-detections
    "confidence_threshold": 0.7,        # Minimum confidence for Toyota detection
    "parking_fee": 5.0,                 # Standard parking fee for non-Toyota
    "free_parking_hours": 2,            # Free parking hours for Toyota
    "paid_parking_rate": 2.0,           # Per hour rate for non-Toyota
    "database_path": "toyota_parking.db",
    "car_brands": ["toyota", "honda", "suzuki", "kia", "ford", "bmw", "mercedes", "hyundai", "nissan", "audi"]
}

# ============================================================================
# DATABASE MANAGER
# ============================================================================
class ParkingDatabase:
    def __init__(self, db_path=CONFIG["database_path"]):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.create_tables()
        self.load_config()
    
    def create_tables(self):
        """Create all necessary database tables"""
        # Vehicle entries table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS vehicle_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                license_plate TEXT,
                vehicle_brand TEXT,
                entry_time DATETIME,
                exit_time DATETIME,
                entry_type TEXT,  -- 'free' or 'paid'
                parking_fee REAL,
                payment_status TEXT,  -- 'paid', 'unpaid', 'free'
                confidence_score REAL,
                vehicle_image_path TEXT,
                logo_detected BOOLEAN,
                plate_detected BOOLEAN
            )
        ''')
        
        # Parking transactions table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS parking_transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_id INTEGER,
                transaction_time DATETIME,
                amount REAL,
                payment_method TEXT,
                status TEXT,
                FOREIGN KEY (entry_id) REFERENCES vehicle_entries(id)
            )
        ''')
        
        # System logs table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                event_type TEXT,
                vehicle_brand TEXT,
                license_plate TEXT,
                action TEXT,
                details TEXT
            )
        ''')
        
        # Configuration table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS configuration (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        
        self.conn.commit()
    
    def load_config(self):
        """Load configuration from database"""
        try:
            self.cursor.execute("SELECT key, value FROM configuration")
            rows = self.cursor.fetchall()
            for key, value in rows:
                if key in CONFIG:
                    try:
                        CONFIG[key] = json.loads(value)
                    except:
                        CONFIG[key] = value
        except:
            pass
    
    def save_config(self):
        """Save configuration to database"""
        for key, value in CONFIG.items():
            if isinstance(value, (list, dict)):
                value_str = json.dumps(value)
            else:
                value_str = str(value)
            self.cursor.execute('''
                INSERT OR REPLACE INTO configuration (key, value) 
                VALUES (?, ?)
            ''', (key, value_str))
        self.conn.commit()
    
    def log_vehicle_entry(self, license_plate, vehicle_brand, is_toyota, confidence, image_path):
        """Log vehicle entry into database"""
        entry_time = datetime.datetime.now()
        entry_type = "free" if is_toyota else "paid"
        parking_fee = 0.0 if is_toyota else CONFIG["parking_fee"]
        payment_status = "free" if is_toyota else "unpaid"
        
        self.cursor.execute('''
            INSERT INTO vehicle_entries 
            (license_plate, vehicle_brand, entry_time, entry_type, parking_fee, 
             payment_status, confidence_score, vehicle_image_path, logo_detected, plate_detected)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (license_plate, vehicle_brand, entry_time, entry_type, parking_fee,
              payment_status, confidence, image_path, True, license_plate is not None))
        
        entry_id = self.cursor.lastrowid
        
        # Log the event
        self.cursor.execute('''
            INSERT INTO system_logs (timestamp, event_type, vehicle_brand, 
                                   license_plate, action, details)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (entry_time, "ENTRY", vehicle_brand, license_plate, 
              "GRANTED" if is_toyota else "CHARGED", 
              f"Confidence: {confidence:.2%}"))
        
        self.conn.commit()
        return entry_id
    
    def log_vehicle_exit(self, entry_id):
        """Log vehicle exit and calculate final fee"""
        exit_time = datetime.datetime.now()
        
        # Get entry details
        self.cursor.execute('''
            SELECT entry_time, entry_type, parking_fee, payment_status 
            FROM vehicle_entries WHERE id = ?
        ''', (entry_id,))
        
        entry = self.cursor.fetchone()
        if not entry:
            return None
        
        entry_time_str, entry_type, initial_fee, payment_status = entry
        
        # Calculate parking duration
        if isinstance(entry_time_str, str):
            entry_time_dt = datetime.datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S")
        else:
            entry_time_dt = entry_time_str
        
        duration = exit_time - entry_time_dt
        hours_parked = duration.total_seconds() / 3600
        
        final_fee = 0.0
        
        if entry_type == "paid":
            # Calculate additional hours beyond initial period
            if hours_parked > 1:  # First hour included in initial fee
                additional_hours = hours_parked - 1
                additional_fee = additional_hours * CONFIG["paid_parking_rate"]
                final_fee = initial_fee + additional_fee
            else:
                final_fee = initial_fee
        
        # Update exit time and final fee
        self.cursor.execute('''
            UPDATE vehicle_entries 
            SET exit_time = ?, parking_fee = ?
            WHERE id = ?
        ''', (exit_time, final_fee, entry_id))
        
        # Log exit event
        self.cursor.execute('''
            INSERT INTO system_logs (timestamp, event_type, action, details)
            VALUES (?, ?, ?, ?)
        ''', (exit_time, "EXIT", "PROCESSED", f"Entry ID: {entry_id}, Hours: {hours_parked:.2f}, Fee: ${final_fee:.2f}"))
        
        self.conn.commit()
        return final_fee
    
    def record_payment(self, entry_id, amount, method="cash"):
        """Record payment for parking"""
        payment_time = datetime.datetime.now()
        
        self.cursor.execute('''
            INSERT INTO parking_transactions (entry_id, transaction_time, amount, payment_method, status)
            VALUES (?, ?, ?, ?, ?)
        ''', (entry_id, payment_time, amount, method, "completed"))
        
        self.cursor.execute('''
            UPDATE vehicle_entries 
            SET payment_status = 'paid' 
            WHERE id = ?
        ''', (entry_id,))
        
        self.conn.commit()
    
    def get_daily_report(self, date=None):
        """Generate daily report"""
        if date is None:
            date = datetime.datetime.now().date()
        
        # Get summary statistics
        self.cursor.execute('''
            SELECT 
                COUNT(*) as total_entries,
                COUNT(CASE WHEN entry_type = 'free' THEN 1 END) as free_entries,
                COUNT(CASE WHEN entry_type = 'paid' THEN 1 END) as paid_entries,
                SUM(CASE WHEN payment_status = 'paid' THEN parking_fee ELSE 0 END) as revenue,
                AVG(confidence_score) as avg_confidence
            FROM vehicle_entries 
            WHERE DATE(entry_time) = ?
        ''', (date,))
        
        result = self.cursor.fetchone()
        if result:
            return result
        return (0, 0, 0, 0.0, 0.0)
    
    def get_current_parked_vehicles(self):
        """Get list of currently parked vehicles"""
        self.cursor.execute('''
            SELECT id, license_plate, vehicle_brand, entry_time, entry_type, parking_fee
            FROM vehicle_entries 
            WHERE exit_time IS NULL
            ORDER BY entry_time DESC
        ''')
        return self.cursor.fetchall()
    
    def get_today_logs(self, limit=50):
        """Get today's system logs"""
        today = datetime.datetime.now().date()
        self.cursor.execute('''
            SELECT timestamp, event_type, vehicle_brand, license_plate, action, details
            FROM system_logs 
            WHERE DATE(timestamp) = ?
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (today, limit))
        return self.cursor.fetchall()
    
    def close(self):
        """Close database connection"""
        self.save_config()
        self.conn.close()

# ============================================================================
# SIMPLE VEHICLE DETECTION ENGINE (Without AI models)
# ============================================================================
class SimpleVehicleDetector:
    def __init__(self):
        # Detection cache to prevent duplicate detections
        self.detection_cache = deque(maxlen=10)
        self.last_detection_time = 0
        
        # Simple color-based Toyota logo detection (red color detection)
        self.toyota_red_lower = np.array([0, 120, 70])
        self.toyota_red_upper = np.array([10, 255, 255])
        
        print("✓ Simple Vehicle Detector initialized (Color-based detection)")
    
    def detect_toyota_color(self, image):
        """Simple color-based Toyota logo detection"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for red color (Toyota logo often has red)
        mask = cv2.inRange(hsv, self.toyota_red_lower, self.toyota_red_upper)
        
        # Count red pixels
        red_pixel_count = np.sum(mask > 0)
        total_pixels = image.shape[0] * image.shape[1]
        red_percentage = red_pixel_count / total_pixels
        
        # If significant red percentage, assume Toyota
        if red_percentage > 0.01:  # 1% of image is red
            confidence = min(red_percentage * 10, 0.95)  # Scale to confidence
            return {
                'is_toyota': True,
                'confidence': confidence,
                'red_percentage': red_percentage
            }
        
        return {
            'is_toyota': False,
            'confidence': 0.7,  # Default confidence for non-Toyota
            'red_percentage': red_percentage
        }
    
    def detect_car_region(self, image):
        """Simple car region detection using edge detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (likely the car)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add padding
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            
            return [x, y, x + w, y + h]
        
        # Fallback: center region
        height, width = image.shape[:2]
        return [width//4, height//4, 3*width//4, 3*height//4]
    
    def simulate_license_plate(self):
        """Simulate license plate detection (for demo)"""
        import random
        import string
        
        # Generate random plate (50% chance)
        if random.random() > 0.5:
            # Format: ABC 123
            letters = ''.join(random.choices(string.ascii_uppercase, k=3))
            numbers = ''.join(random.choices(string.digits, k=3))
            plate = f"{letters} {numbers}"
            return {
                'text': plate,
                'confidence': random.uniform(0.7, 0.95)
            }
        
        return None
    
    def simulate_car_brand(self):
        """Simulate car brand detection (for demo)"""
        import random
        
        brands = CONFIG["car_brands"]
        brand = random.choice(brands)
        is_toyota = (brand == "toyota")
        
        # Toyota has higher confidence
        if is_toyota:
            confidence = random.uniform(0.8, 0.98)
        else:
            confidence = random.uniform(0.7, 0.95)
        
        return {
            'class': brand,
            'confidence': confidence,
            'is_toyota': is_toyota
        }
    
    def process_vehicle(self, image):
        """Main function to process vehicle detection"""
        current_time = time.time()
        
        # Check if enough time has passed since last detection
        if current_time - self.last_detection_time < CONFIG["detection_interval"]:
            return None
        
        self.last_detection_time = current_time
        
        # Step 1: Detect car region
        car_region_box = self.detect_car_region(image)
        
        # Step 2: Simple color-based Toyota detection
        color_result = self.detect_toyota_color(image)
        
        # Step 3: Simulate car brand detection (for demo)
        # In real system, replace with actual AI model
        brand_result = self.simulate_car_brand()
        
        # Step 4: Simulate license plate detection (for demo)
        plate_result = self.simulate_license_plate()
        
        # Combine results - prefer color detection for Toyota
        if color_result['is_toyota']:
            is_toyota = True
            confidence = color_result['confidence']
            vehicle_brand = "toyota"
        else:
            is_toyota = brand_result['is_toyota']
            confidence = brand_result['confidence']
            vehicle_brand = brand_result['class']
        
        # Get license plate text
        license_plate = plate_result['text'] if plate_result else None
        
        # Check cache for duplicate detection
        detection_key = f"{license_plate}_{vehicle_brand}_{int(current_time/10)}"
        if detection_key in self.detection_cache:
            return None
        
        self.detection_cache.append(detection_key)
        
        # Save vehicle image
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        image_dir = "vehicle_images"
        os.makedirs(image_dir, exist_ok=True)
        image_path = f"{image_dir}/{timestamp}_{vehicle_brand}.jpg"
        cv2.imwrite(image_path, image)
        
        return {
            'is_toyota': is_toyota,
            'vehicle_brand': vehicle_brand,
            'license_plate': license_plate,
            'confidence': confidence,
            'image_path': image_path,
            'logo_detected': color_result['is_toyota'],
            'plate_detected': plate_result is not None,
            'detections': {
                'car_region': car_region_box,
                'color_result': color_result
            }
        }

# ============================================================================
# MAIN APPLICATION GUI
# ============================================================================
class ToyotaParkingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Toyota Dealership - AI Parking Management System")
        self.root.geometry("1200x800")
        
        # Initialize components
        self.db = ParkingDatabase()
        self.detector = SimpleVehicleDetector()
        
        # Camera setup
        self.cap = None
        self.camera_active = False
        self.auto_detection_active = False
        self.current_frame = None
        
        # Statistics
        self.stats_update_interval = 5000  # 5 seconds
        
        # Setup UI
        self.setup_ui()
        
        # Start with camera off
        self.toggle_camera()
        
        # Start statistics updater
        self.update_statistics()
        
        # Bind window closing event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_ui(self):
        """Setup the user interface"""
        # Configure styles
        style = ttk.Style()
        style.theme_use('clam')  # Use a modern theme
        
        # Configure custom styles
        style.configure("Title.TLabel", font=("Arial", 18, "bold"))
        style.configure("Status.TLabel", font=("Arial", 12))
        style.configure("Stats.TLabel", font=("Arial", 10))
        style.configure("Granted.TLabel", foreground="green", font=("Arial", 14, "bold"))
        style.configure("Denied.TLabel", foreground="red", font=("Arial", 14, "bold"))
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # ========== LEFT PANEL: Camera and Detection ==========
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Title
        title_frame = ttk.Frame(left_frame)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(title_frame, text="🚗 TOYOTA DEALERSHIP", style="Title.TLabel").pack(side=tk.LEFT)
        ttk.Label(title_frame, text="AI Parking Management System", 
                 font=("Arial", 10)).pack(side=tk.LEFT, padx=(10, 0))
        
        # Camera Display
        camera_frame = ttk.LabelFrame(left_frame, text="Entry Camera Feed", padding="5")
        camera_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.camera_label = ttk.Label(camera_frame, background="black", relief=tk.SUNKEN)
        self.camera_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Camera Controls
        controls_frame = ttk.Frame(left_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.camera_btn = ttk.Button(
            controls_frame, text="Start Camera", 
            command=self.toggle_camera, width=15
        )
        self.camera_btn.pack(side=tk.LEFT, padx=5)
        
        self.auto_detect_btn = ttk.Button(
            controls_frame, text="Auto-Detect: OFF", 
            command=self.toggle_auto_detect, width=15
        )
        self.auto_detect_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            controls_frame, text="Test Detection", 
            command=self.manual_detection, width=15
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            controls_frame, text="Load Image", 
            command=self.load_image_file, width=15
        ).pack(side=tk.LEFT, padx=5)
        
        # Detection Zone
        zone_frame = ttk.LabelFrame(left_frame, text="Detection Info", padding="10")
        zone_frame.pack(fill=tk.X)
        
        ttk.Label(zone_frame, text="Show a photo of a car to the camera", 
                 font=("Arial", 10)).pack(anchor=tk.W)
        ttk.Label(zone_frame, text="Toyota vehicles get FREE parking", 
                 foreground="green", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(5, 0))
        ttk.Label(zone_frame, text="Other brands are charged parking fee", 
                 foreground="red", font=("Arial", 10)).pack(anchor=tk.W)
        
        # ========== RIGHT PANEL: Information and Controls ==========
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, width=400)
        
        # Detection Results
        results_frame = ttk.LabelFrame(right_frame, text="Detection Results", padding="15")
        results_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.result_title = ttk.Label(
            results_frame, text="System Ready", 
            font=("Arial", 12, "bold")
        )
        self.result_title.pack(anchor=tk.W)
        
        self.result_details = ttk.Label(
            results_frame, text="Start camera and show car photo", 
            wraplength=350, justify=tk.LEFT
        )
        self.result_details.pack(anchor=tk.W, pady=5)
        
        self.confidence_label = ttk.Label(
            results_frame, text="", 
            font=("Arial", 10)
        )
        self.confidence_label.pack(anchor=tk.W, pady=5)
        
        self.action_label = ttk.Label(
            results_frame, text="", 
            font=("Arial", 12, "bold")
        )
        self.action_label.pack(anchor=tk.W, pady=10)
        
        # Statistics
        stats_frame = ttk.LabelFrame(right_frame, text="Today's Statistics", padding="15")
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack(fill=tk.BOTH, expand=True)
        
        self.stats_labels = {}
        stats_data = [
            ("Total Entries", "total_entries", "0"),
            ("Toyota (Free)", "toyota_entries", "0"),
            ("Other (Paid)", "other_entries", "0"),
            ("Revenue", "revenue", "$0.00"),
            ("Success Rate", "success_rate", "0%"),
            ("Parked Now", "parked_now", "0")
        ]
        
        for i, (title, key, default) in enumerate(stats_data):
            frame = ttk.Frame(stats_grid)
            frame.grid(row=i//3, column=i%3, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
            
            ttk.Label(frame, text=title, font=("Arial", 9)).pack(anchor=tk.W)
            label = ttk.Label(frame, text=default, font=("Arial", 12, "bold"))
            label.pack(anchor=tk.W)
            self.stats_labels[key] = label
        
        # Currently Parked Vehicles
        parked_frame = ttk.LabelFrame(right_frame, text="Currently Parked Vehicles", padding="10")
        parked_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Treeview for parked vehicles
        columns = ("ID", "Plate", "Brand", "Time", "Fee")
        self.parked_tree = ttk.Treeview(parked_frame, columns=columns, show="headings", height=6)
        
        column_widths = {"ID": 40, "Plate": 80, "Brand": 60, "Time": 70, "Fee": 60}
        for col in columns:
            self.parked_tree.heading(col, text=col)
            self.parked_tree.column(col, width=column_widths[col])
        
        scrollbar = ttk.Scrollbar(parked_frame, orient=tk.VERTICAL, command=self.parked_tree.yview)
        self.parked_tree.configure(yscrollcommand=scrollbar.set)
        
        self.parked_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Exit button
        button_frame = ttk.Frame(parked_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(
            button_frame, text="Mark Exit", 
            command=self.mark_vehicle_exit, width=12
        ).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(
            button_frame, text="Refresh", 
            command=self.update_statistics, width=12
        ).pack(side=tk.LEFT, padx=2)
        
        # Quick Actions
        quick_frame = ttk.LabelFrame(right_frame, text="Quick Actions", padding="10")
        quick_frame.pack(fill=tk.X)
        
        quick_grid = ttk.Frame(quick_frame)
        quick_grid.pack()
        
        ttk.Button(
            quick_grid, text="Generate Report", 
            command=self.generate_report, width=15
        ).grid(row=0, column=0, padx=5, pady=5)
        
        ttk.Button(
            quick_grid, text="View Logs", 
            command=self.view_logs, width=15
        ).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Button(
            quick_grid, text="Manual Entry", 
            command=self.manual_entry, width=15
        ).grid(row=1, column=0, padx=5, pady=5)
        
        ttk.Button(
            quick_grid, text="Clear Display", 
            command=self.clear_display, width=15
        ).grid(row=1, column=1, padx=5, pady=5)
    
    def toggle_camera(self):
        """Toggle camera on/off"""
        if not self.camera_active:
            # Start camera
            self.cap = cv2.VideoCapture(CONFIG["camera_index"])
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["frame_width"])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["frame_height"])
            
            if not self.cap.isOpened():
                messagebox.showerror("Camera Error", 
                    "Could not open camera. Please check:\n"
                    "1. Camera is connected\n"
                    "2. No other app is using camera\n"
                    "3. Camera drivers are installed")
                return
            
            self.camera_active = True
            self.camera_btn.config(text="Stop Camera")
            self.update_camera_feed()
        else:
            # Stop camera
            self.camera_active = False
            self.auto_detection_active = False
            self.auto_detect_btn.config(text="Auto-Detect: OFF")
            self.camera_btn.config(text="Start Camera")
            
            if self.cap:
                self.cap.release()
            
            self.camera_label.config(image='')
    
    def toggle_auto_detect(self):
        """Toggle automatic detection"""
        if not self.camera_active:
            messagebox.showwarning("Camera Off", "Please start camera first.")
            return
        
        self.auto_detection_active = not self.auto_detection_active
        
        if self.auto_detection_active:
            self.auto_detect_btn.config(text="Auto-Detect: ON")
            self.result_title.config(text="🔍 Auto-Detection Active")
            self.result_details.config(text="Looking for vehicles...")
        else:
            self.auto_detect_btn.config(text="Auto-Detect: OFF")
            self.result_title.config(text="Camera Active")
            self.result_details.config(text="Auto-detection turned off")
    
    def update_camera_feed(self):
        """Update camera feed and run auto-detection"""
        if self.camera_active and self.cap:
            ret, frame = self.cap.read()
            
            if ret:
                # Store current frame
                self.current_frame = frame.copy()
                
                # Run auto-detection if active
                if self.auto_detection_active:
                    self.run_auto_detection(frame)
                
                # Draw overlays
                display_frame = self.draw_overlays(frame)
                
                # Convert for display
                display_frame = cv2.resize(display_frame, (640, 480))
                rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_image)
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.camera_label.imgtk = imgtk
                self.camera_label.configure(image=imgtk)
            else:
                # Camera error
                self.camera_active = False
                self.camera_btn.config(text="Start Camera")
                messagebox.showerror("Camera Error", "Failed to read from camera.")
                return
            
            # Schedule next update
            self.root.after(30, self.update_camera_feed)
    
    def draw_overlays(self, frame):
        """Draw overlays on camera frame"""
        overlay = frame.copy()
        height, width = frame.shape[:2]
        
        # Draw detection zone
        zone_color = (0, 255, 255)  # Yellow
        zone_thickness = 2
        
        cv2.rectangle(overlay, (width//4, height//4), (3*width//4, 3*height//4), 
                     zone_color, zone_thickness)
        cv2.putText(overlay, "DETECTION ZONE", (width//4, height//4 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone_color, 2)
        
        # Draw status text
        if self.auto_detection_active:
            status_text = "AUTO-DETECT: ON"
            status_color = (0, 255, 0)  # Green
        else:
            status_text = "CAMERA LIVE"
            status_color = (255, 255, 255)  # White
        
        cv2.putText(overlay, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Draw instructions
        cv2.putText(overlay, "Show car photo here", (width//2 - 100, height//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return overlay
    
    def run_auto_detection(self, frame):
        """Run automatic vehicle detection"""
        try:
            # Process vehicle detection
            result = self.detector.process_vehicle(frame)
            
            if result:
                # Process entry
                self.process_detection_result(result)
        except Exception as e:
            print(f"Auto-detection error: {e}")
    
    def process_detection_result(self, result):
        """Process detection result and update UI/database"""
        is_toyota = result['is_toyota']
        vehicle_brand = result['vehicle_brand']
        license_plate = result['license_plate']
        confidence = result['confidence']
        image_path = result['image_path']
        
        # Log entry in database
        entry_id = self.db.log_vehicle_entry(
            license_plate, vehicle_brand, is_toyota, confidence, image_path
        )
        
        # Update UI
        if is_toyota:
            self.result_title.config(text="✅ TOYOTA DETECTED")
            self.result_details.config(
                text=f"Toyota vehicle identified\n"
                     f"License Plate: {license_plate or 'Not detected'}\n"
                     f"Confidence: {confidence:.1%}"
            )
            self.action_label.config(text="ENTRY GRANTED - FREE PARKING", foreground="green")
            
            # Show success message
            messagebox.showinfo("Access Granted", 
                              f"✅ Toyota vehicle detected!\n"
                              f"Confidence: {confidence:.1%}\n"
                              f"Free entry & parking granted.")
        else:
            fee = CONFIG["parking_fee"]
            self.result_title.config(text=f"⚠ {vehicle_brand.upper()} DETECTED")
            self.result_details.config(
                text=f"{vehicle_brand.upper()} vehicle identified\n"
                     f"License Plate: {license_plate or 'Not detected'}\n"
                     f"Confidence: {confidence:.1%}\n"
                     f"Parking Fee: ${fee:.2f}"
            )
            self.action_label.config(text="ENTRY GRANTED - PAYMENT REQUIRED", foreground="red")
            
            # Show payment message
            messagebox.showwarning("Parking Fee Required", 
                                 f"⚠ {vehicle_brand.upper()} vehicle detected\n"
                                 f"Confidence: {confidence:.1%}\n"
                                 f"Parking Fee: ${fee:.2f}\n"
                                 "Please pay at the counter.")
        
        # Update statistics
        self.update_statistics()
    
    def manual_detection(self):
        """Manually trigger detection"""
        if self.current_frame is None:
            messagebox.showwarning("No Image", "No camera image available. Please start camera.")
            return
        
        try:
            # Process vehicle detection
            result = self.detector.process_vehicle(self.current_frame)
            
            if result:
                self.process_detection_result(result)
            else:
                messagebox.showinfo("Detection", "No vehicle detected in image. Try again.")
                
        except Exception as e:
            messagebox.showerror("Detection Error", str(e))
    
    def load_image_file(self):
        """Load image from file for testing"""
        file_path = filedialog.askopenfilename(
            title="Select Vehicle Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            # Load image
            image = cv2.imread(file_path)
            if image is None:
                messagebox.showerror("Error", "Could not load image.")
                return
            
            # Store as current frame
            self.current_frame = image
            
            # Display image
            display_frame = cv2.resize(image, (640, 480))
            rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_image)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.camera_label.imgtk = imgtk
            self.camera_label.configure(image=imgtk)
            
            # Update status
            self.result_title.config(text="Image Loaded")
            self.result_details.config(text="Click 'Test Detection' to analyze")
            self.action_label.config(text="")
            
            # Enable manual detection
            messagebox.showinfo("Image Loaded", 
                              "Vehicle image loaded successfully.\n"
                              "Click 'Test Detection' to analyze the vehicle.")
    
    def update_statistics(self):
        """Update statistics display"""
        try:
            # Get daily report
            report = self.db.get_daily_report()
            
            if report:
                total, toyota, paid, revenue, avg_conf = report
                
                self.stats_labels["total_entries"].config(text=str(total))
                self.stats_labels["toyota_entries"].config(text=str(toyota))
                self.stats_labels["other_entries"].config(text=str(paid))
                self.stats_labels["revenue"].config(text=f"${revenue:.2f}")
                
                if total > 0:
                    success_rate = (toyota + paid) / total * 100
                else:
                    success_rate = 0
                
                self.stats_labels["success_rate"].config(text=f"{success_rate:.1f}%")
            
            # Update parked vehicles list
            parked_vehicles = self.db.get_current_parked_vehicles()
            self.stats_labels["parked_now"].config(text=str(len(parked_vehicles)))
            
            # Update treeview
            for item in self.parked_tree.get_children():
                self.parked_tree.delete(item)
            
            for vehicle in parked_vehicles:
                entry_id, plate, brand, entry_time, entry_type, fee = vehicle
                
                # Format entry time
                if isinstance(entry_time, str):
                    entry_str = entry_time[11:19]  # Extract time part
                else:
                    entry_str = entry_time.strftime("%H:%M:%S") if hasattr(entry_time, 'strftime') else str(entry_time)[11:19]
                
                # Format fee
                fee_str = "FREE" if entry_type == "free" else f"${fee:.2f}"
                
                self.parked_tree.insert("", "end", values=(
                    entry_id, plate or "N/A", brand.upper(), entry_str, fee_str
                ))
                
        except Exception as e:
            print(f"Error updating statistics: {e}")
    
    def mark_vehicle_exit(self):
        """Mark selected vehicle as exited"""
        selection = self.parked_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a vehicle to mark as exited")
            return
        
        item = self.parked_tree.item(selection[0])
        values = item['values']
        
        if not values:
            return
        
        entry_id = values[0]
        plate = values[1]
        
        try:
            # Calculate and record exit
            final_fee = self.db.log_vehicle_exit(entry_id)
            
            messagebox.showinfo("Exit Recorded", 
                              f"Vehicle {plate} marked as exited.\n"
                              f"Final parking fee: ${final_fee:.2f}")
            
            # Update lists
            self.update_statistics()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to record exit: {e}")
    
    def generate_report(self):
        """Generate daily report"""
        report = self.db.get_daily_report()
        
        if report:
            total, toyota, paid, revenue, avg_conf = report
            
            report_text = f"""
            DAILY PARKING REPORT
            ====================
            Date: {datetime.datetime.now().date()}
            
            Statistics:
            -----------
            Total Entries: {total}
            Toyota Vehicles: {toyota} (Free)
            Other Vehicles: {paid} (Paid)
            Total Revenue: ${revenue:.2f}
            Average Confidence: {avg_conf:.1%}
            
            Current Status:
            --------------
            Camera: {'ACTIVE' if self.camera_active else 'INACTIVE'}
            Auto-Detection: {'ACTIVE' if self.auto_detection_active else 'INACTIVE'}
            Currently Parked: {self.stats_labels['parked_now'].cget('text')}
            """
            
            # Save report to file
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            reports_dir = "reports"
            os.makedirs(reports_dir, exist_ok=True)
            filename = f"{reports_dir}/daily_report_{timestamp}.txt"
            
            with open(filename, "w") as f:
                f.write(report_text)
            
            # Show report
            report_window = tk.Toplevel(self.root)
            report_window.title("Daily Report")
            report_window.geometry("500x400")
            
            text_widget = tk.Text(report_window, wrap=tk.WORD)
            text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            text_widget.insert(tk.END, report_text)
            text_widget.config(state=tk.DISABLED)
            
            # Save button
            ttk.Button(report_window, text="Close", 
                      command=report_window.destroy).pack(pady=10)
            
            messagebox.showinfo("Report Generated", 
                              f"Daily report saved to:\n{filename}")
    
    def view_logs(self):
        """View system logs"""
        try:
            logs = self.db.get_today_logs(limit=50)
            
            # Create log window
            log_window = tk.Toplevel(self.root)
            log_window.title("System Logs - Today")
            log_window.geometry("700x500")
            
            # Text widget for logs
            text_widget = tk.Text(log_window, wrap=tk.WORD)
            text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Add logs to text widget
            text_widget.insert(tk.END, "SYSTEM LOGS - TODAY\n")
            text_widget.insert(tk.END, "="*50 + "\n\n")
            
            for log in logs:
                timestamp, event_type, brand, plate, action, details = log
                
                # Format timestamp
                if isinstance(timestamp, str):
                    time_str = timestamp[11:19]
                else:
                    time_str = timestamp.strftime("%H:%M:%S") if hasattr(timestamp, 'strftime') else str(timestamp)[11:19]
                
                log_entry = f"[{time_str}] {event_type}: {brand or 'N/A'} ({plate or 'N/A'}) - {action}\n"
                if details:
                    log_entry += f"   Details: {details}\n"
                log_entry += "\n"
                text_widget.insert(tk.END, log_entry)
            
            text_widget.config(state=tk.DISABLED)
            
            # Close button
            ttk.Button(log_window, text="Close", 
                      command=log_window.destroy).pack(pady=10)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load logs: {e}")
    
    def manual_entry(self):
        """Manual vehicle entry form"""
        manual_window = tk.Toplevel(self.root)
        manual_window.title("Manual Vehicle Entry")
        manual_window.geometry("400x300")
        
        # Form fields
        form_frame = ttk.Frame(manual_window, padding="20")
        form_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(form_frame, text="Manual Vehicle Entry", 
                 font=("Arial", 14, "bold")).grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # License Plate
        ttk.Label(form_frame, text="License Plate:").grid(row=1, column=0, sticky=tk.W, pady=5)
        plate_var = tk.StringVar()
        ttk.Entry(form_frame, textvariable=plate_var, width=20).grid(row=1, column=1, pady=5)
        
        # Vehicle Brand
        ttk.Label(form_frame, text="Vehicle Brand:").grid(row=2, column=0, sticky=tk.W, pady=5)
        brand_var = tk.StringVar(value="toyota")
        brand_combo = ttk.Combobox(form_frame, textvariable=brand_var, 
                                  values=CONFIG["car_brands"], state="readonly", width=18)
        brand_combo.grid(row=2, column=1, pady=5)
        
        # Entry Type
        ttk.Label(form_frame, text="Entry Type:").grid(row=3, column=0, sticky=tk.W, pady=5)
        type_var = tk.StringVar(value="free")
        type_frame = ttk.Frame(form_frame)
        type_frame.grid(row=3, column=1, pady=5, sticky=tk.W)
        ttk.Radiobutton(type_frame, text="Free (Toyota)", 
                       variable=type_var, value="free").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(type_frame, text="Paid (Other)", 
                       variable=type_var, value="paid").pack(side=tk.LEFT, padx=5)
        
        # Parking Fee
        ttk.Label(form_frame, text="Parking Fee:").grid(row=4, column=0, sticky=tk.W, pady=5)
        fee_var = tk.StringVar(value="5.00")
        ttk.Entry(form_frame, textvariable=fee_var, width=20).grid(row=4, column=1, pady=5)
        
        def save_manual_entry():
            """Save manual entry to database"""
            plate = plate_var.get().strip().upper()
            brand = brand_var.get()
            entry_type = type_var.get()
            fee = float(fee_var.get()) if entry_type == "paid" else 0.0
            
            if not plate:
                messagebox.showwarning("Missing Info", "Please enter license plate number.")
                return
            
            try:
                # Log manual entry
                timestamp = datetime.datetime.now()
                
                self.db.cursor.execute('''
                    INSERT INTO vehicle_entries 
                    (license_plate, vehicle_brand, entry_time, entry_type, 
                     parking_fee, payment_status, confidence_score, logo_detected, plate_detected)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (plate, brand, timestamp, entry_type, fee, 
                      "free" if entry_type == "free" else "unpaid", 
                      1.0, True, True))
                
                # Log system event
                self.db.cursor.execute('''
                    INSERT INTO system_logs 
                    (timestamp, event_type, vehicle_brand, license_plate, action, details)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (timestamp, "MANUAL_ENTRY", brand, plate, 
                      "GRANTED", "Manual entry created"))
                
                self.db.conn.commit()
                
                # Show success message
                if entry_type == "free":
                    messagebox.showinfo("Entry Recorded", 
                                      f"✅ Toyota vehicle manually entered.\n"
                                      f"License Plate: {plate}\n"
                                      f"Free parking granted.")
                else:
                    messagebox.showinfo("Entry Recorded", 
                                      f"⚠ {brand.upper()} vehicle manually entered.\n"
                                      f"License Plate: {plate}\n"
                                      f"Parking Fee: ${fee:.2f}")
                
                # Update statistics
                self.update_statistics()
                
                manual_window.destroy()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save entry: {e}")
        
        # Buttons
        button_frame = ttk.Frame(form_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="Save Entry", 
                  command=save_manual_entry, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", 
                  command=manual_window.destroy, width=12).pack(side=tk.LEFT, padx=5)
    
    def clear_display(self):
        """Clear the detection display"""
        self.result_title.config(text="System Ready")
        self.result_details.config(text="Start camera and show car photo")
        self.confidence_label.config(text="")
        self.action_label.config(text="")
        messagebox.showinfo("Display Cleared", "Detection display has been cleared.")
    
    def on_closing(self):
        """Handle application closing"""
        # Stop camera
        self.camera_active = False
        
        if self.cap:
            self.cap.release()
        
        # Close database
        self.db.close()
        
        # Close window
        self.root.destroy()

# ============================================================================
# SYSTEM INITIALIZATION AND STARTUP
# ============================================================================
def setup_directories():
    """Create necessary directories"""
    directories = [
        "vehicle_images",
        "captured_images",
        "reports"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Directory created/verified: {directory}")

def print_welcome_message():
    """Print welcome message and instructions"""
    print("\n" + "="*70)
    print("TOYOTA DEALERSHIP - AI PARKING MANAGEMENT SYSTEM")
    print("="*70)
    print("Project: AI-Based Vehicle Identification and Parking Management")
    print("Authors: Saad Amjad, Mohtishim Fareed, Agib Seyed")
    print("\nSYSTEM FEATURES:")
    print("1. Vehicle detection using color analysis")
    print("2. Simple Toyota logo detection (red color based)")
    print("3. License plate simulation")
    print("4. Free entry & parking for Toyota vehicles")
    print("5. Paid parking for non-Toyota vehicles")
    print("6. Real-time database logging and reporting")
    print("\nHOW TO USE:")
    print("1. Click 'Start Camera' to begin")
    print("2. Show a photo of a car to the camera")
    print("3. Toyota vehicles (red color) = FREE entry")
    print("4. Other vehicles = PAID entry ($5.00)")
    print("5. Use 'Test Detection' for manual detection")
    print("6. Check statistics in right panel")
    print("="*70 + "\n")

def check_requirements():
    """Check if required packages are installed"""
    try:
        import cv2
        import numpy as np
        import tkinter as tk
        from PIL import Image, ImageTk
        print("✓ All required packages are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        print("\nPlease install required packages:")
        print("pip install opencv-python pillow numpy")
        return False

def main():
    """Main application entry point"""
    # System initialization
    print("Initializing Toyota Dealership Parking System...\n")
    
    # Check requirements
    if not check_requirements():
        response = input("Do you want to continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Setup directories
    setup_directories()
    
    # Print welcome message
    print_welcome_message()
    
    try:
        # Create and start GUI
        root = tk.Tk()
        app = ToyotaParkingApp(root)
        
        # Start the application
        print("✓ System initialized successfully")
        print("✓ Starting GUI application...")
        root.mainloop()
        
    except Exception as e:
        print(f"✗ Error starting application: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have a webcam connected")
        print("2. Close other applications using the camera")
        print("3. Run as administrator if needed")
        input("\nPress Enter to exit...")

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    main()