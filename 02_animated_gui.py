# -*- coding: utf-8 -*-
"""
SIGNAL PROCESSING AND MACHINE LEARNING 2024-2025 FALL

FINAL : HUMAN ACTIVITY RECOGNITION

Script 02 : ANIMATED GUI FOR REAL-TIME PREDICTION


"""

# ==========================================
# 1. IMPORTATIONS
# ==========================================

import sys, serial, serial.tools.list_ports, numpy as np, joblib, threading, os, time
from collections import deque, Counter
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# ==========================================
# 1. KALMAN FILTER CLASS
# ==========================================
class SimpleKalman:
    def __init__(self, q=0.01, r=0.1):
        self.q = q 
        self.r = r 
        self.x = 0 
        self.p = 1 
        self.k = 0 

    def update(self, measurement):
        self.p = self.p + self.q
        self.k = self.p / (self.p + self.r)
        self.x = self.x + self.k * (measurement - self.x)
        self.p = (1 - self.k) * self.p
        return self.x

# ==========================================
# 2. SETTINGS AND CONSTANTS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WINDOW_SIZE = 25 
MODEL_PATH = os.path.join(BASE_DIR, "best_model_ml.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.joblib")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.joblib")

def extract_features_for_ml(segments):
    features_list = []
    for segment in segments:
        mean = np.mean(segment, axis=0); std = np.std(segment, axis=0)
        max_val = np.max(segment, axis=0); min_val = np.min(segment, axis=0)
        gyro_mag = np.sqrt(np.sum(np.square(segment[:, 0:3]), axis=1))
        accel_mag = np.sqrt(np.sum(np.square(segment[:, 3:6]), axis=1))
        mag_stats = [np.mean(gyro_mag), np.std(gyro_mag), np.mean(accel_mag), np.std(accel_mag)]
        zcr = np.mean(np.diff(np.sign(segment), axis=0) != 0, axis=0)
        features = np.concatenate([mean, std, max_val, min_val, mag_stats, zcr])
        features_list.append(features)
    return np.array(features_list)

# ==========================================
# 3. SERIAL WORKER CLASS
# ==========================================
class SerialWorker(QObject):
    data_received = pyqtSignal(str)
    prediction_made = pyqtSignal(str, float)

    def __init__(self, port):
        super().__init__()
        self.port = port; self.running = False
        self.scaler = joblib.load(SCALER_PATH)
        self.le = joblib.load(ENCODER_PATH)
        self.model = joblib.load(MODEL_PATH)
        self.buffer = deque(maxlen=WINDOW_SIZE)
        self.kalmans = [SimpleKalman(q=0.005, r=0.05) for _ in range(6)]
        self.history = deque(maxlen=5) 

    def run(self):
        try:
            ser = serial.Serial(self.port, 115200, timeout=0.001)
            self.running = True
            while self.running:
                if ser.in_waiting > 500: ser.reset_input_buffer()
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    self.data_received.emit(line)
                    try:
                        vals = [float(v) for v in line.split(',')]
                        if len(vals) == 6:
                            filtered_vals = [self.kalmans[i].update(vals[i]) for i in range(6)]
                            self.buffer.append(filtered_vals)
                            if len(self.buffer) == WINDOW_SIZE:
                                window_scaled = self.scaler.transform(np.array(self.buffer))
                                feat = extract_features_for_ml([window_scaled])
                                probs = self.model.predict_proba(feat)[0]
                                act = self.le.inverse_transform([np.argmax(probs)])[0]
                                conf = float(np.max(probs) * 100)
                                self.history.append(act)
                                counts = Counter(self.history)
                                stable_act, count = counts.most_common(1)[0]
                                if count >= 3:
                                    self.prediction_made.emit(str(stable_act), conf)
                    except: continue
                time.sleep(0.05) 
            ser.close()
        except Exception as e: print(f"HATA: {e}")

# ==========================================
# 4. MODERN GUI CLASS
# ==========================================
class HARWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_activity = ""
        self.initUI()

    def initUI(self):
        self.setWindowTitle("211805048_211805054_231805003 - Real Time HAR System")
        self.setFixedSize(1200, 850)
        
        # Global Stil (Glassmorphism ve Dark Theme)
        self.setStyleSheet("""
            QMainWindow { background-color: #0F111A; }
            QLabel { color: #E0E0E0; font-family: 'Segoe UI'; }
            QFrame#Card { background-color: #1C1E26; border-radius: 20px; border: 1px solid #2D303E; }
            QPushButton#StartBtn { background-color: #00D1B2; color: #0F111A; font-weight: bold; border-radius: 10px; padding: 12px; font-size: 11pt; }
            QPushButton#StartBtn:hover { background-color: #00B89C; }
            QPushButton#StopBtn { background-color: #FF2D55; color: white; font-weight: bold; border-radius: 10px; padding: 12px; font-size: 11pt; }
            QComboBox { background-color: #0F111A; color: white; border: 1px solid #3d3d5c; border-radius: 8px; padding: 10px; }
            QTextEdit { background-color: #08090F; color: #39FF14; border: none; border-radius: 10px; font-family: 'Consolas'; font-size: 9pt; }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(25)

        # ---LEFT PANEL (Control System) ---
        left_panel = QVBoxLayout()
        
        # Connection Card
        conn_card = QFrame(); conn_card.setObjectName("Card")
        conn_lay = QVBoxLayout(conn_card)
        conn_lay.setContentsMargins(20, 20, 20, 20)
        
        conn_title = QLabel("📡 SYSTEM CONNECTION")
        conn_title.setStyleSheet("font-size: 12pt; font-weight: bold; color: #00D1B2;")
        
        self.combo_ports = QComboBox(); self.refresh_ports()
        self.btn_start = QPushButton("START SYSTEM"); self.btn_start.setObjectName("StartBtn")
        self.btn_start.clicked.connect(self.toggle_process)
        
        conn_lay.addWidget(conn_title); conn_lay.addSpacing(10)
        conn_lay.addWidget(self.combo_ports); conn_lay.addSpacing(15)
        conn_lay.addWidget(self.btn_start)
        left_panel.addWidget(conn_card)

        # Data Flow Card
        log_card = QFrame(); log_card.setObjectName("Card")
        log_lay = QVBoxLayout(log_card)
        log_lay.addWidget(QLabel("📟 LIVE DATA STREAM (KALMAN)"))
        self.data_log = QTextEdit(); self.data_log.setReadOnly(True)
        log_lay.addWidget(self.data_log)
        left_panel.addWidget(log_card, 1) # Provide flexibility

        # ---RIGHT PANEL (Visual Analysis) ---
        right_panel = QVBoxLayout()

        # Animation Viewer
        anim_card = QFrame(); anim_card.setObjectName("Card")
        anim_card.setMinimumHeight(500)
        anim_lay = QVBoxLayout(anim_card)
        self.anim_label = QLabel()
        self.anim_label.setAlignment(Qt.AlignCenter)
        self.movie = QMovie(os.path.join(BASE_DIR, "animations/idle.gif"))
        self.anim_label.setMovie(self.movie); self.movie.start()
        anim_lay.addWidget(self.anim_label)
        right_panel.addWidget(anim_card)

        # Scorecard
        res_card = QFrame(); res_card.setObjectName("Card")
        res_lay = QVBoxLayout(res_card)
        res_lay.setContentsMargins(30, 20, 30, 20)
        
        self.lbl_prediction = QLabel("SYSTEM STANDBY")
        self.lbl_prediction.setAlignment(Qt.AlignCenter)
        self.lbl_prediction.setStyleSheet("font-size: 32pt; font-weight: bold; color: #58A6FF; letter-spacing: 2px;")
        
        # Modern Progress Bar
        self.prog_conf = QProgressBar()
        self.prog_conf.setFixedSize(600, 15)
        self.prog_conf.setTextVisible(False)
        self.prog_conf.setStyleSheet("""
            QProgressBar { background-color: #08090F; border-radius: 7px; border: none; }
            QProgressBar::chunk { background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00D1B2, stop:1 #3498DB); border-radius: 7px; }
        """)
        
        self.lbl_conf_val = QLabel("Confidence Rate: 0%")
        self.lbl_conf_val.setAlignment(Qt.AlignCenter)
        self.lbl_conf_val.setStyleSheet("font-size: 10pt; color: #8B949E;")

        res_lay.addWidget(self.lbl_prediction)
        res_lay.addWidget(self.prog_conf, 0, Qt.AlignCenter)
        res_lay.addWidget(self.lbl_conf_val)
        right_panel.addWidget(res_card)

        layout.addLayout(left_panel, 2)
        layout.addLayout(right_panel, 3)

    def refresh_ports(self): 
        self.combo_ports.clear()
        self.combo_ports.addItems([p.device for p in serial.tools.list_ports.comports()])

    def toggle_process(self):
        if hasattr(self, 'worker') and self.worker.running: 
            self.worker.running = False
            self.btn_start.setText("START SYSTEM")
            self.btn_start.setObjectName("StartBtn")
            self.btn_start.setStyle(self.btn_start.style()) # Style refresh
        else:
            port = self.combo_ports.currentText()
            if not port: return
            self.worker = SerialWorker(port)
            self.t = threading.Thread(target=self.worker.run, daemon=True)
            self.worker.data_received.connect(self.update_log_safe)
            self.worker.prediction_made.connect(self.handle_prediction)
            self.t.start()
            self.btn_start.setText("DISCONNECT")
            self.btn_start.setObjectName("StopBtn")
            self.btn_start.setStyle(self.btn_start.style())

    def update_log_safe(self, text):
        self.data_log.append(text)
        if self.data_log.document().blockCount() > 25: self.data_log.clear()
        self.data_log.verticalScrollBar().setValue(self.data_log.verticalScrollBar().maximum())

    def handle_prediction(self, activity, conf):
        self.prog_conf.setValue(int(conf))
        self.lbl_conf_val.setText(f"Confidence Rate: %{conf:.1f}")
        
        if activity != self.current_activity:
            self.current_activity = activity
            clean = "_".join(activity.lower().replace("_", " ").split())
            path = os.path.join(BASE_DIR, f"animations/{clean}.gif")
            if os.path.exists(path):
                self.movie.stop(); self.movie.setFileName(path); self.movie.start()
        
        color = "#00D1B2" if conf >= 70 else "#F1C40F"
        self.lbl_prediction.setText(activity.upper().replace("_", " "))
        self.lbl_prediction.setStyleSheet(f"font-size: 32pt; font-weight: bold; color: {color}; letter-spacing: 2px;")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HARWindow()
    window.show()
    sys.exit(app.exec_())