import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
import collections
import time

# ... [Keep the Model Classes (BSLStaticModel, BSLSequenceModel) from before] ...
class BSLStaticModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(BSLStaticModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.network(x)

class BSLSequenceModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(BSLSequenceModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=2, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# --- NEW: Evaluation Counters ---
class LiveEvaluator:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.static_preds = []
        self.seq_preds = []
        self.target = None
        self.start_time = None

evaluator = LiveEvaluator()

# --- [Keep the Loading Logic for model_s, model_q, classes] ---
# --- 2. Load Assets ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load the class names we saved in the notebook
classes = np.load('classes.npy', allow_pickle=True)

model_s = BSLStaticModel(126, len(classes)).to(device)
model_q = BSLSequenceModel(126, 64, len(classes)).to(device)

# Load the trained weights
model_s.load_state_dict(torch.load('bsl_static_model.pth', map_location=device))
model_q.load_state_dict(torch.load('bsl_sequence_model.pth', map_location=device))
model_s.eval()
model_q.eval()

# --- 3. MediaPipe Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils
# --- UPDATED Real-Time Loop ---
cap = cv2.VideoCapture(0)
sequence_buffer = collections.deque(maxlen=10)

# Stability Tracking
last_pred_s = None
last_pred_q = None
flips_s = 0
flips_q = 0
frame_count = 0

print("CONTROLS:")
print("1-9, A-Z: Set Ground Truth Target")
print("Space: Reset Metrics")
print("Q: Quit and Save Report")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    frame_count += 1
    current_pred_s = "None"
    current_pred_q = "None"
    
    if results.multi_hand_landmarks:
        # ... [Keep Landmark Extraction & Normalization Logic] ...

        # --- Prediction A (Static) ---
        input_s = torch.Tensor(norm_land_data).to(device)
        with torch.no_grad():
            out_s = model_s(input_s)
            prob_s = torch.softmax(out_s, dim=1).max().item() # Confidence Score
            current_pred_s = classes[torch.argmax(out_s).item()]
            
            # Count Jitter
            if current_pred_s != last_pred_s:
                flips_s += 1
            last_pred_s = current_pred_s

        # --- Prediction B (Sequence) ---
        sequence_buffer.append(norm_land_data[0])
        if len(sequence_buffer) == 10:
            input_q = torch.Tensor(np.array(sequence_buffer)).unsqueeze(0).to(device)
            with torch.no_grad():
                out_q = model_q(input_q)
                prob_q = torch.softmax(out_q, dim=1).max().item()
                current_pred_q = classes[torch.argmax(out_q).item()]
                
                # Count Jitter
                if current_pred_q != last_pred_q:
                    flips_q += 1
                last_pred_q = current_pred_q

        # --- Log for Accuracy if Target is set ---
        if evaluator.target:
            evaluator.static_preds.append(current_pred_s == evaluator.target)
            evaluator.seq_preds.append(current_pred_q == evaluator.target)

    # --- UI Elements ---
    cv2.rectangle(frame, (0,0), (600, 180), (255, 255, 255), -1)
    
    target_text = f"TARGET: {evaluator.target if evaluator.target else 'Not Set (Press Key)'}"
    cv2.putText(frame, target_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Show Predictions + Stability (Flips per frame)
    cv2.putText(frame, f"STATIC: {current_pred_s} | Jitter: {flips_s}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f"SEQUENCE: {current_pred_q} | Jitter: {flips_q}", (10, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 0), 2)
    
    # Real-time Accuracy Display
    if evaluator.target and len(evaluator.static_preds) > 0:
        acc_s = sum(evaluator.static_preds)/len(evaluator.static_preds)
        acc_q = sum(evaluator.seq_preds)/len(evaluator.seq_preds)
        cv2.putText(frame, f"Live Acc - S: {acc_s:.2%} | Q: {acc_q:.2%}", (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

    cv2.imshow('BSL MSc Real-Time Evaluation', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        evaluator.reset()
        flips_s, flips_q = 0, 0
    elif key != 255: # If any other key is pressed
        evaluator.target = chr(key).upper()
        evaluator.static_preds, evaluator.seq_preds = [], []

# --- 5. Final Report Generation ---
print("\n--- REAL-TIME COMPARISON REPORT ---")
print(f"Total Frames Processed: {frame_count}")
print(f"Static Model Flips (Jitter): {flips_s}")
print(f"Sequence Model Flips (Jitter): {flips_q}")
print(f"Stability Improvement: {((flips_s - flips_q) / (flips_s + 1)) * 100:.1f}% smoother")
# Save this to a text file for the friend to show
with open("realtime_results.txt", "w") as f:
    f.write(f"Static Flips: {flips_s}\nSequence Flips: {flips_q}\n")

cap.release()
cv2.destroyAllWindows()