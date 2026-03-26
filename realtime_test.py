import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
import collections

# --- 1. Model Definitions (Must match your training exactly) ---
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

# --- 4. Real-Time Loop ---
cap = cv2.VideoCapture(0)
sequence_buffer = collections.deque(maxlen=10) # Sliding window of 10 frames

print("System Ready. Press 'q' to exit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame = cv2.flip(frame, 1) # Mirror view
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    pred_s = "Detecting..."
    pred_q = "Detecting..."
    
    land_data = np.zeros(126)

    if results.multi_hand_landmarks:
        # Extract landmarks for up to 2 hands
        for h_idx, hand_lms in enumerate(results.multi_hand_landmarks):
            if h_idx >= 2: break
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            for l_idx, lm in enumerate(hand_lms.landmark):
                base_idx = h_idx * 63 + l_idx * 3
                land_data[base_idx:base_idx+3] = [lm.x, lm.y, lm.z]
        
        # --- Advanced Normalization (Same as training) ---
        land_data_reshaped = land_data.reshape(1, -1)
        # Calculate scale based on Hand 1 (Wrist to Middle Finger Base)
        wrist = land_data_reshaped[0, 0:3]
        m_base = land_data_reshaped[0, 27:30] 
        scale = np.linalg.norm(m_base - wrist) + 1e-6
        
        norm_land_data = np.zeros((1, 126))
        for h in range(2):
            offset = h * 63
            w_coord = land_data_reshaped[0, offset : offset+3]
            for j in range(21):
                idx = offset + j*3
                norm_land_data[0, idx : idx+3] = (land_data_reshaped[0, idx : idx+3] - w_coord) / scale

        # --- Prediction A: Static ---
        input_s = torch.Tensor(norm_land_data).to(device)
        with torch.no_grad():
            out_s = model_s(input_s)
            pred_s = classes[torch.argmax(out_s).item()]

        # --- Prediction B: Sequence ---
        sequence_buffer.append(norm_land_data[0])
        if len(sequence_buffer) == 10:
            input_q = torch.Tensor(np.array(sequence_buffer)).unsqueeze(0).to(device)
            with torch.no_grad():
                out_q = model_q(input_q)
                pred_q = classes[torch.argmax(out_q).item()]

    # --- Overlay Results ---
    # Draw a background box for text readability
    cv2.rectangle(frame, (0,0), (350, 130), (255, 255, 255), -1)
    cv2.putText(frame, f"STATIC (A): {pred_s}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f"SEQUENCE (B): {pred_q}", (10, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 0), 2)
    
    cv2.imshow('BSL MSc Project: Static vs Sequence', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()