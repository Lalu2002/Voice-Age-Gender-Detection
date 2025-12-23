import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import os

# --- 1. Config (Must Match Training) ---
SAMPLE_RATE = 22050
DURATION = 5
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FIXED_LENGTH = SAMPLE_RATE * DURATION

# Label Maps
GENDER_MAP = {0: 'female', 1: 'male'}
AGE_MAP_5 = {0: 'Teens', 1: 'Twenties', 2: 'Thirties', 3: 'Fourties', 4: '50 Plus'}

# --- 2. Model Architectures ---

# Model A (Gender - Baseline)
class VoiceCNN_Old(nn.Module):
    def __init__(self):
        super(VoiceCNN_Old, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1); self.bn1 = nn.BatchNorm2d(16); self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1); self.bn2 = nn.BatchNorm2d(32); self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1); self.bn3 = nn.BatchNorm2d(64); self.pool3 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc_shared = nn.Linear(64 * 16 * 27, 128); self.fc_bn = nn.BatchNorm1d(128); self.dropout = nn.Dropout(0.5)
        self.age_head = nn.Linear(128, 6) 
        self.gender_head = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = self.dropout(F.relu(self.fc_bn(self.fc_shared(x))))
        return self.age_head(x), self.gender_head(x)

# Model B (Age - Specialist)
class AgeCNN_New(nn.Module):
    def __init__(self):
        super(AgeCNN_New, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1); self.bn1 = nn.BatchNorm2d(32); self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1); self.bn2 = nn.BatchNorm2d(64); self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1); self.bn3 = nn.BatchNorm2d(128); self.pool3 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 16 * 27, 512); self.bn4 = nn.BatchNorm1d(512); self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 5)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = self.dropout(F.relu(self.bn4(self.fc1(x))))
        return self.fc2(x)

# --- 3. Load Models ---
device = torch.device('cpu') 

model_a = VoiceCNN_Old().to(device)
model_b = AgeCNN_New().to(device)

# Load weights (Ensure filenames match your upload!)
model_a.load_state_dict(torch.load("best_voice_model.pth", map_location=device))
model_b.load_state_dict(torch.load("best_age_model_50plus.pth", map_location=device))

model_a.eval()
model_b.eval()

# --- 4. Prediction Logic ---
def predict(audio_path):
    if audio_path is None:
        return "Please upload an audio file.", None, None

    # Load and Preprocess
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
    
    # Pad/Trim to exact length
    if len(audio) < FIXED_LENGTH:
        audio = np.pad(audio, (0, int(FIXED_LENGTH - len(audio))), 'constant')
    else:
        audio = audio[:int(FIXED_LENGTH)]
        
    spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    log_spec = librosa.power_to_db(spec, ref=np.max)
    
    # Normalization (Standardize)
    GLOBAL_MEAN = -66.5
    GLOBAL_STD = 14.5
    norm_spec = (log_spec - GLOBAL_MEAN) / GLOBAL_STD
    
    tensor = torch.tensor(norm_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        _, gender_logits = model_a(tensor)
        age_logits = model_b(tensor)
        
    # Format Outputs
    g_probs = torch.softmax(gender_logits, dim=1).cpu().numpy()[0]
    gender_output = {GENDER_MAP[i]: float(g_probs[i]) for i in range(2)}
    
    a_probs = torch.softmax(age_logits, dim=1).cpu().numpy()[0]
    age_output = {AGE_MAP_5[i]: float(a_probs[i]) for i in range(5)}
    
    return "Analysis Successful", gender_output, age_output

# --- 5. UI Interface ---
description = """
<center>
<h2>🎙️ AI Voice Analysis: Age & Gender Detection</h2>
<p>Upload a short voice recording (e.g., say 'Hello, how are you?') to analyze speaker characteristics.</p>
</center>
"""

iface = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="filepath", label="🎤 Record or Upload Audio"),
    outputs=[
        gr.Label(label="Status"),
        gr.Label(label="Gender Prediction"),
        gr.Label(label="Age Prediction")
    ],
    title="Voice Profiler AI",
    description=description,
    theme="default" 
)

iface.launch()