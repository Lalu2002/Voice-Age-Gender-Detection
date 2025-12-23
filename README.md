# Voice-Age-Gender-Detection: End-to-End Deep Learning Pipeline

This project demonstrates the process of building a robust **Age and Gender Classification System** using raw audio data. We utilized **Convolutional Neural Networks (CNNs)** on Mel Spectrograms to treat audio analysis as a computer vision task, achieving high accuracy even on challenging, real-world data from the [Mozilla Common Voice Dataset](https://datacollective.mozillafoundation.org/datasets/cmflnuzw52mzok78yz6woemc1).

The project implements a **Multi-Task Learning** architecture and explores advanced strategies like **On-the-fly Data Augmentation** and **Class Balancing** to handle the massive 89GB dataset.

🔴 **Live Demo:** [Click here to try the deployed model on Hugging Face](https://lalu-prakash-voice-age-gender-detector.hf.space/?__theme=system&deep_link=l3n98TGSheo)

---

## Code Explanation (`Project Workflow`)

The codebase is a comprehensive, technical workflow demonstrating best practices in audio processing, CNN architecture design, and handling imbalanced datasets using PyTorch.

### **1. Data Preprocessing & Feature Engineering**

| Component | Technical Explanation |
| :--- | :--- |
| `Mozilla Common Voice` | The source dataset (~89GB) containing diverse speech samples. **[Download Link](https://datacollective.mozillafoundation.org/datasets/cmflnuzw52mzok78yz6woemc1)** |
| `Librosa` (Spectrogram Generation) | We convert raw `.mp3` audio into **Mel Spectrograms** (visual representations of frequency over time). We use `n_mels=128` and `n_fft=2048` to capture the specific "texture" of human voice timbre, which CNNs are excellent at analyzing. |
| `Creating_Balanced_Dataset.ipynb` | Addresses the **Extreme Class Imbalance** issue (the dataset is heavily skewed towards males in their 20s). This script downsamples the dataset to create a balanced subset of 10k-50k samples, ensuring the model doesn't just memorize the majority class. |
| `power_to_db` | Converts the amplitude of the audio to the **Decibel (dB)** scale. This normalization is crucial because human hearing (and voice distinctiveness) is logarithmic, not linear. |

### **2. Model Configuration (Multi-Task CNN)**

| Component | Technical Explanation |
| :--- | :--- |
| **Multi-Head Architecture** | Instead of training two separate models, we use a shared convolutional backbone that splits into two heads: one for **Gender (Binary)** and one for **Age (Multi-class)**. This improves efficiency and allows the model to learn shared features (like pitch) relevant to both tasks. |
| `Conv2d` & `BatchNorm` | We use 2D Convolutions with kernel size (3x3) followed immediately by **Batch Normalization**. This stabilizes the learning process and allows the model to converge faster on the complex spectrogram patterns. |
| **Dropout (0.5 - 0.6)** | Heavy regularization is applied in the fully connected layers. This was necessary to prevent overfitting, especially on the "Senior (60+)" age group where data was scarce. |

### **3. Training Execution and Efficiency**

| Component | Technical Explanation |
| :--- | :--- |
| `gender_age_model_10ksamples_augmented` | Implements **Data Augmentation** using `audiomentations`. We inject Gaussian Noise and Time Shifts during training. This forces the model to learn the *voice* rather than the silence or background static common in the raw dataset. |
| **Age Specialist Strategy** | To improve recall on older voices, we merged the "60s", "70s", and "80s" classes into a single **"50 Plus"** category. This simplified the problem space and significantly improved the model's ability to identify elderly speakers. |
| `Adam Optimizer` | Used with a learning rate of `0.001` and `CrossEntropyLoss`. We implemented early stopping to save the best weights based on validation loss, preventing the "over-training" that often occurs after epoch 15. |

### **4. Deployment and Evaluation**

| Component | Technical Explanation |
| :--- | :--- |
| `Final_evaluation_3_models.ipynb` | Loads the weights (`.pth`) from the Baseline, Augmented, and Age-Specialist models. It evaluates them on a blind test set to generate Confusion Matrices, proving the robustness of the final ensemble. |
| **Hugging Face Spaces** | The final model is deployed using **Gradio** on a CPU instance. The inference pipeline handles audio loading, trimming (to 5s), and spectrogram conversion in real-time. |

---

## How to Run the Project

### **Prerequisites & Environment Setup**

The project is designed to run in **Google Colab** due to the high computational requirements of spectrogram generation.

### **Prerequisites**

1.  **Hardware:** Requires a GPU (T4 or better). CPU training is not feasible for this architecture.
2.  **Dataset:** Download the [Mozilla Common Voice Dataset](https://datacollective.mozillafoundation.org/datasets/cmflnuzw52mzok78yz6woemc1) (~89GB).
3.  **Storage:** Ensure you have at least 100GB of free space on Google Drive.
4.  **Path Configuration:** You **must** update the file paths in the notebooks (e.g., `/content/drive/MyDrive/...`) to match your folder structure.

### **Execution Order**

Execute the notebooks in the following order to reproduce the pipeline:

```bash
# 1. Extract the massive dataset (Expect 6+ hours):
Run 'Dataset_extraction.ipynb'

# 2. Generate the spectrogram features (.npy files):
Run 'Audio_Feature_Extraction.ipynb'

# 3. Create the balanced train/test splits:
Run 'train_test_val_split.ipynb'

# 4. Train the models (Choose your strategy):
Run 'gender_and_age_model_10ksamples.ipynb'         # Baseline
Run 'best_age_model_50plus_samples.ipynb'           # High Accuracy Age Model

# 5. Evaluate results:
Run 'Final_evaluation_3_models.ipynb'
