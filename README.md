# Bird Species Identification Using Audio Signals  
*A Machine Learning and Neural Network Approach*

The system processes bird audio clips, extracts key features, and classifies them using both traditional ML models and deep learning architectures. It outputs the predicted bird name and its image.
---

---
## 🎯 Features

- Audio input support (.wav format)  
- Noise reduction and signal preprocessing  
- MFCC feature extraction  
- Image display of predicted bird  
---
## 🛠 Technologies Used

- **Programming Language:** Python  
- **Libraries:**  
  - Librosa (Audio Feature Extraction)  
  - TensorFlow / PyTorch (Deep Learning)  
  - Scikit-learn (ML Models)  
  - OpenCV (Display Images)  
  - Matplotlib, Seaborn (Visualization)  
  - Flask for GUI  
---
## ⚙️ Installation (Using Anaconda)

1. **Clone the repository**
```bash
git clone https://github.com/your-username/bird-audio-classification.git
cd bird-audio-classification
Create and activate the conda environment

bash
conda create -n myproject python=3.9.7
conda activate myproject

Install packages
pip install jupyter notebook==1.0.0 pandas==1.4.3 numpy==1.21.5 seaborn==0.11.2 tensorflow==2.8.2 scikit-learn==1.1.1 spacy==3.4.1 mlxtend==0.19.0 openpyxl==3.0.10 Flask==2.2.2 opencv-python==4.5.5.62 Pillow==9.2.0 tqdm==4.64.0 imbalanced-learn==0.9.1 librosa==0.9.1 scikit-image==0.19.2 PyYAML==6.0 python-bidi==0.4.2 torch==1.8.0

Download spaCy language models
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
python -m spacy download en_core_web_lg
 ## 📸 Example Output

### 🟢 Register Page  
This is the registration interface where users can create an account to use the bird identification system.

![Register Page](register_page.png)
![Login Page](login_page.png)
![Bird Name & Image Display](bird_nameimage_display.png)

