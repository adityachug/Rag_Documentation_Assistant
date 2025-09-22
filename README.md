# RAG Documentation Assistant  

A **Streamlit-based app** that integrates with **PyTorch** and supports Retrieval-Augmented Generation (RAG) for documentation assistance.  

This project runs in an isolated **Python 3.11 environment** to ensure compatibility with Streamlit and PyTorch.  

---

## 🚀 Features
- Interactive Streamlit interface  
- PyTorch model integration  
- Retrieval-Augmented Generation (RAG) pipeline for documentation queries  
- Caching for efficient model loading  

---

## 📦 Requirements
- **Python 3.11.8** (do not use 3.12 or 3.13 due to PyTorch incompatibility)  
- Streamlit `1.30.0`  
- PyTorch `2.2.0`  
- Torchvision `0.17.1`  
- Torchaudio `2.2.1`  

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create virtual environment (Python 3.11)
```powershell
# Windows PowerShell
"C:\Path\To\Python311\python.exe" -m venv .venv_py311
.venv_py311\Scripts\Activate.ps1
```

```bash
# Linux / macOS
python3.11 -m venv .venv_py311
source .venv_py311/bin/activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install streamlit==1.30.0
pip install torch==2.2.0 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cpu
```

---

## ▶️ Run the App
```bash
streamlit run app.py --server.runOnSave=false
```

Then open your browser at:  
👉 http://localhost:8501  

---

## 📂 Project Structure
```
.
├── app.py                  # Main Streamlit app
├── requirements.txt        # Dependencies
├── .streamlit/
│   └── config.toml         # Streamlit config
├── .venv_py311/            # Virtual environment (ignored in git)
└── README.md               # Project documentation
```

---

## ⚠️ Notes
- Always activate `.venv_py311` before running the app.  
- If you close VS Code or PowerShell, reactivate it with:  
  ```powershell
  .venv_py311\Scripts\Activate.ps1
  ```  
- For GPU support, install the correct PyTorch wheel from [pytorch.org](https://pytorch.org/get-started/locally/).  
