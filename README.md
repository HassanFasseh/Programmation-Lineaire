# Programmation Lineaire

This project is a linear programming web application built with **FastAPI** and **NumPy**. It provides a simple interface to solve linear optimization problems via a web interface.

## 🚀 Features

- Solve linear programming problems using the Simplex method.
- Web-based interface for ease of use.
- Fast and lightweight backend using FastAPI.
- JSON API endpoint for optimization.

## 🛠️ Technologies Used

- Python 3.x
- FastAPI
- NumPy
- Uvicorn (ASGI server)

## 📦 Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/HassanFasseh/Programmation-Lineaire.git
   cd Programmation-Lineaire
2.Create and activate a virtual environment (optional but recommended):
  python -m venv .venv
  .venv\Scripts\activate  # On Windows
  source .venv/bin/activate  # On macOS/Linux

  pip install -r requirements.txt


 3.Running the App
 uvicorn main:app --reload
 Then open your browser and go to http://127.0.0.1:8000.
