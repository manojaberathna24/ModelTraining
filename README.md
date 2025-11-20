                                # Car Damage Classifier

              Detect **Dent** or **Scratch** from car images using a simple trained SVM model.

---

## How to Run

1. **Clone the repo**

```bash
git clone [https://github.com/username/car_damage_classifier.git](https://github.com/manojaberathna24/ModelTraining.git)
cd car_damage_classifier


Create and activate virtual environment
    python -m venv venv
    venv\Scripts\activate

Install dependencies    
pip install -r requirements.txt

Train model(optional)
    python train.py

Run the web app
    streamlit run app/streamlit_app.py

Command line prediction (optional)
    python app/predict.py dataset/dent/dent_01.jpg
