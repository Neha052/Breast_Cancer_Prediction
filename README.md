# \*\*Breast Cancer Prediction\*\*

## \*\*UX Research + Machine Learning Demo\*\*

🔗 **Live Streamlit App:** [https://your-app-name.streamlit.app](https://breastcancerprediction-6gkmjv3vnbdkbjbcx3nday.streamlit.app/)

!\[Python](https://img.shields.io/badge/Python-3.9+-blue)

!\[scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)

!\[Streamlit](https://img.shields.io/badge/Streamlit-App-red)



This project demonstrates how \*\*machine learning models can be translated into human-centered, decision-support tools\*\*.  
It combines \*\*data analysis, model development, and UX-aware deployment\*\* using an interactive \*\*Streamlit web application\*\*.
The app predicts whether a tumor is \*\*malignant\*\* or \*\*benign\*\* using the \*\*Breast Cancer Wisconsin dataset\*\*.


## \*\*🚀 How to Run the Project (Quick Start)\*\*

You can run this project locally in under \*\*5 minutes\*\*.

### \*\*1. Clone the repository\*\*

git clone https://github.com/<your-username>/breast-cancer-streamlit.git

cd breast-cancer-streamlit

2\. Install dependencies

pip install -r requirements.txt

3\. Run the Streamlit app

streamlit run app.py

4\. Open in your browser

http://localhost:8501

🎯 Why This Project Matters (UX + ML Lens)

This project is intentionally designed as a stakeholder-facing prototype, not just a modeling exercise.

It demonstrates:
* How ML outputs can be communicated clearly to non-technical users
* Use of probability scores to convey uncertainty
* End-to-end ownership: data → model → interface → deployment
* Translation of technical analysis into actionable insights
* This mirrors real-world workflows in UX research, applied ML, and health-tech product teams.
  
📊 Dataset Overview

!Source: sklearn.datasets.load\_breast\_cancer
Samples: 569
Features: 30 numeric clinical measurements
Target: Binary classification
Malignant
Benign

🧠 Machine Learning Workflow

Load dataset using scikit-learn
Exploratory data analysis (EDA) and feature understanding
Feature scaling and train–test split
Train and evaluate multiple classification models
Select the best-performing model
Serialize the final model (.pkl)
Deploy as an interactive Streamlit application
Model training and evaluation are documented in the Jupyter notebook.

🖥️ Streamlit App Capabilities

Dataset preview and basic EDA
Interactive feature input
Real-time predictions
Class probability outputs for interpretability
The interface prioritizes clarity, readability, and decision support over visual complexity.

🧪 UX Research Framing

Methods, Assumptions, and Risks

This project is framed as a decision-support prototype, not a fully autonomous diagnostic system.

Research Methods (Applied)

Although no live user testing was conducted, the project simulates early-stage UX research and prototyping practices:

*Data-informed design: Feature selection reflects clinically relevant variables
*Explainability-first approach: Probability scores help users reason about uncertainty
*Progressive disclosure: Exploration (EDA) and prediction are separated to reduce cognitive load
*Prototype as research artifact: The app supports early stakeholder feedback and iteration

Mitigations in the interface:

* Probabilistic outputs
* Clear demonstration framing
* Separation of exploration and prediction



Ethical Considerations

This project is for educational and demonstration purposes only. It is not intended for medical diagnosis or treatment. Any real-world deployment would require:
Clinical validation, Domain expert review, Ethical and regulatory oversight

📁 Project Structure

BreastCancerPrediction/

│

├── app.py                     # Streamlit application

├──final\_model.pkl            # Trained ML model

├── requirements.txt

├── README.md

└── Breast\_cancer\_prediction.ipynb   # Model training notebook

☁️ Deployment

The application is deployed using Streamlit Community Cloud directly from this GitHub repository.
Automatic redeployment on every push to main.
No manual server configuration required.

🛠️ Tools \& Technologies

* Python
* scikit-learn
* pandas
* NumPy
* Streamlit
* Git \& GitHub
* Pickle (model persistence)



✅ What This Project Demonstrates

* Bridging machine learning and UX research
* Designing interpretable ML-driven interfaces
* Communicating uncertainty responsibly
* Building production-ready ML prototypes
* Applying UX thinking in high-stakes domains

👤 Author

Neha Ingale
UX Research • Machine Learning • Data Analysis

