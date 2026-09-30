# ChurnGuard: Customer Churn Predictor

## Run from Command Prompt (Windows)
1. Extract the zip, then open Command Prompt in the extracted folder:
       cd path\to\churn_app
2. Install requirements (one time):
       pip install -r requirements.txt
3. Start the app:
       streamlit run app.py
   It opens at http://localhost:8501

Shortcut: double-click setup_and_run.bat to do steps 2 and 3.
If you see a model-loading error, run: python train.py   (rebuilds churn_model.pkl)

## What's inside the app
- Predict: live risk gauge, revenue at risk, best retention ideas, what-if simulator, prediction history
- Batch upload: score a whole CSV, download ranked results
- Insights: churn rates by contract, tenure, internet, payment, bill size + feature importance
- Help & model: metrics explained, limits, troubleshooting

## Files
- app.py: Streamlit frontend
- churn_logic.py: predictions, what-if scenarios, insights
- train.py: trains the Random Forest and saves churn_model.pkl
- churn_model.pkl: trained model (full pipeline with encoding)
- customer_churn_dataset.csv: dataset
- .streamlit/config.toml: colour theme
- requirements.txt: dependencies

## Deploy free on Streamlit Community Cloud
1. Upload all files (including the .streamlit folder) to a GitHub repo.
2. Go to share.streamlit.io, sign in with GitHub, click Create app.
3. Choose the repo, main file app.py, then Deploy.
