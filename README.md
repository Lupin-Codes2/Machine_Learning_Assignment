# Machine_Learning_Assignment
A full-stack ML app using FastAPI and scikit-learn. train_models.py cleans the games dataset, trains a LinearRegression (votes) and a LogisticRegression (genre), reports R² and accuracy, and saves both with joblib. app.py loads them and serves two POST endpoints, /predict/regression and /predict/classification. An HTML page has a form for each.
