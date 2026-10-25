# Full-Stack Machine Learning App
- A FastAPI web app that trains and serves two scikit-learn models using the included **Top 1000 games of all time.csv** dataset:

- **Vote-count regression:** LinearRegression predicts the number of votes from game ranking and release year. The held-out **R²** is printed during training and returned by the API.
- **Primary-genre classification:** LogisticRegression predicts a game's first listed genre from ranking, release year, and vote count. Genres with fewer than 15 examples are grouped as `Other`. The held-out **accuracy** is printed during training and returned by the API.

The CSV contains some blank or incomplete rows; the training script skips those rows and trains on the remaining valid records. Both fitted models and their metadata are saved as joblib files under models.

With the included CSV and the script's fixed 80/20 split, the current held-out scores are **R² = 0.4816** and **accuracy = 0.8400**. Training prints fresh scores when rerun. The genre data is imbalanced (most games are Action), so accuracy should be read in that context.

## Run locally

Requires Python 3.10 or newer.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_models.py
uvicorn app:app --reload
```
Open <http://127.0.0.1:8000> for the prediction forms. Interactive API documentation is at <http://127.0.0.1:8000/docs>.

## API
Both endpoints validate input and return a JSON prediction along with the relevant held-out model metric.

## GitHub Pages

The published GitHub Pages site is fully static. It loads the trained model parameters and game lookup data from `model-parameters.json` and runs both predictions in the visitor's browser, so it does not need a running FastAPI server. The classifier lets visitors search for a game by name and fills in its ranking, year, and vote count. Keep the model parameters and game list in this file in sync with the dataset whenever you retrain or update the data.
