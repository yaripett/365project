# 365project
# NFL Game Predictor

## Project Overview

This project uses machine learning to predict the winners of NFL games based on previous team statistics. The goal is to determine whether historical offensive performance can be used to predict the outcome of NFL matchups.

The project uses a Python backend for machine learning and a React frontend that allows users to interact with the predictor.

Two machine learning models are used:
- Logistic Regression
- Random Forest

Both models are compared against a baseline that always predicts the home team will win.

## Technologies Used

- **Python** – Data processing and machine learning
- **React** – Frontend user interface
- **JavaScript** – Frontend functionality
- **Pandas** – Data cleaning and processing
- **NumPy** – Numerical operations
- **Scikit-learn** – Machine learning models
- **Kaggle** – NFL statistics dataset

## Dataset

The dataset used in this project comes from Kaggle.

**Dataset:** [NFL Stats 1999–2022 by Philip Hyde](https://www.kaggle.com/datasets/philiphyde1/nfl-stats-1999-2022)

### Downloading the Dataset

1. Visit the [Kaggle Dataset Page](https://www.kaggle.com/datasets/philiphyde1/nfl-stats-1999-2022).
2. Sign in to your Kaggle account or create one.
3. Click **Download** to download the dataset.
4. Extract the downloaded ZIP file.
5. Locate the file named `weekly_team_stats_offense.csv`.
6. Place the CSV file in your project's data directory or wherever your Python code expects it.

**Important:** The dataset must be downloaded separately from Kaggle before running the predictor.

## Installation and Setup

### Step 1: Clone the Repository

Open a terminal and run:

```bash
git clone YOUR_REPOSITORY_URL
cd YOUR_REPOSITORY_NAME
```

Replace the placeholders with your GitHub repository information.

### Step 2: Install Python Dependencies

Make sure Python 3 is installed.

Install the necessary libraries:

```bash
pip install pandas numpy scikit-learn
```

Install any additional packages required by your backend, such as Flask or FastAPI, if they are used in the project.

### Step 3: Download the NFL Dataset

Download the dataset from Kaggle:

https://www.kaggle.com/datasets/philiphyde1/nfl-stats-1999-2022

Extract the files and place `weekly_team_stats_offense.csv` in the location expected by your Python scripts.

### Step 4: Install React Dependencies

Make sure Node.js and npm are installed.

Open a terminal and navigate to the directory containing your React application's `package.json` file.

Run:

```bash
npm install
```

This installs the dependencies needed for the React frontend.

## Running the Project

The project has two connected components: a Python backend and a React frontend.

Both must be running for the complete web application to function.

### Step 1: Start the Python Backend

Open a terminal and navigate to the Python backend directory.

Start the backend using the command appropriate for your project.

For example, if your backend uses Flask and the server file is `app.py`:

```bash
python app.py
```

**Note:** Replace this command with the actual startup command for your backend.

The backend handles model predictions and provides the results to the React frontend through an API.

### Step 2: Start the React Frontend

Open a second terminal and navigate to your React application directory.

Run:

```bash
npm install
```

If your project uses Vite, start the React application with:

```bash
npm run dev
```

If your project uses Create React App, use:

```bash
npm start
```

Open the local URL displayed in your terminal to access the website.

### Step 3: Use the Predictor

Once both the backend and frontend are running:

1. Open the React application in your browser.
2. Select the NFL teams for the matchup.
3. Submit the matchup for prediction.
4. The React frontend sends the request to the Python backend.
5. The backend processes the information using the machine learning model.
6. The predicted winner is returned and displayed in the React application.

## How the Predictor Works

### 1. Data Preparation

The program loads the NFL dataset and filters the data to include regular-season games.

The project uses statistics from previous games rather than statistics from the game being predicted. This helps prevent data leakage.

### 2. Feature Selection

The predictor uses the average statistics from each team's previous four games.

Features include:

- Passing yards
- Rushing yards
- Interceptions
- Passing touchdowns
- Rushing touchdowns
- Passing first downs
- Rushing first downs
- Total yards

The program calculates the difference between the home team's and away team's previous four-game averages.

### 3. Machine Learning Models

**Logistic Regression:** Uses the statistical relationship between previous team performance and game outcomes to predict the probability of a home-team win.

**Random Forest:** Uses multiple decision trees to identify patterns in historical NFL statistics and predict game outcomes.

### 4. Model Evaluation

The models are evaluated using test games that were not included in the training data.

Their performance is compared against a baseline that always predicts the home team will win.

## Model Results

The following results were obtained after correcting the dataset labels and evaluating the models.

| Model | Accuracy |
|---|---|
| Always Predict Home Team | 52.21% |
| Logistic Regression | 59.31% |
| Random Forest | 59.07% |

A total of **408 games** were used for testing.

Logistic Regression achieved the highest accuracy, slightly outperforming Random Forest.

## Frontend and Backend Connection

The React frontend communicates with the Python backend through API requests.

When a user selects a matchup, the frontend sends the selected teams to the backend.

The backend uses the trained machine learning model to generate a prediction and returns the results to the frontend.

React then displays the predicted winner to the user.

Make sure the frontend API configuration points to the correct backend address.

## Project Goal

The purpose of this project is to explore how machine learning can be used to predict NFL game outcomes.

By using historical team performance statistics, the project compares multiple machine learning approaches and demonstrates how predictions can be presented through an interactive web application.

## Dataset Credit

The NFL dataset was created by **Philip Hyde** and is available on Kaggle.

Dataset Link: https://www.kaggle.com/datasets/philiphyde1/nfl-stats-1999-2022
