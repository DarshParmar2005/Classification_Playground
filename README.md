
# Classification Playground

Welcome to the **Classification Playground** — an interactive Shiny app that lets users explore and compare six popular machine learning classification models using the classic Titanic dataset. This app was developed as a personal data science project to showcase model interpretability, performance visualization, and user interaction.

**Live App**: [https://darshparmar.shinyapps.io/classification_playground](https://darshparmar.shinyapps.io/classification_playground)

---

## Models Included
- Decision Tree
- Random Forest
- Logistic Regression
- K-Nearest Neighbours
- Support Vector Machine
- XGBoost

Each model has its own tab with:
- Predictor selection (dependent on model type)
- Hyperparameter tuning
- Lollipop plot of performance metrics
- In-sample and out-of-sample confusion matrix
- PCA plot (some models)

---

## Features
- **Real-time model training** based on user-selected predictors and parameters
- **Confusion matrices** with visual accuracy summaries
- **Plots**: PCA, ROC curves, and feature importance
- **Performance comparison table** across models
- **Sidebar tooltips** that explain model parameters

---

## Dataset
This app uses the `Titanicp` dataset from the `vcdExtra` package in R, which includes the following features:
- `pclass`: Passenger class (1st, 2nd, 3rd)
- `sex`: Sex of the passenger
- `age`: Age in years
- `sibsp`: Number of siblings/spouses aboard
- `parch`: Number of parents/children aboard
- `survived`: Survival status (binary classification target)

---

## Libraries used
- **Frontend/UI**: R Shiny with `bslib`, `DT`, `reactable`, `plotOutput`, and `card` components
- **Models**: `rpart`, `randomForest`, `glm`, `caret::knn3`, `e1071::svm`, `xgboost`
- **Metrics & Plots**: `yardstick`, `pROC`, `ggplot2`, `reactablefmtr`
- **Data wrangling**: `tidyverse`

---

## How to Run Locally

### 1. Clone the repo:
```bash
git clone https://github.com/DarshParmar2005/classification-playground.git
cd classification-playground
```

### 2. Open RStudio and run:
```r
shiny::runApp()
```

> Make sure you have all required packages installed. You can install any missing ones using `install.packages("package-name")`.

---

## Motivation
This app was created as part of a personal project to:
- Practice implementing multiple ML models in R
- Learn Shiny for interactive data apps
- Provide a reusable and educational tool for students and data science beginners

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Author
**Darsh Parmar**   
The University of Sydney  
