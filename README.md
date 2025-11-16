# Kaggle's Titanic: A Machine Learning Survival Prediction Model

This project is a solution to the classic Kaggle "Titanic: Machine Learning from Disaster" competition. The goal is to use a dataset of passenger information to build a model that predicts survival. This notebook represents my first hands-on project in machine learning, from data cleaning to model submission.

**Final Model Accuracy:** 78.6%

---

## My Workflow & Process

I built this model by following a complete, end-to-end data science process.

### 1. Setup and Data Cleaning (Cell 1)
* Imported `pandas`, `numpy`, and key `sklearn` libraries.
* Loaded `train.csv` and `test.csv`.
* To create a clean dataset for training, I dropped all rows from the training data that had missing values for 'Age' or 'Embarked'.

### 2. Feature Engineering & Imputation (Cell 2)
This was the most important phase. I created new features (`IsAlone`, `Title`) to give the model more information to learn from.

* **`FamilySize` & `IsAlone`:** I combined the `SibSp` (siblings/spouses) and `Parch` (parents/children) columns to create a single `FamilySize` feature. I then used this to create a new boolean feature `IsAlone`, based on the hypothesis that solo travelers had different survival rates.
* **`Title`:** I extracted the social title (e.g., 'Mr', 'Miss', 'Dr') from the `Name` column, as status likely played a role in survival. I also grouped rare titles into a single 'Rare' category to avoid noise.
* **Imputation:**
    * `Age`: Filled missing `Age` values using the median age of the passenger's 'Title' group (e.g., the median age of all 'Miss' passengers).
    * `Fare`: Filled one missing `Fare` value in the test set with the mean fare from the training set.
    * `Embarked`: Filled missing `Embarked` values with 'S' (the most common port).

### 3. Final Feature Selection (Cell 2)
* Selected the final features for the model: `Pclass`, `Sex`, `FamilySize`, `IsAlone`, `Embarked`, `Title`, `Fare`, and `Age`.
* Converted all categorical features (`Sex`, `Embarked`, `Title`) into numerical data using `pandas.get_dummies()`.
* Aligned the training and test set columns to ensure they had the exact same structure.

### 4. Model Training & Hyperparameter Tuning (Cells 3-4)
* **Model:** I chose a `RandomForestClassifier` for its power and reliability.
* **Validation:** I split the training data into a new, smaller training set and a validation set using `train_test_split` (Cell 3).
* **Tuning:** I wrote a helper function to test a list of `candidate_max_leaf_nodes` ([5, 25, 50, 100, 250, 500]). This loop tested each tree size and found that **`best_tree_size = 25`** produced the highest accuracy on the validation set (Cell 4).

### 5. Final Model & Submission (Cell 5)
* I trained a new `RandomForestClassifier` using the `best_tree_size` of 25.
* This final model was trained on the *entire* cleaned training dataset (all of `X` and `y`).
* The model was used to predict survival on the `X_test` data.
* The final predictions were formatted into a `submission.csv` file for the Kaggle competition.
