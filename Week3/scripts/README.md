Data Analysis Project
This repository contains scripts for performing Exploratory Data Analysis (EDA), A/B Hypothesis Testing, and Statistical Modeling. Below is a brief overview of each component and instructions for how to use the provided scripts.

Table of Contents
Exploratory Data Analysis (EDA)
A/B Hypothesis Testing
Statistical Modeling
Usage Instructions
Dependencies
License
Exploratory Data Analysis (EDA)
The EDA scripts provide a comprehensive analysis of the dataset, including data cleaning, summary statistics, and visualizations.

Scripts
eda_analysis.py: This script performs the following tasks:
Loads and preprocesses the data.
Identifies and handles missing values.
Provides summary statistics and basic visualizations.
Performs univariate and bivariate analyses.
Key Functions
load_data(file_path): Loads the dataset from the specified file path.
preprocess_data(): Cleans and preprocesses the data.
generate_summary_statistics(): Computes summary statistics for the dataset.
visualize_data(): Creates visualizations to understand data distributions and relationships.
A/B Hypothesis Testing
The A/B testing scripts help in evaluating the impact of changes or interventions by comparing two groups.

Scripts
ab_testing.py: This script conducts A/B testing using statistical methods to compare the performance of two groups.
Key Functions
perform_ab_test(group_a, group_b): Performs statistical tests to compare the two groups.
Statistical Modeling
The statistical modeling scripts focus on building and evaluating predictive models.

Scripts
statistical_modeling.py: This script includes functions for data preprocessing, model training, and evaluation.
Key Functions
preprocess_data(columns, sample_fraction): Prepares the data for modeling by handling missing values and encoding categorical variables.
split_data(target_column, columns): Splits the data into training and testing sets.
train_model(X_train, y_train): Trains a predictive model.
evaluate_model(X_test, y_test): Evaluates the model’s performance on the test data.
Usage Instructions
Install Dependencies: Ensure you have the required libraries installed. You can use pip to install the necessary packages.

bash
Copy code
pip install -r requirements.txt
Run EDA Analysis:

bash
Copy code
python eda_analysis.py --file_path path/to/your/data.csv
Perform A/B Testing:

bash
Copy code
python ab_testing.py --group_a path/to/group_a.csv --group_b path/to/group_b.csv
Run Statistical Modeling:

bash
Copy code
python statistical_modeling.py --file_path path/to/your/data.csv --target_column TotalPremium --feature_columns Gender_Female Gender_Male Country_South Africa VehicleType_Bus
Dependencies
pandas
numpy
scikit-learn
matplotlib
seaborn
statsmodels
You can install these dependencies using pip:

bash
Copy code
pip install pandas numpy scikit-learn matplotlib seaborn statsmodels
License
This project is licensed under the MIT License. See the LICENSE file for details.