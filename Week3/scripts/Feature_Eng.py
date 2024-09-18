import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self, data, target_variable, categorical_features, numerical_features):
        self.data = data
        self.target_variable = target_variable
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.preprocessor = None
        self.X_preprocessed = None
    
    def preprocess_data(self):
        """
        Preprocess the data by encoding categorical features, scaling numerical features,
        and handling missing values.
        """
        # Separate target variable and features
        X = self.data[self.categorical_features + self.numerical_features]
        y = self.data[self.target_variable]
        
        # Define preprocessing for numerical data
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Define preprocessing for categorical data
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ]
        )

        # Fit and transform the data
        self.X_preprocessed = self.preprocessor.fit_transform(X)
        
        return self.X_preprocessed, y

    def get_preprocessed_data(self):
        """
        Get the preprocessed data. This function assumes that preprocess_data() has been called.
        """
        if self.X_preprocessed is None:
            raise ValueError("Data has not been preprocessed. Please call preprocess_data() first.")
        return self.X_preprocessed

# Example usage
if __name__ == "__main__":
    # Sample dataset
    file = 'your_dataset.csv'
    data = pd.read_csv(file)

    # Specify the target variable and feature columns
    target_variable = 'TotalClaims'
    categorical_features = ['VehicleType', 'Make', 'Model']
    numerical_features = ['SumInsured', 'RegistrationYear', 'NumberOfDoors']

    # Create the preprocessor instance
    preprocessor = DataPreprocessor(data, target_variable, categorical_features, numerical_features)

    # Preprocess the data
    X_preprocessed, y = preprocessor.preprocess_data()

    # Print the results
    print("Selected features for modeling:")
    print(categorical_features + numerical_features)
    print("Preprocessed data:")
    print(X_preprocessed)
    print("Target variable:")
    print(y)