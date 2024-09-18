import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import plotly.graph_objects as go

class InsuranceEDA:
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        # Create directories if they don't exist
        self.images_dir = '../Images'
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)
        self.data_dir = '../Data'
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        # Set modern style
        plt.style.use('seaborn-v0_8-darkgrid') 
        # Path for updated CSV
        self.updated_file_path = os.path.join(self.data_dir, 'updated_data.csv')
        
    def data_structure(self):
        """Check the data types of all columns."""
        print("Data Structure:")
        print(self.data.dtypes)
    
    def descriptive_statistics(self):
        """Display the descriptive statistics of the dataset."""        
        print("Descriptive Statistics:")
        print(self.data.describe())
    
    def check_missing_values(self):
        """Check for missing values in the dataset before filling."""
        print("Missing Values Before Filling:")
        missing_values = self.data.isnull().sum()
        print(missing_values)
        return missing_values

    def fill_missing_values(self):
        """Fill missing values in the dataset."""
        # Fill numerical columns with mean
        numerical_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        self.data[numerical_cols] = self.data[numerical_cols].fillna(self.data[numerical_cols].mean())

        # Fill categorical columns with 'unknown'
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        self.data[categorical_cols] = self.data[categorical_cols].fillna('unknown')
        
        # Save the updated data to a new CSV
        self.data.to_csv(self.updated_file_path, index=False)
        print(f"Updated data with filled missing values saved to {self.updated_file_path}")

    def check_missing_values_after_fill(self):
        """Check for missing values in the updated CSV file after filling."""
        print("Checking Missing Values in Updated CSV:")
        updated_data = pd.read_csv(self.updated_file_path)
        missing_values_after_fill = updated_data.isnull().sum()
        print(missing_values_after_fill)
        return missing_values_after_fill
    
    def univariate_analysis(self):
        """Plot histograms for numerical columns and bar charts for categorical columns."""        
        print("Univariate Analysis:")

        # Define numerical and categorical columns
        numerical_columns = ['TotalPremium', 'TotalClaims', 'SumInsured', 'CalculatedPremiumPerTerm']
        categorical_columns = self.data.select_dtypes(include=['object']).columns.tolist()

        # Plot histograms for numerical columns
        for column in numerical_columns:
            if column in self.data.columns and not self.data[column].isnull().all():
                plt.figure(figsize=(10, 6))
                sns.histplot(self.data[column], bins=50, color='darkgreen', kde=True)
                plt.title(f'Distribution of {column}')
                plt.xlabel(column)
                plt.ylabel('Frequency')
                file_path = os.path.join(self.images_dir, f'{column}_distribution.png')
                plt.savefig(file_path)
                plt.close()
                print(f"Saved histogram of {column} as {file_path}")
            else:
                print(f"No data for {column} or column not found")

        # Plot bar charts for categorical columns
        for column in categorical_columns:
            if column not in numerical_columns:
                plt.figure(figsize=(10, 6))
                sns.countplot(x=self.data[column], color='darkgreen')
                plt.title(f'Distribution of {column}')
                plt.xlabel(column)
                plt.ylabel('Count')
                file_path = os.path.join(self.images_dir, f'{column}_distribution.png')
                plt.savefig(file_path)
                plt.close()
                print(f"Saved bar chart of {column} as {file_path}")
    
    def bivariate_analysis(self):
        """Explore relationships between monthly changes in TotalPremium and TotalClaims by PostalCode."""        
        print("Bivariate Analysis:")

        # Ensure the correct date column exists and is in datetime format
        date_column = 'TransactionMonth'  # Change this to the correct date column if necessary
        if date_column not in self.data.columns:
            raise ValueError(f"The dataset must contain a '{date_column}' column for time series analysis.")
        
        # Convert date column to datetime
        self.data[date_column] = pd.to_datetime(self.data[date_column], errors='coerce')
        
        if self.data[date_column].isnull().any():
            print(f"Warning: There are missing or invalid values in '{date_column}'.")

        # Set date column as index and sort
        self.data.set_index(date_column, inplace=True)
        self.data.sort_index(inplace=True)
        
        # Resample the data to monthly frequency and aggregate TotalPremium and TotalClaims
        monthly_data = self.data.resample('M').agg({
            'TotalPremium': 'sum',
            'TotalClaims': 'sum'
        })
        
        # Calculate monthly percentage change
        monthly_data['TotalPremium_Monthly_Change'] = monthly_data['TotalPremium'].pct_change()
        monthly_data['TotalClaims_Monthly_Change'] = monthly_data['TotalClaims'].pct_change()

        # Drop rows with NaN values that resulted from pct_change()
        monthly_data.dropna(inplace=True)

        # Print correlation matrix of monthly changes
        correlation_matrix = monthly_data[['TotalPremium_Monthly_Change', 'TotalClaims_Monthly_Change']].corr()
        print("\nCorrelation Matrix of Monthly Changes:")
        print(correlation_matrix)

        # Scatter plot of TotalPremium_Monthly_Change vs TotalClaims_Monthly_Change
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='TotalPremium_Monthly_Change', y='TotalClaims_Monthly_Change', data=monthly_data, color='darkgreen')
        plt.title('Monthly Change in Total Premium vs Total Claims')
        plt.xlabel('Total Premium Monthly Change')
        plt.ylabel('Total Claims Monthly Change')
        plt.grid(True)
        file_path = os.path.join(self.images_dir, 'premium_vs_claims_scatter.png')
        plt.savefig(file_path)
        plt.close()
        print(f"Saved scatter plot of monthly changes as {file_path}")

        
    def outlier_detection(self):
        # Check for and remove duplicate indices
        if self.data.index.duplicated().any():
            print("Duplicate index labels detected. Resetting index.")
            self.data = self.data.reset_index(drop=True)

        # Identify columns with numeric data
        numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        
        # Find columns with high number of outliers (based on interquartile range)
        high_outlier_cols = []
        
        for col in numeric_cols:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            # Define outlier criteria
            outliers = ((self.data[col] < (Q1 - 1.5 * IQR)) | (self.data[col] > (Q3 + 1.5 * IQR)))
            # Add column to list if more than 5% of the data are outliers
            if outliers.mean() > 0.05:
                high_outlier_cols.append(col)
        
        print(f"Columns with high number of outliers: {high_outlier_cols}")
        
        # Check and remove duplicate rows from the dataset
        if self.data.duplicated().any():
            print("There are duplicate rows in the dataset, removing them.")
            self.data = self.data.drop_duplicates()

        # Loop through columns with high number of outliers and plot boxplots
        for col in high_outlier_cols:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=self.data[col], color='gold')
            plt.title(f'Outliers in {col}')
            plt.xlabel(col)
            plot_path = os.path.join(self.images_dir, f'{col}_outliers.png')
            plt.savefig(plot_path)
            print(f"Saved boxplot for {col} outliers to {plot_path}")
            plt.close()
        
        return high_outlier_cols

    def handle_outliers(self, high_outlier_cols):
        # Handling outliers: Optionally, remove or adjust outliers
        for col in high_outlier_cols:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            # Remove outliers
            self.data = self.data[(self.data[col] >= lower_bound) & (self.data[col] <= upper_bound)]
            print(f"Removed outliers from {col}.")
        
        # Plot boxplots for columns after handling outliers
        for col in high_outlier_cols:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=self.data[col], color='gold')
            plt.title(f'Outliers Removed in {col}')
            plt.xlabel(col)
            plot_path = os.path.join(self.images_dir, f'{col}_outliers_removed.png')
            plt.savefig(plot_path)
            print(f"Saved boxplot for {col} with outliers removed to {plot_path}")
            plt.close()
    
    def plot_comparison_subplots(self):
            """Create subplots comparing 'mmcode' and 'cubiccapacity' before and after outlier removal."""
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # Plot original 'mmcode'
            sns.boxplot(x=self.data['mmcode'], ax=axes[0, 0], color='skyblue')
            axes[0, 0].set_title('Original mmcode')
            axes[0, 0].text(-0.1, 1.05, 'A', transform=axes[0, 0].transAxes, 
                            fontsize=20, fontweight='bold', va='top', ha='right')

            # Plot original 'cubiccapacity'
            sns.boxplot(x=self.data['cubiccapacity'], ax=axes[0, 1], color='skyblue')
            axes[0, 1].set_title('Original cubiccapacity')
            axes[0, 1].text(-0.1, 1.05, 'B', transform=axes[0, 1].transAxes, 
                            fontsize=20, fontweight='bold', va='top', ha='right')

            # Detect and remove outliers for 'mmcode' and 'cubiccapacity'
            high_outlier_cols = self.outlier_detection()
            self.handle_outliers(['mmcode', 'cubiccapacity'])

            # Plot cleaned 'mmcode'
            sns.boxplot(x=self.data['mmcode'], ax=axes[1, 0], color='lightcoral')
            axes[1, 0].set_title('Cleaned mmcode')
            axes[1, 0].text(-0.1, 1.05, 'C', transform=axes[1, 0].transAxes, 
                            fontsize=20, fontweight='bold', va='top', ha='right')

            # Plot cleaned 'cubiccapacity'
            sns.boxplot(x=self.data['cubiccapacity'], ax=axes[1, 1], color='lightcoral')
            axes[1, 1].set_title('Cleaned cubiccapacity')
            axes[1, 1].text(-0.1, 1.05, 'D', transform=axes[1, 1].transAxes, 
                            fontsize=20, fontweight='bold', va='top', ha='right')

            # Adjust layout and save the figure
            plt.tight_layout()
            plot_path = os.path.join(self.images_dir, 'mmcode_cubiccapacity_comparison.png')
            plt.savefig(plot_path)
            plt.show()
            print(f"Saved comparison subplots to {plot_path}")
    def geographical_distribution(self):
            """Plot distribution of Total Premium and Total Claims by Province."""        
            print("Geographical Distribution Plot:")

            # Check if 'Province' column exists
            if 'Province' not in self.data.columns:
                raise ValueError("The dataset must contain a 'Province' column for geographical distribution analysis.")

            # Group data by Province and calculate average values
            province_data = self.data.groupby('Province')[['TotalPremium', 'TotalClaims']].mean().reset_index()

            # Plotting with Matplotlib
            plt.figure(figsize=(12, 8))
            province_data.plot(x='Province', y=['TotalPremium', 'TotalClaims'], kind='bar', figsize=(12, 6), color=['#1f77b4', '#ff7f0e'])
            plt.title('Average Total Premium and Claims by Province')
            plt.xlabel('Province')
            plt.ylabel('Average Value')
            plt.legend(['Total Premium', 'Total Claims'])
            plt.xticks(rotation=45, ha='right')
            file_path = os.path.join(self.images_dir, 'geographical_distribution.png')
            plt.savefig(file_path)
            plt.close()
            print(f"Saved geographical distribution plot as {file_path}")

    def premium_vs_claims(self):
            """Plot Total Premium vs Total Claims."""        
            print("Premium vs Claims Plot:")

            # Check if necessary columns exist
            if 'TotalPremium' not in self.data.columns or 'TotalClaims' not in self.data.columns:
                raise ValueError("The dataset must contain 'TotalPremium' and 'TotalClaims' columns for this analysis.")

            # Plotting with Seaborn
            plt.figure(figsize=(12, 8))
            sns.scatterplot(x='TotalPremium', y='TotalClaims', data=self.data, color='darkgreen')
            plt.title('Total Premium vs Total Claims')
            plt.xlabel('Total Premium')
            plt.ylabel('Total Claims')
            plt.grid(True)
            file_path = os.path.join(self.images_dir, 'premium_vs_claims_scatter.png')
            plt.savefig(file_path)
            plt.close()
            print(f"Saved Premium vs Claims scatter plot as {file_path}")
     
    def plot_premiums_and_claims_by_province(self):
        # Calculate average premiums and claims by province
        province_summary = self.data.groupby('Province').agg({
            'TotalPremium': 'mean',
            'TotalClaims': 'mean'
        }).reset_index()

        # Create plot
        fig = go.Figure()

        # Add bar for total premiums
        fig.add_trace(go.Bar(
            x=province_summary['Province'],
            y=province_summary['TotalPremium'],
            name='Average Total Premium',
            marker_color='darkgreen'
        ))

        # Add line for total claims
        fig.add_trace(go.Scatter(
            x=province_summary['Province'],
            y=province_summary['TotalClaims'],
            mode='lines+markers',
            name='Average Total Claims',
            marker_color='gold'
        ))

        fig.update_layout(
            title='Average Total Premium and Claims by Province',
            xaxis_title='Province',
            yaxis_title='Amount',
            template='plotly_dark'
        )

        fig.write_image(os.path.join(self.images_dir, 'premiums_and_claims_by_province.png'))
        fig.show()

    def plot_premium_vs_claims(self):
        # Create plot
        fig = go.Figure()

        # Scatter plot of TotalPremium vs TotalClaims
        fig.add_trace(go.Scatter(
            x=self.data['TotalPremium'],
            y=self.data['TotalClaims'],
            mode='markers',
            marker=dict(color='rgba(255, 182, 193, .9)', size=10, line=dict(width=2, color='DarkSlateGrey')),
            name='Premium vs Claims'
        ))

        fig.update_layout(
            title='Distribution of Total Premium vs Total Claims',
            xaxis_title='Total Premium',
            yaxis_title='Total Claims',
            template='plotly_dark'
        )

        fig.write_image(os.path.join(self.images_dir, 'premium_vs_claims.png'))
        fig.show()

    def plot_monthly_changes(self):
        # Convert TransactionMonth to datetime
        self.data['TransactionMonth'] = pd.to_datetime(self.data['TransactionMonth'], format='%Y-%m')

        # Calculate monthly percentage changes
        df_sorted = self.data.sort_values('TransactionMonth')
        df_sorted['TotalPremium_Monthly_Change'] = df_sorted['TotalPremium'].pct_change() * 100
        df_sorted['TotalClaims_Monthly_Change'] = df_sorted['TotalClaims'].pct_change() * 100

        # Create plot
        fig = go.Figure()

        # Line plot for TotalPremium change
        fig.add_trace(go.Scatter(
            x=df_sorted['TransactionMonth'],
            y=df_sorted['TotalPremium_Monthly_Change'],
            mode='lines+markers',
            name='Total Premium Change',
            line=dict(color='darkgreen')
        ))

        # Line plot for TotalClaims change
        fig.add_trace(go.Scatter(
            x=df_sorted['TransactionMonth'],
            y=df_sorted['TotalClaims_Monthly_Change'],
            mode='lines+markers',
            name='Total Claims Change',
            line=dict(color='gold')
        ))

        fig.update_layout(
            title='Monthly Percentage Change in Total Premium and Claims',
            xaxis_title='Month',
            yaxis_title='Percentage Change',
            template='plotly_dark'
        )

        fig.write_image(os.path.join(self.images_dir, 'monthly_changes.png'))
        fig.show()
