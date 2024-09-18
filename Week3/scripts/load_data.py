
import pandas as pd
import os

class InsuranceDataProcessor:
    def __init__(self, text_file_path):
        """
        Initialize the class with the path to the text file.
        
        Args:
        - text_file_path (str): Path to the input text file.
        """
        self.text_file_path = text_file_path
    
    def save_to_csv(self, csv_file_name='insurance_text_data.csv'):
        """
        Load insurance data from a text file and save it as a CSV.
        
        Args:
        - csv_file_name (str): The name of the CSV file to save.
        """
        try:
            # Read the text file into a DataFrame
            df = pd.read_csv(self.text_file_path, delimiter='|', engine='python')
            
            # Create the path for the 'data' folder one level up
            data_folder = os.path.abspath(os.path.join(os.getcwd(), '..', 'data'))
            os.makedirs(data_folder, exist_ok=True)  # Create 'data' folder if it doesn't exist
            
            # Define the full path for the CSV file
            csv_file_path = os.path.join(data_folder, csv_file_name)
            
            # Save the DataFrame to a CSV file
            df.to_csv(csv_file_path, index=False)
            
            print(f"Data saved to {csv_file_path}")
            
        except pd.errors.EmptyDataError:
            print("Error: The input text file is empty.")
        except pd.errors.ParserError as e:
            print(f"Error: Parsing error while reading the text file - {e}")
        except FileNotFoundError:
            print("Error: The specified text file was not found.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")