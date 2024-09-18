import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
import shap

class SHAPAnalysis:
    def __init__(self, file_path, target_col):
        self.file_path = file_path
        self.target_col = target_col
        self.df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        self.X = self.df.drop(target_col, axis=1)
        self.Y = self.df[target_col]
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=44)
        
    def plot_shap_analysis(self):
        # Initialize models
        GB = GradientBoostingRegressor(random_state=42)
        xgb_model = XGBRegressor(random_state=30)
        
        # Fit models
        GB.fit(self.X_train, self.Y_train)
        xgb_model.fit(self.X_train, self.Y_train, verbose=1)
        
        # SHAP values
        ex_gb = shap.Explainer(GB)
        shap_values_gb = ex_gb(self.X_train)
        
        ex_xgb = shap.Explainer(xgb_model)
        shap_values_xgb = ex_xgb(self.X_train)
        
        # Plot SHAP summary plots and dependence plots
        fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(40, 32))
        fig.suptitle('Global SHAP Dependency - Melekasa Station', fontname="Times New Roman", style='italic', fontweight="bold", fontsize=27, y=0.95)
        
        # Gradient Boosting Regressor SHAP summary plots
        axes[0, 0].set_title('GBR Feature Importance')
        shap.summary_plot(shap_values_gb, self.X_train, cmap="viridis", show=False, ax=axes[0, 0])
        axes[0, 0].tick_params(labelsize=18)
        axes[0, 0].set_xlabel("SHAP value", fontsize=18, fontweight="bold")
        
        # XGBoost SHAP summary plots
        axes[0, 1].set_title('XGBoost Feature Importance')
        shap.summary_plot(shap_values_xgb, self.X_train, show=False, ax=axes[0, 1])
        axes[0, 1].tick_params(labelsize=18)
        axes[0, 1].set_xlabel("SHAP value", fontsize=18, fontweight="bold")
        
        # Dependency plots for Gradient Boosting Regressor
        shap.dependence_plot('SH', shap_values_gb, self.X_train, ax=axes[1, 0], cmap=plt.get_cmap("viridis"), show=False)
        axes[1, 0].set_title('Sunshine Hour (GBR)')
        axes[1, 0].tick_params(labelsize=16)
        axes[1, 0].set_xlabel("Sunshine hour", fontweight="bold", fontsize=18)
        axes[1, 0].set_ylabel("SHAP value", fontsize=18, fontweight="bold")

        shap.dependence_plot('Month', shap_values_gb, self.X_train, ax=axes[1, 1], cmap=plt.get_cmap("viridis"), show=False)
        axes[1, 1].set_title('Month (GBR)')
        axes[1, 1].tick_params(labelsize=16)
        axes[1, 1].set_xlabel("Month", fontweight="bold", fontsize=18)
        axes[1, 1].set_ylabel("SHAP value", fontsize=18, fontweight="bold")

        shap.dependence_plot('Tmax', shap_values_gb, self.X_train, ax=axes[1, 2], cmap=plt.get_cmap("viridis"), show=False)
        axes[1, 2].set_title('Tmax (GBR)')
        axes[1, 2].tick_params(labelsize=16)
        axes[1, 2].set_xlabel("Tmax", fontweight="bold", fontsize=18)
        axes[1, 2].set_ylabel("SHAP value", fontsize=18, fontweight="bold")

        shap.dependence_plot('Tmean', shap_values_gb, self.X_train, ax=axes[1, 3], cmap=plt.get_cmap("viridis"), show=False)
        axes[1, 3].set_title('Tmean (GBR)')
        axes[1, 3].tick_params(labelsize=16)
        axes[1, 3].set_xlabel("Tmean", fontweight="bold", fontsize=18)
        axes[1, 3].set_ylabel("SHAP value", fontsize=18, fontweight="bold")

        # Dependency plots for XGBoost
        shap.dependence_plot('SH', shap_values_xgb, self.X_train, ax=axes[2, 0], show=False)
        axes[2, 0].set_title('Sunshine Hour (XGBoost)')
        axes[2, 0].tick_params(labelsize=16)
        axes[2, 0].set_xlabel("Sunshine hour", fontweight="bold", fontsize=16)
        axes[2, 0].set_ylabel("SHAP value", fontsize=16, fontweight="bold")

        shap.dependence_plot('Month', shap_values_xgb, self.X_train, ax=axes[2, 1], show=False)
        axes[2, 1].set_title('Month (XGBoost)')
        axes[2, 1].tick_params(labelsize=16)
        axes[2, 1].set_xlabel("Month", fontweight="bold", fontsize=18)
        axes[2, 1].set_ylabel("SHAP value", fontsize=18, fontweight="bold")

        shap.dependence_plot('Tmean', shap_values_xgb, self.X_train, ax=axes[2, 2], show=False)
        axes[2, 2].set_title('Tmean (XGBoost)')
        axes[2, 2].tick_params(labelsize=16)
        axes[2, 2].set_xlabel("Tmean", fontweight="bold", fontsize=18)
        axes[2, 2].set_ylabel("SHAP value", fontsize=18, fontweight="bold")

        shap.dependence_plot('Tmax', shap_values_xgb, self.X_train, ax=axes[2, 3], show=False)
        axes[2, 3].set_title('Tmax (XGBoost)')
        axes[2, 3].tick_params(labelsize=16)
        axes[2, 3].set_xlabel("Tmax", fontweight="bold", fontsize=18)
        axes[2, 3].set_ylabel("SHAP value", fontsize=18, fontweight="bold")

        # Add text annotations
        for i, ax in enumerate(axes.flat):
            ax.text(0.05, 0.95, chr(65 + i), fontweight="bold", fontsize=24, transform=ax.transAxes)

        # Adjust layout
        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.25, hspace=0.25)

        # Save and show plot
        plt.savefig("MLML.png", dpi=400, bbox_inches='tight')
        plt.show()

# Example usage
shap_analysis = SHAPAnalysis('C:/Users/User/Desktop/PAPERS/Evap/DEVAP.csv', 'Evapoavg')
shap_analysis.plot_shap_analysis()