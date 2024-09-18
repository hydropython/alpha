import pandas as pd
from scipy.stats import chi2_contingency, norm, ttest_ind
import numpy as np

class InsuranceDataAnalyzer:
    def __init__(self, file_path):
        """
        Initializes the InsuranceDataAnalyzer with the given CSV file path.
        :param file_path: Path to the CSV file containing insurance data.
        """
        self.file_path = file_path
        self.data = pd.read_csv(file_path)

    def cramer_v(self, chi2_stat, contingency_table):
        """
        Computes Cramér's V for effect size.
        :param chi2_stat: Chi-square statistic from the chi2 test.
        :param contingency_table: Contingency table used in the test.
        :return: Cramér's V (effect size).
        """
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape) - 1
        return np.sqrt(chi2_stat / (n * min_dim))

    def test_no_risk_differences_across_provinces(self):
        """
        Tests if there are no risk differences in TotalClaims across provinces.
        """
        contingency_table = pd.crosstab(self.data['Province'], self.data['TotalClaims'])
        chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
        cramer_v = self.cramer_v(chi2_stat, contingency_table)

        print(f"Chi2 Stat (Provinces): {chi2_stat:.2f}, P-Value: {p_value:.4f}, Cramér's V: {cramer_v:.4f}")
        self._interpret_p_value(p_value)

    def test_no_risk_differences_between_zip_codes(self):
        """
        Tests if there are no risk differences in TotalClaims between zip codes.
        """
        contingency_table = pd.crosstab(self.data['PostalCode'], self.data['TotalClaims'])
        chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
        cramer_v = self.cramer_v(chi2_stat, contingency_table)

        print(f"Chi2 Stat (Postal Codes): {chi2_stat:.2f}, P-Value: {p_value:.4f}, Cramér's V: {cramer_v:.4f}")
        self._interpret_p_value(p_value)

    def test_no_margin_differences_between_zip_codes(self):
        """
        Tests if there are no significant margin (TotalPremium) differences between zip codes.
        """
        contingency_table = pd.crosstab(self.data['PostalCode'], self.data['TotalPremium'])
        chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
        cramer_v = self.cramer_v(chi2_stat, contingency_table)

        print(f"Chi2 Stat (Premium by Postal Codes): {chi2_stat:.2f}, P-Value: {p_value:.4f}, Cramér's V: {cramer_v:.4f}")
        self._interpret_p_value(p_value)

    def test_no_risk_differences_between_women_and_men(self):
        """
        Tests if there are no significant risk (TotalClaims) differences between women and men.
        """
        contingency_table = pd.crosstab(self.data['Gender'], self.data['TotalClaims'])
        chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
        cramer_v = self.cramer_v(chi2_stat, contingency_table)

        print(f"Chi2 Stat (Gender): {chi2_stat:.2f}, P-Value: {p_value:.4f}, Cramér's V: {cramer_v:.4f}")
        self._interpret_p_value(p_value)

        # Perform a two-proportion z-test for risk differences between genders
        self._two_proportion_z_test('Gender', 'TotalClaims', ['Male', 'Female'])

    def _two_proportion_z_test(self, group_col, target_col, categories):
        """
        Performs a two-proportion z-test to compare the proportions of a binary target across two groups.
        :param group_col: The column representing the groups (e.g., Gender).
        :param target_col: The column representing the target binary outcome (e.g., TotalClaims).
        :param categories: The two categories to compare (e.g., ['Male', 'Female']).
        """
        # Extracting data for the two groups
        group1 = self.data[self.data[group_col] == categories[0]][target_col]
        group2 = self.data[self.data[group_col] == categories[1]][target_col]

        # Calculating proportions
        p1 = group1.mean()  # Proportion of claims in group1
        p2 = group2.mean()  # Proportion of claims in group2
        n1 = group1.count()  # Number of observations in group1
        n2 = group2.count()  # Number of observations in group2

        # Pooled proportion
        p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)

        # Z-score calculation
        z_score = (p1 - p2) / np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))

        # Two-tailed p-value
        p_value = 2 * (1 - norm.cdf(abs(z_score)))

        # Print results
        print(f"Two-Proportion Z-Test between {categories[0]} and {categories[1]}: Z-Score = {z_score:.2f}, P-Value = {p_value:.4f}")
        self._interpret_p_value(p_value)

    def _interpret_p_value(self, p_value, alpha=0.05):
        """
        Interprets the result of a hypothesis test based on the p-value.
        :param p_value: The p-value obtained from the test.
        :param alpha: The significance level, default is 0.05.
        """
        if p_value < alpha:
            print("The result is statistically significant (reject H0).")
        else:
            print("The result is not statistically significant (fail to reject H0).")