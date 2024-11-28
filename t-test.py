import numpy as np
from scipy.stats import ttest_rel, mannwhitneyu, wilcoxon
import torch
from sklearn.metrics import mean_squared_error

# Example accuracies for baseline model and DBN
baseline_accuracies = [78.440, 78.880, 78.710, 78.000, 79.020]

dbn_accuracies_combined = [82.060, 82.600, 82.380, 81.970, 81.930]

dbn_accuracies = [80.500, 80.720, 80.520, 80.330, 80.350]



# Significance threshold
alpha = 0.05

# 1. Paired T-Test (Comparing two models on the same dataset)
statistic, p_value = ttest_rel(baseline_accuracies, dbn_accuracies)
print(f"Paired T-Test Results: Statistic = {statistic}, p-value = {p_value}")
if p_value < alpha:
    print("Result: Statistically significant (p < 0.05)\n")
else:
    print("Result: Not statistically significant (p >= 0.05)\n")


# 2. Mean and Standard Deviation for Both Models
baseline_mean = np.mean(baseline_accuracies)
baseline_std = np.std(baseline_accuracies)

dbn_mean = np.mean(dbn_accuracies)
dbn_std = np.std(dbn_accuracies)

dbn_mean_combined = np.mean(dbn_accuracies_combined)
dbn_std_combined = np.std(dbn_accuracies_combined)

print(f"Baseline Model Mean Accuracy: {baseline_mean:.3f}, Std Dev: {baseline_std:.3f}")
print(f"DBN One Model Mean Accuracy: {dbn_mean:.3f}, Std Dev: {dbn_std:.3f}")
print(f"DBN Model Combined Mean Accuracy: {dbn_mean_combined:.3f}, Std Dev: {dbn_std_combined:.3f}")

# 3. Optional: Mean Squared Error (MSE) Between Models
mse = mean_squared_error(baseline_accuracies, dbn_accuracies)
print(f"Mean Squared Error between models: {mse:.3f}")