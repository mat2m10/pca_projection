import subprocess
# Statistical Analysis
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import math

        
        
def ols_regression(y, X1, covs=None):
    """
    Perform OLS regression with one primary predictor X1 and optional covariates.

    Parameters:
    y (pd.Series): The dependent variable.
    X1 (pd.Series): The primary predictor.
    covs (pd.DataFrame or None): DataFrame where each column is a covariate. Can be None.

    Returns:
    None: Prints the coefficients and p-values.
    """
    # Combine X1 and covariates into a single DataFrame
    if covs is not None:
        X = pd.concat([X1, covs], axis=1)
    else:
        X = X1.to_frame()
    
    X = sm.add_constant(X)  # Adds a column of ones to include an intercept in the model

    # Fit the OLS model
    model = sm.OLS(y, X)
    results = model.fit()

    # Extract coefficients (beta values) and p-values
    beta_values = results.params
    p_values = results.pvalues
    
    return beta_values, p_values

def pca_of_n_snps(path_input, path_output, nr_snps, n_components):
    chroms = os.listdir(path_input)
    nr_snps_for_PCA_per_chrom = math.ceil(nr_snps/len(chroms))
    genos = []
    for chrom in chroms:
        path_chrom = f"{path_input}/{chrom}"
        chunks = os.listdir(path_chrom)
        nr_snps_for_PCA_per_chunks = math.ceil(nr_snps_for_PCA_per_chrom / len(chunks))

        for chunk in chunks:
            path_chunk = f"{path_chrom}/{chunk}"
            geno = pd.read_pickle(path_chunk)
            # Get number of available columns
            num_available_columns = geno.shape[1]

            # Adjust n if needed
            n = min(nr_snps_for_PCA_per_chunks, num_available_columns)
            geno = geno.sample(n=n, axis=1)
            genos.append(geno)
    genos = pd.concat(genos, axis=1)
    # Standardize the data (zero mean, unit variance)
    scaler = StandardScaler()
    genos = scaler.fit_transform(genos)  # Returns a NumPy array

    # Apply PCA
    pca = PCA(n_components=n_components)
    genos_pca = pca.fit_transform(genos)  # Transform the data

    # Convert PCA output to DataFrame
    genos_pca = pd.DataFrame(genos_pca, columns=[f'PC{i+1}' for i in range(n_components)])
    genos_pca.to_pickle(path_output)
    return genos_pca