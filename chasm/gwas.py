import subprocess
# Statistical Analysis
import statsmodels.api as sm
import scipy.stats as stats
import pandas as pd

"""
    This function takes the path to the plink file and the path to the output directory
    and returns the allele frequencies of the plink file(s).
"""

def make_AFs(path_data, name_file_input, path_plink, path_output):
    for chrom in list(range(22)):
        chrom += 1
        name_file_output = f"chrom_{chrom}_AFs_{name_file_input}"
        path_input_data = f"{path_data}/{name_file_input}"
        cmd = (
            f"cd {path_data}; {path_plink}/plink2 "
            f"--bfile {name_file_input} "
            f"--chr {chrom} "
            f"--freq "
            f"--out {path_output}/{name_file_output} "
        )
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        # Decode output if necessary (for Python 3 compatibility)
        stdout = stdout.decode('utf-8')
        stderr = stderr.decode('utf-8')
        print(stderr)
        
        
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