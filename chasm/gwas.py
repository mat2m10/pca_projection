import subprocess
# Statistical Analysis
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os
import math
import shutil
        
        
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

def pca_of_n_snps(path_macro_similar, path_output, temp_ids, nr_snps, n_components):
    chroms = os.listdir(path_macro_similar)
    nr_snps_for_PCA_per_chrom = math.ceil(nr_snps/len(chroms))
    genos = []
    for chrom in chroms:
        path_chrom = f"{path_macro_similar}/{chrom}"
        chunks = os.listdir(path_chrom)
        nr_snps_for_PCA_per_chunks = math.ceil(nr_snps_for_PCA_per_chrom / len(chunks))

        for chunk in chunks:
            path_chunk = f"{path_chrom}/{chunk}"
            geno = pd.read_pickle(path_chunk)
            geno = geno.loc[temp_ids['index']]
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
    genos_pca.to_pickle(f"{path_output}/global_PCs.pkl")
    return genos_pca


def project_on_dimensions(path_macro_similar, path_output, PCs, temp_ids, nr_of_projected_dimensions):
    
    # Ensure the output directory exists
    os.makedirs(path_input, exist_ok=True)
    path_projected = path_input
    path_input = f"{path_input}/to_do/"
    os.makedirs(path_input, exist_ok=True)
    # Copy all contents from input to output
    for item in os.listdir(path_macro_similar):
        
        os.system(f"cp -r {path_input_raw}/{item} {path_input}")
    
    snp_ids = []
    for i in list(range(nr_of_projected_dimensions)):
        n_components = 10
        nr_snps = 20_000
        path_output_dim = f"{path_projected}/dim_{i+1}/"
        os.makedirs(path_output_dim, exist_ok=True)
        genos_pca = pca_of_n_snps(path_input, path_output_dim, nr_snps, n_components)
        
        chroms = [f for f in os.listdir(path_input) if f.startswith('chrom')]

        for chrom in chroms:
            path_chrom = f"{path_input}/{chrom}"
            path_output_chrom = f"{path_output_dim}/{chrom}"
            os.makedirs(path_output_chrom, exist_ok=True)
            chunks = os.listdir(path_chrom)
            for chunk in chunks:
                path_chunk = f"{path_chrom}/{chunk}"
                path_chunk_raw = f"{path_input_raw}/{chrom}/{chunk}"
                path_output_chunk = f"{path_output_chrom}/{chunk}"
                
                geno_raw = pd.read_pickle(path_chunk_raw)
                nr_snps_raw = geno_raw.shape[1]
                to_take = math.ceil(nr_snps_raw/nr_of_projected_dimensions)

                geno = pd.read_pickle(path_chunk)
                
                p_vals = []
                betas = []
                snps = []
                
                common_index = genos_pca.index.intersection(geno.index)
                genos_pca = genos_pca.loc[common_index]
                geno = geno.loc[common_index]
                for snp in geno.columns:
                    [beta_values, p_values] = ols_regression(genos_pca['PC1'], geno[snp], covs=None)
                    p_vals.append(p_values[snp])
                    betas.append(beta_values[snp])
                    snps.append(snp)


                p_vals = pd.DataFrame(data = {'pval': p_vals, 'betas':betas, 'snp_rs':snps})
                p_vals['-logp'] = -np.log10(p_vals['pval'].replace(0, 1e-300))
                
                # Assuming `p_vals` is a pandas DataFrame
                to_keep = p_vals.sort_values(by='-logp', ascending=False).head(to_take)
                to_keep['dim'] = i+1
                snp_ids.append(to_keep)
                
                # Filter out rows that are in `to_keep`
                to_do = p_vals.loc[~p_vals.index.isin(to_keep.index)]
                geno[to_keep['snp_rs']].to_pickle(path_output_chunk)
                geno[to_do['snp_rs']].to_pickle(path_chunk)

    snp_ids = pd.concat(snp_ids, axis=0)
    snp_ids.to_pickle(f"{path_projected}/snp_ids.pkl")
    os.system(f"rm -rf {path_input}")