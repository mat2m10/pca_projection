import pandas as pd
from sklearn.impute import KNNImputer

import numpy as np
import os


def find_Pxy(ps):
    # Compute the mean of each row
    twopq = 2*ps*(1-ps)
    row_means = twopq.mean(axis=1).values  # Convert to a numpy array
    
    # Create a symmetric matrix of pairwise averages
    symmetric_matrix = 0.5 * (row_means[:, np.newaxis] + row_means[np.newaxis, :])
    
    # Convert to DataFrame
    Pxy = pd.DataFrame(symmetric_matrix, index=twopq.index, columns=twopq.index)
    return Pxy

def find_Dxy(ps):
    ps_values = ps.values
    # Compute the pairwise differences
    complement = 1 - ps_values  # Precompute the complements
    Dxy_matrix = (
        (ps_values @ complement.T) + (complement @ ps_values.T)
    ) / ps_values.shape[1]
    
    # Convert to DataFrame for easier handling
    Dxy = pd.DataFrame(Dxy_matrix, index=ps.index, columns=ps.index)
    return Dxy

def make_fst(path_input, file_name, path_output):
    imputer = KNNImputer(n_neighbors=5, metric='nan_euclidean')  # Correct metric

    path_major = f"{path_input}/p2/{file_name}"
    path_het = f"{path_input}/2pq/{file_name}"
    path_minor = f"{path_input}/q2/{file_name}"
    try:
        majs = pd.read_pickle(path_major)
        hets = pd.read_pickle(path_het)
        mins = pd.read_pickle(path_minor)
        esti_ps = ((majs - mins)+1)/2
        esti_qs = 1-esti_ps

        esti_inbreeding = abs(1-(hets/(2*esti_ps*esti_qs)).mean(axis = 1))
        humans = pd.DataFrame()
        humans['inbreeding'] = esti_inbreeding.values
        humans.to_pickle(f"{path_output}/humans_{file_name}")
        Pxy = find_Pxy(esti_ps)
        Dxy = find_Dxy(esti_ps)
        Fst = (Dxy - Pxy)/Dxy
        Fst = Fst.round(5).abs()
        
        # Apply KNN imputation
        Fst = pd.DataFrame(
            imputer.fit_transform(Fst),
            index=Fst.index,
            columns=Fst.columns
        )
        Fst.to_pickle(f"{path_output}/Fst_{file_name}")

    except Exception as e:
        print(e)
        print(f"error there: {path_major}")
        
def make_global_fst(path_input, path_output):
    chrom_folders = [f for f in os.listdir(path_input) if f.startswith('chrom')]
    nr_files = sum(
        len([f for f in os.listdir(f"{path_input}/{chrom}") if f.startswith('Fst')])
        for chrom in chrom_folders
    )
    
    # Get matrix shape from the first Fst file found
    first_file = next(
        f"{path_input}/{chrom}/{file}"
        for chrom in chrom_folders
        for file in os.listdir(f"{path_input}/{chrom}")
        if file.startswith('Fst')
    )
    first = pd.read_pickle(first_file)
    n = first.shape[0]
    mean_Fst_matrix = pd.DataFrame(np.zeros((n, n)), index=first.index, columns=first.columns)
    
    # Accumulate Fst matrices
    for chrom_folder in chrom_folders:
        path_Fsts = f"{path_input}/{chrom_folder}"
        for file in os.listdir(path_Fsts):
            if file.startswith('Fst'):                
                Fst = pd.read_pickle(f"{path_Fsts}/{file}")
                Fst = Fst / nr_files
                mean_Fst_matrix = mean_Fst_matrix.add(Fst)

    
    mean_Fst_matrix.to_pickle(f"{path_output}/global_Fst.pkl")