            
import os
import pandas as pd
import numpy as np

"""
Check if the columns are SNPs
"""
def is_snp(col):
    return any(char.isdigit() for char in col)

def make_df(path_input, path_usefull, path_output):
    geno = pd.read_csv(path_input, sep=" ")
    
    # Identify non-SNP columns
    non_snp_cols = [col for col in geno.columns if not is_snp(col)]
    
    os.makedirs(path_usefull, exist_ok=True)
    
    if non_snp_cols:
        # Save non-SNP columns as a DataFrame
        ids_df = geno[non_snp_cols].copy()  # Avoid SettingWithCopyWarning
        ids_df["index"] = ids_df.index  # Store index as a separate column
        ids_df.reset_index(drop=True, inplace=True)  # Reset index before saving
        ids_df.to_pickle(f"{path_usefull}/ids.pkl")
        
        geno = geno.drop(columns=non_snp_cols)
    else:
        # Save the index as a DataFrame
        ids_df = pd.DataFrame(geno.index, columns=["index"])
        ids_df.to_pickle(f"{path_usefull}/ids.pkl")
    
    os.makedirs(path_output, exist_ok=True)
    
    # Save the SNP genotype data
    geno.to_pickle(f"{path_output}/geno.pkl")
    
    return geno
    
def calculate_AFs(geno):
    # Has to be encoded as 0 (maj), 1 (het), 2 (min)
    
    af = geno.sum(axis=0) / (2 * geno.shape[0])

    af = pd.DataFrame(af, columns=["AF"])
    # Assuming allele_frequencies is a Pandas Series
    af = af.reset_index()
    af.columns = ["snp_rs", "AF"]  # Rename columns
    # Create a new column without the allele suffix
    af['RSID'] = af['snp_rs'].str.replace(r'_[ACTG]$', '', regex=True)
    return af

def merge_AFs_ensembl_build(path_ensembl, path_usefull, afs):
    merged_dfs = []
    for chrom in list(range(22)):
        chrom += 1
        path_ensembl_chrom = f"{path_ensembl}/chrom_{chrom}"
        for build in [f for f in os.listdir(path_ensembl_chrom) if f.startswith('build')]:
            build = pd.read_pickle(f"{path_ensembl_chrom}/{build}")
            build['CHROM'] = chrom
            merged_dfs.append(pd.merge(afs, build, left_on='RSID', right_on='RSID', how='inner'))

    df = pd.concat(merged_dfs, axis=0)
    # Sorting by CHROM and then POS
    df_sorted = df.sort_values(by=["CHROM", "POS"]).reset_index(drop=True)
    df_sorted.to_pickle(f"{path_usefull}/allele_frequencies.pkl")
    
    
def divide_into_chunks(path_input, path_afs, path_output, size_chunck, min_maf):
    geno = pd.read_pickle(f"{path_input}")
    geno.fillna(0, inplace=True)
    geno = (geno - 1)*-1
    
    afs = pd.read_pickle(path_afs)
    
    for chrom in afs['CHROM'].unique():
        path_output_chrom = f"{path_output}/chrom_{chrom}/"
        os.makedirs(path_output_chrom, exist_ok=True)
    
        afs_chrom = afs[afs['CHROM'] == chrom]
        afs_chrom = afs_chrom[afs_chrom['AF'] > min_maf]

        afs_chrom = afs_chrom.sort_values(by=["AF"], ascending=False).reset_index(drop=True)
        nr_snps_total = len(afs_chrom)
        num_subframes = nr_snps_total//size_chunck
        remaining_rows = nr_snps_total%size_chunck
        
        
        try:
            to_divide_in = remaining_rows//num_subframes
            rest = nr_snps_total- ((size_chunck + to_divide_in)*num_subframes)
            snps_per_segments = np.ones(num_subframes) * (size_chunck+ to_divide_in)
            to_add = np.concatenate((np.ones(rest), np.zeros(num_subframes - rest)),axis = 0)
            snps_per_segments = snps_per_segments + to_add
        except Exception as e:
            snps_per_segments = [nr_snps_total]
            
        # Make Chunks per chromosomes
        start = 0
        end = 0
        i = 0
        for nr_snps in snps_per_segments:
            end = int(end + nr_snps)
            AF_chunk = afs_chrom[start:end].copy()
            start = int(start + nr_snps)
            feature_size = AF_chunk.shape[0]
            i+=1
            minaf = np.round(AF_chunk['AF'].min(),2)
            maxaf = np.round(AF_chunk['AF'].max(),2)
            geno[AF_chunk['snp_rs']].to_pickle(f"{path_output_chrom}/chunk_{i}_size_{len(AF_chunk)}_mafs_{minaf}_{maxaf}.pkl")
    
    
def align_dataframes(df1, df2, df3):
    """
    Ensures that df1, df2, and df3 have exactly the same columns.
    Any missing columns in one DataFrame cause those columns to be dropped in all others.
    """
    # Find common columns across all three dataframes
    common_columns = set(df1.columns) & set(df2.columns) & set(df3.columns)
    
    # Keep only common columns in all DataFrames
    df1_aligned = df1[list(common_columns)].copy()
    df2_aligned = df2[list(common_columns)].copy()
    df3_aligned = df3[list(common_columns)].copy()
    
    return df1_aligned, df2_aligned, df3_aligned

