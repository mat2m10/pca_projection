import pandas as pd

from sklearn.preprocessing import StandardScaler
from k_means_constrained import KMeansConstrained
from sklearn.decomposition import PCA

def segmenter(path_input, path_output, name_file, n_components, size_block):
    chunk = pd.read_pickle(f"{path_input}/{name_file}.pkl")
    chunk = chunk.fillna(2)
    
    size_chunck = chunk.shape[1]
    
    snps = chunk.T
    scaler = StandardScaler()
    scaled_snps = scaler.fit_transform(snps)
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(scaled_snps)
    
    if size_block > size_chunck:
        size_block = size_chunck
    n_clusters = int(size_chunck/size_block)

    size_min = size_block - size_block*0.2
    size_max = size_block + size_block*0.2
    if size_max > size_chunck:
        size_max = size_chunck
        size_min = None

    if size_max*n_clusters < size_chunck:
        n_clusters +=1
        
        
    # Apply constrained K-Means clustering
    clf = KMeansConstrained(
        n_clusters=n_clusters,
        size_min=size_min,
        size_max=size_max,
        random_state=0
    )
    
    clf.fit_predict(reduced_data)
    snps['ld_block'] = list(clf.labels_) 
    for ld_block_nr in list(snps['ld_block'].unique()):
        ld_block = snps[snps['ld_block'] == ld_block_nr]
        ld_block = ld_block.drop(columns='ld_block').T
        size_snps = ld_block.shape[1]
        ld_block.to_pickle(f"{path_output}/block_{ld_block_nr}_size_{size_snps}_from_{name_file}.pkl")