import os
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
    
    
import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
    
    
import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
    
    
def reduce_reconstruct(block, n_components, var_threshold=1e-8):
    # Remove near-constant features
    try:
        selector = VarianceThreshold(threshold=var_threshold)
        filtered_block = selector.fit_transform(block)

        # Save column names for reconstruction
        selected_columns = block.columns[selector.get_support()]

        # Check if there are any remaining features
        if filtered_block.shape[1] == 0:
            raise ValueError("All features were removed by VarianceThreshold. Try lowering the threshold.")

        # Standardize the filtered data
        scaler = StandardScaler()
        scaled_snps = scaler.fit_transform(filtered_block)

        # Apply PCA
        pca = PCA(n_components=min(n_components, scaled_snps.shape[1]))  # Ensure n_components isn't larger than features
        reduced_data = pca.fit_transform(scaled_snps)

        # Reconstruction
        reconstructed_scaled_snps = pca.inverse_transform(reduced_data)
        reconstructed_block = scaler.inverse_transform(reconstructed_scaled_snps)

        # Restore DataFrame format
        reconstructed_df = pd.DataFrame(data=reconstructed_block, columns=selected_columns, index=block.index)

        return reconstructed_df
    
    except ValueError as e:
        return block

def linear_abyss(path_input, name_file, path_output, n_components=5, p2=False, twopq = False, q2=False):
    path_ld = f"{path_input}/{name_file}"
    block = pd.read_pickle(f"{path_ld}")
    block = block.fillna(-1.0)

    if q2:
        # Update minor allele mapping
        db_minor = block.copy()
        db_minor = db_minor.applymap(lambda x: 1 if x == -1.0 else 0)

        db_minor_rec = reduce_reconstruct(db_minor, n_components)
        path_minor = f"{path_output}/q2/"
        os.makedirs(path_minor, exist_ok=True)
        db_minor_rec.to_pickle(f"{path_minor}/{name_file}")
    else:
        pass
    
    if twopq:  
        # Update heterozygous allele mapping
        db_het = block.copy()
        db_het = db_het.applymap(lambda x: 1 if x == 0.0 else 0)
        
        db_het_rec = reduce_reconstruct(db_het, n_components)
        path_het = f"{path_output}/2pq/"
        os.makedirs(path_het, exist_ok=True)
        db_het_rec.to_pickle(f"{path_het}/{name_file}")
    
    else:
        pass
    
    if p2:
        # Update major allele mapping
        db_major = block.copy()
        db_major = db_major.applymap(lambda x: 1 if x == 1.0 else 0)
    
        db_major_rec = reduce_reconstruct(db_major, n_components)
        path_major = f"{path_output}/p2/"
        os.makedirs(path_major, exist_ok=True)
        db_major_rec.to_pickle(f"{path_major}/{name_file}")
