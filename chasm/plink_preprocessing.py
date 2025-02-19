
import os
import re
import pandas as pd
import numpy as np
import requests
import gzip
import subprocess

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
        
"""
    This function concatenates the allele frequencies of the different populations
"""

def concat_AFs(path_input, path_output):
    chroms = list(set([f.split('_')[1] for f in os.listdir(f"{path_input}") if f.endswith(f".afreq")]))
    pops = list(set([f.split("_AFs_")[1].split('.')[0] for f in os.listdir(f"{path_input}") if f.endswith('afreq')]))
    for chrom in chroms:
        pop_afs = []
        for pop in pops:
            path_pop_file = f"{path_input}/chrom_{chrom}_AFs_{pop}.afreq"
            df = pd.read_csv(path_pop_file, sep='\s+')
            df.rename(columns={'ALT_FREQS': f"ALT_FREQS_{pop}"}, inplace=True)
            df.rename(columns={'OBS_CT': f"OBS_CT_{pop}"}, inplace=True)
            pop_afs.append(df)
        
        df = pop_afs[0]

        for df_pop in pop_afs[1:]:
            df = pd.merge(df, df_pop, on=['#CHROM', 'ID', 'REF', 'ALT', 'PROVISIONAL_REF?'], how='inner')
        total_count = 0
        total_obs = 0
        for pop in pops:
            df[f"ALT_COUNT_{pop}"] = df[f"ALT_FREQS_{pop}"] * df[f"OBS_CT_{pop}"]
            total_count += df[f"ALT_COUNT_{pop}"]
            total_obs += df[f"OBS_CT_{pop}"]
        
        df['TOTAL_ALT_FREQ'] = total_count / total_obs
        df.to_pickle(f"{path_output}/global_AF_chrom_{chrom}.pkl")

"""
    This function divides the allele frequencies into similar chunks and extracts the genotypes for each chunk
"""

def divide_into_chunks(path_input_afs, path_input_plink_fam, path_plink, path_output, size_chunck, min_maf):
    
    global_afs = [f for f in os.listdir(path_input_afs) if f.startswith('global')]
    global_afs = sorted(global_afs, key=lambda x: int(x.split('_chrom_')[1].split('.pkl')[0]))
    already_done = [f"global_AF_{f}.pkl" for f in os.listdir(path_output) if f.startswith('chrom')]
    to_do = [f for f in global_afs if f not in already_done]
    populations = [f.split('.fam')[0] for f in os.listdir(path_input_plink_fam) if f.endswith('fam')]
    
    
    for chrom in to_do:
        chrom = int(chrom.split('.pkl')[0].split('_')[-1])
        path_output_chrom = f"{path_output}/chrom_{chrom}"
        os.makedirs(path_output_chrom, exist_ok = True)
        name_file = f"global_AF_chrom_{chrom}.pkl"
        AF_global = pd.read_pickle(f"{path_input_afs}/{name_file}")
        AF_global = AF_global.sort_values(by='TOTAL_ALT_FREQ')
        AF_global = AF_global[AF_global['TOTAL_ALT_FREQ'] > min_maf]
        nr_snps_total = len(AF_global)

        num_subframes = nr_snps_total//size_chunck
        remaining_rows = nr_snps_total%size_chunck
        try:
            to_divide_in = remaining_rows//num_subframes
            rest = nr_snps_total- ((size_chunck + to_divide_in)*num_subframes)
            snps_per_segments = np.ones(num_subframes) * (size_chunck+ to_divide_in)
            to_add = np.concatenate((np.ones(rest), np.zeros(num_subframes - rest)),axis = 0)
            snps_per_segments = snps_per_segments + to_add
        except Exception as e:
            print(e)
            snps_per_segments = [nr_snps_total]

        # Make Chunks per chromosomes
        start = 0
        end = 0
        i = 0
        for nr_snps in snps_per_segments:
            end = int(end + nr_snps)
            AF_chunk = AF_global[start:end].copy()
            start = int(start + nr_snps)
            feature_size = AF_chunk.shape[0]
            i+=1
        
            minaf = np.round(AF_chunk['TOTAL_ALT_FREQ'].min(),2)
            maxaf = np.round(AF_chunk['TOTAL_ALT_FREQ'].max(),2)
            snp_ids_to_keep = ",".join(list(AF_chunk['ID']))
        
            snps_file = f"{path_output_chrom}/snps_chunk_{i}.txt"
            with open(snps_file, 'w') as f:
                f.write("\n".join(AF_chunk['ID']))
            try:
                combined_geno = []
                for population in populations:
                    pop_name = population.split('_')[1]
                    pop_path = f"{path_input_plink_fam}/{population}"
            
                    cmd = [
                        f"cd {path_input_plink_fam}; {path_plink}/plink2 "
                        f"--bfile {pop_path} "
                        f"--extract {snps_file} "
                        f"--recode A "
                        f"--out {path_output_chrom}/pop_{pop_name}_chunk_{i}_size_{len(AF_chunk)}_mafs_{minaf}_{maxaf}"
                    ]
            
                    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    stdout, stderr = process.communicate()
                    
                    # Decode output if necessary (for Python 3 compatibility)
                    stdout = stdout.decode('utf-8')
                    stderr = stderr.decode('utf-8')
            
                    pop_geno = pd.read_csv(f"{path_output_chrom}/pop_{pop_name}_chunk_{i}_size_{len(AF_chunk)}_mafs_{minaf}_{maxaf}.raw", sep='\s+')
                    combined_geno.append(pop_geno)
        
                os.system(f"rm -rf {path_output_chrom}/snps_chunk_{i}.txt")
                os.system(f"rm -rf {path_output_chrom}/pop*")
                
                # Find common columns across all DataFrames
                common_columns = set(combined_geno[0].columns).intersection(*[df.columns for df in combined_geno])
                
                # Filter DataFrames to keep only the common columns
                combined_geno = [df[list(common_columns)] for df in combined_geno]

                # Define the columns you want to ensure are first
                priority_columns = ['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE']
                
                # Concatenate filtered DataFrames
                combined = pd.concat(combined_geno, ignore_index=True)
                
                # Ensure priority columns come first, followed by the rest
                ordered_columns = priority_columns + [col for col in combined.columns if col not in priority_columns]
                combined = combined[ordered_columns]
                # Concatenate filtered DataFrames
                combined.to_pickle(f"{path_output_chrom}/chunk_{i}_size_{len(AF_chunk)}_mafs_{minaf}_{maxaf}.pkl")
            except Exception as e:
                print(e)

"""
This function checks if there are some columns which are not SNPs and creates an ID df with them
"""
                
def make_ids(path_input, path_output):    
    chrom1  = os.listdir(path_input)[0]
    chunk1 = os.listdir(f"{path_input}/{chrom1}")[0]
    path_chunk = f"{path_input}/{chrom1}/{chunk1}"
    chunk = pd.read_pickle(path_chunk)
    cols = chunk.columns
    

    # Define a function to check if a column follows SNP naming conventions
    def is_snp(col):
        return col.startswith("rs") or re.match(r"^\d+:\d+$", col)  # Matches "rsID" or "chr:position"

    # Identify non-SNP columns
    non_snp_cols = [col for col in chunk.columns if not is_snp(col)]
    print(non_snp_cols)
    humans = chunk[non_snp_cols]
    humans.to_pickle(f"{path_output}/humans.pkl")
    
    for chrom in os.listdir(path_input):
        for chunk in os.listdir(f"{path_input}/{chrom}"):
            path_chunk = f"{path_input}/{chrom}/{chunk}"
            chunk = pd.read_pickle(path_chunk)       
            chunk = chunk.drop(columns=non_snp_cols)
            chunk.to_pickle(path_chunk)
            
"""
Check if the columns are SNPs
"""
def is_snp(col):
    return any(char.isdigit() for char in col)

"""
This function downloads the VCF files from the Ensembl FTP server
"""

def download_chromosome_vcf(chromosome, output_path):
    # Define the Ensembl FTP URL
    base_url = "ftp://ftp.ensembl.org/pub/release-110/variation/vcf/homo_sapiens/"
    filename = f"homo_sapiens-chr{chromosome}.vcf.gz"
    url = f"{base_url}{filename}"
    
    # Ensure the output path exists
    os.makedirs(output_path, exist_ok=True)
    
    # Define the output file path
    output_file = os.path.join(output_path, filename)
    
    # Use wget to download the file
    try:
        # Suppress the output during download
        subprocess.run(["wget", "-O", output_file, url], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Removed: print(f"Downloaded: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {filename}: {e}")


"""
Parse the vcf
"""

def parse_vcf(path_output, build_name):
    # Extract chromosome name
    chrom = re.search(r'chr(\w+)\.vcf', build_name).group(1)
    os.makedirs(f"{path_output}/chrom_{chrom}",exist_ok = True)
    print(chrom)
    vcf_file = f"{path_output}/{build_name}"

    # Check and create output directory
    os.makedirs(path_output, exist_ok=True)

    # Lists to store data
    pos_list = []
    rsid_list = []
    size_chunk = 500_000

    # Open the VCF file (supports .gz and plain files)
    if vcf_file.endswith(".gz"):
        f = gzip.open(vcf_file, 'rt')
    else:
        f = open(vcf_file, 'r')
    
    # Parse VCF file
    with f:
        for line in f:
            if line.startswith("#"):
                continue
            
            fields = line.strip().split("\t")
            pos = fields[1]
            rsid = fields[2]
            
            pos_list.append(pos)
            rsid_list.append(rsid)
    
    # Total SNPs and chunk calculations
    nr_snps_total = len(pos_list)
    num_subframes = nr_snps_total // size_chunk
    remaining_rows = nr_snps_total % size_chunk
    
    try:
        to_divide_in = remaining_rows // num_subframes
        rest = nr_snps_total - ((size_chunk + to_divide_in) * num_subframes)
        snps_per_segments = np.ones(num_subframes) * (size_chunk + to_divide_in)
        to_add = np.concatenate((np.ones(rest), np.zeros(num_subframes - rest)), axis=0)
        snps_per_segments += to_add
    except Exception as e:
        print(e)
        snps_per_segments = [nr_snps_total]
    
    # Save data in chunks
    start = 0
    i = 1
    for nr_snps in snps_per_segments:
        end = int(start + nr_snps)
        pos_list_temp = pos_list[start:end]
        rsid_list_temp = rsid_list[start:end]

        df = pd.DataFrame({"POS": pos_list_temp, "RSID": rsid_list_temp})
        df.to_pickle(f"{path_output}/chrom_{chrom}/build_nr_{i}.pkl")

        start = end
        i += 1