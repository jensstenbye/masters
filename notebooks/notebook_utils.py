from pyfaidx import Fasta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

#Looks up sequence from annotation file and saves it as fasta. 
#Length of sequence is determined by seq_window_size
#Only considers genes present in scRNA_geneid_list
#Returns a list of remaining geneid's [some may be dropped due to uncalled bases]
def fetch_tss_to_fasta(seq_window_size,
                       scRNA_geneid_list,
                       ann_path,
                       genome_path,
                       out_path, 
                       **kwargs):
    #Read in fasta and index for quick lookup
    genome = Fasta(genome_path, as_raw=True)
    
    tss_df = pd.read_csv(ann_path, sep='\t')
    tss_df = tss_df.sort_values('ccds_id').drop_duplicates('gene_id').sort_index()
    tss_df.set_index('gene_id', inplace = True)
    
    #Only interested in genes found in scRNA dataset
    valid_geneid = list(set(scRNA_geneid_list) & set(tss_df.index))
    final_geneid = []
    #Iterate over all gene ids in TSS file and extract sequence
    with open(out_path, 'w') as out_file:
        for geneid in valid_geneid:
            #Extract tss info
            tss = tss_df.loc[geneid]
            chrom_idx = tss['chrom']
            tss_idx = tss['tss']

            #Look up sequence window
            tss_seq = genome[chrom_idx][tss_idx-seq_window_size//2:tss_idx+seq_window_size//2]
        
            #If containing N's discard gene
            error_counter = 0
            if 'N' in tss_seq:
                error_counter += 1  
                continue
            
            #Write to a fasta file
            out_file.write(f'>{geneid}\n{tss_seq}\n')
            final_geneid.append(geneid)
    
    print(f"{len(valid_geneid)} genes processed, {error_counter} genes dropped due to presence of uncalled bases")
    return final_geneid

#Plots for easy hist and scatter plots
def plot_histo_filter(data, 
                      save_path_svg=None, 
                      title='Default title',
                      ylab='Default y-label',
                      xlab='Default x-label',
                      cutoff_points=[],
                      bins=100,
                      xwin=None,
                      xbot=None,
                      ylog=False):
    #Plot histogram over data
    hist, bins, patches = plt.hist(data, density=False, range=xwin, bins=bins, color = "firebrick", ec="black")
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    if ylog:
        plt.yscale('log')
        plt.ylim(bottom=1)
    if xbot:
        plt.xlim(left=xbot)
    if cutoff_points:
        count = 0
        for cutoff in cutoff_points:
            if count == 0:
                plt.axvline(x=cutoff, ymin=0, ymax=1,linestyle='--', label='Filter cutoff')
            else:
                plt.axvline(x=cutoff, ymin=0, ymax=1,linestyle='--')
            count += 1
        plt.legend()
    
    if save_path_svg:
        plt.savefig(save_path_svg, format='svg')  
    return plt

def plot_scatter_filter(x, y,
                        save_path_svg=None,
                        title='Default title',
                        ylab='Default y-label',
                        xlab='Default x-label',
                        cutoff_x=[], 
                        cutoff_y=[]):
    #Plot scatter over x,y
    plt.scatter(x,y,s=1,color='red')
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
        
    if cutoff_x:
        for cutoff in cutoff_x:
            plt.axvline(x=cutoff, ymin=0, ymax=1,linestyle='--', label='Filter cutoff')
        plt.legend()
    if cutoff_y:
        for cutoff in cutoff_y:
            plt.axhline(y=cutoff, xmin=0, xmax=1,linestyle='--', label='Filter cutoff')
        plt.legend() 
        
    if save_path_svg:
        plt.savefig(save_path_svg, format='svg')  
    return plt                      


#gene id quick fix, dont know what went worng..
# import scanpy as sc
# scRNA_counts = sc.read('/home/jbs/scRNA-seq/steps/preprocessed/redone/PBMC_win600.h5',
# 							cache=True)
# scRNA_geneid = scRNA_counts.var_names.to_list()
# print(len(scRNA_geneid))
# window_size=600
# print('start')
# final_geneid = fetch_tss_to_fasta(seq_window_size=window_size,
#                                scRNA_geneid_list=scRNA_geneid,
#                                ann_path='/home/jbs/scRNA-seq/data/annotation_2kbp.tss',
#                                genome_path='/home/jbs/scRNA-seq/data/GRCh38.primary_assembly.genome.fa',
#                                out_path=f'/home/jbs/scRNA-seq/steps/preprocessed/redone/seq_win{window_size}.fa')
# print('done')