
"""Module for defining command line arguments
"""

class CmdArgs:
    """
    """
    verbose_optional = '--verbose'
    verbose_optional_help='''display logging messages on the terminal.
    '''

    msa_file = 'msa_file'
    msa_file_help = '''FASTA formatted MSA file path containing aligned RNA 
    sequences.
    '''

    matrix_size = '--msize'
    matrix_size_help = '''the size of filter matrix. This value is the size 
    of one of the dimensions of the square matrix and must be 3, 5, or 7.
    '''

    wc_and_nwc_optional = '--wc_and_nwc'
    wc_and_nwc_optional_help = '''use two matrices one for Watson-Crick nucleotide pairs
    and the other for non-Watson-Crick nucleotide pairs.
    '''

    # plmDCA specific args
    on_plm_optional = '--on_plm'
    on_plm_optional_help = """Do convolution on top of plmDCA."""
    
    max_iterations_optional = '--max_iterations'
    max_iterations_help = """Number of iterations for gradient decent 
    for negative pseudolikelihood minimization.
    """
    num_threads_optional = '--num_threads'
    num_threads_help = "Number of threads from plmDCA computation."

    num_trials_optional = '--num_trials'
    num_trials_optional_help = '''Number of repitions for cross validation. Each time a repitition is done, 
        the RNA data is shuffiled to form new sets of RNAs in each fold.'''
    for_all_matrices_optional = '--all_filters'
    for_all_matrices_optional_help = """Train CoCoNet for all filter matrix sizes in one run.
    """

if __name__ == '__main__':
    print('You are running module {}'.format(__file__))