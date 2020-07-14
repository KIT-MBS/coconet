
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

if __name__ == '__main__':
    print('You are running module {}'.format(__file__))