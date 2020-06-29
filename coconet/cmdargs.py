
"""
"""

class CmdArgs:
    """
    """
    msa_file = 'msa_file'
    msa_file_help = '''FASTA formatted MSA file path containing aligned RNA 
    sequences.
    '''

    refseq_file_optional = 'refseq_file'
    refseq_file_help = '''Path to FASTA formatted text file containing RNA 
    reference sequence. If the file contains multiple sequences, the first one is 
    taken.
    '''

    pdb_id = 'pdb_id'
    pdb_id_help = '''The name (identification) of the RNA in the PDB database.
    '''

    pdb_chain_name = 'pdb_chain_name'
    pdb_chain_name_help = '''Chain name of of the RNA in the PDB structure.
    '''