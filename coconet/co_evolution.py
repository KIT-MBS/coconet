from pydca.meanfield_dca.meanfield_dca import MeanFieldDCA
from pydca.plmdca.plmdca import PlmDCA as PseudoLikelihoodMaxDCA
from Bio import AlignIO
import logging 
import os 

"""This module performs RNA nucleotide-nucleotide coevolutionary computation  
using direc coupling analyis (DCA). 

Author:  Mehari B. Zerihun
"""

logger = logging.getLogger(__name__)

class CoEvolutionException(Exception):
    """
    """

def get_mfdca_instance(msa_file, pseudocount=0.5, seqid=0.8):
    """Obtains mean-field DCA instance for RNA DCA computation. 

    Parameters
    ----------
        msa_file : str 
            Path to FASTA formatted MSA file 
        pseudo_count : float 
            The pseudocount used to regularize MSA data 
        seqid : float 
            Sequence similarity cut-off value 

    Returns 
    -------
        mfdca_inst : MeanFieldDCA 
            A MeanFieldDCA instace of pydca implementation mean-field algorithm 
    """
    mfdca_inst = MeanFieldDCA(msa_file, 'rna', pseudocount=pseudocount, seqid=seqid)
    
    return mfdca_inst 


def get_plmdca_instance(msa_file, max_iterations=None, lambda_h=None, lambda_J=None, num_threads=None, verbose=False):
    """
    """
    # set the max iterations as large as possible so that the iteration converges
    # Since the computations are fast for RNA setting max_iterations large does not hurt
    max_iterations = 500000 if max_iterations is None else max_iterations
    # if lambda_h and lambda_J are None, the defaults within plmDCA are used
    plmdca_inst = PseudoLikelihoodMaxDCA(msa_file, 'rna', max_iterations=max_iterations, 
        lambda_h=lambda_h, lambda_J=lambda_J, num_threads=num_threads, verbose=verbose,
    )
    return plmdca_inst 


class MSAData:

    def __init__(self, input_msa_file_path):
        self.__input_msa_file_path = input_msa_file_path
        self.GAP_SYMBOLS = ('-', '.', '~')
        self.STANDARD_NUCLEOTIDES = ('A', 'C', 'G', 'U')
        self.__trimmed_msa_file_path = self.get_trimmed_msa_file_path()
        return None 


    @property
    def refseq(self):
        """
        """
        return self.__refseq


    @property
    def trimmed_msa_file_path(self):
        """
        """
        return self.__trimmed_msa_file_path

    
    def validate_refseq(self, refseq):
        """Within the MSA the reference sequence can have gap symbols if not 
        trimmed. Here the validation is done w.r.t standard nucleotide symbols
        and gap symbols.
        """
        valid_seq_symbols = self.STANDARD_NUCLEOTIDES + self.GAP_SYMBOLS
        for nucl in refseq: 
            if nucl not in valid_seq_symbols:
                logger.error('\n\tReference sequence contains invalid character {}'.format(nucl))
                raise ValueError
        return None 

    
    def get_trimmed_msa_file_path(self):
        """If the MSA data is already trimmed w.r.t the reference sequence,
        the path to trimmed MSA data is the input MSA file path. Otherwise the 
        MSA data is trimmed by reference sequence and saved into a new file in 
        the working directory. 
        """
        alignment_data = self.get_alignment_from_fasta_file()

        if self.msa_is_trimmed(alignment_data):
            self.__refseq = alignment_data[0][1]
            self.validate_refseq(self.__refseq)
            return self.__input_msa_file_path
        else: 
            trimmed_msa_data = self.get_trimmedd_msa_data(alignment_data)
            self.__refseq = trimmed_msa_data[0][1]
            self.validate_refseq(self.__refseq)
            input_msa_file_basename = os.path.basename(self.__input_msa_file_path)
            input_msa_file_basename, _ext = os.path.splitext(input_msa_file_basename)
            trimmed_fasta_file_path = input_msa_file_basename + '_trimmed.fa'
            self.write_trimmed_msa_data_to_fasta_file(trimmed_msa_data, trimmed_fasta_file_path)
            return trimmed_fasta_file_path


    def  msa_is_trimmed(self, alignment_data):
        """Verifies if the MSA file is trimmed by reference sequence or not 
        """
        refseq_name, refseq = alignment_data[0]
        # if the reference seqeunce contains gaps, the MSA is not trimmed
        for nucl in refseq:
            if nucl in self.GAP_SYMBOLS: 
                return False 
        return True 


    def get_alignment_from_fasta_file(self):
        """
        """
        record_iter = AlignIO.read(self.__input_msa_file_path, 'fasta')

        alignment_data = list() 
        for record in record_iter:
            seq = record.seq.upper()
            seqname = record.id 
            seq_id_pair = (seqname, seq)
            alignment_data.append(seq_id_pair)
        return alignment_data


    def get_trimmedd_msa_data(self, alignment_data):
        """Trims the MSA by refeseq is not not already trimmed. 
        """

        logger.info('\n\tTrimming MSA data')
        
        _refseq_name, refseq = alignment_data[0]
        columns_to_execlude = list() 
        for pos, nucl in enumerate(refseq):
            if nucl not in self.STANDARD_NUCLEOTIDES: columns_to_execlude.append(pos)
        
        trimmed_msa_data = list()
        for seq_name, seq in alignment_data:
            trimmed_seq_lst = [nucl for pos, nucl in enumerate(seq) if pos not in columns_to_execlude]
            trimmed_seq = ''.join(trimmed_seq_lst)
            trimmed_msa_data.append((seq_name, trimmed_seq))
        return trimmed_msa_data


    def write_trimmed_msa_data_to_fasta_file(self, trimmed_msa_data, trimmed_fasta_file_path):
        """
        """
        with open(trimmed_fasta_file_path, 'w') as fh:
            for seqname, seq in trimmed_msa_data:
                fh.write('>{}\n{}\n'.format(seqname, seq))
        return None 


if __name__ == '__main__':
    """
    """
    print('You are executing module {}'.format(__file__))
