from pydca.meanfield_dca.meanfield_dca import MeanFieldDCA
from pydca.sequence_backmapper.sequence_backmapper import SequenceBackmapper
import logging 

"""This module performs RNA nucleotide-nucleotide coevolutionary computation  
using direc coupling analyis (DCA). 

Author:  Mehari B. Zerihun
"""

logger = logging.getLogger(__name__)


def get_mfdca_instance(msa_file, pseudo_count=0.5, seqid=0.8):
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

    mfdca_inst = MeanFieldDCA(msa_file, 'rna', pseudo_count=pseudo_count, seqid=seqid)

    return mfdca_inst 



def get_seqbackmapper_instance():
    """
    """
    raise NotImplementedError


