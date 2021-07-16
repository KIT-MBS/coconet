from pydca.contact_visualizer.contact_visualizer import DCAVisualizer
from pydca.meanfield_dca.meanfield_dca import MeanFieldDCA
from pydca.plmdca.plmdca import PlmDCA 
from pydca.fasta_reader import fasta_reader
from pydca.sequence_backmapper import scoring_matrix
from pydca.plmdca.plmdca import PlmDCA as PseudoLikelihoodMaxDCA
from pydca.main import configure_logging
from Bio import pairwise2
import matplotlib.pyplot as plt 
from scipy.optimize import minimize as scipy_minimize
from  .convolution import Convolution
from .inputreader import InputReader
from .cmdargs import CmdArgs
import subprocess
import numpy as np
import logging 
import os, errno
import glob 
from datetime import datetime
import pickle
import random
from pathlib import Path
from argparse import ArgumentParser
import sys 


logger = logging.getLogger(__name__)

class CocoNetException(Exception):
    """Raise exceptions related to CocoNet computation.
    """

class CoCoNet:
    """Implements RNA contact prediction using direct coupling analysis enhanced by
    a simple convolutional neural network.
    """

    def __init__(self, data_dir, linear_dist=None, contact_dist=None):
        """Initializes CocoNet instance. 

        Parameters
        ----------
            self : CocoNet
                An instance of CocoNet class.
            dir_msa_files : str
                Path to directory containing the MSA files. 
            dir_pdb_file : str
                Path to the directory containing the PDB files.
            dir_refseq_files : str
                Path to the directory containing reference sequence files.
            linear_dist : int 
                Distance between sites in reference sequence 
            contact_dist : float
                Maximum distance between two residues in PDB file to be considered
                contacts. 
        """
        self.__data_dir = os.path.abspath(data_dir) 
        self.__linear_dist = linear_dist if linear_dist is not None else 4
        self.__contact_dist = contact_dist if contact_dist is not None else 10.0
     
        pdb_chains_list_file = os.path.join(self.__data_dir, 'CCNListOfPDBChains.txt')
        msa_files_list_file = os.path.join(self.__data_dir, 'CCNListOfMSAFiles.txt')
        pdb_files_list_file = os.path.join(self.__data_dir, 'CCNListOfPDBFiles.txt')
        input_reader = InputReader()
        self.__msa_file_names_list = input_reader.read_from_one_column_text_file(msa_files_list_file)
        self.__pdb_chains_list = input_reader.read_from_one_column_text_file(pdb_chains_list_file)
        self.__pdb_file_names_list = input_reader.read_from_one_column_text_file(pdb_files_list_file)
        self.__msa_files_dir = os.path.join(self.__data_dir, 'MSA')
        self.__refseqs_dir = os.path.join(self.__data_dir, 'sequences')
        self.__pdb_files_dir = os.path.join(self.__data_dir, 'PDBFiles')
        self.__secstruct_files_dir = os.path.join(self.__data_dir, 'secstruct')
        self.__msa_files_list  = [
            os.path.abspath(os.path.join(self.__msa_files_dir, msa_file + '.faclean')) for msa_file in self.__msa_file_names_list
        ]

        self.__refseqs = self.get_refseqs()
        self.__refseqs_len = self.get_refseqs_len()

        logmsg  = """
            Data directory          : {},
            PDB chains list file    : {},
            MSA files list file     : {},
            PDB files list file     : {},
        """.format(self.__data_dir, pdb_chains_list_file, 
            msa_files_list_file, pdb_files_list_file,
        )
        
        logger.info(logmsg)
        return None 

    
    @property 
    def pdb_file_names_list(self):
        return self.__pdb_file_names_list

    @property 
    def msa_file_names_list(self):
        return self.__msa_file_names_list

    @property 
    def pdb_chains_list(self):
        return self.__pdb_chains_list

    
    def map_pdb_id_to_family(self):
        """Mapps PDB ID to family name.

        Parameters
        ----------
            self : CocoNet(self, data_dir, linear_dist=None, contact_dist=None)
        
        Returns
        -------
            pdb_id_to_fam_name : dict
                pdb_id_to_fam_name[pdb_id]=fam_name
        """
        pdb_id_to_fam_name = dict()
        for pdb_id, fam_name in zip(self.__pdb_file_names_list, self.__msa_file_names_list):
            pdb_id_to_fam_name[pdb_id] = fam_name
        return pdb_id_to_fam_name

    
    @staticmethod 
    def _to_dict(files_list):
        """Puts a list of file paths into a dictionary 

        Parameters
        ----------
            files_list : list
                A list of file paths
        
        Returns
        -------
            files_dict : dict 
                A dictionary whose keys are basenames of files and values file path.
        """
        files_dict = dict()
        for f in files_list:
            basename, _ = os.path.splitext(os.path.basename(f))
            files_dict[basename] = f 
        return files_dict


    def get_refseq_files_list(self):
        """
        """
        refseq_files_list = [
            os.path.join(self.__refseqs_dir, pdb_file[:4] + '.fa') for pdb_file in self.__pdb_file_names_list
        ]
        return tuple(refseq_files_list)

    
    def get_pdb_files_list(self):
        """
        """
        pdb_files_list = [
            os.path.join(self.__pdb_files_dir, pdb_file + '.pdb') for pdb_file in self.__pdb_file_names_list
        ]
        return tuple(pdb_files_list)


    def create_directories(self, dir_path):
        """Creates (nested) directory given path.

        Parameters
        ----------
            self : CocoNet 
                An instance of CocoNet class.
            dir_path : str 
                Directory path.

        Returns
        -------
            None : None 
        """
        
        try:
            os.makedirs(dir_path)
        except OSError as e:
            if e.errno !=errno.EEXIST:
                logger.error('Unable to create directory using path {}'.format(
                    dir_path)
                )
                raise
        return None 

    
    def recompute_dca_data(self, msa_files_list=None, pickled_data=None):
        """Checks the last modification time of MSA files and pickled DCA data.
        If any of the MSA files are modified recently compared to pickled DCA
        data or pickled data does not exist, return values is True else False.

        Parameters
        ----------
            self : CocoNet 
                An instance of CocoNet class.
            msa_files_list : list 
                A list of RNA MSA files.
            pickled_data : str 
                Pickled DCA data 

        Returns
        -------
            True or False : bool
        """
    
        if any([msa_files_list, pickled_data]) is None:
            logger.error('\n\tYou need to supply all keyword arguments')
            raise CocoNetException
        if not os.path.exists(pickled_data): return True 
        mtime_msa_files = max([os.path.getmtime(f) for f in msa_files_list])
        mtime_pickled_data = os.path.getmtime(pickled_data)
        if mtime_msa_files > mtime_pickled_data:
            return True 
        return False

    
    def compute_mfdca_DI_scores(self):
        """Computes the mean-field DCA score of all the RNA families in the directory
        Parameters
        ----------
            self : CocoNet
                An instance of CocoNet class

        Returns 
        -------
            all_dca_data : dict 
        """
        all_dca_data = dict()
        for msa_file in self.__msa_files_list:
            logger.info('\n\tMSA file: {}'.format(msa_file))
            famname, _ext = os.path.splitext(os.path.basename(msa_file))
            logger.info('\n\tFamily name: {}'.format(famname))
            try:
                mfdca = MeanFieldDCA(msa_file, 'rna')
                dca_data = mfdca.compute_sorted_DI_APC()
            except Exception:
                raise 
            else:
                all_dca_data[famname] = dca_data
        # The DCA scores are in a list of tuples. Lets convert them to a dict
        all_dca_data_dict = dict()
        for rna_fam in all_dca_data:
            all_dca_data_dict[rna_fam] = dict((pair, score) for pair, score in all_dca_data[rna_fam])
        return all_dca_data_dict

    
    def compute_plmdca_FN_APC_scores(self, max_iterations=500000, num_threads=1, verbose=False):
        """
        """
        all_dca_data = dict()
        for msa_file in self.__msa_files_list:
            logger.info('\n\tMSA file: {}'.format(msa_file))
            famname, _ext = os.path.splitext(os.path.basename(msa_file))
            logger.info('\n\tFamily name: {}'.format(famname))
            try:
                plmdca_inst = PseudoLikelihoodMaxDCA(msa_file, 'rna', max_iterations=max_iterations, verbose=verbose, num_threads=num_threads)
                dca_data = plmdca_inst.compute_sorted_FN_APC()
            except Exception:
                raise 
            else:
                all_dca_data[famname] = dca_data
        # The DCA scores are in a list of tuples. Lets convert them to a dict
        all_dca_data_dict = dict()
        for rna_fam in all_dca_data:
            all_dca_data_dict[rna_fam] = dict((pair, score) for pair, score in all_dca_data[rna_fam])
        return all_dca_data_dict

    #TODO  remove this method. Currently,  its not being used by get_pdb_data() method.
    def recompute_mapped_pdb_data(self, refseq_files_list=None, pdb_files_list=None, pickled_data=None):
        """Checks the last modification time of reference sequence files, PDB structure 
        files and pickled mapped PDB data. If any of the reference sequence files or 
        PDB structure files is recently modified compared to the pickled data or 
        pickled data does not exist the return value is True, else False. 

        Parameters
        ----------
            self : CocoNet 
                An instance of CocoNet class. 
            refseq_files_list : list 
                A list of reference sequences. 
            pdb_files_list : list 
                A list of PDB structure files. 
            pickled_data : str 
                Name of pickled PDB data.
        
        Returns
        -------
            True or False : bool 
        """
        if any([refseq_files_list, pdb_files_list, pickled_data]) is None:
            logger.error('\n\tYou need to provide values for all the keyword arguments')
            raise CocoNetException

        if not os.path.exists(pickled_data): return True 
        mtime_refseq_files = max([os.path.getmtime(f) for f in refseq_files_list])
        mtime_pdb_files = max([os.path.getmtime(f) for f in pdb_files_list])
        mtime_input_files = max(mtime_refseq_files, mtime_pdb_files)
        mtime_pdb_data_dict = os.path.getmtime(pickled_data)
        if mtime_pdb_data_dict < max(mtime_input_files, mtime_refseq_files):
            return True
        return False 

    
    def get_pdb_data(self):
        """Computes mapped PDB contacts for multiple RNA families. The computed 
        mapped PDB data is pickled and only recomputed if any of reference sequence 
        files, PDB files or PDB chain metadata file is updated.

        Parameters
        ----------
            self : CocoNet 
                An instance of CocoNet class
        
        Returns
        -------
            mapped_pdb_data : dict 
                A dictionary whose keys are RNA familiy names and values dictionaries 
                that have site pair keys and PDB data values. 
        """
        refseq_files_list = self.get_refseq_files_list()
        pdb_files_list = self.get_pdb_files_list()
    
        logger.info('\n\tObtaining mapped PDB data')
        txtfreader = InputReader()
        mapped_pdb_data = dict()
        for chain_id, pdb_file, refseq_file, msa_file in zip(self.__pdb_chains_list, pdb_files_list, refseq_files_list, self.__msa_file_names_list):
            curr_pdb_data, _missing, _refseq_len = txtfreader.get_mapped_pdb_data(pdb_chain_id=chain_id, 
                refseq_file=refseq_file, pdb_file=pdb_file, linear_dist=self.__linear_dist, 
                contact_dist=self.__contact_dist
            )
            # self.__msa_file_names_list  contains the list of MSA files, not the full path of the files
            famname, _ext =  os.path.splitext(msa_file)
            mapped_pdb_data[famname] = curr_pdb_data
        return mapped_pdb_data


    def get_refseqs(self):
        """Obtains reference sequences of several RNA famlies from fasta formatted file. 

        Parameters
        ----------
            self : CocoNet 
                An instance of CocoNet class
        
        Returns
        -------
            reference_sequenes_dict : dict()
            reference_sequences_dict[FAMILY_NAME] = sequence 
        """
        
        refseq_files_list = self.get_refseq_files_list()
        logger.info('\n\tObtaining reference sequences from FASTA files')
        reference_sequences_dict = dict()
        for refseq_file, msa_file_basename in zip(refseq_files_list, self.__msa_file_names_list):
            # if reference sequence file contains multiple sequences, take the first one.
            reference_sequences_dict[msa_file_basename.strip()] = fasta_reader.get_alignment_from_fasta_file(refseq_file)[0].strip()
        return reference_sequences_dict

    
    def get_refseqs_len(self):
        """Obtains length of reference sequence for each RNA family

        Parameters 
        ----------
            self : CocoNet 
                An instance of CocoNet class
        
        Returns 
        -------
            refseqs_len_dict : dict()
                refseqs_len_dict[FAMILY_NAME] = refseq_length
        """
        refseqs_dict = self.__refseqs
        logger.info('\n\tObtaining length of reference sequences for {} RNA families'.format(len(refseqs_dict)))

        refseqs_len_dict = {
            fam : len(refseqs_dict[fam]) for fam in refseqs_dict
        }
        return refseqs_len_dict   

    
    def objective_function(self, weight_matrix, dca_data_train, pdb_data_train):
        """Computes the total objective function (for the entire training data set)

        Parameters
        ----------
            self : CocoNet 
                An instance of CocoNet class.
            weight_matrix : np.array
                A 1d numpy array of weights
            dca_data_train : dict 
                DCA data for training.
            pdb_data_train : dict 
                PDB data for training. 

        Returns
        -------
            total_cost : float 
                Total value of the objective/error function.
        """
        fm_dim_size = int(np.sqrt(weight_matrix.size))
        assert fm_dim_size * fm_dim_size == weight_matrix.size
        conv_inst = Convolution(fm_dim_size)
        total_cost = 0.0
        logger.info('\n\tComputing objective function using filter matrix of size: {}'.format(weight_matrix.size))
        for fam in dca_data_train:
            fam_dca_scores = dca_data_train[fam]
            fam_pdb_contacts = pdb_data_train[fam]
            fam_refseq_len = self.__refseqs_len[fam]
            fam_cost = conv_inst.objective_function(fam_pdb_contacts, fam_dca_scores, weight_matrix, fam_refseq_len)
            total_cost += fam_cost
        return total_cost

    
    def objective_function_WC_and_NONWC_pairs(self, weight_matrix, dca_data_train, pdb_data_train):
        """Computes value of cost function when contacts are categorized as WC and non-WC nucleotide 
        pairs.

        Parameters
        ----------
            self :CocoNet  
                CocoNet(self, data_dir, linear_dist=None, contact_dist=None) 
            weight_matrix : np.array()
                1d numpy array of weights. Its size must be twice the size of 
                the filter matrix used so as to accommodate WC and non-WC weights
                together. 
            dca_data_train : dict()
                DCA data for training set families. 
            pdb_data_train : dict()
                PDB data for training set families. 

        Returns
        -------
            total_cost : float 
                value of cost function for at a particular iteration (value of weight matrix).
        """

        fm_dim_size = int(np.sqrt((weight_matrix.size/2)))
        assert weight_matrix.size == 2*fm_dim_size * fm_dim_size
        conv_inst = Convolution(fm_dim_size)
        total_cost = 0.0
        logger.info('\n\tComputing objective function for WC and non-WC residue pairs using filter matrix of size: {}'.format(weight_matrix.size))
        for fam in dca_data_train:
            fam_dca_scores = dca_data_train[fam]
            fam_pdb_contacts = pdb_data_train[fam]
            fam_refseq = self.__refseqs[fam]
            fam_cost = conv_inst.objective_function_WC_and_NONWC_pairs(fam_pdb_contacts, fam_dca_scores, weight_matrix, fam_refseq)
            total_cost += fam_cost
        return total_cost

    
    def total_gradients(self, weight_matrix, dca_data_train, pdb_data_train):
        """Computes the total gradient of the error function. That is, for more than
        one RNA family.

        Parameters
        ----------
            self : CocoNet 
                An instance of CocoNet class.
            weight_matrix : np.array
                A 1d numpy array of weights
            dca_data_train : dict 
                DCA data for training.
            pdb_data_train : dict 
                PDB data for training. 
        
        Returns
        -------
        total_gradients : np.array
            A 1d numpy array of the total gradient

        """
        logger.info('\n\tFilter Matrix\n{}'.format(weight_matrix))
        fm_dim_size = int(np.sqrt(weight_matrix.size)) 
        assert fm_dim_size * fm_dim_size == weight_matrix.size
        conv_inst = Convolution(fm_dim_size)
        total_gradients = 0.0 # will be promoted to numpy array when added to another array
        for fam in dca_data_train:
            fam_dca_scores = dca_data_train[fam]
            fam_pdb_contacts = pdb_data_train[fam]
            fam_refseq_len = self.__refseqs_len[fam]
            fam_gradient = conv_inst.gradients(fam_pdb_contacts, fam_dca_scores, weight_matrix, fam_refseq_len)
            total_gradients += fam_gradient
        return total_gradients 

    
    def total_gradients_WC_and_NONWC_pairs(self, weight_matrix, dca_data_train, pdb_data_train):
        """Computes the total gradient of the error function for WC and non-WC contact 
        pair classification for test RNA families. 

        Parameters
        ----------
            self : CocoNet 
                An instance of CocoNet class.
            weight_matrix : np.array
                A 1d numpy array of weights
            dca_data_train : dict 
                DCA data for training.
            pdb_data_train : dict 
                PDB data for training. 
        
        Returns
        -------
        total_gradients : np.array
            A 1d numpy array of the total gradient

        """
    
        logger.info('\n\tFilter Matrix\n{}'.format(weight_matrix))
        fm_dim_size = int(np.sqrt(weight_matrix.size/2)) 
        assert weight_matrix.size == 2*fm_dim_size * fm_dim_size 
        conv_inst = Convolution(fm_dim_size)
        total_gradients = 0.0 # will be promoted to numpy array when added to another array
        for fam in dca_data_train:
            fam_dca_scores = dca_data_train[fam]
            fam_pdb_contacts = pdb_data_train[fam]
            fam_refseq = self.__refseqs[fam]
            fam_gradient = conv_inst.gradients_WC_and_NONWC_pairs(fam_pdb_contacts, fam_dca_scores, weight_matrix, fam_refseq)
            total_gradients += fam_gradient
        return total_gradients 

    
    def train(self, initial_weight_matrix, dca_data_train, pdb_data_train):
        """Performs gradient decent using scipy.optimize.minimize

        Parameters
        ----------
            self : CocoNet 
                An instance of CocoNet class.
            initial_weight_matrix : np.array
                A 1d numpy array of initial weights.
            dca_data_train : dict 
                DCA data for training.
            pdb_data_train : dict 
                PDB data for training.

        Returns
        -------
            result_lbfgs :  scipy.optimize.optimize.OptimizeResult
                A instance of scipy.optimize.optimize.OptimizeResult class. It has
                the following attributes:
                result_lbfgs.fun  : float 
                    Value of minimized  objective function 
                result_lbfgs.jac : np.array
                    A 1d numpy array of the hessian inverse
                result_lbfgs.nit : int 
                    The number of minimization iterations. 
                result_lbfgs.success : bool
                    Success status of minimization 
                result_lbfgs.x : np.array
                    A 1d numpy array of optimized weights.
                Note: there are more attributes that can be displayed using, for 
                example, print(result_lbfgs_b)
        """
        result_lbfgs = scipy_minimize(self.objective_function, initial_weight_matrix, 
            method='L-BFGS-B', jac=self.total_gradients,
            args=(dca_data_train, pdb_data_train)
        )
        if not result_lbfgs.success: raise CocoNetException('Iteration not converged') 
        return result_lbfgs

    
    def train_3x3(self, dca_data_train, pdb_data_train):
        """Computes a 3x3 trained filter matrix.

        Parameters
        ----------
            self : CocoNet 
                An instance of CocoNet class.
            dca_data_train : dict 
                Training DCA data.
            pdb_data_train : dict 
                Training PDB data.
        
        Returns
        -------
            lbfgs_result.x : np.array
                A 1d numpy array of trained filter matrix elements.
        """
        weight_matrix = np.zeros(shape=(9, ), dtype=np.float64)

        lbfgs_result = self.train(weight_matrix, dca_data_train, pdb_data_train)
        return lbfgs_result.x

    
    def train_5x5(self, dca_data_train, pdb_data_train):
        """Computes a 5x5 trained filter matrix.

        Parameters
        ----------
            self : CocoNet 
                An instance of CocoNet class.
            dca_data_train : dict 
                Training DCA data.
            pdb_data_train : dict
                Training PDB data.
        
        Returns
        -------
            lbfgs_result.x : np.array
                A 1d numpy array of trained filter matrix elements.
        """
        weight_matrix = np.zeros(shape=(25, ), dtype=np.float64)
        lbfgs_result = self.train(weight_matrix, dca_data_train, pdb_data_train)
        return lbfgs_result.x

    
    def train_7x7(self, dca_data_train, pdb_data_train):
        """Computes a 7x7 trained filter matrix.

        Parameters
        ----------
            self : CocoNet 
                An instance of CocoNet class.
            dca_data_train : dict 
                Training DCA data.
            pdb_data_train : dict
                Training DCA data.
    
        Returns
        -------
            lbfgs_results.x : np.array
                A 1d numpy array of trained filter matrix elements.
        """
        weight_matrix = np.zeros(shape=(49, ), dtype=np.float64)
        lbfgs_result = self.train(weight_matrix, dca_data_train, pdb_data_train)
        return lbfgs_result.x

    
    def train_WC_and_NONWC(self, initial_weight_matrix, dca_data_train, pdb_data_train):
        """
        """
        result_lbfgs = scipy_minimize(self.objective_function_WC_and_NONWC_pairs, 
            initial_weight_matrix, method='L-BFGS-B', 
            jac=self.total_gradients_WC_and_NONWC_pairs, 
            args=(dca_data_train, pdb_data_train)
        )
        if not result_lbfgs.success: raise CocoNetException('Iteration not converged')
        return result_lbfgs

    
    def train_WC_and_NONWC_3x3(self, dca_data_train, pdb_data_train):
        """
        """
        weight_matrix = np.zeros(shape=(18, ), dtype=np.float64)
        lbfgs_result = self.train_WC_and_NONWC(weight_matrix, dca_data_train, pdb_data_train)
        return lbfgs_result.x

    
    def train_WC_and_NONWC_5X5(self, dca_data_train, pdb_data_train):
        """
        """
        weight_matrix = np.zeros(shape=(50, ), dtype=np.float64)
        lbfgs_result =  self.train_WC_and_NONWC(weight_matrix, dca_data_train, pdb_data_train)
        return lbfgs_result.x 

    
    def train_WC_and_NONWC_7x7(self, dca_data_train, pdb_data_train):
        """
        """
        weight_matrix = np.zeros(shape=(98, ), dtype=np.float64)
        lbfgs_result = self.train_WC_and_NONWC(weight_matrix, dca_data_train, pdb_data_train)
        return lbfgs_result.x 

    def cross_validation(self, matrix_size=None, wc_and_nwc=False, num_batchs=5, output_dir=None, on_plm=False,
            verbose=False, num_threads=None, max_iterations=None, num_trials=1):
        """Performs cross validation of CocoNet 
        """
        

        pdb_data = self.get_pdb_data() 
        if on_plm:
            dca_data=  self.compute_plmdca_FN_APC_scores(max_iterations=max_iterations, num_threads=num_threads, verbose=verbose)
            if wc_and_nwc and output_dir is None:
                output_dir = f'CoCoNet_plmDCA_CrossValidation_Output_2x{matrix_size}x{matrix_size}-' + datetime.now().strftime('%Y-%m-%d-%H_%M_%S')
            if wc_and_nwc is False and output_dir is None:
                output_dir = f'CoCoNet_plmDCA_CrossValidation_Output_{matrix_size}x{matrix_size}-' + datetime.now().strftime('%Y-%m-%d-%H_%M_%S')
        
        else: # uses mean-field DCA
            dca_data = self.compute_mfdca_DI_scores() 
            if wc_and_nwc and output_dir is None:
                output_dir = f'CoCoNet_mfDCA_CrossValidation_Output_2x{matrix_size}x{matrix_size}-' + datetime.now().strftime('%Y-%m-%d-%H_%M_%S')
            if wc_and_nwc is False and output_dir is None:
                output_dir = f'CoCoNet_mfDCA_CrossValidation_Output_{matrix_size}x{matrix_size}-' + datetime.now().strftime('%Y-%m-%d-%H_%M_%S')
        

        fams_in_DCA = list(dca_data.keys())
        fams_in_PDB = list(pdb_data.keys())
        for fam in fams_in_DCA: assert fam in fams_in_PDB
        batch_len = len(fams_in_PDB)//num_batchs

        for i in range(num_trials):
            # create output destination directories
            
            trial_dir = 'trial_{}'.format(i + 1) 
            trial_output_dir =  os.path.join(output_dir, trial_dir)
            # shuffle the list of RNAs 
            random.shuffle(fams_in_PDB)
            #divide families into batches
            for j in range(num_batchs):
                trial_batch_output_dir = os.path.join(trial_output_dir, 'fold_{}'.format(j + 1))
                self.create_directories(trial_batch_output_dir)
                lower_bound = j * batch_len 
                upper_bound = lower_bound + batch_len
                batch_j = fams_in_PDB[lower_bound:upper_bound] if j < (num_batchs - 1) else fams_in_PDB[lower_bound:]
                # take batch_j as a test set 
                testset_fams = batch_j 
                training_fams = [fam for fam in fams_in_PDB if fam not in testset_fams]
                dca_data_train_j = { fam : dca_data[fam] for fam in training_fams }
                pdb_data_train_j = {fam : pdb_data[fam] for fam in training_fams}
                
                metadata_file = os.path.join(trial_batch_output_dir, 'metadata_fold_{}.txt'.format(j + 1)) 
                
                with open(metadata_file, 'w') as fh: 
                    fh.write('Testset RNA Families \n')
                    for counter, fam in enumerate(testset_fams, start=1): fh.write('{}\t{}\n'.format(counter, fam))
                    fh.write('Training RNA Families\n') 
                    for counter, fam in enumerate(training_fams, start=1): fh.write('{}\t{}\n'.format(counter, fam))
                # perform training
                base_header = 'Coconet cross validation result for {} filter matrix.\nTotal number of training families: {}'
                # 3x3 
                if matrix_size == 3 and not wc_and_nwc:
                    mat_3x3 = self.train_3x3(dca_data_train_j, pdb_data_train_j)
                    outfile_3x3 = os.path.join(trial_batch_output_dir, 'params_3x3.txt')
                    header_mat_3x3 = base_header.format('3x3', len(training_fams))
                    np.savetxt(outfile_3x3, mat_3x3, header=header_mat_3x3)
                if matrix_size == 3 and wc_and_nwc:
                    mat_WCNWC_3x3 = self.train_WC_and_NONWC_3x3(dca_data_train_j, pdb_data_train_j)
                    outfile_WCNWC_3x3 = os.path.join(trial_batch_output_dir, 'params_WCNWC_3x3.txt')
                    header_mat_WCNWC_3x3 = base_header.format('WCNWC 3x3', len(training_fams))
                    np.savetxt(outfile_WCNWC_3x3, mat_WCNWC_3x3, header=header_mat_WCNWC_3x3)
                # 5x5 
                if matrix_size == 5 and not wc_and_nwc:
                    mat_5x5 = self.train_5x5(dca_data_train_j, pdb_data_train_j)
                    outfile_5x5 = os.path.join(trial_batch_output_dir, 'params_5x5.txt')
                    header_mat_5x5 = base_header.format('5x5', len(training_fams))
                    np.savetxt(outfile_5x5, mat_5x5, header=header_mat_5x5)
                if matrix_size == 5 and wc_and_nwc:
                    mat_WCNWC_5x5 = self.train_WC_and_NONWC_5X5(dca_data_train_j, pdb_data_train_j)
                    outfile_WCNWC_5x5 = os.path.join(trial_batch_output_dir, 'params_WCNWC_5x5.txt')
                    header_mat_WCNWC_5x5 = base_header.format('WCNWC 5x5', len(training_fams))
                    np.savetxt(outfile_WCNWC_5x5, mat_WCNWC_5x5, header=header_mat_WCNWC_5x5)
                # 7x7
                if matrix_size == 7 and not wc_and_nwc:
                    mat_7x7 = self.train_7x7(dca_data_train_j, pdb_data_train_j)
                    outfile_7x7 = os.path.join(trial_batch_output_dir, 'params_7x7.txt')
                    header_mat_7x7 = base_header.format('7x7', len(training_fams))
                    np.savetxt(outfile_7x7, mat_7x7, header=header_mat_7x7)
                if matrix_size == 7 and wc_and_nwc:
                    mat_WCNWC_7x7 = self.train_WC_and_NONWC_7x7(dca_data_train_j, pdb_data_train_j)
                    outfile_WCNWC_7x7 = os.path.join(trial_batch_output_dir, 'params_WCNWC_7x7.txt')
                    header_mat_WCNWC_7x7 = base_header.format('WCNWC 7x7', len(training_fams))
                    np.savetxt(outfile_WCNWC_7x7, mat_WCNWC_7x7, header=header_mat_WCNWC_7x7)   
                
        return None 
# end of class CoCoNet 

def execute_from_command_line(matrix_size=None, wc_and_nwc=False, num_trials=1,
        on_plm=False, verbose=False, output_dir=None, max_iterations=None, num_threads=None):
    """
    """
    if matrix_size is None: matrix_size = 3 # use this default values to annotate do ouput 
    #directory names consistent with the default values in argparser.
    
    if verbose: configure_logging()
    logger.info('\n\tTraining CoCoNet')
    dataset_dir = Path(__file__).parent.parent / 'RNA_DATASET'
    coconet_inst = CoCoNet(dataset_dir)
    coconet_inst.cross_validation(matrix_size, num_threads=num_threads, wc_and_nwc=wc_and_nwc, 
        on_plm=on_plm, verbose=verbose, max_iterations=max_iterations, num_trials=num_trials
    )
    
    return None 


def train_coconet():
    """
    """
    parser = ArgumentParser() 

    # This argument is added to help run help message when no positional argument is supplied
    parser.add_argument('run', help='Execute CoCoNet training')
    parser.add_argument(CmdArgs.verbose_optional, help=CmdArgs.verbose_optional_help, action='store_true')
    parser.add_argument(CmdArgs.matrix_size, help=CmdArgs.matrix_size_help,  type=int, choices=(3, 5, 7), default=3)
    parser.add_argument(CmdArgs.wc_and_nwc_optional, help=CmdArgs.wc_and_nwc_optional_help, action='store_true')
    parser.add_argument(CmdArgs.max_iterations_optional, help=CmdArgs.max_iterations_help, type=int, default=500000)
    parser.add_argument(CmdArgs.num_threads_optional, help=CmdArgs.num_threads_help, type=int, default=1)
    parser.add_argument(CmdArgs.on_plm_optional, help=CmdArgs.on_plm_optional_help, action='store_true')
    parser.add_argument(CmdArgs.num_trials_optional, help=CmdArgs.num_trials_optional_help, type=int, default=1)

    args = parser.parse_args(args = None if sys.argv[1:] else ['--help']) 
    args_dict = vars(args)

    execute_from_command_line(
        args_dict.get(CmdArgs.matrix_size[2:]),
        wc_and_nwc = args_dict.get(CmdArgs.wc_and_nwc_optional.strip()[2:]),
        verbose = args_dict.get(CmdArgs.verbose_optional.strip()[2:]),
        max_iterations = args_dict.get(CmdArgs.max_iterations_optional.strip()[2:]),
        num_threads = args_dict.get(CmdArgs.num_threads_optional.strip()[2:]),
        on_plm = args_dict.get(CmdArgs.on_plm_optional.strip()[2:]),
        num_trials = args_dict.get(CmdArgs.num_trials_optional.strip()[2:]),
    )


if __name__ =='__main__':
    train_coconet()

