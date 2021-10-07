from pydca.fasta_reader import fasta_reader
from pydca.contact_visualizer.contact_visualizer import DCAVisualizer
import pathlib 
import glob 
import logging 
import subprocess
import pickle
import numpy as np 
from argparse import ArgumentParser
from pydca.main import configure_logging
    


"""Module to compute PPV for both testset and cross-validation RNA data.
"""

logger = logging.getLogger(__name__)
class FamilyMapper:
    """Mapps family names to coresponding PDB names and PDB chain IDs for the RNA
    dataset 
    """
    def __init__(self, dataset_dir):
        self.__dataset_dir = dataset_dir
        self.__families_list_file = pathlib.Path(self.__dataset_dir) / 'CCNListOfMSAFiles.txt'
        self.__pdb_ids_list_file = pathlib.Path(self.__dataset_dir) / 'CCNListOfPDBFiles.txt' 
        self.__pdb_chain_list_file = pathlib.Path(self.__dataset_dir ) / 'CCNListOfPDBChains.txt'
        return None


    def read_from_one_column_file(self, input_file):
        """
        """
        first_column_data = list()
        with open(input_file, 'r') as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('#'): continue
                line = line.split()
                if len(line) > 1: logger.warning('\n\tLine contains multicolunm data')
                first_column_data.append(line[0])
        return first_column_data


    def get_list_of_families(self):
        """
        """
        list_of_fams = self.read_from_one_column_file(self.__families_list_file)
        logger.info('\n\tFound {} families'.format(len(list_of_fams)))
        return list_of_fams
    

    def get_list_of_pdb_ids(self):
        """
        """
        list_of_pdb_ids = self.read_from_one_column_file(self.__pdb_ids_list_file) 
        logger.info('\n\tFound {} PDB IDs'.format(len(list_of_pdb_ids)))
        return list_of_pdb_ids
    

    def get_list_of_pdb_chains(self):
        """
        """
        list_of_pdb_chains  = self.read_from_one_column_file(self.__pdb_chain_list_file)
        logger.info('\n\tFound {} PDB chains'.format(len(list_of_pdb_chains)))
        return list_of_pdb_chains
    

    def map_fam_name_to_pdb_id(self):
        """
        """
        list_of_fams = self.get_list_of_families()
        list_of_pdb_ids = self.get_list_of_pdb_ids()
        logger.info('\n\tMapping families name to PDB IDs')
        fam_to_pdb = dict(zip(list_of_fams, list_of_pdb_ids))
        return fam_to_pdb
    

    def map_fam_name_to_pdb_chain(self):
        """
        """
        list_of_fams = self.get_list_of_families()
        list_of_pdb_chain = self.get_list_of_pdb_chains()
        logger.info('\n\tMapping families to PDB chains')
        fam_to_pdb_chain = dict(zip(list_of_fams, list_of_pdb_chain))
        return fam_to_pdb_chain

    
    def map_fam_name_to_secstruct_file_path(self):
        """
        """
        logger.info('\n\tMapping family name to secondary structure file paths')
        data_path = pathlib.Path(self.__dataset_dir) / 'secstruct'
        secstruct_files = data_path.glob('secstruct_*') 
        fam_to_pdb_map = self.map_fam_name_to_pdb_id() 
        pdb_to_fam_map = { 
            val:key for key, val in fam_to_pdb_map.items() 
        } 
        ss_file_prefix = 'secstruct_'
        fam_name_to_ss_file = dict()
        for ss_file in secstruct_files:
            ss_basename = pathlib.Path(ss_file).stem
            pdb_id = ss_basename[len(ss_file_prefix):]
            fam_name = pdb_to_fam_map[pdb_id]
            fam_name_to_ss_file[fam_name] = ss_file
        return fam_name_to_ss_file
    

    def map_fam_name_to_refseq_file_path(self):
        """
        """
        fam_to_pdb_id = self.map_fam_name_to_pdb_id()
        pdb_id_to_fam = {val:key for key, val in fam_to_pdb_id.items()}
        mapping = list(pdb_id_to_fam.items()) 
        refseq_files = dict()
        for pdb_id, fam_name in mapping:
            pdb_id_standared = pdb_id[:4] # pdb_ids have _CHAINID  following the standared ID
            fasta_file = (pathlib.Path(self.__dataset_dir) / 'sequences') / (pdb_id_standared + '.fa')
            if not fasta_file.is_file():
                logger.error(f'\n\t{fasta_file} does not exist')
                raise FileNotFoundError
            refseq_files[fam_name] = fasta_file
        return refseq_files
# End of class FamilyMapper 

class RNADataset:
    """
    """
    def __init__(self, dataset_dir):
        """
        """
        self.__dataset_dir = pathlib.Path(dataset_dir)
        return None 

    
    def get_refseqs(self):
        """
        """
        refseq_files = FamilyMapper(self.__dataset_dir).map_fam_name_to_refseq_file_path() 
        refseqs = dict() 
        for fam_name, refeq_file in refseq_files.items():
            seq = fasta_reader.get_alignment_from_fasta_file(refeq_file)
            refseqs[fam_name] =  str(seq[0])
        return refseqs

    
    def get_refseqs_len(self):
        """
        """
        refseqs = self.get_refseqs() 
        refseqs_len = dict() 
        for fam, refseq in  refseqs.items():
            refseqs_len[fam] = len(refseq)
        return refseqs_len

    
    def get_list_of_all_fams(self):
        """
        """
        fam_mapper = FamilyMapper(self.__dataset_dir).map_fam_name_to_pdb_id() 
        list_of_all_fams = list(fam_mapper.keys())
        logger.info('\n\tTotal list of RNA fams obtained: {}'.format(len(list_of_all_fams)))
        return list_of_all_fams

    
    def get_list_of_fams_with_HighMeff(self, list_of_high_meff_file=None):
        """
        """
        if list_of_high_meff_file is None:
            list_of_high_meff_file = self.__dataset_dir / 'FamiliesWithHighMeff.txt'
        logger.info('\n\tObtaining RNAs with HighMeff from file: {}'.format(list_of_high_meff_file))
        list_of_highMeff_fams = list()
        with open(list_of_high_meff_file, 'r') as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('#'):continue
                list_of_highMeff_fams.append(line)
        logger.info('\n\tNumber of RNA fams with HighMeff: {}'.format(len(list_of_highMeff_fams)))
        return list_of_highMeff_fams

    
    def get_list_of_fams_with_LowMeff(self, list_of_high_meff_file=None):
        """
        """
    
        high_Meff_fams_list = self.get_list_of_fams_with_HighMeff(list_of_high_meff_file)
        fam_mapper_inst = FamilyMapper(self.__dataset_dir)
        all_fams_list = fam_mapper_inst.get_list_of_families()
        logger.warning('\n\tTotal number of RNA fams found: {}'.format(len(all_fams_list)))
        low_Meff_fams_list = list()
        for fam in all_fams_list:
            if fam not in high_Meff_fams_list:
                low_Meff_fams_list.append(fam)
        logger.info('\n\tTotal number of LowMeff fams:{}'.format(len(low_Meff_fams_list)))
        return low_Meff_fams_list
#End of class RNADataset 

class PDBData:
    """
    """
    def __init__(self, dataset_dir):
        """
        """
        self.__dataset_dir = pathlib.Path(dataset_dir)
        return None 
    

    def get_pdb_files(self):
        """
        """
        fam_mapper = FamilyMapper(self.__dataset_dir)
        fam_name_to_pdb_id = fam_mapper.map_fam_name_to_pdb_id()
        pdb_files = dict()
        pdbs_dir = self.__dataset_dir / 'PDBFiles'
        for fam_name in fam_name_to_pdb_id:
            pdb_id = fam_name_to_pdb_id[fam_name]
            pdb_file = pdbs_dir / f'{pdb_id}.pdb'
            pdb_files[fam_name] = pdb_file 
        return pdb_files
    

    def get_fam_mapped_pdb_data(self, pdb_chain_id, pdb_file, refseq_file, linear_dist=4, contact_dist=10.0):
        """
        """
        dcavis_inst = DCAVisualizer(
            'rna', pdb_chain_id, pdb_file,
            refseq_file=refseq_file,
            linear_dist=linear_dist,
            contact_dist=contact_dist,
            rna_secstruct_file=None
        )
        
        mapped_site_pairs, missing_sites = dcavis_inst.get_mapped_pdb_contacts()
        return mapped_site_pairs
    

    def get_all_mapped_pdb_data(self):
        """
        """
        refseqs_len = RNADataset(self.__dataset_dir).get_refseqs_len()
        logger.info('\n\tObtaining mapped PDB data')
        dataset_dir_basename = self.__dataset_dir.stem
        if dataset_dir_basename == 'RNA_TESTSET':
            pdb_pickle_file_name =  'MappedPDBDataTestset.pickle'
        elif dataset_dir_basename == 'RNA_DATASET':
            pdb_pickle_file_name = 'MappedPDBDataDataset.pickle'
        else:
            logger.error(f'\n\tInvalid dataset base directory name {dataset_dir_basename}')
            raise ValueError
        pdb_data_pickled = pathlib.Path(pdb_pickle_file_name)

        if pdb_data_pickled.exists():
            with open(pdb_data_pickled, 'rb') as fh:
                all_mapped_pdb_data = pickle.load(fh)
            return all_mapped_pdb_data
        else:
            family_mapper = FamilyMapper(self.__dataset_dir)
            chain_mapping = family_mapper.map_fam_name_to_pdb_chain()
            refseq_files = family_mapper.map_fam_name_to_refseq_file_path()
            pdb_files = self.get_pdb_files() 
            all_mapped_pdb_data = dict()
            for fam_name in pdb_files:
                refseq_file = refseq_files[fam_name]
                pdb_file = pdb_files[fam_name]
                chain_id = chain_mapping[fam_name]
                fam_mapped_pdb_data = self.get_fam_mapped_pdb_data(chain_id, pdb_file, refseq_file)
                all_mapped_pdb_data[fam_name] = fam_mapped_pdb_data
        # pickle mapped PDB data
        with open(pdb_data_pickled, 'wb') as fh:
            pickle.dump(all_mapped_pdb_data, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return all_mapped_pdb_data
#End of class PDBData

class RNASecondaryStructException(Exception):
    """Exceptions related to scondary structure data
    """
#End of calss RNASecondaryStructException

class RNASecStruct:
    """
    """
    def __init__(self, ss_data_file):
       
        self.__data_file = ss_data_file 
        self.__pdb_id, self.__sequence, self.__raw_secstruct = self._get_data_from_file()
        self.__secstruct = self._get_secstruct_without_pseudoknots()
        self._validate_secstruct()
        return None 
    

    @property
    def pdb_id(self):
        
        return self.__pdb_id
    

    @property
    def secstruct(self):
       
        return self.__secstruct
    

    @property
    def sequence(self):
        
        return self.__sequence
    

    def _get_data_from_file(self):
        
        sequence = list()
        secstruct = list()
        with open(self.__data_file) as fh:
            for line in fh:
                line = line.strip()
                if not line: continue
                if line.startswith('SECSTRUCT'):
                    secstruct.append(line.split(':')[1].strip())
                if line.startswith('SEQUENCE'):
                    sequence.append(line.split(':')[1].strip())
                if line.startswith('PDBID'):
                    pdb_id = line.split(':')[1].strip()
        secstruct_str = ''
        for secstruct_segment in secstruct: secstruct_str = secstruct_str + secstruct_segment
        seq_str = ''
        for seq_segment in sequence: seq_str = seq_str + seq_segment
        assert len(seq_str) == len(secstruct_str)
        seq_str = seq_str.upper()
        return pdb_id, seq_str, secstruct_str
    

    def _get_secstruct_without_pseudoknots(self):
       
        secstruct_brackets = ('(', ')')
        non_nested_secstruct = [b if b in secstruct_brackets else '.' for b in self.__raw_secstruct]
        non_nested_secstruct = ''.join(non_nested_secstruct)
        return non_nested_secstruct

    
    def _validate_secstruct(self):
        
        assert len(self.__sequence) == len(self.__secstruct)
        left_brackets = list()
        for b in self.__secstruct:
            if b == '(': left_brackets.append(b)
            if b == ')': del left_brackets[-1]
        if left_brackets:
            raise RNASecondaryStructException('Invalid secondary structure')
        return None 


    def get_non_nested_secstruct_pairs(self):
        
        secstruct_pairs_dict = dict()
        left_brackets = list()
        for k, b in enumerate(self.__secstruct):
            if b == '(' : 
                left_brackets.append(k) # record position of left bracket
            if b == ')' :
                left_bra_pos = left_brackets[-1]
                site_pair = left_bra_pos, k
                assert k > left_bra_pos
                nucl_pair = '{}-{}'.format(self.__sequence[left_bra_pos], self.__sequence[k])
                secstruct_pairs_dict[site_pair] = nucl_pair
                # remove the last matched left bracket position
                del left_brackets[-1] 
        assert len(left_brackets)== 0
        return secstruct_pairs_dict
    

    def _get_neighbors_for_secstruct_pair(self, pair, radius):
        
        if radius < 0 : 
            logger.error('Invalid value for secondary structure pair neighbor radius : {}'.format(radius))
            raise RNASecondaryStructException
        k, l = pair 
        neighbor_pairs_list = list()
        secstruct_len = len(self.__secstruct)
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                m, n = k + i, l + j
                if m==n:continue # cannot self pair 
                # do not include neighbors that do not exist ( < 0 or >= sequence/secstruct lenght))
                if (n >= 0 and n < secstruct_len) and (m >= 0 and m < secstruct_len): 
                    mn_pair = (m, n) if m < n else (n, m)
                    neighbor_pairs_list.append(mn_pair)
        return tuple(neighbor_pairs_list)

    
    def get_list_of_secstruct_pairs_and_neighbors(self, radius):
        
        secstruct_site_pairs = tuple(self.get_non_nested_secstruct_pairs().keys())
        all_secstruct_pair_and_neighbors = list()
        for pair in secstruct_site_pairs:
            curr_pair_and_neighbors = self._get_neighbors_for_secstruct_pair(pair, radius)
            if radius == 0: assert pair == curr_pair_and_neighbors[0]
            all_secstruct_pair_and_neighbors.extend(curr_pair_and_neighbors)
        all_secstruct_pair_and_neighbors = tuple(set(all_secstruct_pair_and_neighbors))

        return all_secstruct_pair_and_neighbors

#End of class RNASecStruct
class DataPathsTestset:
    """
    """
    path_raw_data_testset = pathlib.Path('RAW_COEV_DATA_ALL/RAW_DATA_TESTSET')
    raw_data_dir_EVC = path_raw_data_testset / 'EVC_RAW_DATA_TESTSET'
    raw_data_dir_PSICOV =  path_raw_data_testset / 'PSICOV_RAW_DATA_TESTSET'
    raw_data_dir_BL = path_raw_data_testset / 'BL_RAW_DATA_TESTSET'
    raw_data_dir_mfDCA = path_raw_data_testset / 'mfDCA_RAW_DATA_TESTSET'
    raw_data_dir_plmDCA = path_raw_data_testset / 'plmDCA_RAW_DATA_TESTSET'
    raw_data_dir_CoCoNet = path_raw_data_testset / 'CoCoNet_RAW_DATA_TESTSET'
    raw_data_dir_CoCoNet_single_matrix = raw_data_dir_CoCoNet / 'SingleFilterMatrix'
    raw_data_dir_CoCoNet_two_matrices = raw_data_dir_CoCoNet / 'TwoFilterMatrices'

    dataset_dir = pathlib.Path('RNA_TESTSET')
    pdb_files_dir = dataset_dir / 'PDBFiles'
    sequence_files_dir = dataset_dir / 'sequences'
    secstruct_files_dir = dataset_dir / 'secstruct'

    num_rna_fams = 23


class DataPathsCrossVal:
    """
    """
    """Relative paths of raw coevolutionary score data
    """
    path_raw_data_dataset = pathlib.Path('RAW_COEV_DATA_ALL/RAW_DATA_DATASET')
    raw_data_dir_EVC = path_raw_data_dataset / 'EVC_RAW_DATA_DATASET'
    raw_data_dir_PSICOV = path_raw_data_dataset / 'PSICOV_RAW_DATA_DATASET' 
    raw_data_dir_BL = path_raw_data_dataset / 'BL_RAW_DATA_DATASET'
    raw_data_dir_mfDCA = path_raw_data_dataset / 'mfDCA_RAW_DATA_DATASET'
    raw_data_dir_plmDCA = path_raw_data_dataset / 'plmDCA_RAW_DATA_DATASET'
    raw_data_dir_CoCoNet = path_raw_data_dataset / 'CoCoNet_RAW_DATA_FiveFold_CrossVal' 
    raw_data_dir_CoCoNet_single_matrix = raw_data_dir_CoCoNet / 'SingleFilterMatrix'
    raw_data_dir_CoCoNet_two_matrices = raw_data_dir_CoCoNet / 'TwoFilterMatrices'

    dataset_dir = pathlib.Path('RNA_DATASET')  
    pdb_files_dir = dataset_dir / 'PDBFiles'
    sequence_files_dir = dataset_dir / 'sequences'
    secstruct_files_dir = dataset_dir / 'secstruct'

    num_rna_fams = 57


class CoevData:
    

    def __init__(self, datapath_inst):
        """
        """
        self.__dpath_inst = datapath_inst
    
        return None 
    
    @property
    def datapath(self):
        """
        """
        return self.__dpath_inst

    
    def get_list_of_rawdata_from_single_file(self, file_path):
        """
        """
        data_list = list()
        with open(file_path, 'r') as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('#'):continue
                line = line.split()
                if len(line) != 3:
                    logger.error('\n\tThe file contains data that is not in proper format') 
                    raise ValueError
                i, j, score = int(line[0]), int(line[1]), float(line[2])
                # shift the indexes by one to start indexing from zero 
                i -= 1
                j -= 1
                assert j >= 1
                assert i >= 0
                site_pair_score = ((i, j), score) 
                data_list.append(site_pair_score)
        return data_list


    def get_data_mfDCA(self):
        """
        """
        refseqs_len = RNADataset(self.__dpath_inst.dataset_dir).get_refseqs_len()
        logger.info(f'\n\tObtaining mean-field DCA raw data from directory {self.__dpath_inst.raw_data_dir_mfDCA}')
        mfdca_raw_data_files = list(self.__dpath_inst.raw_data_dir_mfDCA.glob('RF*.txt'))
        mfdca_raw_data_dict = dict()
        for raw_data_file in mfdca_raw_data_files:
            fam_name = raw_data_file.stem[:-6]
            mfdca_raw_data_dict[fam_name] = dict(self.get_list_of_rawdata_from_single_file(raw_data_file))
            L = refseqs_len[fam_name]
            assert len(mfdca_raw_data_dict[fam_name]) == L * (L - 1)//2
        assert len(mfdca_raw_data_dict) == self.__dpath_inst.num_rna_fams
        return mfdca_raw_data_dict
    
    
    def get_data_plmDCA(self):
        """
        """
        refseqs_len = RNADataset(self.__dpath_inst.dataset_dir).get_refseqs_len()
        logger.info(f'\n\tObtaining mean-field DCA raw data from directory {self.__dpath_inst.raw_data_dir_plmDCA}')
        plmdca_raw_data_files = list(self.__dpath_inst.raw_data_dir_plmDCA.glob('RF*.txt'))
        plmdca_raw_data_dict = dict()
        for raw_data_file in plmdca_raw_data_files:
            fam_name = raw_data_file.stem[:-7]
            plmdca_raw_data_dict[fam_name] = dict(self.get_list_of_rawdata_from_single_file(raw_data_file))
            L = refseqs_len[fam_name]
            assert len(plmdca_raw_data_dict[fam_name]) == L * (L - 1)//2
        assert len(plmdca_raw_data_dict) == self.__dpath_inst.num_rna_fams
        return plmdca_raw_data_dict        


    def get_data_EVC(self):
        """
        """
        refseqs_len = RNADataset(self.__dpath_inst.dataset_dir).get_refseqs_len()
        logger.info(f'\n\tObtaining EVCouplings computed raw data from directory {self.__dpath_inst.raw_data_dir_EVC}')
        evc_raw_data_files = list(self.__dpath_inst.raw_data_dir_EVC.glob('RF*.txt'))
        evc_raw_data_dict = dict()
        for raw_data_file in evc_raw_data_files:
            fam_name = raw_data_file.stem[:-4]
            evc_raw_data_dict[fam_name] = dict(self.get_list_of_rawdata_from_single_file(raw_data_file))
            L = refseqs_len[fam_name]
            assert len(evc_raw_data_dict[fam_name]) == L * (L - 1)//2
        assert len(evc_raw_data_dict) == self.__dpath_inst.num_rna_fams
        return evc_raw_data_dict

    
    def get_data_PSICOV(self):
        """
        """
        refseqs_len = RNADataset(self.__dpath_inst.dataset_dir).get_refseqs_len()
        fam_names_list = list(refseqs_len.keys())
        logger.info(f'\n\tObtaining PSICOV computed raw data from directory {self.__dpath_inst.raw_data_dir_PSICOV}')
        psicov_raw_data_files = list(self.__dpath_inst.raw_data_dir_PSICOV.glob('RF*.txt'))
        psicov_raw_data_dict = dict()
        for raw_data_file in psicov_raw_data_files:
            fam_name = raw_data_file.stem[:-7]
            psicov_raw_data_dict[fam_name] = dict(self.get_list_of_rawdata_from_single_file(raw_data_file))
            L = refseqs_len[fam_name] 
            assert len(psicov_raw_data_dict[fam_name]) == L * (L - 1)//2
        assert len(psicov_raw_data_dict) == self.__dpath_inst.num_rna_fams
        return psicov_raw_data_dict

    
    def get_data_BL(self):
        """
        """
        refseqs_len = RNADataset(self.__dpath_inst.dataset_dir).get_refseqs_len()
        logger.info(f'\n\tObtaining Boltzmann learning raw data from directory {self.__dpath_inst.raw_data_dir_BL}')
        bl_raw_data_files = list(self.__dpath_inst.raw_data_dir_BL.glob('RF*.txt'))
        bl_raw_data_dict = dict()
        for raw_data_file in bl_raw_data_files:
            fam_name = raw_data_file.stem[:-3]
            bl_raw_data_dict[fam_name] = dict(self.get_list_of_rawdata_from_single_file(raw_data_file))
            L = refseqs_len[fam_name]
            assert len(bl_raw_data_dict[fam_name]) == L * (L - 1)//2
        assert len(bl_raw_data_dict) == self.__dpath_inst.num_rna_fams
        return bl_raw_data_dict


    def _get_coconet_rawdata_files_single_matrix_crossval(self, filter_mat=None):
        """
        """
        if filter_mat not in ['3x3', '5x5', '7x7']:
            logger.error(f'\n\tUnknown filter matrix type {filter_mat}')
            raise ValueError
        trial_dirs = self.__dpath_inst.raw_data_dir_CoCoNet_single_matrix.glob('trial_*')
        
        raw_data_files_dict = dict()
        for trial_dir in trial_dirs:
            trial_dir_basename = trial_dir.stem
            raw_data_files_dict[trial_dir_basename] = list()
            fold_dirs = trial_dir.glob('fold_*')
            for fold_dir in fold_dirs:
                raw_data_files = fold_dir.glob(f'RF*_{filter_mat}.txt')
                for raw_data_file in raw_data_files:
                    raw_data_files_dict[trial_dir_basename].append(raw_data_file)
            assert len(raw_data_files_dict[trial_dir_basename]) == self.__dpath_inst.num_rna_fams
        return raw_data_files_dict

    
    def _get_coconet_rawdata_files_two_matrices_cross_val(self, filter_mat=None):
        """
        """
        if filter_mat not in ['2x3x3', '2x5x5', '2x7x7']:
            logger.error(f'\n\tUnknown filter matrix type {filter_mat}')
            raise ValueError
        trial_dirs = self.__dpath_inst.raw_data_dir_CoCoNet_two_matrices.glob('trial_*')
        raw_data_files_dict = dict()
        for trial_dir in trial_dirs:
            trial_dir_basename = trial_dir.stem
            raw_data_files_dict[trial_dir_basename] = list()
            fold_dirs = trial_dir.glob('fold_*')
            for fold_dir in fold_dirs:
                raw_data_files = fold_dir.glob(f'RF*_{filter_mat}.txt')
                for raw_data_file in raw_data_files:
                    raw_data_files_dict[trial_dir_basename].append(raw_data_file)
            assert len(raw_data_files_dict[trial_dir_basename]) == self.__dpath_inst.num_rna_fams
        return raw_data_files_dict


    def _get_average_fam_coconet_data_over_trials(self, fams_raw_data):
        """fams_raw_data[trail][fam_name] = dca_data_dict
        """
        refseqs_len = RNADataset(self.__dpath_inst.dataset_dir).get_refseqs_len()
        fams_raw_data_average = dict()
        rna_fams_list = RNADataset(self.__dpath_inst.dataset_dir).get_list_of_all_fams()
        for fam_name in rna_fams_list:
            L = refseqs_len[fam_name]
            fams_raw_data_average[fam_name] = fams_raw_data['trial_1'][fam_name]
            for trial in fams_raw_data:
                if trial == 'trial_1': continue
                fam_trial_data_dict = fams_raw_data[trial][fam_name]
                site_pair_counter = 0
                for i in range(L - 1):
                    for j in range(i + 1, L):
                        site_pair = (i, j)
                        site_pair_counter += 1
                        fams_raw_data_average[fam_name][site_pair] += fam_trial_data_dict[site_pair]
            assert site_pair_counter ==  len(fams_raw_data_average[fam_name])
        # divide the scores by number of trials
        num_trials = len(fams_raw_data)
        for fam_name in fams_raw_data_average:
            for site_pair in fams_raw_data_average[fam_name]:
                fams_raw_data_average[fam_name][site_pair] /= float(num_trials)
        return fams_raw_data_average


    def get_average_coconet_data_single_matrix_cross_val(self, filter_mat=None):
        """
        """
        if filter_mat not in ['3x3', '5x5', '7x7']:
            logger.error(f'\n\tInvalid filter matrix name {filter_mat}')
            raise ValueError
        fams_raw_data = dict()
        logger.info(f'\n\tObtaining average coevolution data for filter matrix {filter_mat}')
        raw_data_files_dict = self._get_coconet_rawdata_files_single_matrix_crossval(filter_mat=filter_mat)
        for trial in raw_data_files_dict:
            list_of_raw_data_files = raw_data_files_dict[trial]
            fams_raw_data[trial] = dict()
            for raw_data_file in list_of_raw_data_files:
                fam_name = raw_data_file.stem[:-4]
                fams_raw_data[trial][fam_name] = dict(self.get_list_of_rawdata_from_single_file(raw_data_file))
            assert len(fams_raw_data[trial]) == self.__dpath_inst.num_rna_fams
        #get averaged scores 
        average_fams_data= self._get_average_fam_coconet_data_over_trials(fams_raw_data)
        return average_fams_data

    
    def get_average_coconet_data_two_matrices_cross_val(self, filter_mat=None):
        """
        """
        if filter_mat not in ['2x3x3', '2x5x5', '2x7x7']:
            logger.error(f'\n\t Invalid filter matrix name {filter_mat}')
            raise ValueError
        fams_raw_data = dict()
        logger.info(f'\n\tObtaining average coevolution data for filter matrix {filter_mat}')
        raw_data_files_dict = self._get_coconet_rawdata_files_two_matrices_cross_val(filter_mat=filter_mat)
        for trial in raw_data_files_dict:
            list_of_raw_data_files = raw_data_files_dict[trial]
            fams_raw_data[trial] = dict()
            for raw_data_file in list_of_raw_data_files:
                fam_name = raw_data_file.stem[:-6]
                fams_raw_data[trial][fam_name] = dict(self.get_list_of_rawdata_from_single_file(raw_data_file))
            assert len(fams_raw_data[trial]) == self.__dpath_inst.num_rna_fams
        #get average scores
        average_fams_data = self._get_average_fam_coconet_data_over_trials(fams_raw_data)
        return average_fams_data
    
    def get_coconet_data_single_matrix_testset(self, filter_mat=None):
        """
        """
        files_pattern = f'RF*_{filter_mat}.txt'
        raw_data_files = list(self.__dpath_inst.raw_data_dir_CoCoNet_single_matrix.glob(files_pattern))
        if not raw_data_files: raise FileNotFoundError
        raw_data_dict = dict()
        for data_file in raw_data_files:
            fam_name = data_file.stem[:-len(filter_mat) - 1]
            raw_data_dict[fam_name] = dict(self.get_list_of_rawdata_from_single_file(data_file))
        return raw_data_dict
    
    def get_coconet_data_two_matrices_testset(self, filter_mat=None):
        """
        """
        files_pattern = f'RF*{filter_mat}.txt'
        raw_data_files = list(self.__dpath_inst.raw_data_dir_CoCoNet_two_matrices.glob(files_pattern))
        if not raw_data_files: raise FileNotFoundError
        raw_data_dict = dict()
        for data_file in raw_data_files:
            fam_name = data_file.stem[:-len(filter_mat) -1]
            raw_data_dict[fam_name] = dict(self.get_list_of_rawdata_from_single_file(data_file))
        return raw_data_dict

#End of class RawCoevData 


class PosPredVal:

    def __init__(self, coev_data_inst, contact_dist=10.0, linear_dist=4):
        """
        """
        self.__coev_data_inst = coev_data_inst
        self.__contact_dist = contact_dist
        self.__linear_dist = linear_dist
        return None 


    def get_mapped_pdb_data(self):
        """
        """
        mapped_pdb_data = PDBData(self.__coev_data_inst.datapath.dataset_dir).get_all_mapped_pdb_data()
        return mapped_pdb_data

    def get_dataset_one_algo_coev_data(self, algo_name=None):
        """
        """
        valid_algo_names = ('mfdca', 'plmdca', 'bl','evc', 'psicov', '3x3', '5x5', '7x7', '2x3x3', '2x5x5', '2x7x7')
        if algo_name not in valid_algo_names:
            logger.error(f'\n\tInvalid choice {algo_name}')
            raise ValueError
        if algo_name == 'evc': dca_data = self.__coev_data_inst.get_data_EVC()
        if algo_name == 'mfdca': dca_data = self.__coev_data_inst.get_data_mfDCA()
        if algo_name == 'plmdca': dca_data = self.__coev_data_inst.get_data_plmDCA()
        if algo_name == 'bl': dca_data = self.__coev_data_inst.get_data_BL()
        if algo_name == 'psicov': dca_data = self.__coev_data_inst.get_data_PSICOV()

        dataset_dir_name = self.__coev_data_inst.datapath.dataset_dir.stem
        if algo_name == '3x3': 
            if dataset_dir_name == 'RNA_DATASET': 
                dca_data = self.__coev_data_inst.get_average_coconet_data_single_matrix_cross_val(filter_mat=algo_name)
            else:
                dca_data = self.__coev_data_inst.get_coconet_data_single_matrix_testset(filter_mat=algo_name)

        if algo_name == '5x5':
            if dataset_dir_name == 'RNA_DATASET': 
                dca_data = self.__coev_data_inst.get_average_coconet_data_single_matrix_cross_val(filter_mat=algo_name)
            else:
                dca_data = self.__coev_data_inst.get_coconet_data_single_matrix_testset(filter_mat=algo_name)

        if algo_name == '7x7': 
            if dataset_dir_name == 'RNA_DATASET': 
                dca_data = self.__coev_data_inst.get_average_coconet_data_single_matrix_cross_val(filter_mat=algo_name)
            else:
                dca_data = self.__coev_data_inst.get_coconet_data_single_matrix_testset(filter_mat=algo_name)

        if algo_name == '2x3x3': 
            if dataset_dir_name == 'RNA_DATASET': 
                dca_data = self.__coev_data_inst.get_average_coconet_data_two_matrices_cross_val(filter_mat=algo_name)
            else:
                dca_data = self.__coev_data_inst.get_coconet_data_two_matrices_testset(filter_mat=algo_name)

        if algo_name == '2x5x5': 
            if dataset_dir_name == 'RNA_DATASET': 
                dca_data = self.__coev_data_inst.get_average_coconet_data_two_matrices_cross_val(filter_mat=algo_name)
            else:
                dca_data = self.__coev_data_inst.get_coconet_data_two_matrices_testset(filter_mat=algo_name)

        if algo_name == '2x7x7': 
            if dataset_dir_name == 'RNA_DATASET': 
                dca_data = self.__coev_data_inst.get_average_coconet_data_two_matrices_cross_val(filter_mat=algo_name)
            else:
                dca_data = self.__coev_data_inst.get_coconet_data_two_matrices_testset(filter_mat=algo_name)

        return dca_data

    
    def compute_ppv_all_contact_types(self, dca_data, mapped_pdb_data, list_of_RNAs=None):
        """
        """
        if list_of_RNAs is None:
            list_of_RNAs = RNADataset(self.__coev_data_inst.datapath.dataset_dir).get_list_of_all_fams()
        fams_ppv = dict()
        for fam_name in dca_data:
            if fam_name in list_of_RNAs:
                fam_dca_data = dca_data[fam_name]
                fam_dca_data_sorted = sorted(fam_dca_data.items(), key=lambda k : k[1], reverse=True)
                fam_dca_data_sorted_lin_dist = [
                    site_pair for site_pair, _score in fam_dca_data_sorted if site_pair[1] - site_pair[0] > self.__linear_dist
                ] 
                fam_pdb_data = mapped_pdb_data[fam_name]
                pdb_data_lin_dist = [
                    (site_pair, _data[-1]) for site_pair, _data in fam_pdb_data.items() if site_pair[1] - site_pair[0] > self.__linear_dist
                    
                ]
                pdb_contacts = [
                    site_pair for site_pair, _distance in pdb_data_lin_dist if _distance < self.__contact_dist 
                ]

                # compute ppv by iterationg over sorted dca putative contacts filtered by linear distance
                num_true_contacts = 0
                fams_ppv[fam_name] = list()
                for rank, site_pair in enumerate(fam_dca_data_sorted_lin_dist, start=1):
                    if site_pair in pdb_contacts: num_true_contacts += 1
                    ppv_at_rank = float(num_true_contacts)/float(rank)
                    fams_ppv[fam_name].append(ppv_at_rank)
        return fams_ppv

    
    def compute_ppv_tertiary_contacts(self, dca_data, mapped_pdb_data, radius, list_of_RNAs=None):
        """
        """
        fams_ppv_ter = dict()
        fam_name_to_ss_file = FamilyMapper(self.__coev_data_inst.datapath.dataset_dir).map_fam_name_to_secstruct_file_path()
        for fam_name in dca_data:
            if fam_name in list_of_RNAs:
                ss_file = fam_name_to_ss_file[fam_name]
                fam_dca_data = dca_data[fam_name]
                fam_dca_data_sorted = sorted(fam_dca_data.items(), key=lambda k : k[1], reverse=True)
                fam_dca_data_sorted_lin_dist = [
                    site_pair for site_pair, _score in fam_dca_data_sorted if site_pair[1] - site_pair[0] > self.__linear_dist
                ] 
                
                secstruct_pairs_and_neighbors = RNASecStruct(ss_file).get_list_of_secstruct_pairs_and_neighbors(radius=radius)
                fam_dca_data_sorted_lin_dist_ter = [
                    site_pair for site_pair in fam_dca_data_sorted_lin_dist if site_pair not in secstruct_pairs_and_neighbors
                
                ]
                fam_pdb_data = mapped_pdb_data[fam_name]
                pdb_data_lin_dist = [
                    (site_pair, _data[-1]) for site_pair, _data in fam_pdb_data.items() if site_pair[1] - site_pair[0] > self.__linear_dist
                    
                ]
                pdb_contacts = [
                    site_pair for site_pair, _distance in pdb_data_lin_dist if _distance < self.__contact_dist 
                ]

                pdb_contacts_ter = [
                    site_pair for site_pair in pdb_contacts if site_pair not in secstruct_pairs_and_neighbors
                ]

                # compute ppv by iterationg over sorted dca putative contacts filtered by linear distance
                num_true_contacts = 0
                fams_ppv_ter[fam_name] = list()
                for rank, site_pair in enumerate(fam_dca_data_sorted_lin_dist_ter, start=1):
                    if site_pair in pdb_contacts_ter: num_true_contacts += 1
                    ppv_at_rank = float(num_true_contacts)/float(rank)
                    fams_ppv_ter[fam_name].append(ppv_at_rank)
        return fams_ppv_ter
 
    
    def get_ppv_all_contact_types_one_algo_full_rank(self, algo_name=None):
        """
        """
        dca_data = self.get_dataset_one_algo_coev_data(algo_name=algo_name)
        pdb_data = self.get_mapped_pdb_data()
        fams_ppv = self.compute_ppv_all_contact_types(dca_data, pdb_data)
        return fams_ppv
    
    def get_ppv_tertiary_contacts_one_algo_full_rank(self, radius, algo_name=None):
        """
        """
        dca_data = self.get_dataset_one_algo_coev_data(algo_name=algo_name)
        pdb_data = self.get_mapped_pdb_data()
        ter_fams_ppv = self.compute_ppv_tertiary_contacts(dca_data, pdb_data, radius=radius)
        return ter_fams_ppv

    
    def get_average_ppv_all_contact_types_one_algo_at_rank_L_all_RNAs(self, algo_name=None):
        """
        """
        dca_data = self.get_dataset_one_algo_coev_data(algo_name=algo_name)
        pdb_data = self.get_mapped_pdb_data()
        fams_ppv = self.compute_ppv_all_contact_types(dca_data, pdb_data)
        refeqs_len = RNADataset(self.__coev_data_inst.datapath.dataset_dir).get_refseqs_len()
        fams_ppv_at_L_list = list()
        for fam_name in fams_ppv:
            L = refeqs_len[fam_name]
            curr_fam_ppv = fams_ppv[fam_name][L]
            fams_ppv_at_L_list.append(curr_fam_ppv)
        average_ppv = np.mean(fams_ppv_at_L_list)
        #print(f'Algo = {algo_name},  Average PPV = {average_ppv}')
        return round(average_ppv*100, 1)

    
    def get_average_ppv_all_contact_types_one_algo_at_rank_L_HighMeff_RNAs(self, algo_name=None):
        """
        """
        dca_data = self.get_dataset_one_algo_coev_data(algo_name=algo_name)
        pdb_data = self.get_mapped_pdb_data()
        list_of_HMeff_RNAs = RNADataset(self.__coev_data_inst.datapath.dataset_dir).get_list_of_fams_with_HighMeff()
        fams_ppv = self.compute_ppv_all_contact_types(dca_data, pdb_data, list_of_RNAs=list_of_HMeff_RNAs)
        refeqs_len = RNADataset(self.__coev_data_inst.datapath.dataset_dir).get_refseqs_len()
        fams_ppv_at_L_list = list()
        for fam_name in fams_ppv:
            L = refeqs_len[fam_name]
            curr_fam_ppv = fams_ppv[fam_name][L]
            fams_ppv_at_L_list.append(curr_fam_ppv)
        average_ppv = np.mean(fams_ppv_at_L_list)
        #print(f'Algo = {algo_name},  HighMeff RNAs Average PPV = {average_ppv}')
        return round(average_ppv*100, 1)

    
    def get_average_ppv_all_contact_types_one_algo_at_rank_L_LowMeff_RNAs(self, algo_name=None):
        """
        """
        dca_data = self.get_dataset_one_algo_coev_data(algo_name=algo_name)
        pdb_data = self.get_mapped_pdb_data()
        list_of_LowMeff = RNADataset(self.__coev_data_inst.datapath.dataset_dir).get_list_of_fams_with_LowMeff()
        fams_ppv = self.compute_ppv_all_contact_types(dca_data, pdb_data, list_of_RNAs=list_of_LowMeff)
        refeqs_len = RNADataset(self.__coev_data_inst.datapath.dataset_dir).get_refseqs_len()
        fams_ppv_at_L_list = list()
        for fam_name in fams_ppv:
            L = refeqs_len[fam_name]
            curr_fam_ppv = fams_ppv[fam_name][L]
            fams_ppv_at_L_list.append(curr_fam_ppv)
        average_ppv = np.mean(fams_ppv_at_L_list)
        #print(f'Algo = {algo_name},  LowMeff RNAs Average PPV = {average_ppv}')
        return round(average_ppv*100, 1)


    def get_average_ppv_one_algo_at_rank_L_all_RNAs_TERTIARY(self, algo_name=None, radius=2):
        """
        """
        dca_data = self.get_dataset_one_algo_coev_data(algo_name=algo_name)
        pdb_data = self.get_mapped_pdb_data()
        all_RNAs_list = RNADataset(self.__coev_data_inst.datapath.dataset_dir).get_list_of_all_fams()
        fams_ppv_ter = self.compute_ppv_tertiary_contacts(dca_data, pdb_data, radius=radius, list_of_RNAs=all_RNAs_list)
        refseqs_len = RNADataset(self.__coev_data_inst.datapath.dataset_dir).get_refseqs_len()
        fams_ppv_at_L_list = list()
        for fam_name in fams_ppv_ter:
            L = refseqs_len[fam_name]
            curr_fam_ppv = fams_ppv_ter[fam_name][L]
            fams_ppv_at_L_list.append(curr_fam_ppv)
        avearge_ppv_ter = np.mean(fams_ppv_at_L_list)
        #print(f'Algo = {algo_name}, Average tertiary PPV = {avearge_ppv_ter}')
        return round(avearge_ppv_ter*100, 1)
    
    def get_average_ppv_one_algo_at_rank_L_HighMeff_TERTIARY(self, algo_name=None, radius=2):
        """
        """
        dca_data = self.get_dataset_one_algo_coev_data(algo_name=algo_name)
        pdb_data = self.get_mapped_pdb_data()
        HighMeff_RNAs_list = RNADataset(self.__coev_data_inst.datapath.dataset_dir).get_list_of_fams_with_HighMeff()
        fams_ppv_ter = self.compute_ppv_tertiary_contacts(dca_data, pdb_data, radius=radius, list_of_RNAs=HighMeff_RNAs_list)
        refseqs_len = RNADataset(self.__coev_data_inst.datapath.dataset_dir).get_refseqs_len()
        fams_ppv_at_L_list = list()
        for fam_name in fams_ppv_ter:
            L = refseqs_len[fam_name]
            curr_fam_ppv = fams_ppv_ter[fam_name][L]
            fams_ppv_at_L_list.append(curr_fam_ppv)
        avearge_ppv_ter = np.mean(fams_ppv_at_L_list)
        #print(f'Algo = {algo_name}, Average tertiary PPV = {avearge_ppv_ter}')
        return round(avearge_ppv_ter*100, 1)
    

    def get_average_ppv_one_algo_at_rank_L_LowMeff_TERTIARY(self, algo_name=None, radius=2):
        """
        """
        dca_data = self.get_dataset_one_algo_coev_data(algo_name=algo_name)
        pdb_data = self.get_mapped_pdb_data()
        LowMeff_RNAs_list = RNADataset(self.__coev_data_inst.datapath.dataset_dir).get_list_of_fams_with_LowMeff()
        fams_ppv_ter = self.compute_ppv_tertiary_contacts(dca_data, pdb_data, radius=radius, list_of_RNAs=LowMeff_RNAs_list)
        refseqs_len = RNADataset(self.__coev_data_inst.datapath.dataset_dir).get_refseqs_len()
        fams_ppv_at_L_list = list()
        for fam_name in fams_ppv_ter:
            L = refseqs_len[fam_name]
            curr_fam_ppv = fams_ppv_ter[fam_name][L]
            fams_ppv_at_L_list.append(curr_fam_ppv)
        avearge_ppv_ter = np.mean(fams_ppv_at_L_list)
        #print(f'Algo = {algo_name}, Average tertiary PPV = {avearge_ppv_ter}')
        return round(avearge_ppv_ter*100, 1)


#End of class PosPredVal


DCA_ALGOS_LIST = ('mfdca', 'plmdca', 'psicov', 'evc', 'bl')
COCONET_ALGOS_LIST = ('3x3', '5x5', '7x7', '2x3x3', '2x5x5', '2x7x7')
ALL_ALGOS_LIST = DCA_ALGOS_LIST + COCONET_ALGOS_LIST


def get_ppv_instance(dataset_type='crossval'):
    """
    """
    if dataset_type=='crossval':
        ppv_inst = PosPredVal(CoevData(DataPathsCrossVal))
    elif dataset_type == 'testset':
        ppv_inst = PosPredVal(CoevData(DataPathsTestset))
    else:
        logger.error(f"\n\tUnknown dataset_type {dataset_type}. Must be 'crossval' or 'testset>'")
        raise ValueError
    return ppv_inst


def get_all_algos_average_ppv_Meff_category_all_contact_types(dataset_type='crossval', Meff_category='all'):
    """
    """
    logger.info(f'\n\tComputing all contacts average PPV at rank L for dataset {dataset_type} and {Meff_category} Meff RNAs')
    ppv_inst = get_ppv_instance(dataset_type=dataset_type)
    average_ppv_dict = dict()
    
    for algo_name in ALL_ALGOS_LIST:
        if Meff_category == 'all':
            algo_ppv = ppv_inst.get_average_ppv_all_contact_types_one_algo_at_rank_L_all_RNAs(algo_name=algo_name)
        elif Meff_category == 'high':
            algo_ppv  = ppv_inst.get_average_ppv_all_contact_types_one_algo_at_rank_L_HighMeff_RNAs(algo_name=algo_name)
        elif Meff_category == 'low':
            algo_ppv = ppv_inst.get_average_ppv_all_contact_types_one_algo_at_rank_L_LowMeff_RNAs(algo_name=algo_name)
        else:
            logger.error(f'\n\t Meff_category should be one of [All, HighMeff, LowMeff]. {Meff_category} given')
            raise ValueError
        average_ppv_dict[algo_name] = algo_ppv
    
    return average_ppv_dict


def get_all_algos_average_ppv_Meff_category_TERTIARY(dataset_type = 'crossval', Meff_category='all'):
    """
    """
    logger.info(f'\n\tComputing tertiary contacts average PPV at rank L for dataset {dataset_type} for {Meff_category} Meff RNAs')
    ppv_inst = get_ppv_instance(dataset_type=dataset_type)
    average_ppv_dict = dict()
    
    for algo_name in ALL_ALGOS_LIST:
        if Meff_category == 'all':
            algo_ppv = ppv_inst.get_average_ppv_one_algo_at_rank_L_all_RNAs_TERTIARY(algo_name = algo_name)
        elif Meff_category == 'high':
            algo_ppv  = ppv_inst.get_average_ppv_one_algo_at_rank_L_HighMeff_TERTIARY(algo_name = algo_name)
        elif Meff_category == 'low':
            algo_ppv = ppv_inst.get_average_ppv_one_algo_at_rank_L_LowMeff_TERTIARY(algo_name = algo_name)
        else:
            logger.error(f'\n\t Meff_category should be one of [All, HighMeff, LowMeff]. {Meff_category} given')
            raise ValueError
        average_ppv_dict[algo_name] = algo_ppv
    
    return average_ppv_dict

        
def execute_from_command_line(Meff=None, tertiary=False, verbose=False, testset=False):
    """
    """
    if verbose: configure_logging()
    dataset_type = 'crossval' if not testset else 'testset'
    if tertiary:
        algos_ppv_at_L = get_all_algos_average_ppv_Meff_category_TERTIARY(dataset_type=dataset_type, Meff_category=Meff)
    else:
        algos_ppv_at_L = get_all_algos_average_ppv_Meff_category_all_contact_types(dataset_type=dataset_type, Meff_category=Meff)

    for algo_name, ppv in algos_ppv_at_L.items():
        print(algo_name, f'PPV = {ppv}')

    return None 


def main():
    """
    """
    parser = ArgumentParser(description='Parser for computing poitive predictive values')
    parser.add_argument('--testset', help = 'compute PPV for teset RNAs', action = 'store_true')
    parser.add_argument('--Meff', help = 'compute PPV for a set of RNAs categorized by effective number of sequences', 
        choices = ('all', 'high', 'low'), default='all'
    )
    parser.add_argument('--tertiary', help = 'compute PPV of tertiary contacts', action='store_true')
    parser.add_argument('--verbose', help = 'show logging messages on the terminal', action='store_true')

    args_dict = vars(parser.parse_args())
    execute_from_command_line(
        testset = args_dict.get('testset'),
        Meff = args_dict.get('Meff'),
        tertiary = args_dict.get('tertiary'),
        verbose = args_dict.get('verbose'),
    )
    return None 

if __name__ == '__main__':
    
    main()
