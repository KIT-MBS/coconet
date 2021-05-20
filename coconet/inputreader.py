from pydca.contact_visualizer.contact_visualizer import DCAVisualizer 
import logging
import glob
import os 


"""Defines InputReader class

Author: Mehari B. Zerihun
"""

logger = logging.getLogger(__name__)

class InputReaderException(Exception):
    """Raises exceptions related to reading input data from text files.
    """


class InputReader:
    """Reads data from text files related to CocoNet computation.
    """

    def read_from_one_column_text_file(self, file_name):
        """Reads string data from one column text file.

        Parameters 
        ----------
            self : InputReader 
                An instance of InputReader class.
            file_name : str 
                Path to file 

        Retruns 
        -------
            data_list : tuple 
                A tuple of data read from the text file.
        """
        logger.info('\n\tReading from one column data text file : {}'.format(file_name))
        data_list = list()
        with open(file_name, 'r') as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("#"): continue 
                data_list.append(line)
        return tuple(data_list)               


    def get_mapped_pdb_data(self, pdb_chain_id=None, refseq_file=None, pdb_file=None, linear_dist=4, contact_dist=10.0):
        """Obtains PDB contact data from a PDB file 

        Parameters
        ----------
            self : InputReader
                An instance of InputReader class.
            pdb_chain_id : str 
                ID of a PDB structure chain. 
            refseq_file : str
                Path to file containing reference sequence. 
            pdb_file : str
                Path to PDB structure file. 
            linear_dist : int 
                Minimum distance between sites in reference sequence. 
            contact_dist : float 
                Cut-off value of PDB contacts.

        Returns
        -------
            mapped_site_pairs, missing_sites, refseq_len : tuple
            A tuple of dictionary of mapped site pairs, a list of sites missing in PDB
            structure and the length of the reference sequence.
        """
        #pdb_files_list = get_files_from_dir(self.__dir_pdb_files, 'RF*') 
        #refseq_files_list = get_files_from_dir(self.__dir_refseq_files, 'refseq_RF*')
        #pdb_chain_id_list = self.read_pdb_chain_metadata(pdb_chain_metadata_file)

        if pdb_chain_id is None or refseq_file is None or pdb_file is None:
            logger.error('\n\tYou need to pass a value (not None) to all the keyword arguments')
            raise InputReaderException
        
        dcavis_inst = DCAVisualizer(
            'rna', pdb_chain_id, pdb_file,
            refseq_file=refseq_file,
            linear_dist=linear_dist,
            contact_dist=contact_dist,
            rna_secstruct_file=None
        )
        refseq_len = len(dcavis_inst.get_matching_refseq_to_biomolecule())
        #wc_pairs = dcavis_inst.rna_secstruct_content.wcpairs
        mapped_site_pairs, missing_sites = dcavis_inst.get_mapped_pdb_contacts()
        return mapped_site_pairs, missing_sites, refseq_len


    def get_files_from_dir(self, the_dir, pattern):
        """Obtains list of files from a directory

        Parameters
        ----------
            self : InputReader 
                An instance of InputReader class.
            the_dir : str
                Path to the directory contains PDB files
    
        Returns 
        -------
            files_list : list 
                A list of files that are in directory the_dir.
        """
        if os.path.isdir(the_dir):
            the_dir = os.path.abspath(the_dir)
        else:
            logger.error('\n\t{} is not a directory'.format(the_dir))
            raise ValueError 

        files_list = list()
        for f in glob.glob(os.path.join(the_dir, pattern)):
            if os.path.isfile(f): files_list.append(f)
        logger.info('\n\tTotal number of files found in directory {}: {}'.format(the_dir, len(files_list)))
        return files_list
    
    #TODO  change method name to get_pdb_chain_metadata
    def read_pdb_chain_metadata(self, pdb_chain_metadata_file):
        """Reads PDB chain IDs from text file. The file should contain the RNA 
        family name in the first column and the PDB chain ID(s) in the second
        colum with multiple IDs separated by comma.

        Parameters
        ----------
            self : InputReader  
                An instance of InputReader class
            pdb_chain_metadata_file : str
                Path to text file containing PBD chain metadata

        Returns 
        -------
            pdb_chain_id_dict : dict
                A dictionary whose keys are the basenames of PDB file without .pdb
                extension and whose keys are the PDB chain ID.
        """

        pdb_chain_id_dict = dict()
        with open(pdb_chain_metadata_file, 'r') as fh:
            for line in fh:
                line = line.strip()
                if line.startswith('#'): continue 
                line = line.split()
                if line[1] == '-': continue
                base_name, _ = os.path.splitext(line[0]) # get rid of .pdb extension
                pdb_chain_id_dict[base_name] = line[1].split(',')
        logger.info('\n\tNumber of RNA families with valid PDB chain id found:{}'.format(len(pdb_chain_id_dict.keys())))
        return pdb_chain_id_dict
    

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

    
    def get_site_pair_data_from_file(self, site_pair_file):
        """Reads from a three column text file. The first two columns correspond 
        to sites i and j such that j > i, and the third column can be a DCA score or
        PDB contact data.

        Parameters
        ----------
            self : InputReader 
                An instance of InputReader class. 
            site_pair_file : str
                Path to text file containing sites i, j and their DCA score.

        Returns
        -------
            site_pair_dict : dict 
                A dictionary whose keys are site pairs (i, j) such that j > i and 
                whose values are the corresponding DCA score or PDB contact value. 
        """

        logger.info('\n\tObtaining site-pair data from file: {}'.format(site_pair_file))
        site_pair_dict = dict()
        with open(site_pair_file, 'r') as fh:
            for line in fh:
                line = line.strip()
                if not line: continue 
                if line.startswith('#'): continue 
                i, j , score = line.split()
                i, j, score = int(i), int(j), float(score)
                i, j = i - 1, j - 1  # so that indexing starts from 0
                assert j > i 
                site_pair = (i, j)
                site_pair_dict[site_pair] = score

        return site_pair_dict

    
    def read_all_dca_data(self, dca_dir, pattern):
        """Reads DCA data from multiple files. 

        Parameters
        ----------
            self : InputReader
                An instance of InputReader class. 
            dca_dir : str
                Path to directory containing DCA output text files. 
            pattern : str 
                DCA output files name pattern.

        Returns 
        -------
            all_dca_data : dict 
                A dictionary whose keys are RNA family names and values dictionaries
                of DCA files. 
        """
        dca_files_list = self.get_files_from_dir(dca_dir, pattern)
        all_dca_data = dict()
        for dcaf in dca_files_list:
            dcaf_basename = os.path.basename(dcaf)
            family_name, _  = os.path.splitext(dcaf_basename[20:])  # works only for the current pattern (MFDCA_apc_di_scores_RF*.txt), need to be changed  if pattern is PLMDCA_apc_di_scores_RF*.txt
            all_dca_data[family_name] = self.get_site_pair_data_from_file(dcaf)
        return all_dca_data

    
    def read_all_contact_data(self, contact_dir, pattern):
        """Reads all contact data from contact text files.

        Parameters
        ----------
            self : InputReader 
                An instance of InputReader class.
            contact_dir : str
                Path to directory containing contact files.
            pattern : str
                Patter in contat files base name.

        Returns
        -------
            all_contact_data : dict 
                A dictionary of all contact data whose keys are RNA family names
                and values are the contact dictionaries. 
        """

        contacts_file_list = self.get_files_from_dir(contact_dir, pattern)
        all_contact_data = dict()
        for contf in contacts_file_list:
            contf_basename = os.path.basename(contf)
            family_name, _ = os.path.splitext(contf_basename[12:]) # works only for the current pattern(PDBcontacts_RF*)
            # Also note that family_name  has postfix of the form _<PDBID> compared to the DCA filies naming
            all_contact_data[family_name] = self.get_site_pair_data_from_file(contf)
        return all_contact_data

    
    def read_dca_scores_from_sinlge_file(self, file_path):
        """Read DCA scores from text file. The file is assumed to follow
        pydca's DCA output file format.

        Parameters 
        ----------
            self : InputReader
                InputReader()
            file_path: str
                '/path/to/input/dca/file'

        Returns
        -------
        dca_data : dict
            dict[(i, j)] = score 
        """
        dca_data = dict()
        with open(file_path) as fh:
            for line in fh:
                line = line.strip()
                if line.startswith('#'): continue 
                line = line.split()
                i,j, score = int(line[0]) - 1 , int(line[1]) - 1, float(line[2]) 
                assert j > i 
                assert i >= 0
                assert j > 0
                dca_data[(i, j)] = score 
        return dca_data        



if __name__ == '__main__':
    
    def configure_logging():
        """Configures logging. When configured, the logging level is INFO and
        messages are logged to stream handler. Log level name are colored whenever
        the terminal supports that. INFO level is Green, WARNING level is Yellow and
        ERROR level is Red.
        """
        from pydca.config_dca.config_log import LOGGING_CONFIG
        from pydca.config_dca.config_log import ConsoleColor as c_color
        import logging.config

        logging.config.dictConfig(LOGGING_CONFIG)
        logging.addLevelName(logging.INFO, '{}{}{}'.format(
            c_color.green, logging.getLevelName(logging.INFO), c_color.nocolor))
        logging.addLevelName(logging.WARNING, '{}{}{}'.format(
            c_color.yellow, logging.getLevelName(logging.WARNING), c_color.nocolor))
        logging.addLevelName(logging.ERROR, '{}{}{}'.format(
            c_color.red, logging.getLevelName(logging.ERROR), c_color.nocolor))
        return None
    
    configure_logging()
    test_input_reader = InputReader()
    test_input_reader.get_files_from_dir('./', '*')
