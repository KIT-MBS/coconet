import logging 
import numpy as np 

"""Performs computations on the convolutional layer of coconet. 

Author: Mehari B. Zerihun 
"""

logger = logging.getLogger(__name__)


class ConvolutionException(Exception):
    """Raises exceptions for the Convolution class.
    """

class Convolution:
    """Performs convolution per family given a dictionary of PDB contacts and DCA scores.
    """         
    def __init__(self, fm_size, linear_dist=4, contact_dist=10.0):
        """Initializes Convolution instances

        Parameters
        ----------
            self : Convolution
                An instance of Convolution class. 
            fm_size : int 
                Number of rows or columns of a square filter matrix.
        
        Returns 
        -------
            None : None 
                No return value
        """
        if fm_size < 0:
            logger.error('\n\tInvalid value for convolution matrix size')
            raise ConvolutionException
        if fm_size % 2 == 0:
            logger.error('\n\tConvolution matrix size should be odd integer. Example: 3, 5, or 7')
            raise ConvolutionException
        self.__fm_size = fm_size 
        self.__linear_dist = linear_dist 
        self.__contact_dist = contact_dist 
        self.__wc_pairs = ('G-C', 'C-G', 'A-U', 'U-A')
        return None

    @property
    def fmsize(self):
        """
        """
        return self.__fm_size


    def get_weight_matrix_pairs(self, i, j):
        """Obtains the site-pairs around pair (i, j) in a contact map for a given weight
        matrix size. 

        Parameters
        ----------
            self : Convolution
                An instance of Convolution class.
            i : int 
                First site in site-pair (i, j) for j > i
            j : int 
                Second site in site-pair (i, j) for j > i
        
        Returns 
        -------
            weight_matrix_pairs : dict 
                weight_matrix_pairs[ms_counter] = (k, l)
                ms_counter starts from 1
        """
        assert j > i 
        weight_matrix_pairs = dict()
        ms_counter = 0
        for a in range(-self.__fm_size//2 + 1,  self.__fm_size//2 + 1):
            for b in range(-self.__fm_size//2 + 1, self.__fm_size//2 + 1):
                ms_counter += 1
                weight_matrix_pairs[ms_counter] = (i + a, j + b)
        return weight_matrix_pairs
    

    def get_WC_pairs_factor(self, refseq, i, j):
        """For site pairs in the filter matrix, i.e., for all neighboring site 
        around (i, j) and including site pair (i, j), finds out wheather 
        site-pair residues are WC pairs or not. 

        Parameters
        ----------
            self : Convolution
                Convolution(self, fm_size, linear_dist=4, contact_dist=10.0)
            refseq : str 
                Reference sequence in character representation, all letters in 
                uppercase.
        
        Returns
        -------
            wc_factor : np.array((self.__fm_size * self.__fm_size,))
            wc_factor[index] = 1.0 if residues are WC pairs else 0.0
        """

        wc_factor = np.zeros((self.__fm_size * self.__fm_size, ))
        weight_matrix_site_pairs = self.get_weight_matrix_pairs(i, j)
        weight_matrix_site_pairs_size = len(weight_matrix_site_pairs)
        
        assert wc_factor.size == weight_matrix_site_pairs_size

        for counter in range(1, weight_matrix_site_pairs_size + 1):
            k, l = weight_matrix_site_pairs[counter]
            try:
                res_pair = '{}-{}'.format(refseq[k], refseq[l])
            except IndexError:
                # If indices are out-of-bounds we keep zero values for factor matrix
                pass 
            else:
                # Change WC pairs factor to 1.0
                if res_pair in self.__wc_pairs:
                    wc_factor[counter - 1] = 1.0
        return wc_factor

    
    def get_NONWC_pairs_factor(self, refseq, i, j):
        """For site pairs in the filter matrix, i.e., for all neighboring site 
        around (i, j) and including site pair (i, j), finds out wheather 
        site-pair residues are NONWC pairs or not. 

        Parameters
        ----------
            self : Convolution
                Convolution(self, fm_size, linear_dist=4, contact_dist=10.0)
            refseq : str 
                Reference sequence in character representation, all letters in 
                uppercase.
        
        Returns
        -------
            nonwc_factor : np.array((self.__fm_size * self.__fm_size,))
            nonwc_factor[index] = 1.0 if residues are NONWC pairs else 0.0
        """

        nonwc_factor = np.zeros((self.__fm_size * self.__fm_size, ))
        weight_matrix_site_pairs = self.get_weight_matrix_pairs(i, j)
        weight_matrix_site_pairs_size = len(weight_matrix_site_pairs)

        assert nonwc_factor.size == weight_matrix_site_pairs_size
        
        for counter in range(1, weight_matrix_site_pairs_size + 1):
            k, l = weight_matrix_site_pairs[counter]
            try:
                res_pair = '{}-{}'.format(refseq[k], refseq[l]) 
            except IndexError:
                # If indices are out-of-bounds we keep zero values for factor matrix
                pass 
            else:
                # Change NONWC pairs factor to 1.0 
                if res_pair not in self.__wc_pairs:
                    nonwc_factor[counter-1] = 1.0
        return nonwc_factor

    
    def validate_dca_and_pdb_data(self, dca_scores, pdb_contacts, refseq_len):
        """Validates if all site pairs are within dca scores data and pdb contacts 
        data dictionaries. 

        Parameters
        ----------
            self : Convolution 
                An instance of Convolution class.
            pdb_contacts : dict 
                A dictionary whose keys are site pairs (i, j) such that j > i and 
                values are  lists of PDB data.  The last element of each list 
                is the contact distance between sites.
            dca_scores : dict 
                A dictionary whose keys are site pairs (i, j) such that j > i and 
                whose values are the DCA scores.

        Returns 
        -------
            None : None    
        """
        site_pairs_list = dca_scores.keys()# Length of reference sequence
        L = max(site_pairs_list, key=lambda k : k[1])[1] 
        L = L + 1 # site pairs in dca_scores_dict are indexed from 0
        assert L == refseq_len
        #veryfy that all site-pairs are included in both pdb_contacts and dca_scores
        for i in range(refseq_len - 1):
            for j in range(i + 1, refseq_len):
                site_pair = i, j 
                dca_data = dca_scores.get(site_pair)
                pdb_data = pdb_contacts.get(site_pair)
                if dca_data is None:
                    logger.error('\n\tSite pair {} is missing in dca_contacts dictionary'.format(site_pair))
                    raise ConvolutionException
                if pdb_data is None:
                    logger.error('\n\tSite pair {} is missing in pdb_contacts dictionary'.format(site_pair))
                    raise ConvolutionException
        return None 


    def validate_dca_scores(self, dca_scores, refseq_len):
        """Validates if all site pairs are within dca scores data. 

        Parameters
        ----------
            self : Convolution 
                An instance of Convolution class.
            dca_scores : dict 
                A dictionary whose keys are site pairs (i, j) such that j > i and 
                whose values are the DCA scores.

        Returns 
        -------
            None : None    
        """
        site_pairs_list = dca_scores.keys() 
        L = max(site_pairs_list, key=lambda k : k[1])[1]
        L = L + 1 # site pairs in dca_scores_dict are indexed from 0
        assert L == refseq_len
        #veryfy that all site-pairs are included in both pdb_contacts and dca_scores
        for i in range(refseq_len - 1):
            for j in range(i + 1, refseq_len):
                site_pair = i, j 
                dca_data = dca_scores.get(site_pair)
                if dca_data is None:
                    logger.error('\n\tSite pair {} is missing in dca_contacts dictionary'.format(site_pair))
                    raise ConvolutionException
        return None 


    def convolutionNxN_reweigh_dca_scores_WC_and_NONWC(self, unweighed_dca_scores, weights, refseq):
        """Computes weighed DCA scores. 

        Parameters
        ----------
            dca_scores : dict 
                A dictionary whose keys are site pairs (i, j) such that j > i and 
                whose values are the DCA scores.
            weights : list 
                A  list if weights

         Returns 
        -------
            weighed_dca_scores : dict 
                A dictionary whose keys are weighed site pairs (i, j) such that j - i > 4 
                and values are reweighed DCA scores. 
        """
        # Split weights array into WC and non-WC
        size_weights = weights.size 
        assert size_weights % 2 == 0
        half_size_weights = size_weights//2
        weights_WC = weights[:half_size_weights]
        assert weights_WC.size == half_size_weights  
        weights_NONWC = weights[half_size_weights:]
        assert weights_NONWC.size == half_size_weights 
        
        logger.info('\n\tPerforming convolution using a {}x{} matrix'.format(
            self.__fm_size, self.__fm_size)
        )
        site_pairs_list = unweighed_dca_scores.keys()# Length of reference sequence
        L = max(site_pairs_list, key=lambda k : k[1])[1] 
        L = L + 1 # site pairs in dca_scores_dict are indexed from 0
        assert L == len(refseq)
        logger.info('\n\tNumber of sites: {} '.format(L))
        #verify that all site-pairs are included in both pdb_contacts and dca_scores
        self.validate_dca_scores(unweighed_dca_scores, L)
        # reweigh DCA scors
        weighed_dca_scores = dict()
        wm_pairs = dict()
        for i in range(L-1):
            for j in range(i + 1, L):
                if j-i > self.__linear_dist:
                    # generate the NxN matrix site pairs.
                    wm_pairs = self.get_weight_matrix_pairs(i, j)
                    wc_pairs_factor = self.get_WC_pairs_factor(refseq, i, j)
                    nonwc_pairs_factor = self.get_NONWC_pairs_factor(refseq, i, j)
                    # obtain DCA score of site pairs corresponding to the NxN matrix. 
                    # If the maxtrix is out of range, we set the DCA score to zero (zero padding).
                    weighed_ij = 0
                    for k in range(1, self.__fm_size * self.__fm_size + 1):
                        dij = unweighed_dca_scores.get(wm_pairs[k], 0) 
                        #weighed_ij += unweighed_dca_scores.get(wm_pairs[k], 0) * weights[k-1]
                        weighed_ij += dij * weights_WC[k-1] * wc_pairs_factor[k-1] + dij * weights_NONWC[k-1] * nonwc_pairs_factor[k-1]
                    weighed_dca_scores[(i, j)] = weighed_ij
                else: # if j-i <= 4
                    # capture the unweighed 'diagonal' DCA scores
                    weighed_dca_scores[(i, j)] = unweighed_dca_scores[(i,j)]                      
        return weighed_dca_scores


    def convolutionNxN_reweigh_dca_scores(self, unweighed_dca_scores, weights, refseq_len):
        """Computes weighed DCA scores. 

        Parameters
        ----------
            dca_scores : dict 
                A dictionary whose keys are site pairs (i, j) such that j > i and 
                whose values are the DCA scores.
            weights : list 
                A  list if weights

         Returns 
        -------
            weighed_dca_scores : dict 
                A dictionary whose keys are weighed site pairs (i, j) such that j - i > 4 
                and values are reweighed DCA scores. 
        """

        logger.info('\n\tPerforming convolution using a {}x{} matrix'.format(
            self.__fm_size, self.__fm_size)
        )
        
        logger.info('\n\tNumber of sites: {} '.format(refseq_len))
        #verify that all site-pairs are included in both pdb_contacts and dca_scores
        self.validate_dca_scores(unweighed_dca_scores, refseq_len)
        # reweigh DCA scors
        weighed_dca_scores = dict()
        wm_pairs = dict()
        for i in range(refseq_len - 1):
            for j in range(i + 1, refseq_len):
                if j-i > self.__linear_dist:
                    # generate the NxN matrix site pairs.
                    wm_pairs = self.get_weight_matrix_pairs(i, j)
                    # obtain DCA score of site pairs corresponding to the NxN matrix. 
                    # If the maxtrix is out of range, we set the DCA score to zero (zero padding).
                    weighed_ij = 0
                    for k in range(1, self.__fm_size * self.__fm_size + 1): 
                        weighed_ij += unweighed_dca_scores.get(wm_pairs[k], 0) * weights[k-1]
                    weighed_dca_scores[(i, j)] = weighed_ij
                else: # if j-i <= 4
                    # capture the unweighed 'diagonal' DCA scores
                    weighed_dca_scores[(i, j)] = unweighed_dca_scores[(i,j)]                      
        return weighed_dca_scores