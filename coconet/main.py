
from .params import COCONET_PARAMS_DICT
from . import co_evolution
from .convolution import Convolution 
from .co_evolution import MSAData
from .cmdargs import CmdArgs
import logging 
import sys
from argparse import ArgumentParser
import os, errno

"""Main module for coconet 

Author: Mehari B. Zerihun 
"""


logger = logging.getLogger(__name__)
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


def get_weight_matrix(matrix_size, wc_and_nwc=False):
    """
    """
    if wc_and_nwc: 
        mat_key = 'coconet_mat_2x{}x{}'.format(matrix_size, matrix_size)
    else:
        mat_key = 'coconet_mat_{}x{}'.format(matrix_size, matrix_size) 
    return COCONET_PARAMS_DICT[mat_key]


def get_dca_data_dict(msa_file):
    """
    """
    mfdca_inst =  co_evolution.get_mfdca_instance(msa_file)
    fam_dca_data_list = mfdca_inst.compute_sorted_DI_APC() 
    fam_dca_data_dict = dict()
    for pair, score in fam_dca_data_list:
        fam_dca_data_dict[pair] = score 
    return fam_dca_data_dict


def write_site_pair_score_data_to_file(sorted_data_list, output_file_path):
    """Since site indices are starting from zero within python we add one to
    each of them when they are being written to output file.
    """
    with open(output_file_path, 'w') as fh:
        for site_pair, score in sorted_data_list:
            i, j = site_pair[0] + 1, site_pair[1] + 1
            fh.write('{}\t{}\t{}\n'.format(i, j, score))
    return None 


def write_output_data(dca_data, coconet_data, input_msa_file_basename, matrix_used):
    """
    """
    output_dir_name = 'Results_' + input_msa_file_basename
    try:
        os.makedirs(output_dir_name)
    except OSError as e:
        if e.errno !=errno.EEXIST:
            logger.error('\n\tUnable to create directory using path {}'.format(output_dir_name))
            raise
    
    # write data 
    dca_results_file  = 'DCA_' + input_msa_file_basename  + '.txt'
    dca_results_file = os.path.join(output_dir_name, dca_results_file)
    coconet_results_file = 'COCONET' +  matrix_used + '_' + input_msa_file_basename + '.txt'
    coconet_results_file = os.path.join(output_dir_name, coconet_results_file)
    sorted_dca_data = sorted(dca_data.items(), key = lambda d : d[1], reverse = True)
    sorted_coconet_data = sorted(coconet_data.items(), key = lambda d : d[1], reverse = True)
    logger.info('\n\tWriting DCA results to file {}'.format(dca_results_file))
    write_site_pair_score_data_to_file(sorted_dca_data, dca_results_file)
    write_site_pair_score_data_to_file(sorted_coconet_data, coconet_results_file)
    logger.info('\n\tWriting coconet results to file {}'.format(coconet_results_file))
    return None 





def execute_from_command_line(msa_file, matrix_size, wc_and_nwc= False, verbose=False):
    """
    """
    if verbose: configure_logging() 
    logger.info('\n\tExecuting coconet from command line')
    weight_matrix = get_weight_matrix(matrix_size, wc_and_nwc=wc_and_nwc) 
    
    alignment_data_inst = MSAData(msa_file)
    refseq = alignment_data_inst.refseq
    trimmed_msa_file = alignment_data_inst.trimmed_msa_file_path
    
    dca_data = get_dca_data_dict(trimmed_msa_file)
    
    conv_inst  = Convolution(matrix_size)
    if not wc_and_nwc:
        coconet_data = conv_inst.convolutionNxN_reweigh_dca_scores(dca_data, weight_matrix, len(refseq))
        matrix_used = '{}x{}'.format(matrix_size, matrix_size)
    else:
        coconet_data =conv_inst.convolutionNxN_reweigh_dca_scores_WC_and_NONWC(dca_data, weight_matrix, refseq)
        matrix_used = '2x{}x{}'.format(matrix_size, matrix_size)
    # write results to file   
    input_msa_file_basename, _ext = os.path.splitext(os.path.basename(msa_file))
    write_output_data(dca_data, coconet_data, input_msa_file_basename, matrix_used)

    logger.info('\n\tDone')

    return None 



def run_coconet():
    """
    """
    parser = ArgumentParser() 

    parser.add_argument(CmdArgs.verbose_optional, help=CmdArgs.verbose_optional_help, action='store_true')
    parser.add_argument(CmdArgs.msa_file, help=CmdArgs.msa_file_help)
    parser.add_argument(CmdArgs.matrix_size, help=CmdArgs.matrix_size_help,  type=int, choices=(3, 5, 7), default=3)
    parser.add_argument(CmdArgs.wc_and_nwc_optional, help=CmdArgs.wc_and_nwc_optional_help, action='store_true')


    args = parser.parse_args(args = None if sys.argv[1:] else ['--help']) 
    args_dict = vars(args)

    execute_from_command_line(
        args_dict.get(CmdArgs.msa_file),
        args_dict.get(CmdArgs.matrix_size[2:]),
        wc_and_nwc = args_dict.get(CmdArgs.wc_and_nwc_optional.strip()[2:]),
        verbose = args_dict.get(CmdArgs.verbose_optional.strip()[2:])
    )

    return None 

    
if __name__ == '__main__':
    """
    """
    run_coconet()