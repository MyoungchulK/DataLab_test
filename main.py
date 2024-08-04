"""This script is designed to execute the 2 pipelines, registration
and 3d process, that saved in the pcd_register path. Each pipeline 
that saved in the wrappers path will be executed by importlib package.
The script in the wrappers path will access the class that saved in
the tools pass to do necessary calculation. The results will be saved
in the hdf5 format for now. The argument is controlled by the click
package.

    * main - execute each pipeline and save the results.
"""

import os
import sys
import click
from importlib import import_module

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path + '/pcd_register/')
from pcd_register.tools.pcd_loader import get_data_info
from pcd_register.tools.utility import h5_savor

@click.command()
@click.option('-p', '--pipe', type = str)
@click.option('-s', '--dat_src', default='', type=str)
@click.option('-t', '--dat_tar', default='', type=str)
@click.option('-o', '--output', default='', type=str)
@click.option('-i', '--index', default=0, type=int)
@click.option('-r', '--radius', default=0.1, type=float)
@click.option('-v', '--verbose', default=False, type=bool)
@click.option('-e', '--use_dat_ex', default=False, type=bool)
@click.option('-u', '--use_debug', default=False, type=bool)
@click.pass_context
def main(ctx: click.core.Context, 
         pipe: str,
         dat_src: str,
         dat_tar: str,
         output: str,
         index: int,
         radius: float,
         verbose: bool,
         use_dat_ex: bool,
         use_debug: bool):
    """Execute each pipeline and save the results.

    Parameters
    ----------
    ctx : click.core.Context
        The Click object that connects to other functions that decorated
        by the Click command. 
    pipe : str
        The name of the pipeline. regi (registration) or proc_3d (3d process).
    dat_src : str
        The input source file path (default is ''). 
        In case of the proc_3d option, use dat_src for input file option.
    dat_tar : str
        The input target file path (default is '').
    output : str
        The path for storing output file. If user doesn't specify the path,
        It saves the output in the DataLab_test/output/ path (default is '').
    index : int
        The index for selecting the point (default is 0).
    radius : float
        The boundary condition to select the neighboring points.
        (default is 0.1)
    verbose : bool
        Boolean statement to control the print (default is False).
    use_dat_ex : bool
        Boolean statement for using example ICP dataset (default is False).
        If use_dat_ex is True, the dat_src and dat_tar will be overwritten by
        ICP dataset.
    use_debug : bool
        By changing its to True, use can check and svae the all middle step
        of the calculation. It is useful for the debugging (default is False).   
    """

    # Check the sanity of the data path.
    dat_src, dat_tar, dat_key = get_data_info(pipe, dat_src, dat_tar, 
                                              use_dat_ex, verbose) 
    
    # Choose script based on the `pipe` option.
    module = import_module(f'pcd_register.wrappers.{pipe}_wrappers')
    method = getattr(module, f'{pipe}_main')

    # Excute the relevant pipeline.
    if pipe == 'proc_3d':
        results = ctx.invoke(
            method, dat_src=dat_src, output=output, index=index,
            radius=radius, verbose=verbose, use_dat_ex=use_dat_ex, 
            use_debug=use_debug)

    # Until I confirm the conventional file format for saving the results,
    # It will be saved in the hdf5 format.
    file_name = f'{dat_key}_{pipe}_idx{index}_rad{radius}.h5'
    h5_savor(output, file_name, results, verbose=verbose)
    
if __name__ == "__main__":
    # here your code does its job
    main()
