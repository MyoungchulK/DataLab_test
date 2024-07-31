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
from pcd_register.tools.utility import h5_savor

@click.command()
@click.option('-p', '--pipe', type = str)
@click.option('-d', '--data', default='', type=str)
@click.option('-o', '--output', default='', type=str)
@click.option('-i', '--index', default=0, type=int)
@click.option('-r', '--radius', default=0.1, type=float)
@click.option('-v', '--verbose', default=False, type=bool)
@click.option('-u', '--use_debug', default=False, type=bool)
@click.pass_context
def main(ctx: click.core.Context, 
         pipe: str,
         data: str,
         output: str,
         index: int,
         radius: float,
         verbose: bool,
         use_debug: bool):
    """Execute each pipeline and save the results.

    Parameters
    ----------
    ctx : click.core.Context
        The Click object that connects to other functions that decorated
        by the Click command. 
    pipe : str
        The name of the pipeline. registration and 3d process.
    data : str
        The input file path (default is '')
    output : str
        The path for storing output file. If user doesn't specify the path,
        It saves the output in the DataLab_test/output/ path. (default is '')
    index : int
        The index for selecting the point (default is 0)
    radius : float
        The boundary condition to select the neighboring points
        (default is 0.1)
    verbose : bool
        Boolean statement to control the print (default is False)
    use_debug : bool
        By changing its to True, use can check and svae the all middle step
        of the calculation. It is useful for the debugging (default is False)   
    """

    # choose script based on the `pipe` option.
    module = import_module(f'pcd_register.wrappers.{pipe}_wrappers')
    method = getattr(module, f'{pipe}_main')

    # excute the relevant pipeline.
    if pipe == 'proc_3d':
        results = ctx.invoke(
            method, data=data, output=output, index=index,
            radius=radius, verbose=verbose, use_debug=use_debug)

    # Until I confirm the conventional file format for saving the results,
    # It will be saved in the hdf5 format.
    if len(data) == 0:
        dat_name = 'EaglePointCloud'
    else:
        dat_base = os.path.basename(data)
        dat_name = os.path.splitext(dat_base)[0]
    dat_name_full = f'{dat_name}_{pipe}_idx{index}_rad{radius}.h5'
    h5_savor(output, dat_name_full, results, verbose=verbose)

if __name__ == "__main__":
    # here your code does its job
    main()
