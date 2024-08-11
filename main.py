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
@click.option('-v', '--dat_var', default='', type=str)
@click.pass_context
def main(ctx: click.core.Context, dat_var: str):
    """Execute each pipeline and save the results.

    Parameters
    ----------
    ctx : click.core.Context
        The Click object that connects to other functions that decorated
        by the Click command. 
    dat_var : str
        The text file path that contains all the variables for the pipeline 
        process (default is ''). User can control the script by changing the 
        contents of the text file. By doing this way, we don't have to create
        infinite arguments at the terminal. The variables will be stored in the
        dictionary. If dat_var is empty, use icp examples in the examples path. 
    """
    
    # Check the sanity of the data path.
    dat_dict = get_data_info(dat_var)
     
    # Choose script based on the `pipe` option.
    pipe = dat_dict['pipe_name']
    module = import_module(f'pcd_register.wrappers.{pipe}_wrappers')
    method = getattr(module, f'{pipe}_main')

    # Excute the relevant script in the wrapper path.
    results = ctx.invoke(method, dat_dict=dat_dict)
    
    # Until I confirm the conventional file format for saving the results,
    # it will be saved in the hdf5 format.
    h5_savor(dat_dict, results)
    
if __name__ == "__main__":
    # here your code does its job
    main()
