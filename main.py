# here you import from your library "pcd_register"
import os
import sys
import click
from importlib import import_module

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path + '/pcd_register/')
from pcd_register.tools.utility import h5_savor

@click.command()
@click.option('-t', '--test', type = str)
@click.option('-d', '--data', default='', type=str)
@click.option('-o', '--output', default='', type=str)
@click.option('-i', '--index', default=0, type=int)
@click.option('-r', '--radius', default=0.1, type=float)
@click.option('-v', '--verbose', default=False, type=bool)
@click.option('-u', '--use_debug', default=False, type=bool)
@click.pass_context
def main(ctx, 
         test: str,
         data: str,
         output: str,
         index: int,
         radius: float,
         verbose: bool,
         use_debug: bool):

    # choose script based on the `test` option.
    module = import_module(f'pcd_register.wrappers.{test}_wrappers')
    method = getattr(module, f'{test}_main')

    # excute the related function
    if test == 'proc_3d':
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
    dat_name_full = f'{dat_name}_{test}_idx{index}_rad{radius}.h5'
    h5_savor(output, dat_name_full, results, verbose=verbose)

if __name__ == "__main__":
    # here your code does its job
    main()
