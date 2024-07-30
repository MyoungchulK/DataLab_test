# here you import from your library "pcd_register"
import os
import sys
import h5py
import click
from importlib import import_module

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path + '/pcd_register/')
from pcd_register.tools.utility import size_checker

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
            radius=radius, verbose=verbose, 
            use_debug=use_debug)

    # Make output path
    if len(output) == 0:
        code_path = os.path.dirname(os.path.realpath(__file__)) # bit old method
        output = os.path.join(code_path, 'output')
    if not os.path.exists(output):
        os.makedirs(output)

    # Set the file name
    if len(data) == 0:
        dat_name = 'EaglePointCloud'
    else:
        dat_base = os.path.basename(data)
        dat_name = os.path.splitext(dat_base)[0]
    dat_name_full = f'{dat_name}_proc_3d_idx{index}_rad{radius}.h5'
    output_name = os.path.join(output, dat_name_full)

    # Creates the hdf5 file.
    hf = h5py.File(output_name, 'w')
    for r in results:
        if verbose:
            print(r, results[r].shape) # Checking what is saving in the file.
            hf.create_dataset(r, data=results[r], compression="gzip"
                              , compression_opts=9)
    hf.close()
    print(f'Output is in {output_name}. {size_checker(output_name)}')

if __name__ == "__main__":
    # here your code does its job
    main()
