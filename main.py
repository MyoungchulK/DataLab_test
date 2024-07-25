# here you import from your library "pcd_register"
import os
import h5py
import click
import importlib import import_module

@click.command()
@click.option('-t', '--test', type = str)
@click.option('-d', '--data', default = '', type = str)
@click.option('-v', '--verbose', default = False, type = bool)
@click.option('-u', '--use_debug', default = False, type = bool)
@click.option('-p', '--num_pts', default = 100, type = int)
@click.option('-i', '--idx', default = 0, type = int)
def main(test, data, verbose, use_debug):

    # choose script based on the `test` option.
    module = import_module(f'pcd_register.wrappers.{test}_wrappers')
    method = getattr(module, f'main')

    # excute the related function
    if test == 'proc_3d':
        result = method(data, idx, num_pts = num_pts, verbose = verbose, use_debug = use_debug)

    # output path
    abs_path = os.path.abspath('.')
    output_path = os.path.joinpath(abs_path, 'output/')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # output file name
    if len(data) == 0:
        dat_name = 'random'
    else:
        dat_base = os.path.basename(data)   
        dat_name = os.path.splitext(dat_base)
    output_name = os.path.joinpath(output_path, f'{dat_name}_{test}.h5')
    
    # saving
    hf = h5py.File(output_name, 'w')
    for r in results:
        print(r, results[r].shape)
        hf.create_dataset(r, data=results[r], compression="gzip", compression_opts=9)
    hf.close()

if __name__ == "__main__":
    # here your code does its job
    main()
