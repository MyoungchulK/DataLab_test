import numpy as np
import os, sys
import click

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.pcd_loader import pcd_loader
from tools.proc_3d_loader import proc_3d_loader

@click.command()
@click.option('-d', '--data', default = '', type = str)
@click.option('-i', '--index', default = 0, type = int)
@click.option('-r', '--radius', default = 100, type = float)
@click.option('-v', '--verbose', default = False, type = bool)
@click.option('-u', '--use_debug', default = False, type = bool)
def main(data, index, radius, verbose, use_debug):

    pcd = pcd_loader(data, verbose = verbose)
    pcd.get_pts_in_np()
    pts = pcd.pts
  
    proc_3d = proc_3d_loader(verbose = verbose, use_debug = use_debug) 
    proc_3d.get_3d_process(pts, index, radius)  

if __name__ == "__main__":

    main()
