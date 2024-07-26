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
@click.option('-r', '--radius', default = 0.1, type = float)
@click.option('-v', '--verbose', default = False, type = bool)
@click.option('-u', '--use_debug', default = False, type = bool)
def main(data, index, radius, verbose, use_debug):

    pcd = pcd_loader(data, verbose = verbose)
    pcd.get_pts_in_np()
    pts = pcd.pts
    del pcd
  
    proc_3d = proc_3d_loader(verbose = verbose, use_debug = use_debug, use_KDTree = False) 
    proc_3d.get_3d_process(pts, index, radius)  
    covar_mtx = proc_3d.pts_nei_cov
    approx_curv = proc_3d.pts_nei_curv
    del proc_3d
    
    print(f'Covariance matrix @ index {index}: \n{covar_mtx}')
    print(f'Approximate Curvature @ index {index}: {approx_curv}')
    #print(f'Projection of points to the plane @ index {index}: {}')

if __name__ == "__main__":

    main()
