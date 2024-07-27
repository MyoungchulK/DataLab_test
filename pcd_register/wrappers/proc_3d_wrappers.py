import os, sys
import click
import numpy as np

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path + '/../')
from tools.pcd_loader import pcd_loader
from tools.proc_3d_loader import proc_3d_loader

@click.command()
@click.option('-d', '--data', default='', type=str)
@click.option('-i', '--index', default=0, type=int)
@click.option('-r', '--radius', default=0.1, type=float)
@click.option('-v', '--verbose', default=False, type=bool)
@click.option('-u', '--use_debug', default=False, type=bool)
def main(data: str, 
         index: int, 
         radius: float, 
         verbose: bool, 
         use_debug: bool) -> dict:

    pcd = pcd_loader(data, verbose=verbose)
    pcd.get_pts_in_np()
    pts = pcd.pts
    del pcd
  
    proc_3d = proc_3d_loader(verbose=verbose, 
                             use_debug=use_debug, 
                             use_KDTree=False) 
    proc_3d.get_3d_process(pts, index, radius)  
    covar_mtx = proc_3d.pts_nei_cov
    approx_curv = proc_3d.pts_nei_curv
    proj_pts = proc_3d.pts_nei_proj
    if use_debug:
        centriod = proc_3d.pts_nei_cen
        eigen_val = proc_3d.eig_val_nei
        eigen_vec = proc_3d.eig_vec_nei
        nomal_vec = proc_3d.nor_vec_nei
        displace_vec = proc_3d.dis_vec_nei
        pts_i = proc_3d.pts_i
        pts_nei = proc_3d.pts_nei
    del proc_3d
   
    np.set_printoptions(threshold=0) # turncate numpy print. just for cosmetic
    print(f'Covariance matrix @ index {index}:\n{covar_mtx}')
    print(f'Approximate Curvature @ index {index}: {approx_curv}')
    print(f'Projection of points to the plane @ index {index}:\n{proj_pts}\nSize of array: {proj_pts.shape}')
        
if __name__ == "__main__":

    main()
