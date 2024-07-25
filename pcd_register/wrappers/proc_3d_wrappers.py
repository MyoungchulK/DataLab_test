import numpy as np
import os
import h5py


# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter

def main(data, idx, num_pts = 100, verbose = False, use_debug = False):

    if len(data) == 0:
        pts_3d = np.random.rand(num_pts, 3) 
    else:
        hf = h5py.File(data, 'r')
        pts_3d = hf['pts_3d'][:]

    

if __name__ == "__main__":

    main(data, idx)
