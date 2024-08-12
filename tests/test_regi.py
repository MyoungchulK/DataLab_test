from pcd_register.tools.regi_loader import regi_loader
import numpy as np

"""
# Construct the 3d process class
proc_3d = proc_3d_loader(use_debug=True)

# Performs the 3d process
pts = np.arange(12).reshape(4, 3) # fake 3d point cloud
idx = 0 # fake index i
rad = 10 # fake radius
proc_3d.get_3d_process(pts, idx, rad)

def test_pts_sel():
    print('Check the point selection based on the index')
    assert np.array_equal(proc_3d.pts_i, pts[0])

def test_pts_nei_sel():
    print('Check the point selection based on the radius')
    assert np.array_equal(proc_3d.pts_nei, pts[:2])

def test_covar_mtx():
    print('Check the shape of the covariance matrix')
    assert np.array_equal(proc_3d.pts_nei_cov.shape, (3, 3))

def test_curv():
    print('Check the location of the smallest egien value')
    assert proc_3d.min_idx == 0

def test_proj():
    print('Check the shape of the projected points')
    assert np.array_equal(proc_3d.pts_nei_proj.shape, (2, 3))
"""
