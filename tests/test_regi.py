from pcd_register.tools.regi_loader import regi_loader
import numpy as np
import open3d as o3d

# Get icp dataset
paths = o3d.data.DemoICPPointClouds().paths[:2]
pcd_list = []
for p in paths:
    pcd_list.append(o3d.io.read_point_cloud(p))

# Construct the registration class
regi = regi_loader(pcd_list, use_debug=True)

# Performs preprocessing
regi.get_pre_process(0, 1)

# Performs the RANSAC registration
reg_ran = regi.get_ransac_regi(0, 1)

# Performs the ICP registration
reg_icp = regi.get_icp_regi(0, 1)

def test_down_shape():
    print('Check the shape of the down sampled pcd file')
    assert np.array_equal(np.asarray(regi.pcd_down[0].points).shape, (8715, 3))

def test_fpfh_shape():
    print('Check the fpfh dimension is 33.')
    assert regi.pcd_fpfh[0].data.shape[0] == 33

def test_ran_trans_nan():
    print('Check the transformation matrix from the RANSAC is including nan')
    assert ~np.isnan(np.sum(reg_ran.transformation))

def test_icp_trans_nan():
    print('Check the transformation matrix from the ICP is including nan')
    assert ~np.isnan(np.sum(reg_icp.transformation))
    
