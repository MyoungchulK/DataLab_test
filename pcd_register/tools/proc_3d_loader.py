import numpy as np

class proc_3d_loader:

    def __init__(self, verbose = False, use_debug = False, use_KDTree = False):

        self.verbose = verbose
        self.use_debug = use_debug
        self.use_KDTree = use_KDTree
        if self.use_KDTree:
            import open3d as o3d # I dont like to load the heavy lib just for the point selection

    def get_covar_mtx(self):

        pts_nei_cen = np.nanmean(self.pts_nei, axis = 0)
        self.pts_nei_cov = np.cov((self.pts_nei - pts_nei_cen).T)
        if self.use_debug:
            self.pts_nei_cen = pts_nei_cen
        del pts_nei_cen      
 
    def get_approx_curv(self):

        eig_val_nei, self.eig_vec_nei = np.linalg.eigh(self.pts_nei_cov)
        self.min_idx = np.nanargmin(eig_val_nei)
        self.pts_nei_curv = eig_val_nei[self.min_idx] / np.nansum(eig_val_nei)
        if self.use_debug:
            self.eig_val_nei = eig_val_nei
        del eig_val_nei

    def get_proj_pts_to_plane(self):

        nor_vec_nei = self.eig_vec_nei[:, self.min_idx]
        dis_vec_nei = self.pts_nei - self.pts_i
        dis_to_plane = np.dot(dis_vec_nei, nor_vec_nei) # (# of pts, xyz) * (xyz, 1) = (# of pts, 1). But the result is just (# of pts)
        self.pts_nei_proj = self.pts_nei - dis_to_plane[:, np.newaxis] * nor_vec_nei[np.newaxis, :]
        if self.use_debug:
            self.nor_vec_nei = nor_vec_nei
            self.dis_vec_nei = dis_vec_nei
            self.dis_to_plane = dis_to_plane 
        del nor_vec_nei, dis_vec_nei

    def get_3d_process(self, pts, idx, rad):

        if self.verbose:
            print(f'Size of total points: {pts.shape}')
            print(f'Selected index: {idx}')
            print(f'Selected radius: {rad}')

        self.pts_i = pts[idx]
        if self.use_KDTree:
            kdtree = o3d.geometry.KDTreeFlann(pts)
            rad_nei_idx = kdtree.search_radius_vector_3d(self.pts_i, rad)[1]
            del kdtree
        else:
            rad_all = np.sqrt(np.nansum((pts - self.pts_i[np.newaxis, :]) ** 2, axis = 1))
            rad_nei_idx = rad_all <= rad # technically boolean array
            if self.use_debug:
                self.rad_all = rad_all
            if self.verbose:
                print(f'Just you know the median of all radius is {np.round(np.nanmedian(rad_all), 2)}')
            del rad_all

        self.pts_nei = pts[rad_nei_idx]
        if self.verbose:
            print(f'Size of points within radius {rad}: {self.pts_nei.shape}')
        if self.use_debug:
            self.rad_nei_idx = rad_nei_idx
        del rad_nei_idx

        self.get_covar_mtx()

        self.get_approx_curv()

        self.get_proj_pts_to_plane()
        if self.use_debug == False:
            del self.pts_nei, self.eig_vec_nei, self.pts_i, self.min_idx
