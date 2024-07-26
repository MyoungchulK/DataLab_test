import numpy as np

class proc_3d_loader:

    def __init__(self, verbose = False, use_debug = False, use_KDTree = False):

        self.verbose = verbose
        self.use_debug = use_debug
        self.use_KDTree = use_KDTree
        if self.use_KDTree:
            from scipy.spatial import KDTree

    def get_covar_mtx(self):

        pts_nei_cen = np.nanmean(self.pts_nei, axis = 0)
        self.pts_nei_cov = np.cov((self.pts_nei - pts_nei_cen).T)
        if self.use_debug:
            self.pts_nei_cen = pts_nei_cen
        del pts_nei_cen      
 
    def get_approx_curv(self):

        eig_val_nei = np.linalg.eigh(self.pts_nei_cov)[0]
        self.pts_nei_curv = np.nanmin(eig_val_nei) / np.nansum(eig_val_nei)
        if self.use_debug:
            self.eig_val_nei = eig_val_nei
        del eig_val_nei

    def get_3d_process(self, pts, idx, rad):

        if self.verbose:
            print(f'Size of total points: {pts.shape}')
            print(f'Selected index: {idx}')
            print(f'Selected radius: {rad}')

        pts_i = pts[idx]
        if self.use_KDTree:
            rad_nei_idx = tree.query_ball_point(pts_i, rad)
        else:
            rad_all = np.sqrt(np.nansum((pts - pts_i[np.newaxis, :]) ** 2, axis = 1))
            rad_nei_idx = rad_all <= rad
            if self.use_debug:
                self.rad_all = rad_all
            if self.verbose:
                print(f'Just you know the median of all radius is {np.round(np.nanmedian(rad_all), 2)}')
            del rad_all

        self.pts_nei = pts[rad_nei_idx]
        if self.verbose:
            print(f'Size of points within radius {rad}: {self.pts_nei.shape}')
        if self.use_debug:
            self.pts_i = pts_i
            self.rad_nei_idx = rad_nei_idx
        del pts_i, rad_nei_idx

        self.get_covar_mtx()
        if self.use_debug == False:
            del self.pts_nei

        self.get_approx_curv()

