import numpy as np

class proc_3d_loader:

    def __init__(self, verbose = False, use_debug = False):

        self.verbose = verbose
        self.use_debug = use_debug

    def get_covar_mtx(self):

        pts_nei_cen = np.nanmean(self.pts_nei, axis = 0)
        self.pts_cov = np.cov(self.pts_nei - pts_nei_cen)
        self.pts_cov = np.dot(pts_nei_cen.T, pts_nei_cen) / (len(self.pts_nei)-1)
        if self.use_debug:
            self.pts_nei_cen = pts_nei_cen
        del pts_nei_cen      
        print(self.pts_cov)
 
    def get_appro_curv(self):

        eig_val, _ = np.linalg.eigh(self.pts_cov)
        print(eig_val)
        print(eig_val.shape)
        del eig_val

    def get_3d_process(self, pts, idx, rad):

        pts_i = pts[idx]
        rad_all = np.sqrt(np.nansum((pts - pts_i[np.newaxis, :]) ** 2, axis = 1))
        if self.verbose:
            medi_rad = np.nanmedian(rad_all)
            print(f'Just you know the median of all radius is {np.round(medi_rad, 2)}')
            del medi_rad

        self.pts_nei = pts[rad_all <= rad]
        if self.use_debug:
            self.pts_i = pts_i
            self.rad_all = rad_all
        del pts_i, rad_all

        self.get_covar_mtx()

        #self.get_appro_curv()


def compute_curvature(pcd, radius=0.5):

    points = np.asarray(pcd.points)

    from scipy.spatial import KDTree
    tree = KDTree(points)

    curvature = [ 0 ] * points.shape[0]

    for index, point in enumerate(points):
        indices = tree.query_ball_point(point, radius)

        # local covariance
        M = np.array([ points[i] for i in indices ]).T
        M = np.cov(M)

        # eigen decomposition
        V, E = np.linalg.eig(M)
        # h3 < h2 < h1
        h1, h2, h3 = V

        curvature[index] = h3 / (h1 + h2 + h3)

    return curvature
