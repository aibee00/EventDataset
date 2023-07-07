"""
Reference: https://github.com/alyssaq/3Dreconstruction/blob/master/camera.py
"""
import cv2
import numpy as np


class CameraInfoPacket(object):

    def __init__(self, P=None, K=None, R=None, t=None, dist_coeff=None):
        """
        P = K[R|t]
        One must either supply P or K, R, t
        :param P: camera matrix, (3, 4)
        :param K: intrinsic matrix, (3, 3)
        :param R: rotation matrix, (3, 3)
        :param t: translation vector, (3, 1)
        """
        if P is None:
            P = np.dot(K, np.hstack([R, t]))

        self.P = P     # camera matrix
        self.K = K     # intrinsic matrix
        self.R = R     # rotation matrix
        self.t = t     # translation vector
        self.dist_coeff = dist_coeff
        self.cam_orig_world = self.get_cam_coord_world()
        # self.Rw2c = R.astype(np.float64)
        # self.Tw2c = t.astype(np.float64)
    
    def get_cam_coord_world(self):
        """
        return world coordinate of camera origin
        # https://en.wikipedia.org/wiki/Camera_resectioning
        :return:
        """
        return -self.R.T @ self.t

    def project(self, X):
        """
        Project 3D homogenous points X (4 * n) and normalize coordinates.
        Return projected 2D points (2 x n coordinates)
        :param X:
        :return:
        """
        if len(X.shape) == 3:
            x = X @ self.P.T
            x[..., 0] = x[..., 0] / x[..., 2]
            x[..., 1] = x[..., 1] / x[..., 2]
            return x[..., :2]
        x = np.dot(self.P, X)
        x[0, :] /= x[2, :]
        x[1, :] /= x[2, :]

        return x[:2, :]

    def qr_to_rq_decomposition(self):
        """
        Convert QR to RQ decomposition with numpy.
        Note that this could be done by passing in a square matrix with scipy:
        K, R = scipy.linalg.rq(self.P[:, :3])
        :return:
        """
        Q, R = np.linalg.qr(np.flipud(self.P).T)
        R = np.flipud(R.T)
        return R[:, ::-1], Q.T[::-1, :]

    def factor(self):
        """
        Factorize the camera matrix P into K,R,t with P = K[R|t] using RQ-factorization
        :return:
        """

        # already been factorized or supplied
        if self.K is not None and self.R is not None:
            return self.K, self.R, self.t

        K, R = self.qr_to_rq_decomposition()

        # make diagonal of K positive
        T = np.diag(np.sign(np.diag(K)))
        if np.linalg.det(T) < 0:
            T[1, 1] *= -1

        self.K = np.dot(K, T)
        self.R = np.dot(T, R)[:, :3]
        self.t = np.dot(np.linalg.inv(self.K), self.P[:, 3]).reshape(-1, 1)

        # sanity check
        assert (np.isclose(self.P, np.dot(self.K, np.hstack([self.R, self.t])), rtol=1e-8)).any()

    @staticmethod
    def cart2hom(arr):
        """
        Convert catesian to homogenous points by appending a row of 1s
        :param arr: array of shape (num_dimension x num_points)
        :returns: array of shape ((num_dimension+1) x num_points)
        """
        if len(arr.shape) == 3:
            return np.concatenate((arr, np.ones(arr.shape[:-1] + (1,))), axis=-1)

        if arr.ndim == 1:
            return np.hstack([arr, 1])
        return np.asarray(np.vstack([arr, np.ones(arr.shape[1])]))

    @staticmethod
    def hom2cart(arr):
        """
        Convert homogenous to catesian by dividing each row by the last row
        :param arr: array of shape (num_dimension x num_points)
        :returns: array of shape ((num_dimension-1) x num_points) iff d > 1
        """
        # arr has shape: dimensions x num_points
        num_rows = len(arr)
        if num_rows == 1 or arr.ndim == 1:
            return arr

        return np.asarray(arr[:num_rows - 1] / arr[num_rows - 1])

    @staticmethod
    def euler2rotation(thetas):
        """
        Calculate rotation matrix based on euler angles
        :param thetas:
        :return:
        """
        return cv2.Rodrigues(thetas)