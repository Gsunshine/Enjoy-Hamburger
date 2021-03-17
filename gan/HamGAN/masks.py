import numpy as np
import random
import math
import tensorflow as tf
from itertools import cycle
from space_filling_curves import Manhattan, Hilbert

def allow_non_square(fn):
    '''
        Decorator: Augments nO for square masks or allows non square masks.
    '''
    def wrap(self, nI, **kwargs):
        nO = kwargs.pop('nO', None)
        if nO is None:
            nO = nI
        return fn(self, nI, nO, **kwargs)
    return wrap


def numpy(fn):
    '''
        Decorator: Converts a list to a numpy array of float32.
    '''
    def wrap(self, *args, **kwargs):
        indices = fn(self, *args, **kwargs)
        tensor = np.array(indices, dtype=np.float32)
        return tensor
    return wrap


def disallow_downsampling(fn):
    '''
        Decorator: Raises ValueError when the number of output nodes is
                   less than the number of input nodes.
    '''
    def wrap(self, nI, **kwargs):
        nO = kwargs.pop('nO', None)
        if nO is None:
            nO = nI
        if nO < nI:
            raise ValueError('Downsampling not supported.')
        return fn(self, nI, nO, **kwargs)
    return wrap


def disallow_non_square(fn):
    '''
        Decorator: Raises ValueError for non square masks.
    '''
    def wrap(self, nI, **kwargs):
        nO = kwargs.pop('nO', None)
        if nO is not None and nO != nI:
            raise ValueError('Non square masks not supported')
        return fn(self, nI, **kwargs)
    return wrap


def compute_stride(fn):
    '''
        Decorator: Computes a default value for stride, based on number
                   of nodes.
    '''
    def wrap(self, nL, nO=None, **kwargs):
        stride = kwargs.pop('stride', None)
        if stride is None:
            stride = math.floor(math.sqrt(nL))
        return fn(self, nL, nO=nO, stride=stride, **kwargs)
    return wrap


class SparseMask:
    def convert_to_1d(i, j, cols):
        return i * cols + j

    @classmethod
    def get_indices(self, **kwargs):
        raise NotImplementedError()

    @classmethod
    @allow_non_square
    def get_mask(self, nI, nO, **kwargs):
        indices = self.get_indices(nI, nO=nO, **kwargs)
        tensor = np.zeros([nO, nI], dtype=np.float32)
        tensor[indices[:, 0].astype(int), indices[:, 1].astype(int)] = 1
        return tensor

    @classmethod
    @allow_non_square
    @numpy
    def get_grid_mask(self, gridI, gridO, **kwargs):
        '''
            Returns a mask that corresponds to grid inputs.
        '''
        rowsI, colsI = gridI
        rowsO, colsO = gridO
        nI = rowsI * colsI
        nO = rowsO * colsO

        indices = self.get_indices(gridI, nO=gridO, **kwargs)
        tensor = np.zeros([nO, nI], dtype=np.float32)
        tensor[indices[:, 0].astype(int), indices[:, 1].astype(int)] = 1
        return tensor

    def validate_bounds(indices, nO, nI):
        '''
            Decorator: Removes out of bounds for a grid.
        '''
        for index, (i, j) in enumerate(indices):
            if i >= nO or j >= nI:
                indices.pop(index)
        return indices

    @classmethod
    @disallow_non_square
    @numpy
    def get_square_grid_indices_from_1d(self, grid, filling_curve='manhattan'):
        if filling_curve == 'manhattan':
            curve = Manhattan()
        else:
            curve = Hilbert()
        rows, cols = grid
        enumeration = curve.enumerate_cells(rows, cols)
        mask_indices = self.get_indices(grid[0] * grid[1])
        indices = []
        for sO, sI in mask_indices:
            x_i, x_j = enumeration[sO]
            y_i, y_j = enumeration[sI]

            x = self.convert_to_1d(x_i, x_j, cols)
            y = self.convert_to_1d(y_i, y_j, cols)
            indices.append([x, y])
        return indices

    @classmethod
    @allow_non_square
    @numpy
    def get_grid_indices_from_1d(self, gridI, gridO, filling_curve='manhattan'):
        rowsI, colsI = gridI
        rowsO, colsO = gridO

        if rowsI == rowsO and colsI == colsO:
            return self.get_square_grid_indices_from_1d(gridI, filling_curve=filling_curve)

        blocks_ratio = (rowsO * colsO) // (rowsI * colsI)
        offset = (rowsI * colsI)
        indices = self.get_square_grid_indices_from_1d(gridI, filling_curve=filling_curve)
        offset_array = np.zeros(indices.shape)
        for i in range(1, blocks_ratio):
            new_indices = self.get_square_grid_indices_from_1d(gridI, filling_curve=filling_curve)
            offset_array[:, 0] += offset
            new_indices = offset_array + new_indices
            indices = np.concatenate([indices, new_indices])
        return indices

    @classmethod
    @allow_non_square
    def get_grid_mask_from_1d(self, gridI, gridO, **kwargs):
        rowsO, colsO = gridO
        nO = rowsO * colsO
        rowsI, colsI = gridI
        nI = rowsI * colsI

        indices = self.get_grid_indices_from_1d(gridI, nO=gridO, **kwargs)
        tensor = np.zeros([nO, nI], dtype=np.float32)
        tensor[indices[:, 0].astype(int), indices[:, 1].astype(int)] = 1
        return tensor


class SubsequentMask(SparseMask):
    @classmethod
    @disallow_non_square
    def get_indices(self, nL, **kwargs):
        mask = self.get_mask(nL)
        return tf.where(tf.equal(mask, 1))

    @classmethod
    @disallow_non_square
    def get_mask(self, nL):
        reverse = tf.convert_to_tensor(np.triu(np.ones((nL, nL)), k=1),
                                       dtype=tf.float32)
        return tf.dtypes.cast(tf.equal(reverse, 0), tf.float32)


class LeftFloorMask(SparseMask):
    @classmethod
    @disallow_downsampling
    @compute_stride
    @numpy
    def get_indices(self, nI, nO, stride=None, **kwargs):
        indices = []
        for row in range(nO):
            lower = max(0, min(row - (row % stride), nI))
            higher = min(row + 1, nI)
            for col in range(lower, higher):
                indices.append([row, col])
        return indices


class RightFloorMask(SparseMask):
    @classmethod
    @disallow_downsampling
    @compute_stride
    @numpy
    def get_indices(self, nI, nO, stride=None, **kwargs):
        indices = []
        for row in range(nO):
            lower = max(0, min(row - (row % stride), nI))
            higher = min(row + 1, nI)
            for col in range(lower, higher):
                indices.append([col, row])
        return indices


class LeftRepetitiveMask(SparseMask):
    @classmethod
    @compute_stride
    @allow_non_square
    @numpy
    def get_indices(self, nI, nO, stride=None, overlap=1, **kwargs):
        indices = []
        row_indices = np.arange(0, nI)
        indxs = (row_indices % stride) >= (stride - overlap)
        col_indices = row_indices[indxs]

        for row in np.arange(0, nO):
            if nI == nO:
                indices.append([row, row])
            for col in col_indices:
                indices.append([row, col])
        return indices


class StridedMask1(SparseMask):
    @classmethod
    @disallow_non_square
    @compute_stride
    @numpy
    def get_indices(self, nL, stride=None, **kwargs):
        indices = []
        for row in range(nL):
            low = max(0, row - stride + 1)
            for col in range(low, row + 1):
                indices.append([row, col])
        return indices


class StridedMask2(SparseMask):
    @classmethod
    @disallow_non_square
    @compute_stride
    @numpy
    def get_indices(self, nL, stride=None, **kwargs):
        indices = []
        for row in range(nL):
            for col in range(nL):
                if abs(row - col) % stride == 0:
                    indices.append([row, col])
        return indices


class OneDirectionalRepetitiveMask(SparseMask):
    @classmethod
    def get_mask(self, nL, stride=None, overlap=1):
        repetitive_mask = \
            LeftRepetitiveMask.get_mask(nL, stride=stride, overlap=overlap)
        repetitive_mask = SubsequentMask.get_mask(nL) * repetitive_mask
        return repetitive_mask


class RightRepetitiveMask(SparseMask):
    @classmethod
    @allow_non_square
    @compute_stride
    @numpy
    def get_indices(self, nI, nO, stride=None, overlap=1, **kwargs):
        indices = []
        row_indices = np.arange(0, nI)
        indxs = (row_indices % stride) >= (stride - overlap)
        col_indices = row_indices[indxs]
        col_indices = col_indices - (stride - overlap)

        for row in np.arange(0, nO):
            if nI == nO:
                indices.append([row, row])
            for col in col_indices:
                indices.append([row, col])
        return indices


class CircleMask(SparseMask):
    @classmethod
    @numpy
    def get_indices(self, nΙ, nO, *args, **kwargs):
        for i in range(nO):
            # left neighbor
            if i == 0:
                indices.append([i, nI - 1])
            else:
                indices.append([i, i - 1])

            # right neighbor
            if i != last_indx:
                indices.append([i, i + 1])
            else:
                indices.append([i, 0])
        return indices


class CircleGridMask(SparseMask):
    def get_neighbors(i, j, nO, nI):
        neighbors = []
        if i == 0:
            # up neighbor folds
            neighbors.append([nO - 1, j])
        else:
            neighbors.append([i - 1, j])
        if j == 0:
            # left neighbor folds
            neighbors.append([i, nI - 1])
        else:
            neighbors.append([i, j - 1])

        if i == nO - 1:
            # down neighbor folds
            neighbors.append([0, j])
        else:
            neighbors.append([i + 1, j])

        if j == nI - 1:
            # right neighbor folds
            neighbors.append([i, 0])
        else:
            neighbors.append([i, j + 1])
        return neighbors

    @classmethod
    @allow_non_square
    @numpy
    def get_indices(self, gridI, gridO, **kwargs):
        '''
            gridI: (rowsI, colsI)
            gridO: (rowsO, colsO)
        '''
        rowsI, colsI = gridI
        rowsO, colsO = gridO
        indices = []
        for i in range(rowsO):
            for j in range(colsO):
                neighbors = self.get_neighbors(i, j, rowsI, colsI)
                x_index = self.convert_to_1d(i, j, colsO)
                indices += [[x_index, self.convert_to_1d(*x, colsI)] for x in neighbors]
            # if rowsI == rowsO and colsI == colsO:
            #     indices.append([x_index, x_index])
        return indices


class DirectRelayMask(SparseMask):
    @classmethod
    @numpy
    def get_indices(self, nL, **kwargs):
        indices = []
        for i in range(nL - 1):
            indices.append([nL - 1, i])
        return indices


class ReverseRelayMask(SparseMask):
    @classmethod
    @numpy
    def get_indices(self, nL, **kwargs):
        indices = []
        for i in range(nL - 1):
            indices.append([i, nL - 1])
        return indices


class ConstantDegreeRandomMask(SparseMask):
    @classmethod
    @numpy
    def get_indices(self, nI, nO, deg, **kwargs):
        # send edges from nI to nO
        # indices containts tuples with max: [nO, nI]
        indices = []
        for i in range(nI):
            for _ in range(deg):
                # find a random neighbor
                j = random.randint(0, nO - 1)
                indices.append([j, i])
        return indices


class SlidingMask(SparseMask):
    @classmethod
    @allow_non_square
    @numpy
    def get_indices(self, nI, nO, overlap=1, **kwargs):
        deg = math.ceil(((nI + (nO - 1) * overlap) / nO))
        indices = []
        start = 0
        for i in range(nO):
            for k in range(deg):
                y_index = k + start
                if y_index >= nI:
                    y_index = nI - y_index
                indices.append([i, y_index])
            start += deg - 1
            if start >= nI:
                start = nI - start
        return indices


class MasksCollection:
    def __init__(self, masks, mode='interleave'):
        self.masks = masks
        self.mode = mode
        self.merged_head = None
        self.cycled_masks = cycle(self.masks)

    def __call__(self):
        with tf.init_scope():
            if self.mode == 'interleave':
                return next(self.cycled_masks)
            elif self.mode == 'merged_head':
                # avoid re-computation
                if self.merged_head is None:
                    nL = self.masks[0].shape[0]
                    self.merged_head = tf.ones((nL, nL), dtype=tf.int32)
                    for mask in self.masks:
                        self.merged_head = self.merged_head * mask
                return self.merged_head
            elif self.mode == 'heads':
                return np.array(self.masks)
            else:
                raise ValueError('Not supported attention mode')
