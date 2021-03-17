from collections import defaultdict
from hilbertcurve.hilbertcurve import HilbertCurve


class SpaceFillingCurve:
    def enumerate_cells(*args, **kwargs):
        '''
            Abstract method that takes (rows, cols) as input and returns
            a dict that maps 1d numbers to 2d locations.
        '''
        raise NotImplementedError()


class Manhattan(SpaceFillingCurve):
    def enumerate_cells(self, rows, cols):
        # maps distances to cells
        distances = defaultdict(list)

        # maps numbers to cells
        enumeration = {}

        for i in range(rows):
            for j in range(cols):
                distance = i + j
                distances[distance].append([i, j])

        sorted_distances = sorted(list(distances.keys()))

        numbers = list(range(rows * cols))
        for distance in sorted_distances:
            cells = distances[distance]
            for cell in cells:
                enumeration[numbers.pop(0)] = cell
        return enumeration


class Hilbert(SpaceFillingCurve):
    def enumerate_cells(self, rows, cols):
        enumeration = {}
        curve = HilbertCurve(rows, 2)
        for i in range(rows * cols):
            enumeration[i] = curve.coordinates_from_distance(i)
        return enumeration
