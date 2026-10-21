import numpy as np


class Detector:
    def __init__(self, R=2.0, r=0.01, a=0.5, N=40):
        self.R = R
        self.r = r
        self.a = a
        self.N = N

        self.pmt_coords = self._generate_pmt_positions()

    def _generate_pmt_positions(self):
        indices = np.arange(0, self.N, dtype=float) + 0.5
        z = self.R - 2 * self.R * indices / self.N
        rho = np.sqrt(self.R**2 - z**2)
        golden_angle = np.pi * (3-np.sqrt(5))
        theta = golden_angle * indices
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        coords = np.column_stack((x, y, z))
        return coords

    def get_pmt_index(self, point: np.ndarray) -> int:
        point = np.asarray(point, dtype=float)
        distances = np.linalg.norm(self.pmt_coords - point, axis=1)
        min_idx = int(np.argmin(distances))
        if distances[min_idx] <= self.r:
            return min_idx
        return -1

    def is_in_pmt(self, point: np.ndarray) -> bool:
        return self.get_pmt_index(point) != -1


det = Detector(R=2.0, N=40, r=0.1, a=0.5)


