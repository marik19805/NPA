import numpy as np
import matplotlib.pyplot as plt


class Source:
    def __init__(self, Y: float = 10000.0):
        self.Y = Y

    def __repr__(self) -> str:
        return f'Source(Y={self.Y})'

    def generate_directions(self, E: float) -> np.ndarray:
        N_float = self.Y*E
        N = int(np.floor(N_float))
        probability = N_float - N
        N = N + int(np.random.rand() < probability)

        phi = np.random.uniform(0, 2 * np.pi, size=N)
        cos_theta = np.random.uniform(-1.0, 1.0, size=N)
        sin_theta = np.sqrt(1.0 - cos_theta ** 2)

        nx = sin_theta * np.cos(phi)
        ny = sin_theta * np.sin(phi)
        nz = cos_theta

        return np.column_stack((nx,ny,nz))


source = Source(Y=10000)
directions = source.generate_directions(E=1.0)

print(f"Количество фотонов при E = 1 МэВ: {len(directions)}")

# Гистограмма проекций направлений (проверка изотропности)
plt.hist(directions[:, 0], bins=50, histtype='step', label='nx')
plt.hist(directions[:, 1], bins=50, histtype='step', label='ny')
plt.hist(directions[:, 2], bins=50, histtype='step', label='nz')
plt.xlabel('Проекция единичного вектора направления')
plt.ylabel('Число фотонов')
plt.legend()
plt.show()