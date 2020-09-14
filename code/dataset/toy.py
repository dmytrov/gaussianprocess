import numpy as np
from dataset.mocap import *


def load_sines(nparts=1, nchunks=3, tchunk=30):
    T = nchunks * tchunk
    t = np.linspace(0.0, 2 * np.pi, num=T)
    f1 = nchunks * 2.0  # cycles
    f2 = nchunks * 2.0  # cycles
    f3 = nchunks * 3.14  # cycles
    y1 = np.vstack((5.0 * np.sin(f1 * t + 0.0), 5.0 * np.sin(f1 * t + 1.5),
                    4.1 * np.sin(f1 * t + 0.4), 4.1 * np.sin(f1 * t + 1.8),
                    3.1 * np.sin(f1 * t + 0.4), 3.1 * np.sin(f1 * t + 1.8),
                    2.1 * np.sin(f1 * t + 0.4), 2.1 * np.sin(f1 * t + 1.8),)).T
    y2 = np.vstack((5.0 * np.sin(f2 * t + 0.0), 5.0 * np.sin(f2 * t + 1.5),
                    4.1 * np.sin(f2 * t + 0.4), 4.1 * np.sin(f2 * t + 1.8),
                    3.1 * np.sin(f2 * t + 0.4), 3.1 * np.sin(f2 * t + 1.8),
                    2.1 * np.sin(f2 * t + 0.4), 2.1 * np.sin(f2 * t + 1.8),)).T
    y3 = np.vstack((5.0 * np.sin(f3 * t + 0.0), 5.0 * np.sin(f3 * t + 3.5),
                    4.1 * np.sin(f3 * t + 0.4), 4.1 * np.sin(f3 * t + 2.8),
                    3.1 * np.sin(f3 * t + 0.4), 3.1 * np.sin(f3 * t + 3.8),
                    2.1 * np.sin(f3 * t + 0.4), 2.1 * np.sin(f3 * t + 4.8),)).T
    ys = [y1, y2, y3]
    y = np.hstack(ys[:nparts])

    np.random.seed(0)
    # Add noise and random displacement
    y += 0.5 * np.reshape(np.random.normal(size=y.size), y.shape) \
         + 10.0 * np.random.normal(size=y.shape[1])

    parts_IDs = ([0]*8 + [1]*8 + [2]*8)[:nparts*8]
    starts_ends = [[i * tchunk, (i+1) * tchunk] for i in range(nchunks)]
    trial = MotionTrial(y, starts_ends, MotionType.TOY_SINES)
    return None, parts_IDs, trial


if __name__ == "__main__":
    load_sines()
