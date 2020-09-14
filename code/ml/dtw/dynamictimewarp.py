import numpy as np
import unittest
import matplotlib.pyplot as plt

def dynTimeWarp(observed,  latent,  latentStddev=None,  wndlen=1000,  transitions=[(1,  1,  np.log(2.0)),  (0,  1,  np.log(2.0))]):
    """
    Dynamic time warping of observed onto latent using 
    quadratic error function with latentStddev if supplied.
    wndlen: maximum window size
    transitions: list of tuples (dObserved,  dLatent,  cost) with stepsizes 
    along observed and latent trajectory, and cost for each possible transition. 
    Default: match latent and observed, or skip a latent point.
    Distance errors are only accumulated if an observed point is matched 
    to a latent point (i.e. for a transition like (1, 1, ...)), otherwise only path costs are accumulated.
    returns: minimal distance and index associations between x and y
    """
    distances = np.ones((len(latent) + 1, len(observed) + 1)) * np.inf
    distanceIdx = np.zeros((len(latent) + 1, len(observed) + 1), dtype="object")

    distances[0, 0] = 0.0
    skipLatent = list(filter(lambda tr:tr[:2] == (0, 1), transitions))
    if len(skipLatent) == 1:  # special case: skipped points in the latent trajectory at the beginning of the interval
        distances[1:, 0] = np.arange(1, len(latent) + 1, 1) * skipLatent[0][-1]
    elif len(skipLatent) > 1: 
        raise RuntimeError("More than one latent skipping transition detected -- this case is not implemented")

    wndlen = max(wndlen, abs(len(latent) - len(observed)))
    if latentStddev is None: 
        latentStddev = np.ones(latent.shape)

    # Find optimal distance
    for i in range(1, len(latent) + 1):
        jmin, jmax = max(1, i - wndlen), min(len(observed), i + wndlen) + 1
        for j in range(jmin, jmax):
            curDists = []
            for tr in transitions:
                curDists += [distances[i - tr[1], j - tr[0]] + tr[2]]
                if tr[:2] == (1, 1):
                    curDists[-1] += np.linalg.norm((observed[j-1] - latent[i-1]) / latentStddev[i-1])**2

            minDist = min(curDists)
            distances[i, j] = minDist
            distanceIdx[i, j] = transitions[curDists.index(minDist)]

    # Backtracking to find the optimal path
    i = len(latent)
    j = len(observed)
    optPath = []
    while i > 0 and j > 0:
        dj, di = distanceIdx[i, j][:2]
        if (dj, di) == (1, 1):
            optPath += [(i-1, j-1)]
        i -= di
        j -= dj

    optPath.reverse()

    while len(optPath) > 0 and optPath[0][1] < 0:
        optPath = optPath[1:]  # prune skipped points in latent traj. at beginning of interval
    while len(optPath) > 1 and optPath[-2][1] == len(observed) - 1: 
        optPath = optPath[:-1]  # prune skipped points latent traj.  at end of interval

    return distances[-1, -1], optPath
    

class TestDynTimeWarp(unittest.TestCase):
    def drawData(self, noiseStddev):
        # Target signal
        t = np.arange(0, 15.0, 0.1)
        x = np.vstack([np.sin(2 * np.pi * t / 5.0), np.sin(2.0 * np.pi * t / 2.0) + t]).T
        x[:, 0] *= 3

        # Noisy observed signal
        y = x[9:-40]
        ridx = np.random.random((len(y), )) < 0.3
        y = y[ridx, :]
        y += np.random.normal(0.0, noiseStddev, size=y.shape)
        ty = np.arange(0.0, 0.1 * (len(y) + 10), 0.1)[:len(y)]

        return t, x, ty, y
    

    def testWarping(self):
        """
        Test if time-warping of a subsampled reference trajectory onto the reference trajectory works with different amounts of noise. 
        This test has stochastic components and should work in about 90% of the cases
        """
        for noiseStddev in [1e-6, 0.1, 0.3]:
            ttarget, target, ty, y = self.drawData(noiseStddev)
            transitions = [(1, 1, 0.0), (0, 1, 0.0)]
            #transitions = [(1, 1, 0.0), (0, 1, 1.0), (1, 0, 1.0)]
            odi, op = dynTimeWarp(observed=y, latent=target, transitions=transitions)

            self.assertTrue(odi < noiseStddev ** 2 * len(y) * 2.56)

            plt.clf()
            for ci in range(2):
                plt.subplot(2, 1, ci + 1)
                plt.hold(True)
                plt.plot(ttarget, target[:, ci], linewidth=3, label="target")
                plt.plot(ty, y[:, ci], linewidth=2.0, marker="o", linestyle="--", label="signal")

                wy = np.array([y[i[1]] for i in op])
                wt = [ttarget[i[0]] for i in op]
                plt.plot(wt, wy[:, ci], color="red", marker="o", label="opt. warp")
                plt.legend()
            plt.suptitle("Noise stddev {}, distance {}".format(noiseStddev, odi))
            plt.savefig("dtwtest{0:f}.png".format(noiseStddev))




if __name__ == "__main__":

    suite = unittest.TestLoader().loadTestsFromTestCase(TestDynTimeWarp)
    unittest.TextTestRunner(verbosity=2).run(suite)

