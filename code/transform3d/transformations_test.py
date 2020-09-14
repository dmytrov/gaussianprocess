from transform3d.transformations import *


def exmaps_to_matrices(exmaps):
    quats = expmaps_to_quaternions(exmaps, ns=nt.NumpyLinalg)
    matrices = quaternions_to_matrices(quats, ns=nt.NumpyLinalg)
    return matrices

def angles_to_matrices(axis, angles):
    exmaps = np.tile(axis, [angles.size, 1]) * angles[:, np.newaxis]
    return exmaps_to_matrices(exmaps)

def test_euler_angles(quaternions_to_ZXY_euler_angles_function):
    axis = np.array([[1.0, 0.0, 0.0]])
    angles = np.pi / 180 * 10 * np.arange(10)
    exmaps = np.tile(axis, [angles.size, 1]) * angles[:, np.newaxis]
    quats = expmaps_to_quaternions(exmaps, ns=nt.NumpyLinalg)
    res = 180.0 / np.pi * quaternions_to_ZXY_euler_angles_function(quats)
    print("Rotation along X axis:")
    print(res)

    axis = np.array([[0.0, 1.0, 0.0]])
    angles = np.pi / 180 * 10 * np.arange(10)
    exmaps = np.tile(axis, [angles.size, 1]) * angles[:, np.newaxis]
    quats = expmaps_to_quaternions(exmaps, ns=nt.NumpyLinalg)
    res = 180.0 / np.pi * quaternions_to_ZXY_euler_angles_function(quats)
    print("Rotation along Y axis:")
    print(res)

    axis = np.array([[0.0, 0.0, 1.0]])
    angles = np.pi / 180 * 10 * np.arange(10)
    exmaps = np.tile(axis, [angles.size, 1]) * angles[:, np.newaxis]
    quats = expmaps_to_quaternions(exmaps, ns=nt.NumpyLinalg)
    res = 180.0 / np.pi * quaternions_to_ZXY_euler_angles_function(quats)
    print("Rotation along Z axis:")
    print(res)

    exmaps = 0.1 * np.array([[1, 2, 3],
                             [4, -5, 6],
                             [-7, 8, -9],
                             [-5, -2, 1]])
    quats = expmaps_to_quaternions(exmaps, ns=nt.NumpyLinalg)
    ZXY_euler_angles = quaternions_to_ZXY_euler_angles_function(quats)

    angles_Z = ZXY_euler_angles[:,0]
    angles_X = ZXY_euler_angles[:,1]
    angles_Y = ZXY_euler_angles[:,2]

    matrices_X = angles_to_matrices(axis=np.array([[1.0, 0.0, 0.0]]), angles = angles_X)
    matrices_Y = angles_to_matrices(axis=np.array([[0.0, 1.0, 0.0]]), angles = angles_Y)
    matrices_Z = angles_to_matrices(axis=np.array([[0.0, 0.0, 1.0]]), angles = angles_Z)

    mZXY = multiply_matrices(matrices_Z, multiply_matrices(matrices_X, matrices_Y, ns=nt.NumpyLinalg), ns=nt.NumpyLinalg)
    print(mZXY - exmaps_to_matrices(exmaps))

def test_quaternions_difference():
    axis = np.array([[1.0, 0.0, 0.0]])
    angles = np.pi / 180 * 10 * np.arange(1, 90)
    exmaps = np.tile(axis, [angles.size, 1]) * angles[:, np.newaxis]
    quats = expmaps_to_quaternions(exmaps, ns=nt.NumpyLinalg)
    quats_diff = quaternions_difference(quats[0, :][np.newaxis, :], quats, ns=nt.NumpyLinalg)
    an, ax = quaternions_to_angles_axes(quats_diff, ns=nt.NumpyLinalg)
    an_mod = np.mod(an + np.pi, 2.0 * np.pi) - np.pi
    print(np.hstack((ax, an[:, np.newaxis], an_mod[:, np.newaxis])))


if __name__ == "__main__":
    test_quaternions_difference()
    test_euler_angles(quaternions_to_ZXY_euler_angles)