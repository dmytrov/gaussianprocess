import numpy as np
import numerical.numpytheano as nt
import theano.tensor as tt


class EulerOrder(object):
    ZXY = 0  # ZXYv
    YXZ = 1  # YXZv
    XYZ = 2  # XYZv


def multiply_matrices(matrices_a, matrices_b, ns=nt.TheanoLinalg):
    """
    Matrix-wise multiplication
    :param matrices_a: N matrices [N*Q*W]
    :param matrices_b: N matrices [N*W*E]
    :return: N matrices [N*Q*E] 
    """
    ax = matrices_a[:, :, :, np.newaxis]  # [N*4*4*4]
    bx = matrices_b[:, np.newaxis, :, :]  # [N*4*4*4]
    matrices_a_times_b = ns.sum(ax*bx, axis=2)  # [N*4*4]
    return matrices_a_times_b


def quaternions(angles, axes_normalized, ns=nt.TheanoLinalg):
    """
    Quaternions from angles and normalized rotation axes
    :param angles: vector [N]
    :param axes_normalized: matrix [N*3]
    :return: [N*4]
    """
    if ns == nt.TheanoLinalg:
        return tt.concatenate([tt.sin(0.5*angles)[:, np.newaxis] * axes_normalized, 
                               tt.cos(0.5*angles)[:, np.newaxis]], axis=1)
    else:
        return np.concatenate([np.sin(0.5*angles)[:, np.newaxis] * axes_normalized, 
                               np.cos(0.5*angles)[:, np.newaxis]], axis=1)

def quaternions_conjugate(quats, ns=nt.TheanoLinalg):
    return quats * np.array([-1.0, -1.0, -1.0, 1.0])[np.newaxis, :]    

def quaternions_inverse(quats, ns=nt.TheanoLinalg):
    return quaternions_conjugate(quats) / ns.sum(quats**2, axis=1)[:, np.newaxis]

def quaternions_product(quats1, quats2, ns=nt.TheanoLinalg):
    a = quats1[:, :, np.newaxis] * quats2[:, np.newaxis, :]
    x = a[:, 3, 0] + a[:, 0, 3] + a[:, 1, 2] - a[:, 2, 1] 
    y = a[:, 3, 1] + a[:, 1, 3] + a[:, 2, 0] - a[:, 0, 2] 
    z = a[:, 3, 2] + a[:, 2, 3] + a[:, 0, 1] - a[:, 1, 0] 
    w = a[:, 3, 3] - a[:, 0, 0] - a[:, 1, 1] - a[:, 2, 2]
    if ns == nt.TheanoLinalg:
        return tt.stacklists([x, y, z, w]).T
    else:
        return np.vstack((x, y, z, w)).T

def quaternions_difference(quats1, quats2, ns=nt.TheanoLinalg):
    return quaternions_product(quaternions_inverse(quats1, ns), quats2, ns)

def quaternions_to_angles_axes(quats, ns=nt.TheanoLinalg):
    xyz = quats[:, 0:3]
    w = quats[:, 3]
    if ns == nt.TheanoLinalg:
        mag = tt.sqrt(tt.sum(xyz**2, axis=1))
        angles = 2.0 * tt.arccos(w)
        axes = xyz / mag[:, np.newaxis]
    else:
        mag = np.sqrt(np.sum(xyz**2, axis=1))
        #angles = 2.0 * np.arctan2(mag, w)
        angles = 2.0 * np.arccos(w)
        axes = xyz / mag[:, np.newaxis]
    return angles, axes

def quaternion(angle, axis_normalized):
    """
    Quaternion from angl and normalized rotation axis
    :param angle: scalar [1]
    :param axis_normalized: vector [3]
    :return: [4]
    """
    return tt.concatenate([tt.sin(0.5*angle) * axis_normalized, tt.cos(0.5*angle)[np.newaxis]])


def quaternions_to_expmaps(quats):
    """
    :param quats: matrix [N*4]
    :return: exponantial maps [N*3]
    """
    angles, axes = quaternions_to_angles_axes(quats, ns=nt.NumpyLinalg)
    expmaps = angles[:, np.newaxis] * axes
    expmaps[np.isnan(expmaps)] = 0.0
    return expmaps


def expmaps_to_quaternions(expmaps, ns=nt.TheanoLinalg):
    """
    :param expmaps: matrix [N*3]
    :return: quaternions [N*4]
    """
    lengths = ns.sqrt(ns.sum(expmaps**2, axis=1))  # [N]
    return quaternions(angles=lengths, axes_normalized=expmaps / lengths[:, np.newaxis], ns=ns)


def expmap_to_quaternion(expmap):
    """
    :param expmap: vector [3]
    :return: quaternion [4]
    """
    length = tt.sqrt(tt.sum(expmap**2))  # [1]
    return quaternion(angle=length, axis_normalized=expmap / length)

def expmaps_remove_discontinuities(expmaps, threshold=None):
    """
    Removes 2Pi discontinuities
    :param expmaps: matrix [N*3]
    :return: exponential maps [N*4]
    """
    if threshold is None:
        threshold = 1.0
    res = np.copy(expmaps)
    diffs = np.concatenate([[0], np.sqrt(np.sum((expmaps[1:, :] - expmaps[:-1, :])**2, axis=1))])
    peaks = diffs > threshold
    i = 1
    n = expmaps.shape[0]
    windowsize = 10
    while i < n:
        if peaks[i]:
            k = i + 1
            while np.any(diffs[k:k+windowsize] > 0.5*threshold) and k < n-windowsize-1:
                k += 1
            if k < n:
                vd = -(expmaps[k] - expmaps[i-1])
                vd_length = np.sqrt(np.sum(vd**2))
                vd_normalized = vd / vd_length
                correction_factor = np.floor_divide((vd_length + np.pi), 2*np.pi) * 2*np.pi
                vd_correction = correction_factor * vd_normalized
                res[i:, :] += vd_correction
                res[i:k, 0] = np.linspace(res[i-1, 0], res[k, 0], k-i)
                res[i:k, 1] = np.linspace(res[i-1, 1], res[k, 1], k-i)
                res[i:k, 2] = np.linspace(res[i-1, 2], res[k, 2], k-i)
            i = k
        else:
            i += 1
    return res


def quaternions_to_matrices(quats, ns=nt.TheanoLinalg):
    """
    :param quats: [N*4]
    :return: [N*3*3]
    """
    assert quats.ndim == 2
    x = quats[:, 0]  # [N*1]
    y = quats[:, 1]  # [N*1]
    z = quats[:, 2]  # [N*1]
    w = quats[:, 3]  # [N*1]
    m = ns.stacklists([[1.0 - 2.0*(y**2 + z**2),              2.0*(x*y - z*w),              2.0*(x*z + y*w)],
                       [        2.0*(x*y + z*w),      1.0 - 2.0*(x**2 + z**2),              2.0*(y*z - x*w)],
                       [        2.0*(x*z - y*w),              2.0*(y*z + x*w),      1.0 - 2.0*(x**2 + y**2)]])  # [3*3*N]
    return ns.dimshuffle(m, [2, 0, 1])  # [N*3*3]


def quaternion_to_matrix(quat):
    """
    :param quat: [4]
    :return: [3*3]
    """
    assert quat.ndim == 1
    x = quat[0]  # [1]
    y = quat[1]  # [1]
    z = quat[2]  # [1]
    w = quat[3]  # [1]
    m = tt.stacklists([[1.0 - 2.0*(y**2 + z**2),              2.0*(x*y - z*w),              2.0*(x*z + y*w)],
                       [        2.0*(x*y + z*w),      1.0 - 2.0*(x**2 + z**2),              2.0*(y*z - x*w)],
                       [        2.0*(x*z - y*w),              2.0*(y*z + x*w),      1.0 - 2.0*(x**2 + y**2)]])  # [3*3]
    return m
    

def quaternions_to_ZXY_euler_angles(quats):
    """
    For ZXYv order as in BVH file format
    See https://www.geometrictools.com/Documentation/EulerAngles.pdf
    :param quats: [N*4]
    :return: ZXY rotation angles [N*3] 
    """
    assert quats.ndim == 2
    m = quaternions_to_matrices(quats, ns=nt.NumpyLinalg)
    anglesY = np.arctan2(-m[:, 2, 0], m[:, 2, 2])
    anglesX = np.arcsin(m[:, 2, 1])
    anglesZ = np.arctan2(-m[:, 0, 1], m[:, 1, 1])
    return np.vstack([anglesZ, anglesX, anglesY]).T


def quaternions_to_YXZ_euler_angles(quats):
    """
    For YXZv order
    See https://www.geometrictools.com/Documentation/EulerAngles.pdf
    :param quats: [N*4]
    :return: YXZ rotation angles [N*3] 
    """
    assert quats.ndim == 2
    m = quaternions_to_matrices(quats, ns=nt.NumpyLinalg)
    anglesZ = np.arctan2(m[:, 1, 0], m[:, 1, 1])
    anglesX = np.arcsin(-m[:, 1, 2])
    anglesY = np.arctan2(m[:, 0, 2], m[:, 2, 2])
    return np.vstack([anglesY, anglesX, anglesZ]).T


def quaternions_to_XYZ_euler_angles(quats):
    """
    For YXZv order
    See https://www.geometrictools.com/Documentation/EulerAngles.pdf
    :param quats: [N*4]
    :return: YXZ rotation angles [N*3] 
    """
    assert quats.ndim == 2
    m = quaternions_to_matrices(quats, ns=nt.NumpyLinalg)
    anglesZ = np.arctan2(-m[:, 0, 1], m[:, 0, 0])
    anglesY = np.arcsin(m[:, 0, 2])
    anglesX = np.arctan2(-m[:, 1, 2], m[:, 2, 2])
    return np.vstack([anglesX, anglesY, anglesZ]).T


def quaternions_to_euler_angles(quats, rotmode=EulerOrder.ZXY):
    """
    See https://www.geometrictools.com/Documentation/EulerAngles.pdf
    """
    if rotmode == EulerOrder.ZXY:
        return quaternions_to_ZXY_euler_angles(quats)
    elif rotmode == EulerOrder.YXZ:
        return quaternions_to_YXZ_euler_angles(quats)
    elif rotmode == EulerOrder.XYZ:
        return quaternions_to_XYZ_euler_angles(quats)


class RotMode(object):
    raw = 0  # any unstructured
    exponential = 1
    matrix = 2
    quaternion = 3
    euler = 4  # vector order is XYZ, rotation order is ZXYv. In degrees. For BVH
    euler_ydifference = 5  # vector order is XYZ, rotation order is ZXYv. In radians. Differences in Y axis. For the relative root case


class TranslMode(object):
    raw = 0
    absolute = 1  # in cm
    difference = 2  # in 0.1m
    basis_difference = 3  # in 0.1m. Relative translation w.r.t the previous position/rotation basis. For the relative root case


def euler_to_quats(euler, order=EulerOrder.ZXY):
    qx = quaternions(euler[:, 0], np.tile(np.array([1.0, 0.0, 0.0]), [euler.shape[0], 1]), ns=nt.NumpyLinalg)
    qy = quaternions(euler[:, 1], np.tile(np.array([0.0, 1.0, 0.0]), [euler.shape[0], 1]), ns=nt.NumpyLinalg)
    qz = quaternions(euler[:, 2], np.tile(np.array([0.0, 0.0, 1.0]), [euler.shape[0], 1]), ns=nt.NumpyLinalg)
    if order == EulerOrder.ZXY:
        quats = quaternions_product(qz, quaternions_product(qx, qy, ns=nt.NumpyLinalg), ns=nt.NumpyLinalg)
    elif order == EulerOrder.YXZ:
        quats = quaternions_product(qy, quaternions_product(qx, qz, ns=nt.NumpyLinalg), ns=nt.NumpyLinalg)
    return quats


def quats_to_euler(quats, order=EulerOrder.ZXY):
    if order == EulerOrder.ZXY:
        euler = quaternions_to_ZXY_euler_angles(quats)
        euler = np.vstack([euler[:, 1], euler[:, 2], euler[:, 0]]).T  # ZXY order to XYZ order
    elif order == EulerOrder.YXZ:
        euler = quaternions_to_YXZ_euler_angles(quats)
        euler = np.vstack([euler[:, 1], euler[:, 0], euler[:, 2]]).T  # YXZ order to XYZ order
    else:
        raise NotImplementedError()
    euler[np.isnan(euler)] = 0.0
    return euler


def exponential_to_euler(expmaps, order=EulerOrder.ZXY):
    quats = expmaps_to_quaternions(expmaps, ns=nt.NumpyLinalg)
    return quats_to_euler(quats, order)


def euler_to_exponential(euler, order=EulerOrder.ZXY):
    quats = euler_to_quats(euler, order)
    expmaps = quaternions_to_expmaps(quats)
    return expmaps


def euler_to_matrices(eulerangles, order=EulerOrder.ZXY):
    expmaps = euler_to_exponential((np.pi / 180.0) * eulerangles)
    quats = expmaps_to_quaternions(expmaps, ns=nt.NumpyLinalg)
    quats[np.isnan(quats)] = 0.0
    matrices = quaternions_to_matrices(quats, ns=nt.NumpyLinalg)  # [N*3*3]
    return matrices


def difference_relative_basis_translation(eulerangles, positions):
    matrices = euler_to_matrices(eulerangles)
    A = np.zeros([matrices.shape[0], 4, 4])
    A[:, :3, :3] = matrices
    A[:, 3, 3] = 1.0
    A[:, :3, 3] = positions
    Ainv = np.linalg.inv(A)
    AinvA = np.sum(Ainv[:-1, :, :, np.newaxis] * A[1:, np.newaxis, :, :], axis=-2)
    diffs = AinvA[:, :3, 3]
    diffs = np.vstack([diffs[0, :], diffs])
    return diffs


def integrate_relative_basis_translation(eulerangles, translations, startpoint=None):
    if startpoint is None:
        startpoint = np.array([0.0, 0.0, 0.0])
    matrices = euler_to_matrices(eulerangles)
    translations_global_basis = np.sum(matrices[:-1, :, :] * translations[1:, np.newaxis, :], axis=-1)
    translations_global_basis = np.vstack([startpoint, translations_global_basis])
    positions = np.cumsum(translations_global_basis, axis=0)
    return positions


def eulerZXYv_to_eulerYXZv(eulerZXY):
    return quats_to_euler(euler_to_quats(eulerZXY, EulerOrder.ZXY), EulerOrder.YXZ)


def eulerYXZv_to_eulerZXYv(eulerYXZ):
    return quats_to_euler(euler_to_quats(eulerYXZ, EulerOrder.YXZ), EulerOrder.ZXY)
