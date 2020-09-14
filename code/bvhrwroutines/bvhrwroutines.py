import numpy as np
import numerical.numpytheano as nt
import bvhrw.bvhrw as bvh
import transform3d.transformations as tr


def get_node_position_channels(node):
    return node.position_channels


def get_node_position_offset(node):
    position = node.offset
    return np.array(position)


def get_node_rotation_channels(node):
    return node.get_rotation_channels_ordered()


def all_nodes_to_leafs(node):
    nodes = []
    if node.position_channels is not None or node.rotation_channels is not None:
        nodes.append(node)
    for child in node.children:
        nodes.extend(all_nodes_to_leafs(child))
    return nodes

def create_motion_channels(node, parent_channel=None, include_endsites=False):
    channels = []
    if node.position_channels is not None or node.rotation_channels is not None or include_endsites:
        channel = MotionChannel(node.name, parent_channel, node)
        channels.append(channel)
        for child in node.children:
            channels.extend(create_motion_channels(child, channel, include_endsites))
    return channels


class MotionChannel(object):
    def __init__(self, name, parent_channel, bvhnode=None):
        self.name = name
        self.parent_channel = parent_channel
        self.bvhnode = bvhnode
        self.rotation = None
        self.rot_mode = tr.RotMode.raw
        self.translation = None
        self.transl_mode = tr.TranslMode.raw
        self.translation_startpoint = None  # in m
        self.is_root = False
        self.is_endsite = False
        if bvhnode is not None:
            self.from_bhv_node(bvhnode)

    def from_bhv_node(self, bvhnode):
        self.is_root = bvhnode.is_root()
        self.is_endsite = bvhnode.is_endsite()
        translation = get_node_position_channels(bvhnode)
        self.set_translation(translation, tr.TranslMode.absolute)
        rotation = get_node_rotation_channels(bvhnode)
        self.set_rotation(rotation, tr.RotMode.euler)
        if translation is not None:
            self.translation_startpoint = 0.01 * translation[0, :]
            self.translation_startpoint[0] = 0.0
            self.translation_startpoint[2] = 0.0

    def set_rotation(self, rotation, mode):
        self.rotation = rotation
        self.rot_mode = mode

    def get_rotation_as(self, mode):
        if self.rotation is None:
            return None
        if self.rot_mode == mode:
            return self.rotation.copy()
        if self.rot_mode == tr.RotMode.euler:
            if mode == tr.RotMode.exponential:
                return tr.euler_to_exponential((np.pi / 180.0) * self.rotation)
            elif mode == tr.RotMode.euler_ydifference:
                eulerYXZ = tr.eulerZXYv_to_eulerYXZv((np.pi / 180.0) * self.rotation)
                ydiffs = eulerYXZ[1:, 1] - eulerYXZ[:-1, 1]
                ydiffs = np.hstack([ydiffs[0], ydiffs])
                ydiffs[ydiffs > np.pi] = ydiffs[ydiffs > np.pi] - 2.0 * np.pi
                ydiffs[ydiffs < -np.pi] = ydiffs[ydiffs < -np.pi] + 2.0 * np.pi
                res = np.vstack([eulerYXZ[:, 0], ydiffs, eulerYXZ[:, 2]]).T
                return res
        elif self.rot_mode == tr.RotMode.exponential:
            if mode == tr.RotMode.euler:
                return (180.0 / np.pi) * tr.exponential_to_euler(self.rotation)
        elif self.rot_mode == tr.RotMode.euler_ydifference:
            if mode == tr.RotMode.euler:
                y = np.cumsum(self.rotation, axis=0)[:, 1]
                eulerYXZ = np.vstack([self.rotation[:, 0], y, self.rotation[:, 2]]).T
                eulerZXY = tr.eulerYXZv_to_eulerZXYv(eulerYXZ)
                return (180.0 / np.pi) * eulerZXY
        raise NotImplementedError()

    def set_translation(self, translation, mode):
        self.translation = translation
        self.transl_mode = mode

    def get_transformation_relative(self):
        m_rot = self.get_rotation_as(tr.RotMode.euler)
        if m_rot is None:
            m_rot = np.identity(3)[np.newaxis, :, :]
        else:
            m_rot = tr.euler_to_matrices(self.get_rotation_as(tr.RotMode.euler))
        m_translation = self.get_translation_as(tr.TranslMode.absolute)
        if m_translation is None:
            m_translation = np.array(self.bvhnode.offset)
        m = np.zeros([m_rot.shape[0], 4, 4])
        m[:, :3, :3] = m_rot
        m[:, 3, 3] = 1.0
        m[:, :3, 3] = m_translation
        return m

    def get_transformation_from_root(self):
        relative_transformation = self.get_transformation_relative()
        if self.parent_channel is not None:
            parent_transformation = self.parent_channel.get_transformation_from_root()
            return tr.multiply_matrices(parent_transformation, relative_transformation, ns=nt.NumpyLinalg)
        return relative_transformation

    def get_trajectory_from_root(self):
        transformations = self.get_transformation_from_root()
        return transformations[:, :3, 3]

    def get_translation_as(self, mode):
        if self.translation is None:
            return None
        if self.transl_mode == mode:
            return self.translation
        if self.transl_mode == tr.TranslMode.absolute:
            if mode == tr.TranslMode.difference:
                res = self.translation[1:, :] - self.translation[:-1, :]
                res = np.vstack([res[0, :], res])
                return 0.1 * res
            if mode == tr.TranslMode.basis_difference:
                res = tr.difference_relative_basis_translation(self.get_rotation_as(tr.RotMode.euler),
                                                            self.translation)
                return 0.1 * res
        elif self.transl_mode == tr.TranslMode.difference:
            if mode == tr.TranslMode.absolute:
                res = np.cumsum(self.translation, axis=0)
                return 10.0 * res
        elif self.transl_mode == tr.TranslMode.basis_difference:
            if mode == tr.TranslMode.absolute:
                res = tr.integrate_relative_basis_translation(self.get_rotation_as(tr.RotMode.euler),
                                                           self.translation,
                                                           startpoint=10.0*self.translation_startpoint)
                return 10.0 * res

        raise NotImplementedError()

def remove_linear_trend(x):
    return x - np.linspace(0, x[-1] - x[0], len(x))

class MotionData(object):
    def __init__(self):
        self.bvhdata = None
        self.channels = []
        self.endsites = []

    def read_BVH(self, bvhfilename):
        bvhdata = bvh.BVH()
        bvhdata.read_file(bvhfilename)
        self.from_BVHData(bvhdata)

    def from_BVHData(self, bvhdata):
        assert isinstance(bvhdata, bvh.BVH) 
        self.bvhdata = bvhdata
        channels = create_motion_channels(self.bvhdata.root, include_endsites=True)
        self.channels = [channel for channel in channels if not channel.is_endsite]
        self.endsites = [channel for channel in channels if channel.is_endsite]

    def write_BVH(self, filename):
        N = self.channels[0].rotation.shape[0]
        for channel in self.channels:
            assert channel.translation is None or channel.translation.shape[0] == N
            assert channel.rotation is None or channel.rotation.shape[0] == N
        nodes = all_nodes_to_leafs(self.bvhdata.root)
        node = {n.name: n for n in nodes}  # name->node dictionary
        for channel in self.channels:
            if channel.translation is not None:
                translation = channel.get_translation_as(tr.TranslMode.absolute)
                if channel.is_root:
                    translation[:, 1] = remove_linear_trend(translation[:, 1])
                node[channel.name].position_channels = translation 
            if channel.rotation is not None:
                node[channel.name].set_rotation_channels_ordered(channel.get_rotation_as(tr.RotMode.euler))
        self.bvhdata.write_file(filename)


def hstack_nonone(arrays):
    return np.hstack([item for item in arrays if item is not None])


class BVH_Bridge(MotionData):
    def __init__(self):
        self.partnames = []
        self.channelnames = {}  # partname->[channelname]  dictionary

    def add_part(self, partname, channelsnames):
        self.partnames.append(partname)
        self.channelnames[partname] = channelsnames

    def _get_rot_mode(self, channel):
        if channel.is_root:
            return tr.RotMode.euler_ydifference
        else:
            return tr.RotMode.exponential

    def _get_tranls_mode(self, channel):
        if channel.is_root:
            return tr.TranslMode.basis_difference
        else:
            return tr.TranslMode.difference

    def get_part_data(self, partname):
        res = None
        channelsdict = {channel.name:channel for channel in self.channels}
        for channelname in self.channelnames[partname]:
            channel = channelsdict[channelname]
            rotation = channel.get_rotation_as(self._get_rot_mode(channel))
            res = hstack_nonone([res, rotation])
            translation = channel.get_translation_as(self._get_tranls_mode(channel))
            res = hstack_nonone([res, translation])
        return res

    def set_part_data(self, partname, data):
        channelsdict = {channel.name:channel for channel in self.channels}
        for channelname in self.channelnames[partname]:
            channel = channelsdict[channelname]
            channel.set_rotation(data[:, :3], self._get_rot_mode(channel))
            data = data[:, 3:]
            if channel.translation is not None:
                channel.set_translation(data[:, :3], self._get_tranls_mode(channel))
                data = data[:, 3:]
        return data  # remainig dimensions

    def get_all_parts_data_and_IDs(self):
        data = None
        IDs = []
        i = 0
        for partname in self.partnames:
            partdata = self.get_part_data(partname)
            ndims = partdata.shape[1]
            data = hstack_nonone([data, partdata])
            IDs.extend(ndims * [i])
            i += 1
        data[np.isnan(data)] = 0.0  # TODO: figure out why it is nan
        return data, IDs

    def set_all_parts_data(self, data):
        for partname in self.partnames:
            data = self.set_part_data(partname, data)


class BVH_Partitioner(object):
    def __init__(self, motiondata):
        assert isinstance(motiondata, MotionData)
        self.motiondata = motiondata
        self.partnames = []
        self.channelnames = {}  # partname->[channelname]  dictionary

    def add_part(self, partname, channelsnames):
        self.partnames.append(partname)
        self.channelnames[partname] = channelsnames

    def _get_rot_mode(self, channel):
        if channel.is_root:
            return tr.RotMode.euler_ydifference
        else:
            return tr.RotMode.exponential

    def _get_tranls_mode(self, channel):
        if channel.is_root:
            return tr.TranslMode.basis_difference
        else:
            return tr.TranslMode.difference

    def get_part_data(self, partname):
        res = None
        channelsdict = {channel.name:channel for channel in self.motiondata.channels}
        for channelname in self.channelnames[partname]:
            channel = channelsdict[channelname]
            rotation = channel.get_rotation_as(self._get_rot_mode(channel))
            res = hstack_nonone([res, rotation])
            translation = channel.get_translation_as(self._get_tranls_mode(channel))
            res = hstack_nonone([res, translation])
        return res

    def set_part_data(self, partname, data):
        channelsdict = {channel.name:channel for channel in self.motiondata.channels}
        for channelname in self.channelnames[partname]:
            channel = channelsdict[channelname]
            channel.set_rotation(data[:, :3], self._get_rot_mode(channel))
            data = data[:, 3:]
            if channel.translation is not None:
                channel.set_translation(data[:, :3], self._get_tranls_mode(channel))
                data = data[:, 3:]
        return data  # remainig dimensions

    def get_all_parts_data_and_IDs(self):
        data = None
        IDs = []
        i = 0
        for partname in self.partnames:
            partdata = self.get_part_data(partname)
            ndims = partdata.shape[1]
            data = hstack_nonone([data, partdata])
            IDs.extend(ndims * [i])
            i += 1
        data[np.isnan(data)] = 0.0  # TODO: figure out why it is nan
        return data, IDs

    def set_all_parts_data(self, data):
        for partname in self.partnames:
            data = self.set_part_data(partname, data)


def IDs_to_indexes(parts_IDs):
    nparts = np.max(parts_IDs) + 1
    return [np.array([j for j, index in enumerate(parts_IDs) if i == index]) for i in range(nparts)]

if __name__ == "__main__":
    N = 10000
    eulerangles = 460 * np.reshape(np.random.uniform(-1.0, 1.0, size=N*3), [N, 3])
    q1 = tr.euler_to_quats(np.pi / 180.0 * eulerangles)
    q2 = tr.euler_to_quats(tr.quats_to_euler(q1))
    diff = 0.5 * np.sqrt(np.sum((q1-q2)**2, axis=1))
    diff[np.abs(diff - 0.0) < 1.0e-10] = 0.0
    diff[np.abs(diff - 1.0) < 1.0e-10] = 0.0
    assert np.sum(diff) == 0.0
    
    N = 10000
    expmaps1 = 2*np.pi * np.reshape(np.random.uniform(-1.0, 1.0, size=N*3), [N, 3])
    eulerangles = tr.exponential_to_euler(expmaps1)
    expmaps2 = tr.euler_to_exponential(eulerangles)
    diff = np.sqrt(np.sum((expmaps1-expmaps2)**2, axis=1)) / (2.0 * np.pi)
    diff[np.abs(diff - 0.0) < 1.0e-10] = 0.0
    diff[np.abs(diff - 1.0) < 1.0e-10] = 0.0
    diff[np.abs(diff - 2.0) < 1.0e-10] = 0.0
    assert np.sum(diff) == 0.0

    N = 10000
    eulerangles = np.reshape(np.random.uniform(-180.0, 180.0, size=N*3), [N, 3])
    positions = np.reshape(np.random.uniform(-1.0, 1.0, size=N*3), [N, 3])
    translations = tr.difference_relative_basis_translation(eulerangles, positions)
    path = tr.integrate_relative_basis_translation(eulerangles, translations, startpoint=positions[0, :])
    assert np.allclose(path, positions)

    bvhfilename = "../../../data/phasespace/2016.05.03_bjoern/015_skeleton.bvh"
    md = MotionData()
    md.read_BVH(bvhfilename)
    for endsite in md.endsites:
        trajectory = endsite.get_trajectory_from_root()
        
    x = np.array([1.0, 2.0, 3.0, 4.0, 9.0])
    x = remove_linear_trend(x)
    eulerangles = np.array([[0.0, 0.0, 0.0],
                            [10.0, 20.0, 30.0],
                            [40.0, 50.0, 60.0],
                            [70.0, 80.0, 90.0]])
    positions = np.array([[0.0, 0.0, 0.0],
                          [1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0],
                          [7.0, 8.0, 9.0]])
    translations = tr.difference_relative_basis_translation(eulerangles, positions)
    translations[0, :] = 0
    path = tr.integrate_relative_basis_translation(eulerangles, translations)

    eulerYXZ = tr.eulerZXYv_to_eulerYXZv((np.pi / 180.0) * eulerangles)
    eulerZXY = (180.0 / np.pi) * tr.eulerYXZv_to_eulerZXYv(eulerYXZ)


    bvhfilename = "../../../data/phasespace/2016.05.03_bjoern/003_skeleton.bvh"
    md = MotionData()
    md.read_BVH(bvhfilename)
    md.write_BVH("test_in.bvh")
    for channel in md.channels:
        channel.rotation = np.vstack([channel.rotation, channel.rotation])
        if channel.translation is not None:
            channel.translation = np.vstack([channel.translation, channel.translation])
            diffs = channel.get_translation_as(tr.TranslMode.difference)
            channel.set_translation(diffs, tr.TranslMode.difference)
        expmap = channel.get_rotation_as(tr.RotMode.exponential)
        channel.set_rotation(2.0 * expmap, tr.RotMode.exponential)
    md.write_BVH("test_out1.bvh")

    motion = BVH_Bridge()
    motion.read_BVH(bvhfilename)
    motion.add_part("upper", ["pelvis_spine1",
                          "spine1",
                          "spine2",
                          "neck",
                          "right_manubrium",
                          "right_clavicle",
                          "right_humerus",
                          "right_radius",
                          "left_manubrium",
                          "left_clavicle",
                          "left_humerus",
                          "left_radius",
                          ])
    motion.add_part("lower", ["pelvis",
                          "pelvis_right_femur",
                          "right_femur_tibia",
                          "right_tibia_foot",
                          "pelvis_left_femur",
                          "left_femur_tibia",
                          "left_tibia_foot",
                          ])

    y, indexes = motion.get_all_parts_data_and_IDs()
    motion.set_all_parts_data(y)
    motion.write_BVH("test_out2.bvh")

