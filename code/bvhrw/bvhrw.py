# Fast BVH file format reader and writer.
# Has minimal dependencies.

import numpy as np


def _write_indented(ostream, level, line): 
    if level > 0:
        ostream.write("  "*level)
    ostream.write(line + "\n")


def _strs_to_floats(lst):
    return [float(item) for item in lst]


def _floats_to_strs(lst):
    return " ".join(["{:.7g}".format(item) for item in lst])


def _hstack_nonone(lst):
    return np.hstack([i for i in lst if i is not None])


class TokenStream(object):
    def __init__(self, istream):
        self.istream = istream
        self.iline = 0
        self._tokens = []


    def is_endline(self):
        return len(self._tokens) == 0


    def get_next_token(self, pop=True):
        if self.is_endline():
            self._tokens = self.get_next_line_tokens()
            if len(self._tokens) == 0:
                return ""
            else:
                return self.get_next_token()
        if pop:
            return self._tokens.pop(0) 
        else:
            return self._tokens[0]

    def push_token(self, token):
        self._tokens.insert(0, token)


    def get_remaining_line_tokens(self):
        res = self._tokens
        self._tokens = []
        return res



    def get_next_line_tokens(self):
        assert self.is_endline()
        self.iline += 1
        s = self.istream.readline()
        return s.strip().split()




class BNFNode(object):

    def read(self, tokenstream):
        raise NotImplementedError()


    def write(self, ostream):
        raise NotImplementedError()




class TerminalNode(BNFNode):

    def __init__(self, value=None, fmt=None):
        self.value = value  
        if fmt is None:
            fmt = "{}"         
        self.fmt = fmt


    def write(self, ostream):
        ostream.write(self.fmt.format(self.value))



class ConstNode(TerminalNode):

    def read(self, tokenstream):
        s = tokenstream.get_next_token()
        assert s == self.value



class NewLineNode(TerminalNode):
    def __init__(self):
        super(NewLineNode, self).__init__("\n")

    def read(self, tokenstream):
        pass 



class VariableNode(TerminalNode):

    def read(self, tokenstream):
        self.value = tokenstream.get_next_token()


class CompoundNode(BNFNode):

    def __init__(self):
        self.parent = None
        self.children = []


    def read(self, tokenstream):
        for child in self.children:
            child.read(tokenstream)


    def write(self, ostream):
        for child in self.children:
            child.write(ostream)



HIERARCHY = ConstNode("HIERARCHY")
ROOT = ConstNode("ROOT")
JOINT = ConstNode("JOINT")
OFFSET = ConstNode("OFFSET")
CHANNELS = ConstNode("CHANNELS")
END = ConstNode("End")
SITE = ConstNode("Site")
OPEN_BRACE = ConstNode("{")
CLOSE_BRACE = ConstNode("}")
MOTION = ConstNode("MOTION")
NEWLINE = NewLineNode()


class BVHNode(CompoundNode):

    def __init__(self):
        super(BVHNode, self).__init__()
        self.name = None
        self.offset = None
        self.channels = []
        self.rotation_channels = None
        self.position_channels = None
        self.rotation_order = "ZXY"


    def read(self, tokenstream):
        s = tokenstream.get_next_token()
        if s == ROOT.value or s == JOINT.value:
            self._read_joint(tokenstream)
        elif s == END.value:
            self._read_endsite(tokenstream)


    def _read_joint(self, tokenstream):
        self.name = tokenstream.get_next_token()
        OPEN_BRACE.read(tokenstream)
        s = tokenstream.get_next_token()
        while s != CLOSE_BRACE.value:
            if s == OFFSET.value:
                self.offset = _strs_to_floats(tokenstream.get_remaining_line_tokens())
            elif s == CHANNELS.value:
                nchannels = tokenstream.get_next_token()
                self.channels = tokenstream.get_remaining_line_tokens()
            elif s == JOINT.value or s == END.value:
                tokenstream.push_token(s)
                child = BVHNode()
                child.read(tokenstream)
                child.parent = self
                self.children.append(child)
            s = tokenstream.get_next_token()


    def get_level(self):
        if self.parent is None:
            return 0
        return self.parent.get_level() + 1

    def is_root(self):
        return self.parent is None

    def is_endsite(self):
        return len(self.children) == 0

    def write(self, ostream):
        level = self.get_level()
        if self.is_root():
            _write_indented(ostream, level, "{} {}".format(ROOT.value, self.name))
        elif self.is_endsite():
            _write_indented(ostream, level, "{} {}".format(END.value, SITE.value))
        else:
            _write_indented(ostream, level, "{} {}".format(JOINT.value, self.name))

        _write_indented(ostream, level, "{")
        if self.offset is not None:
            _write_indented(ostream, level+1, "{} {}".format(OFFSET.value, _floats_to_strs(self.offset)))
        if len(self.channels) > 0:
            _write_indented(ostream, level+1, "{} {} {}".format(CHANNELS.value, len(self.channels), " ".join(self.channels)))

        for child in self.children:
            child.write(ostream)
        _write_indented(ostream, level, "}")


    def _read_endsite(self, tokenstream):
        _ = tokenstream.get_next_token()
        OPEN_BRACE.read(tokenstream)
        OFFSET.read(tokenstream)
        self.offset = _strs_to_floats(tokenstream.get_remaining_line_tokens())
        CLOSE_BRACE.read(tokenstream)


    def distribute_data_to_nodes(self, data):
        if len(self.channels) > 0:
            self.rotation_order = "".join([ch_name[0] for i, ch_name in enumerate(self.channels) if ch_name[1:] == "rotation"])
            indexes = [i for i, ch_name in enumerate(self.channels) if ch_name[1:] == "rotation"]
            if len (indexes) > 0:
                self.rotation_channels = data[:, indexes]
            indexes = [i for i, ch_name in enumerate(self.channels) if ch_name[1:] == "position"]
            if len (indexes) > 0:
                self.position_channels = data[:, indexes]
        data = data[:, len(self.channels):]
        for child in self.children:
            data = child.distribute_data_to_nodes(data)
        return data

    def collect_data_from_nodes(self):
        data = None
        if len(self.channels) > 0:
            data = np.zeros([self.rotation_channels.shape[0], len(self.channels)])
            indexes = [i for i, ch_name in enumerate(self.channels) if ch_name[1:] == "rotation"]
            if len (indexes) > 0:
                data[:, indexes] = self.rotation_channels
            indexes = [i for i, ch_name in enumerate(self.channels) if ch_name[1:] == "position"]
            if len (indexes) > 0:
                data[:, indexes] = self.position_channels
        for child in self.children:
            data = _hstack_nonone([data, child.collect_data_from_nodes()])
        return data

    def get_rotation_channels_raw(self):
        return np.copy(self.rotation_channels)

    def get_rotation_channels_ordered(self, order=None):
        if self.rotation_channels is None:
            return None
        if order is None:
            order = "XYZ"
        indexes = [self.rotation_order.find(i) for i in order]
        return self.rotation_channels[:, indexes]

    def set_rotation_channels_raw(self, value):
        self.rotation_channels = np.copy(value)

    def set_rotation_channels_ordered(self, value, order=None):
        if order is None:
            order = "XYZ"
        indexes = [order.find(i) for i in self.rotation_order]
        self.rotation_channels = value[:, indexes]




class MotionInfoNode(VariableNode):

    def __init__(self):
        super(MotionInfoNode, self).__init__()
        self.nframes = 0
        self.frametime = 1.0

    def read(self, tokenstream):
        MOTION.read(tokenstream)
        tokens = tokenstream.get_next_line_tokens()
        assert(tokens[0] == "Frames:")
        self.nframes = int(tokens[-1])
        tokens = tokenstream.get_next_line_tokens()
        assert(tokens[0] == "Frame")
        self.frametime = float(tokens[-1])

    def write(self, ostream):
        MOTION.write(ostream)
        ostream.write("\n")
        ostream.write("Frames: {}\n".format(self.nframes))
        ostream.write("Frame Time: {}\n".format(self.frametime))
        


class MotionDataNode(VariableNode):
    
    def read(self, tokenstream):
        self.value = []
        while True:
            tokens = tokenstream.get_next_line_tokens()
            if len(tokens) == 0:
                break
            self.value.append(_strs_to_floats(tokens))
        

    def write(self, ostream):
        for v in self.value:
            ostream.write(_floats_to_strs(v))
            ostream.write("\n")


class BVH(CompoundNode):

    def __init__(self, filename=None):
        super(BVH, self).__init__()
        self.root = BVHNode()
        self.motion_info = MotionInfoNode()
        self.motion_data = MotionDataNode()
        self.children.extend([HIERARCHY, NEWLINE, self.root, self.motion_info, self.motion_data])
        if filename is not None:
            self.read_file(filename)

    def read_file(self, filename):
        with open(filename, "r") as fstream:
            tokenstream = TokenStream(fstream)
            self.read(tokenstream)
        remaining = self.root.distribute_data_to_nodes(np.array(self.motion_data.value))
        assert remaining.shape[1] == 0


    def write_file(self, filename):
        self.motion_data.value = self.root.collect_data_from_nodes()
        self.motion_info.nframes = self.motion_data.value.shape[0]
        with open(filename, "w") as fstream:
            self.write(fstream)


    def read(self, tokenstream):
        super(BVH, self).read(tokenstream)


    def write(self, ostream):
        super(BVH, self).write(ostream)



if __name__ == "__main__":
    filename = "../../../data/phasespace/2016.02.26_bjoern/IC-001_skeleton.bvh"
    bvh = BVH()
    bvh.read_file(filename)
    bvh.root.distribute_data_to_nodes(0.0 * bvh.root.collect_data_from_nodes())
    bvh.write_file("out.bvh")
    
