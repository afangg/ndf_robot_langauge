"""LCM type definitions
This file automatically generated by lcm.
DO NOT MODIFY BY HAND!!!!
"""

try:
    import cStringIO.StringIO as BytesIO
except ImportError:
    from io import BytesIO
import struct

import rndf_lcm.point_cloud_t

class point_cloud_array_t(object):
    __slots__ = ["num_point_clouds", "point_clouds"]

    __typenames__ = ["int32_t", "rndf_lcm.point_cloud_t"]

    __dimensions__ = [None, ["num_point_clouds"]]

    def __init__(self):
        self.num_point_clouds = 0
        self.point_clouds = []

    def encode(self):
        buf = BytesIO()
        buf.write(point_cloud_array_t._get_packed_fingerprint())
        self._encode_one(buf)
        return buf.getvalue()

    def _encode_one(self, buf):
        buf.write(struct.pack(">i", self.num_point_clouds))
        for i0 in range(self.num_point_clouds):
            assert self.point_clouds[i0]._get_packed_fingerprint() == rndf_lcm.point_cloud_t._get_packed_fingerprint()
            self.point_clouds[i0]._encode_one(buf)

    def decode(data):
        if hasattr(data, 'read'):
            buf = data
        else:
            buf = BytesIO(data)
        if buf.read(8) != point_cloud_array_t._get_packed_fingerprint():
            raise ValueError("Decode error")
        return point_cloud_array_t._decode_one(buf)
    decode = staticmethod(decode)

    def _decode_one(buf):
        self = point_cloud_array_t()
        self.num_point_clouds = struct.unpack(">i", buf.read(4))[0]
        self.point_clouds = []
        for i0 in range(self.num_point_clouds):
            self.point_clouds.append(rndf_lcm.point_cloud_t._decode_one(buf))
        return self
    _decode_one = staticmethod(_decode_one)

    _hash = None
    def _get_hash_recursive(parents):
        if point_cloud_array_t in parents: return 0
        newparents = parents + [point_cloud_array_t]
        tmphash = (0x64c4832571d653b2+ rndf_lcm.point_cloud_t._get_hash_recursive(newparents)) & 0xffffffffffffffff
        tmphash  = (((tmphash<<1)&0xffffffffffffffff) + (tmphash>>63)) & 0xffffffffffffffff
        return tmphash
    _get_hash_recursive = staticmethod(_get_hash_recursive)
    _packed_fingerprint = None

    def _get_packed_fingerprint():
        if point_cloud_array_t._packed_fingerprint is None:
            point_cloud_array_t._packed_fingerprint = struct.pack(">Q", point_cloud_array_t._get_hash_recursive([]))
        return point_cloud_array_t._packed_fingerprint
    _get_packed_fingerprint = staticmethod(_get_packed_fingerprint)

