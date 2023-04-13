from yacs.config import CfgNode as CN 

_C = CN()

_C.CAMERA_NAME_PREFIX = 'cam_'
# _C.SERIAL_NUMBERS = ['143122065292']
# _C.SERIAL_NUMBERS = ['843112073228']

# # two that Alina and Ant have been using for a while
# _C.SERIAL_NUMBERS = [
#     '143122065292',
#     '843112073228']

# # two that Alina and Ant have been using for a while + Dan's
# _C.SERIAL_NUMBERS = [
#     '143122065292',
#     '843112073228',
#     '215122255998']

# two that Alina and Ant have been using for a while + Dan's + new D-415 (long mount)
_C.SERIAL_NUMBERS = [
    '143122065292', # back right
    '843112073228', # front right bottom
    '215122255998', # fron right top
    '840412060551'] # side

# Dan's
# _C.SERIAL_NUMBERS = [
#     '215122255998']

#_C.SERIAL_NUMBERS = [
#    '023322060722', # right corner
#    '045422061046', # front
#    '810512062562', # left corner
#]

_C.RGB_LCM_TOPIC_NAME_SUFFIX = 'rgb_image'
_C.DEPTH_LCM_TOPIC_NAME_SUFFIX = 'depth_image'
_C.INFO_LCM_TOPIC_NAME_SUFFIX = 'info'
_C.POSE_LCM_TOPIC_NAME_SUFFIX = 'pose'

_C.WIDTH = 640
_C.HEIGHT = 480
_C.FRAME_RATE = 30

def get_default_multi_realsense_cfg():
    return _C.clone()
