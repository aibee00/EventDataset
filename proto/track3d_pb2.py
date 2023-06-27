# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: track3d.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import version_pb2 as version__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='track3d.proto',
  package='tracks3d',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n\rtrack3d.proto\x12\x08tracks3d\x1a\rversion.proto\"\x1f\n\x07Point2D\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\"*\n\x07Point3D\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\x12\t\n\x01z\x18\x03 \x01(\x02\"U\n\x05\x42ox3D\x12#\n\x08top_left\x18\x01 \x01(\x0b\x32\x11.tracks3d.Point3D\x12\'\n\x0c\x62ottom_right\x18\x02 \x01(\x0b\x32\x11.tracks3d.Point3D\"3\n\x05\x42ox2D\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\x12\t\n\x01w\x18\x03 \x01(\x02\x12\t\n\x01h\x18\x04 \x01(\x02\"\xa7\x01\n\x10\x43\x61meraProjection\x12\n\n\x02id\x18\x01 \x01(\x05\x12\x13\n\x0bvideo_index\x18\x02 \x01(\x05\x12\x13\n\x0b\x66rame_index\x18\x03 \x01(\x05\x12\x13\n\x0btracklet_id\x18\x04 \x01(\x05\x12\x1c\n\x03\x62ox\x18\x05 \x01(\x0b\x32\x0f.tracks3d.Box2D\x12\x17\n\x0ftracklet_id_str\x18\x06 \x01(\t\x12\x11\n\tsit_stand\x18\x07 \x01(\x05\"*\n\x0c\x43\x61meraStream\x12\n\n\x02id\x18\x01 \x01(\x05\x12\x0e\n\x06videos\x18\x02 \x03(\t\"5\n\x05\x46loor\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0c\n\x04name\x18\x03 \x01(\t\x12\x12\n\nimage_path\x18\x02 \x01(\t\"\xf5\x06\n\x05Track\x12\x0b\n\x03pid\x18\x01 \x01(\r\x12\x0f\n\x07pid_str\x18\r \x01(\t\x12\x0b\n\x03\x64id\x18\x0c \x01(\x05\x12\x15\n\rany_multiview\x18\x02 \x01(\x08\x12\x13\n\x0bmean_height\x18\x03 \x01(\x02\x12\x13\n\x0b\x66irst_frame\x18\x04 \x01(\x05\x12\x12\n\nlast_frame\x18\x05 \x01(\x05\x12\x15\n\rn_good_frames\x18\x06 \x01(\x05\x12\x18\n\x10\x66irst_good_frame\x18\x07 \x01(\x05\x12\x17\n\x0flast_good_frame\x18\x08 \x01(\x05\x12%\n\x06\x66rames\x18\n \x03(\x0b\x32\x15.tracks3d.Track.Frame\x12\x0f\n\x07\x63\x61meras\x18\x0b \x03(\x05\x1a\xe9\x04\n\x05\x46rame\x12\r\n\x05\x66rame\x18\x01 \x01(\x05\x12\x14\n\x0cis_multiview\x18\x02 \x01(\x08\x12&\n\x0bnose_pixels\x18\x03 \x01(\x0b\x32\x11.tracks3d.Point2D\x12\"\n\x07nose_3d\x18\x04 \x01(\x0b\x32\x11.tracks3d.Point3D\x12\x1b\n\x13keypoint_confidence\x18\x05 \x01(\x02\x12\x12\n\ngood_frame\x18\x06 \x01(\x08\x12\x11\n\tgood_mask\x18\x07 \x01(\x08\x12\x1a\n\x12good_kp_confidence\x18\x08 \x01(\x08\x12\x15\n\rgood_distance\x18\t \x01(\x08\x12\x10\n\x08\x64istance\x18\n \x01(\x02\x12\x11\n\ttimestamp\x18\x0b \x01(\x02\x12/\n\x0bprojections\x18\x0c \x03(\x0b\x32\x1a.tracks3d.CameraProjection\x12\x10\n\x08\x66loor_id\x18\r \x01(\x05\x12\x12\n\nfloor_name\x18\x0e \x01(\t\x12\x11\n\tsit_stand\x18\x0f \x01(\x05\x12\x1a\n\x12measurement_source\x18\x10 \x01(\t\x12&\n\x0bmeasurement\x18\x11 \x01(\x0b\x32\x11.tracks3d.Point2D\x12\x17\n\x0fheight_estimate\x18\x12 \x01(\x02\x12>\n\x16\x64\x65tection_posture_type\x18\x13 \x01(\x0e\x32\x1e.tracks3d.DetectionPostureType\x12\x38\n\x13\x64\x65tection_view_type\x18\x14 \x01(\x0e\x32\x1b.tracks3d.DetectionViewType\x12\x12\n\nhead_angle\x18\x15 \x01(\x02\"\xfc\x01\n\x06Tracks\x12\x1f\n\x06tracks\x18\x01 \x03(\x0b\x32\x0f.tracks3d.Track\x12\'\n\x07streams\x18\x02 \x03(\x0b\x32\x16.tracks3d.CameraStream\x12\x11\n\tfloor_map\x18\x03 \x01(\t\x12\x1f\n\x06\x66loors\x18\x04 \x03(\x0b\x32\x0f.tracks3d.Floor\x12\x38\n\x0emodel_versions\x18\x05 \x01(\x0b\x32 .tracking_pipeline.ModelVersions\x12:\n\x0fmodule_versions\x18\x06 \x01(\x0b\x32!.tracking_pipeline.ModuleVersions*s\n\x14\x44\x65tectionPostureType\x12\x1d\n\x19\x44\x45TECTION_POSTURE_UNKNOWN\x10\x00\x12\x1e\n\x1a\x44\x45TECTION_POSTURE_STANDING\x10\x01\x12\x1c\n\x18\x44\x45TECTION_POSTURE_SEATED\x10\x02*{\n\x11\x44\x65tectionViewType\x12\x1a\n\x16\x44\x45TECTION_VIEW_UNKNOWN\x10\x00\x12\x18\n\x14\x44\x45TECTION_VIEW_FRONT\x10\x01\x12\x17\n\x13\x44\x45TECTION_VIEW_REAR\x10\x02\x12\x17\n\x13\x44\x45TECTION_VIEW_SIDE\x10\x03')
  ,
  dependencies=[version__pb2.DESCRIPTOR,])

_DETECTIONPOSTURETYPE = _descriptor.EnumDescriptor(
  name='DetectionPostureType',
  full_name='tracks3d.DetectionPostureType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='DETECTION_POSTURE_UNKNOWN', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='DETECTION_POSTURE_STANDING', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='DETECTION_POSTURE_SEATED', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1671,
  serialized_end=1786,
)
_sym_db.RegisterEnumDescriptor(_DETECTIONPOSTURETYPE)

DetectionPostureType = enum_type_wrapper.EnumTypeWrapper(_DETECTIONPOSTURETYPE)
_DETECTIONVIEWTYPE = _descriptor.EnumDescriptor(
  name='DetectionViewType',
  full_name='tracks3d.DetectionViewType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='DETECTION_VIEW_UNKNOWN', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='DETECTION_VIEW_FRONT', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='DETECTION_VIEW_REAR', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='DETECTION_VIEW_SIDE', index=3, number=3,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1788,
  serialized_end=1911,
)
_sym_db.RegisterEnumDescriptor(_DETECTIONVIEWTYPE)

DetectionViewType = enum_type_wrapper.EnumTypeWrapper(_DETECTIONVIEWTYPE)
DETECTION_POSTURE_UNKNOWN = 0
DETECTION_POSTURE_STANDING = 1
DETECTION_POSTURE_SEATED = 2
DETECTION_VIEW_UNKNOWN = 0
DETECTION_VIEW_FRONT = 1
DETECTION_VIEW_REAR = 2
DETECTION_VIEW_SIDE = 3



_POINT2D = _descriptor.Descriptor(
  name='Point2D',
  full_name='tracks3d.Point2D',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='x', full_name='tracks3d.Point2D.x', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='y', full_name='tracks3d.Point2D.y', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=42,
  serialized_end=73,
)


_POINT3D = _descriptor.Descriptor(
  name='Point3D',
  full_name='tracks3d.Point3D',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='x', full_name='tracks3d.Point3D.x', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='y', full_name='tracks3d.Point3D.y', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='z', full_name='tracks3d.Point3D.z', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=75,
  serialized_end=117,
)


_BOX3D = _descriptor.Descriptor(
  name='Box3D',
  full_name='tracks3d.Box3D',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='top_left', full_name='tracks3d.Box3D.top_left', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='bottom_right', full_name='tracks3d.Box3D.bottom_right', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=119,
  serialized_end=204,
)


_BOX2D = _descriptor.Descriptor(
  name='Box2D',
  full_name='tracks3d.Box2D',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='x', full_name='tracks3d.Box2D.x', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='y', full_name='tracks3d.Box2D.y', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='w', full_name='tracks3d.Box2D.w', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='h', full_name='tracks3d.Box2D.h', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=206,
  serialized_end=257,
)


_CAMERAPROJECTION = _descriptor.Descriptor(
  name='CameraProjection',
  full_name='tracks3d.CameraProjection',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='tracks3d.CameraProjection.id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='video_index', full_name='tracks3d.CameraProjection.video_index', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='frame_index', full_name='tracks3d.CameraProjection.frame_index', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='tracklet_id', full_name='tracks3d.CameraProjection.tracklet_id', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='box', full_name='tracks3d.CameraProjection.box', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='tracklet_id_str', full_name='tracks3d.CameraProjection.tracklet_id_str', index=5,
      number=6, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sit_stand', full_name='tracks3d.CameraProjection.sit_stand', index=6,
      number=7, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=260,
  serialized_end=427,
)


_CAMERASTREAM = _descriptor.Descriptor(
  name='CameraStream',
  full_name='tracks3d.CameraStream',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='tracks3d.CameraStream.id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='videos', full_name='tracks3d.CameraStream.videos', index=1,
      number=2, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=429,
  serialized_end=471,
)


_FLOOR = _descriptor.Descriptor(
  name='Floor',
  full_name='tracks3d.Floor',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='tracks3d.Floor.id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='name', full_name='tracks3d.Floor.name', index=1,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='image_path', full_name='tracks3d.Floor.image_path', index=2,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=473,
  serialized_end=526,
)


_TRACK_FRAME = _descriptor.Descriptor(
  name='Frame',
  full_name='tracks3d.Track.Frame',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='frame', full_name='tracks3d.Track.Frame.frame', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='is_multiview', full_name='tracks3d.Track.Frame.is_multiview', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='nose_pixels', full_name='tracks3d.Track.Frame.nose_pixels', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='nose_3d', full_name='tracks3d.Track.Frame.nose_3d', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='keypoint_confidence', full_name='tracks3d.Track.Frame.keypoint_confidence', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='good_frame', full_name='tracks3d.Track.Frame.good_frame', index=5,
      number=6, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='good_mask', full_name='tracks3d.Track.Frame.good_mask', index=6,
      number=7, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='good_kp_confidence', full_name='tracks3d.Track.Frame.good_kp_confidence', index=7,
      number=8, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='good_distance', full_name='tracks3d.Track.Frame.good_distance', index=8,
      number=9, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='distance', full_name='tracks3d.Track.Frame.distance', index=9,
      number=10, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='timestamp', full_name='tracks3d.Track.Frame.timestamp', index=10,
      number=11, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='projections', full_name='tracks3d.Track.Frame.projections', index=11,
      number=12, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='floor_id', full_name='tracks3d.Track.Frame.floor_id', index=12,
      number=13, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='floor_name', full_name='tracks3d.Track.Frame.floor_name', index=13,
      number=14, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sit_stand', full_name='tracks3d.Track.Frame.sit_stand', index=14,
      number=15, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='measurement_source', full_name='tracks3d.Track.Frame.measurement_source', index=15,
      number=16, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='measurement', full_name='tracks3d.Track.Frame.measurement', index=16,
      number=17, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='height_estimate', full_name='tracks3d.Track.Frame.height_estimate', index=17,
      number=18, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='detection_posture_type', full_name='tracks3d.Track.Frame.detection_posture_type', index=18,
      number=19, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='detection_view_type', full_name='tracks3d.Track.Frame.detection_view_type', index=19,
      number=20, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='head_angle', full_name='tracks3d.Track.Frame.head_angle', index=20,
      number=21, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=797,
  serialized_end=1414,
)

_TRACK = _descriptor.Descriptor(
  name='Track',
  full_name='tracks3d.Track',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='pid', full_name='tracks3d.Track.pid', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pid_str', full_name='tracks3d.Track.pid_str', index=1,
      number=13, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='did', full_name='tracks3d.Track.did', index=2,
      number=12, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='any_multiview', full_name='tracks3d.Track.any_multiview', index=3,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mean_height', full_name='tracks3d.Track.mean_height', index=4,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='first_frame', full_name='tracks3d.Track.first_frame', index=5,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='last_frame', full_name='tracks3d.Track.last_frame', index=6,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='n_good_frames', full_name='tracks3d.Track.n_good_frames', index=7,
      number=6, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='first_good_frame', full_name='tracks3d.Track.first_good_frame', index=8,
      number=7, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='last_good_frame', full_name='tracks3d.Track.last_good_frame', index=9,
      number=8, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='frames', full_name='tracks3d.Track.frames', index=10,
      number=10, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='cameras', full_name='tracks3d.Track.cameras', index=11,
      number=11, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_TRACK_FRAME, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=529,
  serialized_end=1414,
)


_TRACKS = _descriptor.Descriptor(
  name='Tracks',
  full_name='tracks3d.Tracks',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='tracks', full_name='tracks3d.Tracks.tracks', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='streams', full_name='tracks3d.Tracks.streams', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='floor_map', full_name='tracks3d.Tracks.floor_map', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='floors', full_name='tracks3d.Tracks.floors', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='model_versions', full_name='tracks3d.Tracks.model_versions', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='module_versions', full_name='tracks3d.Tracks.module_versions', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1417,
  serialized_end=1669,
)

_BOX3D.fields_by_name['top_left'].message_type = _POINT3D
_BOX3D.fields_by_name['bottom_right'].message_type = _POINT3D
_CAMERAPROJECTION.fields_by_name['box'].message_type = _BOX2D
_TRACK_FRAME.fields_by_name['nose_pixels'].message_type = _POINT2D
_TRACK_FRAME.fields_by_name['nose_3d'].message_type = _POINT3D
_TRACK_FRAME.fields_by_name['projections'].message_type = _CAMERAPROJECTION
_TRACK_FRAME.fields_by_name['measurement'].message_type = _POINT2D
_TRACK_FRAME.fields_by_name['detection_posture_type'].enum_type = _DETECTIONPOSTURETYPE
_TRACK_FRAME.fields_by_name['detection_view_type'].enum_type = _DETECTIONVIEWTYPE
_TRACK_FRAME.containing_type = _TRACK
_TRACK.fields_by_name['frames'].message_type = _TRACK_FRAME
_TRACKS.fields_by_name['tracks'].message_type = _TRACK
_TRACKS.fields_by_name['streams'].message_type = _CAMERASTREAM
_TRACKS.fields_by_name['floors'].message_type = _FLOOR
_TRACKS.fields_by_name['model_versions'].message_type = version__pb2._MODELVERSIONS
_TRACKS.fields_by_name['module_versions'].message_type = version__pb2._MODULEVERSIONS
DESCRIPTOR.message_types_by_name['Point2D'] = _POINT2D
DESCRIPTOR.message_types_by_name['Point3D'] = _POINT3D
DESCRIPTOR.message_types_by_name['Box3D'] = _BOX3D
DESCRIPTOR.message_types_by_name['Box2D'] = _BOX2D
DESCRIPTOR.message_types_by_name['CameraProjection'] = _CAMERAPROJECTION
DESCRIPTOR.message_types_by_name['CameraStream'] = _CAMERASTREAM
DESCRIPTOR.message_types_by_name['Floor'] = _FLOOR
DESCRIPTOR.message_types_by_name['Track'] = _TRACK
DESCRIPTOR.message_types_by_name['Tracks'] = _TRACKS
DESCRIPTOR.enum_types_by_name['DetectionPostureType'] = _DETECTIONPOSTURETYPE
DESCRIPTOR.enum_types_by_name['DetectionViewType'] = _DETECTIONVIEWTYPE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Point2D = _reflection.GeneratedProtocolMessageType('Point2D', (_message.Message,), {
  'DESCRIPTOR' : _POINT2D,
  '__module__' : 'track3d_pb2'
  # @@protoc_insertion_point(class_scope:tracks3d.Point2D)
  })
_sym_db.RegisterMessage(Point2D)

Point3D = _reflection.GeneratedProtocolMessageType('Point3D', (_message.Message,), {
  'DESCRIPTOR' : _POINT3D,
  '__module__' : 'track3d_pb2'
  # @@protoc_insertion_point(class_scope:tracks3d.Point3D)
  })
_sym_db.RegisterMessage(Point3D)

Box3D = _reflection.GeneratedProtocolMessageType('Box3D', (_message.Message,), {
  'DESCRIPTOR' : _BOX3D,
  '__module__' : 'track3d_pb2'
  # @@protoc_insertion_point(class_scope:tracks3d.Box3D)
  })
_sym_db.RegisterMessage(Box3D)

Box2D = _reflection.GeneratedProtocolMessageType('Box2D', (_message.Message,), {
  'DESCRIPTOR' : _BOX2D,
  '__module__' : 'track3d_pb2'
  # @@protoc_insertion_point(class_scope:tracks3d.Box2D)
  })
_sym_db.RegisterMessage(Box2D)

CameraProjection = _reflection.GeneratedProtocolMessageType('CameraProjection', (_message.Message,), {
  'DESCRIPTOR' : _CAMERAPROJECTION,
  '__module__' : 'track3d_pb2'
  # @@protoc_insertion_point(class_scope:tracks3d.CameraProjection)
  })
_sym_db.RegisterMessage(CameraProjection)

CameraStream = _reflection.GeneratedProtocolMessageType('CameraStream', (_message.Message,), {
  'DESCRIPTOR' : _CAMERASTREAM,
  '__module__' : 'track3d_pb2'
  # @@protoc_insertion_point(class_scope:tracks3d.CameraStream)
  })
_sym_db.RegisterMessage(CameraStream)

Floor = _reflection.GeneratedProtocolMessageType('Floor', (_message.Message,), {
  'DESCRIPTOR' : _FLOOR,
  '__module__' : 'track3d_pb2'
  # @@protoc_insertion_point(class_scope:tracks3d.Floor)
  })
_sym_db.RegisterMessage(Floor)

Track = _reflection.GeneratedProtocolMessageType('Track', (_message.Message,), {

  'Frame' : _reflection.GeneratedProtocolMessageType('Frame', (_message.Message,), {
    'DESCRIPTOR' : _TRACK_FRAME,
    '__module__' : 'track3d_pb2'
    # @@protoc_insertion_point(class_scope:tracks3d.Track.Frame)
    })
  ,
  'DESCRIPTOR' : _TRACK,
  '__module__' : 'track3d_pb2'
  # @@protoc_insertion_point(class_scope:tracks3d.Track)
  })
_sym_db.RegisterMessage(Track)
_sym_db.RegisterMessage(Track.Frame)

Tracks = _reflection.GeneratedProtocolMessageType('Tracks', (_message.Message,), {
  'DESCRIPTOR' : _TRACKS,
  '__module__' : 'track3d_pb2'
  # @@protoc_insertion_point(class_scope:tracks3d.Tracks)
  })
_sym_db.RegisterMessage(Tracks)


# @@protoc_insertion_point(module_scope)
