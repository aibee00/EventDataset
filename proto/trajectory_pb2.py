# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: trajectory.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='trajectory.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x10trajectory.proto\"\xda\x01\n\x0bPersonTrack\x12\x0b\n\x03pid\x18\x01 \x01(\t\x12\x1b\n\x02xy\x18\x02 \x03(\x0b\x32\x0f.PersonTrack.XY\x1a\x1d\n\x05Point\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\x1a\x81\x01\n\x02XY\x12\x0f\n\x07\x61rea_id\x18\x01 \x01(\t\x12\x1e\n\x02pt\x18\x02 \x01(\x0b\x32\x12.PersonTrack.Point\x12\x11\n\ttimestamp\x18\x03 \x01(\x05\x12\x0f\n\x07\x63hannel\x18\x04 \x03(\t\x12&\n\nchannel_pt\x18\x05 \x03(\x0b\x32\x12.PersonTrack.Point\"[\n\nTrajectory\x12\x0c\n\x04\x64\x61te\x18\x01 \x01(\t\x12\x0f\n\x07mall_id\x18\x02 \x01(\t\x12\x11\n\tmall_name\x18\x03 \x01(\t\x12\x1b\n\x05track\x18\x04 \x03(\x0b\x32\x0c.PersonTrackb\x06proto3')
)




_PERSONTRACK_POINT = _descriptor.Descriptor(
  name='Point',
  full_name='PersonTrack.Point',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='x', full_name='PersonTrack.Point.x', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='y', full_name='PersonTrack.Point.y', index=1,
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
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=78,
  serialized_end=107,
)

_PERSONTRACK_XY = _descriptor.Descriptor(
  name='XY',
  full_name='PersonTrack.XY',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='area_id', full_name='PersonTrack.XY.area_id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pt', full_name='PersonTrack.XY.pt', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='timestamp', full_name='PersonTrack.XY.timestamp', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='channel', full_name='PersonTrack.XY.channel', index=3,
      number=4, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='channel_pt', full_name='PersonTrack.XY.channel_pt', index=4,
      number=5, type=11, cpp_type=10, label=3,
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
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=110,
  serialized_end=239,
)

_PERSONTRACK = _descriptor.Descriptor(
  name='PersonTrack',
  full_name='PersonTrack',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='pid', full_name='PersonTrack.pid', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='xy', full_name='PersonTrack.xy', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_PERSONTRACK_POINT, _PERSONTRACK_XY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=21,
  serialized_end=239,
)


_TRAJECTORY = _descriptor.Descriptor(
  name='Trajectory',
  full_name='Trajectory',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='date', full_name='Trajectory.date', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mall_id', full_name='Trajectory.mall_id', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mall_name', full_name='Trajectory.mall_name', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='track', full_name='Trajectory.track', index=3,
      number=4, type=11, cpp_type=10, label=3,
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
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=241,
  serialized_end=332,
)

_PERSONTRACK_POINT.containing_type = _PERSONTRACK
_PERSONTRACK_XY.fields_by_name['pt'].message_type = _PERSONTRACK_POINT
_PERSONTRACK_XY.fields_by_name['channel_pt'].message_type = _PERSONTRACK_POINT
_PERSONTRACK_XY.containing_type = _PERSONTRACK
_PERSONTRACK.fields_by_name['xy'].message_type = _PERSONTRACK_XY
_TRAJECTORY.fields_by_name['track'].message_type = _PERSONTRACK
DESCRIPTOR.message_types_by_name['PersonTrack'] = _PERSONTRACK
DESCRIPTOR.message_types_by_name['Trajectory'] = _TRAJECTORY
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

PersonTrack = _reflection.GeneratedProtocolMessageType('PersonTrack', (_message.Message,), {

  'Point' : _reflection.GeneratedProtocolMessageType('Point', (_message.Message,), {
    'DESCRIPTOR' : _PERSONTRACK_POINT,
    '__module__' : 'trajectory_pb2'
    # @@protoc_insertion_point(class_scope:PersonTrack.Point)
    })
  ,

  'XY' : _reflection.GeneratedProtocolMessageType('XY', (_message.Message,), {
    'DESCRIPTOR' : _PERSONTRACK_XY,
    '__module__' : 'trajectory_pb2'
    # @@protoc_insertion_point(class_scope:PersonTrack.XY)
    })
  ,
  'DESCRIPTOR' : _PERSONTRACK,
  '__module__' : 'trajectory_pb2'
  # @@protoc_insertion_point(class_scope:PersonTrack)
  })
_sym_db.RegisterMessage(PersonTrack)
_sym_db.RegisterMessage(PersonTrack.Point)
_sym_db.RegisterMessage(PersonTrack.XY)

Trajectory = _reflection.GeneratedProtocolMessageType('Trajectory', (_message.Message,), {
  'DESCRIPTOR' : _TRAJECTORY,
  '__module__' : 'trajectory_pb2'
  # @@protoc_insertion_point(class_scope:Trajectory)
  })
_sym_db.RegisterMessage(Trajectory)


# @@protoc_insertion_point(module_scope)
