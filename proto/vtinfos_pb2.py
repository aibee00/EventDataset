# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: vtinfos.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='vtinfos.proto',
  package='mall.vtinfos',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\rvtinfos.proto\x12\x0cmall.vtinfos\"1\n\x07VTInfos\x12&\n\x08vt_infos\x18\x01 \x03(\x0b\x32\x14.mall.vtinfos.VTInfo\"\xb3\x01\n\x06VTInfo\x12\n\n\x02vt\x18\x01 \x01(\t\x12\x17\n\x0fpred_pid_labels\x18\x02 \x03(\x05\x12\x1e\n\x16pred_pid_probabilities\x18\x03 \x03(\x02\x12\x12\n\nreid_feats\x18\x04 \x03(\x02\x12\x0b\n\x03pid\x18\x05 \x01(\t\x12\x15\n\rhead_features\x18\x06 \x03(\x02\x12\x12\n\nmask_probs\x18\x07 \x03(\x02\x12\x18\n\x10\x66\x61\x63\x65_patches_url\x18\x08 \x01(\tb\x06proto3')
)




_VTINFOS = _descriptor.Descriptor(
  name='VTInfos',
  full_name='mall.vtinfos.VTInfos',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='vt_infos', full_name='mall.vtinfos.VTInfos.vt_infos', index=0,
      number=1, type=11, cpp_type=10, label=3,
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
  serialized_start=31,
  serialized_end=80,
)


_VTINFO = _descriptor.Descriptor(
  name='VTInfo',
  full_name='mall.vtinfos.VTInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='vt', full_name='mall.vtinfos.VTInfo.vt', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pred_pid_labels', full_name='mall.vtinfos.VTInfo.pred_pid_labels', index=1,
      number=2, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pred_pid_probabilities', full_name='mall.vtinfos.VTInfo.pred_pid_probabilities', index=2,
      number=3, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='reid_feats', full_name='mall.vtinfos.VTInfo.reid_feats', index=3,
      number=4, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pid', full_name='mall.vtinfos.VTInfo.pid', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='head_features', full_name='mall.vtinfos.VTInfo.head_features', index=5,
      number=6, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='mask_probs', full_name='mall.vtinfos.VTInfo.mask_probs', index=6,
      number=7, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='face_patches_url', full_name='mall.vtinfos.VTInfo.face_patches_url', index=7,
      number=8, type=9, cpp_type=9, label=1,
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
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=83,
  serialized_end=262,
)

_VTINFOS.fields_by_name['vt_infos'].message_type = _VTINFO
DESCRIPTOR.message_types_by_name['VTInfos'] = _VTINFOS
DESCRIPTOR.message_types_by_name['VTInfo'] = _VTINFO
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

VTInfos = _reflection.GeneratedProtocolMessageType('VTInfos', (_message.Message,), {
  'DESCRIPTOR' : _VTINFOS,
  '__module__' : 'vtinfos_pb2'
  # @@protoc_insertion_point(class_scope:mall.vtinfos.VTInfos)
  })
_sym_db.RegisterMessage(VTInfos)

VTInfo = _reflection.GeneratedProtocolMessageType('VTInfo', (_message.Message,), {
  'DESCRIPTOR' : _VTINFO,
  '__module__' : 'vtinfos_pb2'
  # @@protoc_insertion_point(class_scope:mall.vtinfos.VTInfo)
  })
_sym_db.RegisterMessage(VTInfo)


# @@protoc_insertion_point(module_scope)
