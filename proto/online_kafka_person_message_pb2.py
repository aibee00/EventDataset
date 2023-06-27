# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: online_kafka_person_message.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import online_kafka_pb2 as online__kafka__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='online_kafka_person_message.proto',
  package='tracking_pipeline.online.kafka',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n!online_kafka_person_message.proto\x12\x1etracking_pipeline.online.kafka\x1a\x12online_kafka.proto\"\xc2\x01\n\rPersonMessage\x12\x32\n\x04uuid\x18\x01 \x01(\x0b\x32$.tracking_pipeline.online.kafka.UUID\x12\x45\n\x0emessage_header\x18\x02 \x01(\x0b\x32-.tracking_pipeline.online.kafka.MessageHeader\x12\x36\n\x06person\x18\x08 \x01(\x0b\x32&.tracking_pipeline.online.kafka.Personb\x06proto3'
  ,
  dependencies=[online__kafka__pb2.DESCRIPTOR,])




_PERSONMESSAGE = _descriptor.Descriptor(
  name='PersonMessage',
  full_name='tracking_pipeline.online.kafka.PersonMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='uuid', full_name='tracking_pipeline.online.kafka.PersonMessage.uuid', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='message_header', full_name='tracking_pipeline.online.kafka.PersonMessage.message_header', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='person', full_name='tracking_pipeline.online.kafka.PersonMessage.person', index=2,
      number=8, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=90,
  serialized_end=284,
)

_PERSONMESSAGE.fields_by_name['uuid'].message_type = online__kafka__pb2._UUID
_PERSONMESSAGE.fields_by_name['message_header'].message_type = online__kafka__pb2._MESSAGEHEADER
_PERSONMESSAGE.fields_by_name['person'].message_type = online__kafka__pb2._PERSON
DESCRIPTOR.message_types_by_name['PersonMessage'] = _PERSONMESSAGE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

PersonMessage = _reflection.GeneratedProtocolMessageType('PersonMessage', (_message.Message,), {
  'DESCRIPTOR' : _PERSONMESSAGE,
  '__module__' : 'online_kafka_person_message_pb2'
  # @@protoc_insertion_point(class_scope:tracking_pipeline.online.kafka.PersonMessage)
  })
_sym_db.RegisterMessage(PersonMessage)


# @@protoc_insertion_point(module_scope)
