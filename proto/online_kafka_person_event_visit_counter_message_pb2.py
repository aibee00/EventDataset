# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: online_kafka_person_event_visit_counter_message.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import online_kafka_pb2 as online__kafka__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='online_kafka_person_event_visit_counter_message.proto',
  package='tracking_pipeline.online.kafka',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n5online_kafka_person_event_visit_counter_message.proto\x12\x1etracking_pipeline.online.kafka\x1a\x12online_kafka.proto\"\xe3\x01\n\x1ePersonEventVisitCounterMessage\x12\x32\n\x04uuid\x18\x01 \x01(\x0b\x32$.tracking_pipeline.online.kafka.UUID\x12\x45\n\x0emessage_header\x18\x02 \x01(\x0b\x32-.tracking_pipeline.online.kafka.MessageHeader\x12\x46\n\x05\x65vent\x18\x08 \x01(\x0b\x32\x37.tracking_pipeline.online.kafka.PersonEventVisitCounterb\x06proto3'
  ,
  dependencies=[online__kafka__pb2.DESCRIPTOR,])




_PERSONEVENTVISITCOUNTERMESSAGE = _descriptor.Descriptor(
  name='PersonEventVisitCounterMessage',
  full_name='tracking_pipeline.online.kafka.PersonEventVisitCounterMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='uuid', full_name='tracking_pipeline.online.kafka.PersonEventVisitCounterMessage.uuid', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='message_header', full_name='tracking_pipeline.online.kafka.PersonEventVisitCounterMessage.message_header', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='event', full_name='tracking_pipeline.online.kafka.PersonEventVisitCounterMessage.event', index=2,
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
  serialized_start=110,
  serialized_end=337,
)

_PERSONEVENTVISITCOUNTERMESSAGE.fields_by_name['uuid'].message_type = online__kafka__pb2._UUID
_PERSONEVENTVISITCOUNTERMESSAGE.fields_by_name['message_header'].message_type = online__kafka__pb2._MESSAGEHEADER
_PERSONEVENTVISITCOUNTERMESSAGE.fields_by_name['event'].message_type = online__kafka__pb2._PERSONEVENTVISITCOUNTER
DESCRIPTOR.message_types_by_name['PersonEventVisitCounterMessage'] = _PERSONEVENTVISITCOUNTERMESSAGE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

PersonEventVisitCounterMessage = _reflection.GeneratedProtocolMessageType('PersonEventVisitCounterMessage', (_message.Message,), {
  'DESCRIPTOR' : _PERSONEVENTVISITCOUNTERMESSAGE,
  '__module__' : 'online_kafka_person_event_visit_counter_message_pb2'
  # @@protoc_insertion_point(class_scope:tracking_pipeline.online.kafka.PersonEventVisitCounterMessage)
  })
_sym_db.RegisterMessage(PersonEventVisitCounterMessage)


# @@protoc_insertion_point(module_scope)
