# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: store_events.proto

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
  name='store_events.proto',
  package='events',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x12store_events.proto\x12\x06\x65vents\x1a\rversion.proto\"6\n\x06Region\x12 \n\x04type\x18\x01 \x01(\x0e\x32\x12.events.RegionType\x12\n\n\x02id\x18\x02 \x01(\t\"\xc3\x01\n\x06\x45vents\x12#\n\x0cstore_events\x18\x01 \x03(\x0b\x32\r.events.Event\x12\x10\n\x08store_id\x18\x02 \x01(\t\x12\x0c\n\x04\x64\x61te\x18\x03 \x01(\t\x12\x38\n\x0emodel_versions\x18\x04 \x01(\x0b\x32 .tracking_pipeline.ModelVersions\x12:\n\x0fmodule_versions\x18\x05 \x01(\x0b\x32!.tracking_pipeline.ModuleVersions\"\x8b\x01\n\x0fInOutProperties\x12\x12\n\nin_door_id\x18\x01 \x01(\t\x12&\n\x0cin_door_type\x18\x02 \x01(\x0e\x32\x10.events.DoorType\x12\x13\n\x0bout_door_id\x18\x03 \x01(\t\x12\'\n\rout_door_type\x18\x04 \x01(\x0e\x32\x10.events.DoorType\"(\n\x13ReceptionProperties\x12\x11\n\tstaff_pid\x18\x01 \x03(\t\"5\n\x13\x43ompanionProperties\x12\x0b\n\x03pid\x18\x01 \x03(\t\x12\x11\n\tstaff_pid\x18\x02 \x03(\t\"I\n\x13\x41ttributeProperties\x12\x12\n\nproduct_id\x18\x01 \x01(\t\x12\x1e\n\x06\x61\x63tion\x18\x02 \x01(\x0e\x32\x0e.events.Action\"\xf4\x02\n\x05\x45vent\x12\n\n\x02id\x18\x01 \x01(\t\x12\x1f\n\x04type\x18\x02 \x01(\x0e\x32\x11.events.EventType\x12\x12\n\nstart_time\x18\x03 \x01(\x04\x12\x10\n\x08\x65nd_time\x18\x04 \x01(\x04\x12\x1e\n\x06region\x18\x05 \x01(\x0b\x32\x0e.events.Region\x12\x33\n\x10inout_properties\x18\x06 \x01(\x0b\x32\x17.events.InOutPropertiesH\x00\x12;\n\x14reception_properties\x18\x07 \x01(\x0b\x32\x1b.events.ReceptionPropertiesH\x00\x12;\n\x14\x63ompanion_properties\x18\x08 \x01(\x0b\x32\x1b.events.CompanionPropertiesH\x00\x12;\n\x14\x61ttribute_properties\x18\t \x01(\x0b\x32\x1b.events.AttributePropertiesH\x00\x42\x0c\n\nproperties*\xbd\x01\n\tEventType\x12\x0f\n\x0bSTORE_INOUT\x10\x00\x12\x10\n\x0cREGION_INOUT\x10\x01\x12\x10\n\x0cREGION_VISIT\x10\x02\x12\r\n\tRECEPTION\x10\x03\x12\x17\n\x13\x41SSISTIVE_RECEPTION\x10\x04\x12\x07\n\x03SIT\x10\x05\x12\r\n\tCOMPANION\x10\x06\x12\n\n\x06PASSBY\x10\x07\x12\x0b\n\x07PRODUCT\x10\x08\x12\x08\n\x04STAY\x10\t\x12\x18\n\x14INDIVIDUAL_RECEPTION\x10\n*\x81\x01\n\nRegionType\x12\t\n\x05STORE\x10\x00\x12\x0b\n\x07\x43OUNTER\x10\x01\x12\x07\n\x03\x43\x41R\x10\x02\x12\x11\n\rINTERNAL_ROOM\x10\x03\x12\x13\n\x0fINTERNAL_REGION\x10\x04\x12\x16\n\x12TRANSACTION_REGION\x10\x05\x12\x12\n\x0ePRODUCT_REGION\x10\x06*X\n\x08\x44oorType\x12\x0e\n\nFRONT_DOOR\x10\x00\x12\r\n\tBACK_DOOR\x10\x01\x12\x11\n\rINTERNAL_DOOR\x10\x02\x12\r\n\tCONNECTOR\x10\x03\x12\x0b\n\x07UNKNOWN\x10\x04*\'\n\x06\x41\x63tion\x12\t\n\x05TRYON\x10\x00\x12\x08\n\x04MOVE\x10\x01\x12\x08\n\x04TAKE\x10\x02\x62\x06proto3')
  ,
  dependencies=[version__pb2.DESCRIPTOR,])

_EVENTTYPE = _descriptor.EnumDescriptor(
  name='EventType',
  full_name='events.EventType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='STORE_INOUT', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='REGION_INOUT', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='REGION_VISIT', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RECEPTION', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ASSISTIVE_RECEPTION', index=4, number=4,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SIT', index=5, number=5,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='COMPANION', index=6, number=6,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PASSBY', index=7, number=7,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PRODUCT', index=8, number=8,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='STAY', index=9, number=9,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='INDIVIDUAL_RECEPTION', index=10, number=10,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=989,
  serialized_end=1178,
)
_sym_db.RegisterEnumDescriptor(_EVENTTYPE)

EventType = enum_type_wrapper.EnumTypeWrapper(_EVENTTYPE)
_REGIONTYPE = _descriptor.EnumDescriptor(
  name='RegionType',
  full_name='events.RegionType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='STORE', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='COUNTER', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CAR', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='INTERNAL_ROOM', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='INTERNAL_REGION', index=4, number=4,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TRANSACTION_REGION', index=5, number=5,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PRODUCT_REGION', index=6, number=6,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1181,
  serialized_end=1310,
)
_sym_db.RegisterEnumDescriptor(_REGIONTYPE)

RegionType = enum_type_wrapper.EnumTypeWrapper(_REGIONTYPE)
_DOORTYPE = _descriptor.EnumDescriptor(
  name='DoorType',
  full_name='events.DoorType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='FRONT_DOOR', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BACK_DOOR', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='INTERNAL_DOOR', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CONNECTOR', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='UNKNOWN', index=4, number=4,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1312,
  serialized_end=1400,
)
_sym_db.RegisterEnumDescriptor(_DOORTYPE)

DoorType = enum_type_wrapper.EnumTypeWrapper(_DOORTYPE)
_ACTION = _descriptor.EnumDescriptor(
  name='Action',
  full_name='events.Action',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='TRYON', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MOVE', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TAKE', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1402,
  serialized_end=1441,
)
_sym_db.RegisterEnumDescriptor(_ACTION)

Action = enum_type_wrapper.EnumTypeWrapper(_ACTION)
STORE_INOUT = 0
REGION_INOUT = 1
REGION_VISIT = 2
RECEPTION = 3
ASSISTIVE_RECEPTION = 4
SIT = 5
COMPANION = 6
PASSBY = 7
PRODUCT = 8
STAY = 9
INDIVIDUAL_RECEPTION = 10
STORE = 0
COUNTER = 1
CAR = 2
INTERNAL_ROOM = 3
INTERNAL_REGION = 4
TRANSACTION_REGION = 5
PRODUCT_REGION = 6
FRONT_DOOR = 0
BACK_DOOR = 1
INTERNAL_DOOR = 2
CONNECTOR = 3
UNKNOWN = 4
TRYON = 0
MOVE = 1
TAKE = 2



_REGION = _descriptor.Descriptor(
  name='Region',
  full_name='events.Region',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='events.Region.type', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='id', full_name='events.Region.id', index=1,
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
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=45,
  serialized_end=99,
)


_EVENTS = _descriptor.Descriptor(
  name='Events',
  full_name='events.Events',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='store_events', full_name='events.Events.store_events', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='store_id', full_name='events.Events.store_id', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='date', full_name='events.Events.date', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='model_versions', full_name='events.Events.model_versions', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='module_versions', full_name='events.Events.module_versions', index=4,
      number=5, type=11, cpp_type=10, label=1,
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
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=102,
  serialized_end=297,
)


_INOUTPROPERTIES = _descriptor.Descriptor(
  name='InOutProperties',
  full_name='events.InOutProperties',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='in_door_id', full_name='events.InOutProperties.in_door_id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='in_door_type', full_name='events.InOutProperties.in_door_type', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='out_door_id', full_name='events.InOutProperties.out_door_id', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='out_door_type', full_name='events.InOutProperties.out_door_type', index=3,
      number=4, type=14, cpp_type=8, label=1,
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
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=300,
  serialized_end=439,
)


_RECEPTIONPROPERTIES = _descriptor.Descriptor(
  name='ReceptionProperties',
  full_name='events.ReceptionProperties',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='staff_pid', full_name='events.ReceptionProperties.staff_pid', index=0,
      number=1, type=9, cpp_type=9, label=3,
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
  serialized_start=441,
  serialized_end=481,
)


_COMPANIONPROPERTIES = _descriptor.Descriptor(
  name='CompanionProperties',
  full_name='events.CompanionProperties',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='pid', full_name='events.CompanionProperties.pid', index=0,
      number=1, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='staff_pid', full_name='events.CompanionProperties.staff_pid', index=1,
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
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=483,
  serialized_end=536,
)


_ATTRIBUTEPROPERTIES = _descriptor.Descriptor(
  name='AttributeProperties',
  full_name='events.AttributeProperties',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='product_id', full_name='events.AttributeProperties.product_id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='action', full_name='events.AttributeProperties.action', index=1,
      number=2, type=14, cpp_type=8, label=1,
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
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=538,
  serialized_end=611,
)


_EVENT = _descriptor.Descriptor(
  name='Event',
  full_name='events.Event',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='events.Event.id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='type', full_name='events.Event.type', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='start_time', full_name='events.Event.start_time', index=2,
      number=3, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='end_time', full_name='events.Event.end_time', index=3,
      number=4, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='region', full_name='events.Event.region', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='inout_properties', full_name='events.Event.inout_properties', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='reception_properties', full_name='events.Event.reception_properties', index=6,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='companion_properties', full_name='events.Event.companion_properties', index=7,
      number=8, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='attribute_properties', full_name='events.Event.attribute_properties', index=8,
      number=9, type=11, cpp_type=10, label=1,
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
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='properties', full_name='events.Event.properties',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=614,
  serialized_end=986,
)

_REGION.fields_by_name['type'].enum_type = _REGIONTYPE
_EVENTS.fields_by_name['store_events'].message_type = _EVENT
_EVENTS.fields_by_name['model_versions'].message_type = version__pb2._MODELVERSIONS
_EVENTS.fields_by_name['module_versions'].message_type = version__pb2._MODULEVERSIONS
_INOUTPROPERTIES.fields_by_name['in_door_type'].enum_type = _DOORTYPE
_INOUTPROPERTIES.fields_by_name['out_door_type'].enum_type = _DOORTYPE
_ATTRIBUTEPROPERTIES.fields_by_name['action'].enum_type = _ACTION
_EVENT.fields_by_name['type'].enum_type = _EVENTTYPE
_EVENT.fields_by_name['region'].message_type = _REGION
_EVENT.fields_by_name['inout_properties'].message_type = _INOUTPROPERTIES
_EVENT.fields_by_name['reception_properties'].message_type = _RECEPTIONPROPERTIES
_EVENT.fields_by_name['companion_properties'].message_type = _COMPANIONPROPERTIES
_EVENT.fields_by_name['attribute_properties'].message_type = _ATTRIBUTEPROPERTIES
_EVENT.oneofs_by_name['properties'].fields.append(
  _EVENT.fields_by_name['inout_properties'])
_EVENT.fields_by_name['inout_properties'].containing_oneof = _EVENT.oneofs_by_name['properties']
_EVENT.oneofs_by_name['properties'].fields.append(
  _EVENT.fields_by_name['reception_properties'])
_EVENT.fields_by_name['reception_properties'].containing_oneof = _EVENT.oneofs_by_name['properties']
_EVENT.oneofs_by_name['properties'].fields.append(
  _EVENT.fields_by_name['companion_properties'])
_EVENT.fields_by_name['companion_properties'].containing_oneof = _EVENT.oneofs_by_name['properties']
_EVENT.oneofs_by_name['properties'].fields.append(
  _EVENT.fields_by_name['attribute_properties'])
_EVENT.fields_by_name['attribute_properties'].containing_oneof = _EVENT.oneofs_by_name['properties']
DESCRIPTOR.message_types_by_name['Region'] = _REGION
DESCRIPTOR.message_types_by_name['Events'] = _EVENTS
DESCRIPTOR.message_types_by_name['InOutProperties'] = _INOUTPROPERTIES
DESCRIPTOR.message_types_by_name['ReceptionProperties'] = _RECEPTIONPROPERTIES
DESCRIPTOR.message_types_by_name['CompanionProperties'] = _COMPANIONPROPERTIES
DESCRIPTOR.message_types_by_name['AttributeProperties'] = _ATTRIBUTEPROPERTIES
DESCRIPTOR.message_types_by_name['Event'] = _EVENT
DESCRIPTOR.enum_types_by_name['EventType'] = _EVENTTYPE
DESCRIPTOR.enum_types_by_name['RegionType'] = _REGIONTYPE
DESCRIPTOR.enum_types_by_name['DoorType'] = _DOORTYPE
DESCRIPTOR.enum_types_by_name['Action'] = _ACTION
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Region = _reflection.GeneratedProtocolMessageType('Region', (_message.Message,), {
  'DESCRIPTOR' : _REGION,
  '__module__' : 'store_events_pb2'
  # @@protoc_insertion_point(class_scope:events.Region)
  })
_sym_db.RegisterMessage(Region)

Events = _reflection.GeneratedProtocolMessageType('Events', (_message.Message,), {
  'DESCRIPTOR' : _EVENTS,
  '__module__' : 'store_events_pb2'
  # @@protoc_insertion_point(class_scope:events.Events)
  })
_sym_db.RegisterMessage(Events)

InOutProperties = _reflection.GeneratedProtocolMessageType('InOutProperties', (_message.Message,), {
  'DESCRIPTOR' : _INOUTPROPERTIES,
  '__module__' : 'store_events_pb2'
  # @@protoc_insertion_point(class_scope:events.InOutProperties)
  })
_sym_db.RegisterMessage(InOutProperties)

ReceptionProperties = _reflection.GeneratedProtocolMessageType('ReceptionProperties', (_message.Message,), {
  'DESCRIPTOR' : _RECEPTIONPROPERTIES,
  '__module__' : 'store_events_pb2'
  # @@protoc_insertion_point(class_scope:events.ReceptionProperties)
  })
_sym_db.RegisterMessage(ReceptionProperties)

CompanionProperties = _reflection.GeneratedProtocolMessageType('CompanionProperties', (_message.Message,), {
  'DESCRIPTOR' : _COMPANIONPROPERTIES,
  '__module__' : 'store_events_pb2'
  # @@protoc_insertion_point(class_scope:events.CompanionProperties)
  })
_sym_db.RegisterMessage(CompanionProperties)

AttributeProperties = _reflection.GeneratedProtocolMessageType('AttributeProperties', (_message.Message,), {
  'DESCRIPTOR' : _ATTRIBUTEPROPERTIES,
  '__module__' : 'store_events_pb2'
  # @@protoc_insertion_point(class_scope:events.AttributeProperties)
  })
_sym_db.RegisterMessage(AttributeProperties)

Event = _reflection.GeneratedProtocolMessageType('Event', (_message.Message,), {
  'DESCRIPTOR' : _EVENT,
  '__module__' : 'store_events_pb2'
  # @@protoc_insertion_point(class_scope:events.Event)
  })
_sym_db.RegisterMessage(Event)


# @@protoc_insertion_point(module_scope)
