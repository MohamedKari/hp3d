# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: hp3d/rpc/human_pose_detection_service.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='hp3d/rpc/human_pose_detection_service.proto',
  package='com.porsche.realtery.humanpose',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n+hp3d/rpc/human_pose_detection_service.proto\x12\x1e\x63om.porsche.realtery.humanpose\"*\n\x14StartSessionResponse\x12\x12\n\nsession_id\x18\x01 \x01(\x05\"z\n\rDetectRequest\x12\x12\n\nsession_id\x18\x01 \x01(\x05\x12\x10\n\x08\x66rame_id\x18\x02 \x01(\x05\x12\x14\n\x0c\x66ocal_length\x18\x03 \x01(\x02\x12\r\n\x05\x66rame\x18\x04 \x01(\x0c\x12\x1e\n\x16request_visualizations\x18\x05 \x01(\x08\"\x9f\x01\n\x0e\x44\x65tectResponse\x12\x12\n\nsession_id\x18\x01 \x01(\x05\x12\x10\n\x08\x66rame_id\x18\x02 \x01(\x05\x12\x33\n\x05poses\x18\x03 \x03(\x0b\x32$.com.porsche.realtery.humanpose.Pose\x12\x18\n\x10visualization_2d\x18\x04 \x01(\x0c\x12\x18\n\x10visualization_3d\x18\x05 \x01(\x0c\"\x8c\x01\n\x08Keypoint\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0f\n\x07visible\x18\x02 \x01(\x08\x12\x0b\n\x03p_x\x18\x03 \x01(\x05\x12\x0b\n\x03p_y\x18\x04 \x01(\x05\x12\x0f\n\x07p_score\x18\x05 \x01(\x02\x12\x0b\n\x03\x63_x\x18\x06 \x01(\x05\x12\x0b\n\x03\x63_y\x18\x07 \x01(\x05\x12\x0b\n\x03\x63_z\x18\x08 \x01(\x05\x12\x0f\n\x07\x63_score\x18\t \x01(\x02\"X\n\x04Pose\x12\x13\n\x0btracking_id\x18\x01 \x01(\x05\x12;\n\tkeypoints\x18\x02 \x03(\x0b\x32(.com.porsche.realtery.humanpose.Keypoint\"(\n\x12StopSessionRequest\x12\x12\n\nsession_id\x18\x01 \x01(\x05\"\x07\n\x05\x45mpty2\xb2\x03\n\x12HumanPoseDetection\x12V\n\x04Ping\x12%.com.porsche.realtery.humanpose.Empty\x1a%.com.porsche.realtery.humanpose.Empty\"\x00\x12m\n\x0cStartSession\x12%.com.porsche.realtery.humanpose.Empty\x1a\x34.com.porsche.realtery.humanpose.StartSessionResponse\"\x00\x12i\n\x06\x44\x65tect\x12-.com.porsche.realtery.humanpose.DetectRequest\x1a..com.porsche.realtery.humanpose.DetectResponse\"\x00\x12j\n\x0bStopSession\x12\x32.com.porsche.realtery.humanpose.StopSessionRequest\x1a%.com.porsche.realtery.humanpose.Empty\"\x00\x62\x06proto3'
)




_STARTSESSIONRESPONSE = _descriptor.Descriptor(
  name='StartSessionResponse',
  full_name='com.porsche.realtery.humanpose.StartSessionResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='session_id', full_name='com.porsche.realtery.humanpose.StartSessionResponse.session_id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
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
  serialized_start=79,
  serialized_end=121,
)


_DETECTREQUEST = _descriptor.Descriptor(
  name='DetectRequest',
  full_name='com.porsche.realtery.humanpose.DetectRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='session_id', full_name='com.porsche.realtery.humanpose.DetectRequest.session_id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='frame_id', full_name='com.porsche.realtery.humanpose.DetectRequest.frame_id', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='focal_length', full_name='com.porsche.realtery.humanpose.DetectRequest.focal_length', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='frame', full_name='com.porsche.realtery.humanpose.DetectRequest.frame', index=3,
      number=4, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='request_visualizations', full_name='com.porsche.realtery.humanpose.DetectRequest.request_visualizations', index=4,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
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
  serialized_start=123,
  serialized_end=245,
)


_DETECTRESPONSE = _descriptor.Descriptor(
  name='DetectResponse',
  full_name='com.porsche.realtery.humanpose.DetectResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='session_id', full_name='com.porsche.realtery.humanpose.DetectResponse.session_id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='frame_id', full_name='com.porsche.realtery.humanpose.DetectResponse.frame_id', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='poses', full_name='com.porsche.realtery.humanpose.DetectResponse.poses', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='visualization_2d', full_name='com.porsche.realtery.humanpose.DetectResponse.visualization_2d', index=3,
      number=4, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='visualization_3d', full_name='com.porsche.realtery.humanpose.DetectResponse.visualization_3d', index=4,
      number=5, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
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
  serialized_start=248,
  serialized_end=407,
)


_KEYPOINT = _descriptor.Descriptor(
  name='Keypoint',
  full_name='com.porsche.realtery.humanpose.Keypoint',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='com.porsche.realtery.humanpose.Keypoint.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='visible', full_name='com.porsche.realtery.humanpose.Keypoint.visible', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='p_x', full_name='com.porsche.realtery.humanpose.Keypoint.p_x', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='p_y', full_name='com.porsche.realtery.humanpose.Keypoint.p_y', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='p_score', full_name='com.porsche.realtery.humanpose.Keypoint.p_score', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='c_x', full_name='com.porsche.realtery.humanpose.Keypoint.c_x', index=5,
      number=6, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='c_y', full_name='com.porsche.realtery.humanpose.Keypoint.c_y', index=6,
      number=7, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='c_z', full_name='com.porsche.realtery.humanpose.Keypoint.c_z', index=7,
      number=8, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='c_score', full_name='com.porsche.realtery.humanpose.Keypoint.c_score', index=8,
      number=9, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
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
  serialized_start=410,
  serialized_end=550,
)


_POSE = _descriptor.Descriptor(
  name='Pose',
  full_name='com.porsche.realtery.humanpose.Pose',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='tracking_id', full_name='com.porsche.realtery.humanpose.Pose.tracking_id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='keypoints', full_name='com.porsche.realtery.humanpose.Pose.keypoints', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=552,
  serialized_end=640,
)


_STOPSESSIONREQUEST = _descriptor.Descriptor(
  name='StopSessionRequest',
  full_name='com.porsche.realtery.humanpose.StopSessionRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='session_id', full_name='com.porsche.realtery.humanpose.StopSessionRequest.session_id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
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
  serialized_start=642,
  serialized_end=682,
)


_EMPTY = _descriptor.Descriptor(
  name='Empty',
  full_name='com.porsche.realtery.humanpose.Empty',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
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
  serialized_start=684,
  serialized_end=691,
)

_DETECTRESPONSE.fields_by_name['poses'].message_type = _POSE
_POSE.fields_by_name['keypoints'].message_type = _KEYPOINT
DESCRIPTOR.message_types_by_name['StartSessionResponse'] = _STARTSESSIONRESPONSE
DESCRIPTOR.message_types_by_name['DetectRequest'] = _DETECTREQUEST
DESCRIPTOR.message_types_by_name['DetectResponse'] = _DETECTRESPONSE
DESCRIPTOR.message_types_by_name['Keypoint'] = _KEYPOINT
DESCRIPTOR.message_types_by_name['Pose'] = _POSE
DESCRIPTOR.message_types_by_name['StopSessionRequest'] = _STOPSESSIONREQUEST
DESCRIPTOR.message_types_by_name['Empty'] = _EMPTY
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

StartSessionResponse = _reflection.GeneratedProtocolMessageType('StartSessionResponse', (_message.Message,), {
  'DESCRIPTOR' : _STARTSESSIONRESPONSE,
  '__module__' : 'hp3d.rpc.human_pose_detection_service_pb2'
  # @@protoc_insertion_point(class_scope:com.porsche.realtery.humanpose.StartSessionResponse)
  })
_sym_db.RegisterMessage(StartSessionResponse)

DetectRequest = _reflection.GeneratedProtocolMessageType('DetectRequest', (_message.Message,), {
  'DESCRIPTOR' : _DETECTREQUEST,
  '__module__' : 'hp3d.rpc.human_pose_detection_service_pb2'
  # @@protoc_insertion_point(class_scope:com.porsche.realtery.humanpose.DetectRequest)
  })
_sym_db.RegisterMessage(DetectRequest)

DetectResponse = _reflection.GeneratedProtocolMessageType('DetectResponse', (_message.Message,), {
  'DESCRIPTOR' : _DETECTRESPONSE,
  '__module__' : 'hp3d.rpc.human_pose_detection_service_pb2'
  # @@protoc_insertion_point(class_scope:com.porsche.realtery.humanpose.DetectResponse)
  })
_sym_db.RegisterMessage(DetectResponse)

Keypoint = _reflection.GeneratedProtocolMessageType('Keypoint', (_message.Message,), {
  'DESCRIPTOR' : _KEYPOINT,
  '__module__' : 'hp3d.rpc.human_pose_detection_service_pb2'
  # @@protoc_insertion_point(class_scope:com.porsche.realtery.humanpose.Keypoint)
  })
_sym_db.RegisterMessage(Keypoint)

Pose = _reflection.GeneratedProtocolMessageType('Pose', (_message.Message,), {
  'DESCRIPTOR' : _POSE,
  '__module__' : 'hp3d.rpc.human_pose_detection_service_pb2'
  # @@protoc_insertion_point(class_scope:com.porsche.realtery.humanpose.Pose)
  })
_sym_db.RegisterMessage(Pose)

StopSessionRequest = _reflection.GeneratedProtocolMessageType('StopSessionRequest', (_message.Message,), {
  'DESCRIPTOR' : _STOPSESSIONREQUEST,
  '__module__' : 'hp3d.rpc.human_pose_detection_service_pb2'
  # @@protoc_insertion_point(class_scope:com.porsche.realtery.humanpose.StopSessionRequest)
  })
_sym_db.RegisterMessage(StopSessionRequest)

Empty = _reflection.GeneratedProtocolMessageType('Empty', (_message.Message,), {
  'DESCRIPTOR' : _EMPTY,
  '__module__' : 'hp3d.rpc.human_pose_detection_service_pb2'
  # @@protoc_insertion_point(class_scope:com.porsche.realtery.humanpose.Empty)
  })
_sym_db.RegisterMessage(Empty)



_HUMANPOSEDETECTION = _descriptor.ServiceDescriptor(
  name='HumanPoseDetection',
  full_name='com.porsche.realtery.humanpose.HumanPoseDetection',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=694,
  serialized_end=1128,
  methods=[
  _descriptor.MethodDescriptor(
    name='Ping',
    full_name='com.porsche.realtery.humanpose.HumanPoseDetection.Ping',
    index=0,
    containing_service=None,
    input_type=_EMPTY,
    output_type=_EMPTY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='StartSession',
    full_name='com.porsche.realtery.humanpose.HumanPoseDetection.StartSession',
    index=1,
    containing_service=None,
    input_type=_EMPTY,
    output_type=_STARTSESSIONRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='Detect',
    full_name='com.porsche.realtery.humanpose.HumanPoseDetection.Detect',
    index=2,
    containing_service=None,
    input_type=_DETECTREQUEST,
    output_type=_DETECTRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='StopSession',
    full_name='com.porsche.realtery.humanpose.HumanPoseDetection.StopSession',
    index=3,
    containing_service=None,
    input_type=_STOPSESSIONREQUEST,
    output_type=_EMPTY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_HUMANPOSEDETECTION)

DESCRIPTOR.services_by_name['HumanPoseDetection'] = _HUMANPOSEDETECTION

# @@protoc_insertion_point(module_scope)
