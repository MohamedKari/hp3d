# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from hp3d.rpc import human_pose_detection_service_pb2 as hp3d_dot_rpc_dot_human__pose__detection__service__pb2


class HumanPoseDetectionStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Ping = channel.unary_unary(
                '/com.porsche.realtery.humanpose.HumanPoseDetection/Ping',
                request_serializer=hp3d_dot_rpc_dot_human__pose__detection__service__pb2.Empty.SerializeToString,
                response_deserializer=hp3d_dot_rpc_dot_human__pose__detection__service__pb2.Empty.FromString,
                )
        self.StartSession = channel.unary_unary(
                '/com.porsche.realtery.humanpose.HumanPoseDetection/StartSession',
                request_serializer=hp3d_dot_rpc_dot_human__pose__detection__service__pb2.Empty.SerializeToString,
                response_deserializer=hp3d_dot_rpc_dot_human__pose__detection__service__pb2.StartSessionResponse.FromString,
                )
        self.Detect = channel.unary_unary(
                '/com.porsche.realtery.humanpose.HumanPoseDetection/Detect',
                request_serializer=hp3d_dot_rpc_dot_human__pose__detection__service__pb2.DetectRequest.SerializeToString,
                response_deserializer=hp3d_dot_rpc_dot_human__pose__detection__service__pb2.DetectResponse.FromString,
                )
        self.StopSession = channel.unary_unary(
                '/com.porsche.realtery.humanpose.HumanPoseDetection/StopSession',
                request_serializer=hp3d_dot_rpc_dot_human__pose__detection__service__pb2.StopSessionRequest.SerializeToString,
                response_deserializer=hp3d_dot_rpc_dot_human__pose__detection__service__pb2.Empty.FromString,
                )


class HumanPoseDetectionServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Ping(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def StartSession(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Detect(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def StopSession(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_HumanPoseDetectionServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Ping': grpc.unary_unary_rpc_method_handler(
                    servicer.Ping,
                    request_deserializer=hp3d_dot_rpc_dot_human__pose__detection__service__pb2.Empty.FromString,
                    response_serializer=hp3d_dot_rpc_dot_human__pose__detection__service__pb2.Empty.SerializeToString,
            ),
            'StartSession': grpc.unary_unary_rpc_method_handler(
                    servicer.StartSession,
                    request_deserializer=hp3d_dot_rpc_dot_human__pose__detection__service__pb2.Empty.FromString,
                    response_serializer=hp3d_dot_rpc_dot_human__pose__detection__service__pb2.StartSessionResponse.SerializeToString,
            ),
            'Detect': grpc.unary_unary_rpc_method_handler(
                    servicer.Detect,
                    request_deserializer=hp3d_dot_rpc_dot_human__pose__detection__service__pb2.DetectRequest.FromString,
                    response_serializer=hp3d_dot_rpc_dot_human__pose__detection__service__pb2.DetectResponse.SerializeToString,
            ),
            'StopSession': grpc.unary_unary_rpc_method_handler(
                    servicer.StopSession,
                    request_deserializer=hp3d_dot_rpc_dot_human__pose__detection__service__pb2.StopSessionRequest.FromString,
                    response_serializer=hp3d_dot_rpc_dot_human__pose__detection__service__pb2.Empty.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'com.porsche.realtery.humanpose.HumanPoseDetection', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class HumanPoseDetection(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Ping(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/com.porsche.realtery.humanpose.HumanPoseDetection/Ping',
            hp3d_dot_rpc_dot_human__pose__detection__service__pb2.Empty.SerializeToString,
            hp3d_dot_rpc_dot_human__pose__detection__service__pb2.Empty.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def StartSession(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/com.porsche.realtery.humanpose.HumanPoseDetection/StartSession',
            hp3d_dot_rpc_dot_human__pose__detection__service__pb2.Empty.SerializeToString,
            hp3d_dot_rpc_dot_human__pose__detection__service__pb2.StartSessionResponse.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Detect(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/com.porsche.realtery.humanpose.HumanPoseDetection/Detect',
            hp3d_dot_rpc_dot_human__pose__detection__service__pb2.DetectRequest.SerializeToString,
            hp3d_dot_rpc_dot_human__pose__detection__service__pb2.DetectResponse.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def StopSession(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/com.porsche.realtery.humanpose.HumanPoseDetection/StopSession',
            hp3d_dot_rpc_dot_human__pose__detection__service__pb2.StopSessionRequest.SerializeToString,
            hp3d_dot_rpc_dot_human__pose__detection__service__pb2.Empty.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)
