# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import crosszone_service_pb2 as crosszone__service__pb2


class CrossZoneServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.InsertZoneData = channel.unary_unary(
                '/tracking_pipeline.online.CrossZoneService/InsertZoneData',
                request_serializer=crosszone__service__pb2.ZoneDataRequest.SerializeToString,
                response_deserializer=crosszone__service__pb2.ZoneDataResponse.FromString,
                )


class CrossZoneServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def InsertZoneData(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_CrossZoneServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'InsertZoneData': grpc.unary_unary_rpc_method_handler(
                    servicer.InsertZoneData,
                    request_deserializer=crosszone__service__pb2.ZoneDataRequest.FromString,
                    response_serializer=crosszone__service__pb2.ZoneDataResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'tracking_pipeline.online.CrossZoneService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class CrossZoneService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def InsertZoneData(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/tracking_pipeline.online.CrossZoneService/InsertZoneData',
            crosszone__service__pb2.ZoneDataRequest.SerializeToString,
            crosszone__service__pb2.ZoneDataResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
