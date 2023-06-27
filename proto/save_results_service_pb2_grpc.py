# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import save_results_service_pb2 as save__results__service__pb2


class SaveResultsServiceStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.savepb = channel.unary_unary(
        '/tracking_pipeline.online.SaveResultsService/savepb',
        request_serializer=save__results__service__pb2.singleview_tracks_results.SerializeToString,
        response_deserializer=save__results__service__pb2.SaveResultsResponse.FromString,
        )


class SaveResultsServiceServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def savepb(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_SaveResultsServiceServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'savepb': grpc.unary_unary_rpc_method_handler(
          servicer.savepb,
          request_deserializer=save__results__service__pb2.singleview_tracks_results.FromString,
          response_serializer=save__results__service__pb2.SaveResultsResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'tracking_pipeline.online.SaveResultsService', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
