import logging
from io import BytesIO
from pathlib import Path
from argparse import ArgumentParser
import datetime
import time

import grpc
from PIL import Image
import cv2

from hp3d.modules.input_reader import VideoReader

from .human_pose_detection_service_pb2_grpc import HumanPoseDetectionStub
from .human_pose_detection_service_pb2 import (
    Empty,
    StartSessionResponse, 
    DetectRequest, 
    DetectResponse,
    StopSessionRequest)

class UnexpectedResponseType(Exception):
    pass

class HumanPoseDetectionClient():
    def __init__(self, ip_address: str, port: str, ping_timeout_in_seconds: float):
        self.ping_timeout_in_seconds = ping_timeout_in_seconds
        
        self.channel = grpc.insecure_channel(
            target=f"{ip_address}:{port}", 
            options=[
                ("grpc.max_send_message_length", 10_000_000),
                ("grpc.max_receive_message_length", 10_000_000),
                ("grpc.max_message_length", 10_000_000)
            ])    

        self.stub = HumanPoseDetectionStub(self.channel)
        self.Ping()

    def Ping(self):
        request = Empty()

        logging.getLogger(__name__).info("Sending Ping to server with timeout %s ...)", self.ping_timeout_in_seconds)

        response = self.stub.Ping(
            request, 
            wait_for_ready=True, 
            timeout=self.ping_timeout_in_seconds)

        if not isinstance(response, Empty):
            raise UnexpectedResponseType()

    
    def StartSession(self) -> int:
        request = Empty()
        response: StartSessionResponse = self.stub.StartSession(request)
        return response.session_id

    def Detect(self, session_id: int, frame_id: int, frame_image: Image.Image):
        frame_bytesio = BytesIO()
        frame_image.save(frame_bytesio, format="jpeg")
        frame_bytes = frame_bytesio.getvalue()

        request = DetectRequest(
            session_id=session_id,
            frame_id=frame_id,
            frame=frame_bytes,
            request_visualizations=True)
        
        response: DetectResponse = self.stub.Detect(request)

        session_id_resp, frame_id_resp, poses, visualization_2d, visualization_3d = \
            response.session_id, response.frame_id, response.poses, response.visualization_2d, response.visualization_3d

        assert session_id == session_id_resp
        assert frame_id == frame_id_resp

        return poses, visualization_2d, visualization_3d

    def StopSession(self, session_id: int) -> None:
        request = StopSessionRequest(
            session_id=session_id)

        response = self.stub.StopSession(request)
        
        if not isinstance(response, Empty):
            raise UnexpectedResponseType()


if __name__ == "__main__":

    # python -m hp3d.rpc.client raw.mp4 3.123.206.99 50053

    parser = ArgumentParser()
    
    parser.add_argument("path_to_video")
    parser.add_argument("remote_ip")
    parser.add_argument("remote_port")
    parser.add_argument("--timeout", default=60, type=int)

    args = parser.parse_args()

    human_pose_detection_client = HumanPoseDetectionClient(
        args.remote_ip,
        args.remote_port,
        60)

    _session_id = human_pose_detection_client.StartSession()
    _timestamp = str(datetime.datetime.fromtimestamp(time.time())).replace(":", "_").replace(" ", "_").replace(".", "_")
    _output_dir = Path("output", _timestamp)
    _output_dir.mkdir(exist_ok=False, parents=True)

    frame_provider = VideoReader(args.path_to_video)
    for _i, _frame_np in enumerate(frame_provider):
        _frame_image = Image.fromarray(cv2.cvtColor(_frame_np, cv2.COLOR_RGB2BGR))

        poses, viz_2d, viz_3d = human_pose_detection_client.Detect(_session_id, _i,  _frame_image)

        Image.open(BytesIO(viz_2d)).save(str(Path(_output_dir, f"viz_2d_{_i}.jpg")))
        Image.open(BytesIO(viz_3d)).save(str(Path(_output_dir, f"viz_3d_{_i}.jpg")))

    human_pose_detection_client.StopSession(_session_id)
