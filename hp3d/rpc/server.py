from typing import Dict
from concurrent.futures import ThreadPoolExecutor
import logging
from io import BytesIO
import signal
import sys

import grpc
from PIL import Image
import cv2
import numpy as np

from . import human_pose_detection_service_pb2_grpc

from .human_pose_detection_service_pb2 import (
    Empty, 
    DetectRequest, 
    DetectResponse, 
    Keypoint, 
    Pose, 
    StartSessionResponse, 
    StopSessionRequest)

from ..app import Hp3dSession, Hp3dDetection, keypoint_names_by_id

class RpcHandlingException(Exception):
    def __init__(self, rpc_name: str):
        super().__init__()
        self.rpc_name = rpc_name

    def __str__(self):
        error_str = (
            f" \n"
            f"Error during handling RPC with name {self.rpc_name}.\n" 
            f"Error type was: {type(self.__cause__)}\n"
            f"Error message was: {self.__cause__} \n"
        )
            
        logging.error(error_str)
        return error_str

class Hp3dServicer(human_pose_detection_service_pb2_grpc.HumanPoseDetectionServicer):

    def __init__(self):
        super().__init__()

        self.sessions: Dict[int, Hp3dSession] = {}
        self.total_session_counter: int = -1
        self.framecounter = -1

    def log_request(self, rpc_name, request, context): # pylint: disable=unused-argument
        logging.getLogger(__name__).debug(
            "Received gRPC request for method %s by peer %s with metadata %s", 
            rpc_name,
            context.peer(),
            context.invocation_metadata())

    def Ping(self, request: Empty, context) -> Empty:
        try:
            # Input
            self.log_request("Ping", request, context)

            # Process

            # Output

            return Empty()
        except Exception as e:
            raise RpcHandlingException("Ping") from e


    def StartSession(self, request: Empty, context) -> StartSessionResponse:
        try:
            # Input
            self.log_request("StartSession", request, context)
            
            # Process
            self.total_session_counter += 1
            session_id = self.total_session_counter

            self.sessions[session_id] = Hp3dSession()

            # Output
            start_session_response = StartSessionResponse(
                session_id=session_id)

            return start_session_response
        except Exception as e:
            raise RpcHandlingException("StartSession") from e


    def Detect(self, request: DetectRequest, context) -> DetectResponse:
        try:
            # Input
            ## Input Request
            self.framecounter += 1
            self.log_request("Detect", request, context)
            
            session_id: int = request.session_id
            frame_id: int = request.frame_id
            focal_length: float = request.focal_length
            frame: bytes = request.frame
            request_visualizations: bool = request.request_visualizations

            ## Input Deserialization
            logging.getLogger(__name__).debug("Frame with frame_id %s has length %s", frame_id, len(frame))
            frame_image: Image = Image.open(BytesIO(frame))
            frame_np = np.array(frame_image)
            cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR) 
           
            # Process 
            session: Hp3dSession = self.sessions[session_id]
            hp3d_detections = session.detect(frame_id, frame_np, request_visualizations=request_visualizations)

            # Output
            ## Output Serialization
            logging.getLogger(__name__).debug("Serializing hp3d_detections.")
            
            proto_poses = []
            for i, pose_dict in enumerate(hp3d_detections.poses_struct):
                proto_keypoints = []

                for keypoint_name, keypoint_values in pose_dict.items():
                    if not keypoint_name in list(keypoint_names_by_id.values()):
                        continue

                    proto_keypoints.append(
                        Keypoint(
                            name=keypoint_name,
                            visible=keypoint_values["visible"],
                            p_x=int(keypoint_values["p_x"]),
                            p_y=int(keypoint_values["p_y"]),
                            p_score=keypoint_values["p_score"],
                            c_x=int(keypoint_values["c_x"]),
                            c_y=int(keypoint_values["c_y"]),
                            c_z=int(keypoint_values["c_z"]),
                            c_score=keypoint_values["c_score"]))

                proto_poses.append(
                    Pose(
                        tracking_id=pose_dict["tracking_id"],
                        keypoints=proto_keypoints))
            
            cv2.cvtColor(hp3d_detections.pixel_space_image, cv2.COLOR_BGR2RGB)
            visualization_2d_image = Image.fromarray(hp3d_detections.pixel_space_image)            
            visualization_2d_bytesio = BytesIO()
            visualization_2d_image.save(visualization_2d_bytesio, format="jpeg")
            visualization_2d_bytes = visualization_2d_bytesio.getvalue()
            
            cv2.cvtColor(hp3d_detections.camera_space_image, cv2.COLOR_BGR2RGB)
            visualization_3d_image = Image.fromarray(hp3d_detections.camera_space_image)
            visualization_3d_bytesio = BytesIO()
            visualization_3d_image.save(visualization_3d_bytesio, format="jpeg")
            visualization_3d_bytes = visualization_3d_bytesio.getvalue()

            ## Output Response
            detect_response = DetectResponse(
                session_id=session_id,
                frame_id=frame_id,
                poses=proto_poses, 
                visualization_2d=visualization_2d_bytes,
                visualization_3d=visualization_3d_bytes)

            return detect_response
        except Exception as e:
            raise RpcHandlingException("Detect") from e


    def StopSession(self, request: StopSessionRequest, context) -> Empty:
        try:
            # Input
            self.log_request("StopSession", request, context)
            
            session_id: int = request.session_id

            # Process

            hp3d_session = self.sessions[session_id]
            hp3d_session.stop_session()

            del self.sessions[session_id]

            # Output
            stop_session_response = Empty()

            return stop_session_response
        except Exception as e:
            raise RpcHandlingException("StopSession") from e


def register_stop_signal_handler(grpc_server):

    logging.getLogger(__name__).info("Registering signal handler for server %s ", grpc_server)

    def signal_handler(signalnum, _):
        logging.getLogger(__name__).info("Processing signal %s received...", signalnum)
        grpc_server.stop(None)
        sys.exit("Exiting after cancel request.")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def serve():
    server = grpc.server(
        ThreadPoolExecutor(max_workers=1),
        options=[
            ("grpc.max_send_message_length", 10_000_000),
            ("grpc.max_receive_message_length", 10_000_000),
            ("grpc.max_message_length", 10_000_000)
        ])

    human_pose_detection_service_pb2_grpc.add_HumanPoseDetectionServicer_to_server(
        Hp3dServicer(),
        server)

    port = 50053
    server.add_insecure_port(f"[::]:{port}")
    register_stop_signal_handler(server)
    server.start()

    logging.getLogger(__name__).info("Serving 3D Human Pose Detection on port %s!", port)
    server.wait_for_termination()

if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    serve()