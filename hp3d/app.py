from argparse import ArgumentParser
import json
import os
import time
import datetime
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np

from .modules.input_reader import VideoReader
from .modules.draw import Plotter3d, draw_poses
from .modules.parse_poses import parse_poses
from .modules.inference_engine_pytorch import InferenceEnginePyTorch

keypoint_names_by_id = {
        0: "neck",
        1: "nose",
        2: "torso",
        3: "l_sho",
        4: "l_elb",
        5: "l_wri",
        6: "l_hip",
        7: "l_knee",
        8: "l_ank",
        9: "r_sho",
        10: "r_elb",
        11: "r_wri",
        12: "r_hip",
        13: "r_knee",
        14: "r_ank",
        15: "r_eye",
        16: "l_eye",
        17: "r_ear",
        18: "l_ear"
    }


def get_path_compatible_date_string():
    return str(datetime.datetime.fromtimestamp(time.time())).replace(":", "_").replace(" ", "_").replace(".", "_")

def rotate_poses(poses_3d, R, t):
    R_inv = np.linalg.inv(R)
    for pose_id in range(len(poses_3d)):
        pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
        pose_3d[0:3, :] = np.dot(R_inv, pose_3d[0:3, :] - t)
        poses_3d[pose_id] = pose_3d.transpose().reshape(-1)

    return poses_3d

def get_poses_struct(poses_2d, poses_3d, tracking_ids) -> List[Dict[str, Dict[str, Any]]]:
    num_poses = poses_2d.shape[0]
    num_keypoints = 18
    num_unused_keypoints = 1

    poses_list = list()
    
    for pose_2d, pose_3d, tracking_id in zip(poses_2d, poses_3d, tracking_ids):
        current_pose = dict()
        current_pose["tracking_id"] = tracking_id
        keypoints_2d = pose_2d[:-1].reshape((-1, 3))
        keypoints_3d = pose_3d.reshape((-1, 4))
        
        for keypoint_id, (keypoint_2d, keypoint_3d) in enumerate(zip(keypoints_2d, keypoints_3d)):
            p_x, p_y, p_score = score_2d = keypoint_2d # p = pixel space
            c_x, c_y, c_z, c_score = keypoint_3d # c = camera space
            
            current_pose[keypoint_names_by_id[keypoint_id]] = {
                "visible": bool(p_score != -1),
                "p_x": float(p_x),
                "p_y": float(p_y),
                "p_score": float(p_score),
                "c_x": float(c_x),
                "c_y": float(c_y),
                "c_z": float(c_z),
                "c_score": float(c_score)
            }

        poses_list.append(current_pose)
    
    return poses_list


class Hp3dDetection():

    def __init__(self, 
            scaled_input_image: np.ndarray, 
            poses_struct: list,
            pixel_space_image: np.ndarray,
            camera_space_image: np.ndarray):
        
        self.scaled_input_image = scaled_input_image
        self.poses_struct = poses_struct 
        self.pixel_space_image = pixel_space_image
        self.camera_space_image = camera_space_image
        

    def save(self, output_dir: Path, frame_id: int):
        cv2.imwrite(
            os.path.join(output_dir, f"input_{frame_id:04}.jpg"), 
            self.scaled_input_image)

        Path(output_dir, f"poses_{frame_id:04}.json").write_text(
             json.dumps(self.poses_struct, indent=4))   

        cv2.imwrite(
            os.path.join(output_dir, f"pixel_space_{frame_id:04}.jpg"),
            self.pixel_space_image)

        cv2.imwrite(
            os.path.join(output_dir, f"camera_space_{frame_id:04}.jpg"),
            self.camera_space_image)
         


class Hp3dSession():


    def __init__(
            self, 
            device: str = "cuda:0", 
            path_to_model: str = "human-pose-estimation-3d.pth", 
            base_height: int = 256, 
            path_to_extrinsics = 'hp3d/data/extrinsics.json',
            fx: np.float32 = -1): 
        """
        device: str. E. g. cpu, cuda:0, cuda:1, ...
        path_to_model: Path to PyTorch pth file
        fx: Camera focal length.
        """
        self.stride = 8
        self.base_height = base_height
        self.fx = fx

        self.net = InferenceEnginePyTorch(path_to_model, device)

        self.canvas_3d_size = (720, 1280, 3)
        self.plotter = Plotter3d(self.canvas_3d_size)
        
        extrinsics = json.loads(Path(path_to_extrinsics).read_text())
        self.R = np.array(extrinsics['R'], dtype=np.float32)
        self.t = np.array(extrinsics['t'], dtype=np.float32)
        
        self.timestamp = get_path_compatible_date_string()
        self.mean_time = 0
    
    def detect(
            self, 
            frame_id: int, 
            frame_np: np.array, 
            request_visualizations: bool = True) -> Hp3dDetection:
        current_time = cv2.getTickCount()

        input_scale = self.base_height / frame_np.shape[0]
        scaled_img = cv2.resize(frame_np, dsize=None, fx=input_scale, fy=input_scale)
        scaled_img = scaled_img[:, 0:scaled_img.shape[1] - (scaled_img.shape[1] % self.stride)]  # better to pad, but cut out for demo

        if self.fx < 0:  # Focal length is unknown
            fx = np.float32(0.8 * frame_np.shape[1])

        inference_result = self.net.infer(scaled_img)
        poses_3d, poses_2d, tracking_ids = parse_poses(inference_result, input_scale, self.stride, fx, is_video=True)
        
        poses_struct = get_poses_struct(poses_2d, poses_3d, tracking_ids)
        
        canvas_3d = None
        viz_2d = None
        if request_visualizations:
            canvas_3d = np.zeros(self.canvas_3d_size, dtype=np.uint8)

            edges = []
            if len(poses_3d):
                poses_3d = rotate_poses(poses_3d, self.R, self.t)
                poses_3d_copy = poses_3d.copy()
                x = poses_3d_copy[:, 0::4]
                y = poses_3d_copy[:, 1::4]
                z = poses_3d_copy[:, 2::4]
                poses_3d[:, 0::4], poses_3d[:, 1::4], poses_3d[:, 2::4] = -z, x, -y

                poses_3d = poses_3d.reshape(poses_3d.shape[0], 19, -1)[:, :, 0:3]
                edges = (Plotter3d.SKELETON_EDGES + 19 * np.arange(poses_3d.shape[0]).reshape((-1, 1, 1))).reshape((-1, 2))
            self.plotter.plot(canvas_3d, poses_3d, edges)

            viz_2d = frame_np
            draw_poses(viz_2d, poses_2d)

            # Write FPS to image
            current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()
            if self.mean_time == 0:
                self.mean_time = current_time
            else:
                self.mean_time = self.mean_time * 0.95 + current_time * 0.05
            cv2.putText(viz_2d, 'FPS: {}'.format(int(1 / self.mean_time * 10) / 10),
                        (40, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))

        return Hp3dDetection(scaled_img, poses_struct, viz_2d, canvas_3d)

    def stop_session(self):
        del self.net

def run(path_to_video, path_to_output_dir, device):
    hp3d_session = Hp3dSession(device, "human-pose-estimation-3d.pth")
    output_dir = Path(path_to_output_dir, hp3d_session.timestamp)
    output_dir.mkdir(exist_ok=False, parents=False)

    frame_provider = VideoReader(path_to_video)
    for i, frame in enumerate(frame_provider):
        hp3d_detection = hp3d_session.detect(i, frame)
        hp3d_detection.save(output_dir, i)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path_to_video")
    parser.add_argument("path_to_output_dir")
    parser.add_argument("--device", default="cuda:0", choices=[f"cuda:{i}" for i in range(0, 8)])
    
    args = parser.parse_args()
    run(args.path_to_video, args.path_to_output_dir, args.device)
