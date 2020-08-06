from argparse import ArgumentParser
import json
import os
import time
import datetime
from pathlib import Path
from time import timezone
from PIL import Image

import cv2
import numpy as np

from .modules.input_reader import VideoReader, ImageReader
from .modules.draw import Plotter3d, draw_poses
from .modules.parse_poses import parse_poses

def get_path_compatible_date_string():
    return str(datetime.datetime.fromtimestamp(time.time())).replace(":", "_").replace(" ", "_").replace(".", "_")

def rotate_poses(poses_3d, R, t):
    R_inv = np.linalg.inv(R)
    for pose_id in range(len(poses_3d)):
        pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
        pose_3d[0:3, :] = np.dot(R_inv, pose_3d[0:3, :] - t)
        poses_3d[pose_id] = pose_3d.transpose().reshape(-1)

    return poses_3d

def get_poses_struct(poses_2d, poses_3d, tracking_ids):
    num_poses = poses_2d.shape[0]
    num_keypoints = 18
    num_unused_keypoints = 1

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

class ProgrammaticArgs():
    model = "human-pose-estimation-3d.pth"
    device = "cuda:0"
    output_path = "/share"

class Hp3dSession():

    def __init__(self, path_to_model: str, device: str, output_path: str):
        pass

    def detect(self, frame_id: int, frame_image: Image.Image):
        pass


def run():
    parser = ArgumentParser(description='Lightweight 3D human pose estimation demo. '
                                        'Press esc to exit, "p" to (un)pause video or process next image.')
    parser.add_argument('-m', '--model',
                        help='Required. Path to checkpoint with a trained model '
                             '(or an .xml file in case of OpenVINO inference).',
                        type=str, required=True)
    parser.add_argument('--video', help='Optional. Path to video file or camera id.', type=str, default='')
    parser.add_argument('-d', '--device',
                        help='Optional. Specify the target device to infer on: CPU or GPU. '
                             'The demo will look for a suitable plugin for device specified '
                             '(by default, it is GPU).',
                        type=str, default='GPU')
    parser.add_argument('--use-openvino',
                        help='Optional. Run network with OpenVINO as inference engine. '
                             'CPU, GPU, FPGA, HDDL or MYRIAD devices are supported.',
                        action='store_true')
    parser.add_argument('--images', help='Optional. Path to input image(s).', nargs='+', default='')
    parser.add_argument('--height-size', help='Optional. Network input layer height size.', type=int, default=256)
    parser.add_argument('-o', '--output-path', help="Optional. Path to output image(s)", default=".")
    parser.add_argument('--extrinsics-path',
                        help='Optional. Path to file with camera extrinsics.',
                        type=str, default=None)
    parser.add_argument('--fx', type=np.float32, default=-1, help='Optional. Camera focal length.')
    args = parser.parse_args()

    if args.video == '' and args.images == '':
        raise ValueError('Either --video or --image has to be provided')

    stride = 8
    if args.use_openvino:
        from modules.inference_engine_openvino import InferenceEngineOpenVINO
        net = InferenceEngineOpenVINO(args.model, args.device)
    else:
        from .modules.inference_engine_pytorch import InferenceEnginePyTorch
        net = InferenceEnginePyTorch(args.model, args.device)

    canvas_3d = np.zeros((720, 1280, 3), dtype=np.uint8)
    plotter = Plotter3d(canvas_3d.shape[:2])

    file_path = args.extrinsics_path
    if file_path is None:
        file_path = os.path.join('hp3d', 'data', 'extrinsics.json')
    with open(file_path, 'r') as f:
        extrinsics = json.load(f)
    R = np.array(extrinsics['R'], dtype=np.float32)
    t = np.array(extrinsics['t'], dtype=np.float32)

    frame_provider = ImageReader(args.images)
    is_video = False
    if args.video != '':
        frame_provider = VideoReader(args.video)
        is_video = True
    base_height = args.height_size
    fx = args.fx

    output_dir = Path(args.output_path, get_path_compatible_date_string())
    output_dir.mkdir(exist_ok=False, parents=False)
    delay = 1
    esc_code = 27
    p_code = 112
    space_code = 32
    mean_time = 0
    for i, frame in enumerate(frame_provider):
        
        current_time = cv2.getTickCount()
        if frame is None:
            break
        input_scale = base_height / frame.shape[0]
        scaled_img = cv2.resize(frame, dsize=None, fx=input_scale, fy=input_scale)
        scaled_img = scaled_img[:, 0:scaled_img.shape[1] - (scaled_img.shape[1] % stride)]  # better to pad, but cut out for demo
        if fx < 0:  # Focal length is unknown
            fx = np.float32(0.8 * frame.shape[1])

        scaled_img_path = os.path.join(output_dir, f"input_{i:04}.jpg")
        cv2.imwrite(scaled_img_path, scaled_img)

        inference_result = net.infer(scaled_img)
        poses_3d, poses_2d, tracking_ids = parse_poses(inference_result, input_scale, stride, fx, is_video)
        
        poses_struct = get_poses_struct(poses_2d, poses_3d, tracking_ids)
        
        with open(f"poses_{i:04}.json", "wt") as json_file:
            json.dump(poses_struct, json_file, indent=4)
        
        edges = []
        if len(poses_3d):
            poses_3d = rotate_poses(poses_3d, R, t)
            poses_3d_copy = poses_3d.copy()
            x = poses_3d_copy[:, 0::4]
            y = poses_3d_copy[:, 1::4]
            z = poses_3d_copy[:, 2::4]
            poses_3d[:, 0::4], poses_3d[:, 1::4], poses_3d[:, 2::4] = -z, x, -y

            poses_3d = poses_3d.reshape(poses_3d.shape[0], 19, -1)[:, :, 0:3]
            edges = (Plotter3d.SKELETON_EDGES + 19 * np.arange(poses_3d.shape[0]).reshape((-1, 1, 1))).reshape((-1, 2))
        plotter.plot(canvas_3d, poses_3d, edges)
        cv2.imwrite(os.path.join(output_dir, f"camera_space_{i:04}.jpg"), canvas_3d)

        draw_poses(frame, poses_2d)
        current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()
        if mean_time == 0:
            mean_time = current_time
        else:
            mean_time = mean_time * 0.95 + current_time * 0.05
        cv2.putText(frame, 'FPS: {}'.format(int(1 / mean_time * 10) / 10),
                    (40, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
        cv2.imwrite(os.path.join(output_dir, f"pixel_space_{i:04}.jpg"), frame)

        key = cv2.waitKey(delay)
        print("key", key)
        if key == esc_code:
            print("key == esc_code", key == esc_code)
            break
        if key == p_code:
            print("key == p_code", key == p_code)
            if delay == 1:
                print("delay == 1", delay == 1)
                delay = 0
            else:
                print("delay == 1", delay == 1)
                delay = 1
        if delay == 0 or not is_video:  # allow to rotate 3D canvas while on pause
            print("delay == 0 or not is_video", delay == 0 or not is_video)
            key = 0
            while (key != p_code
                   and key != esc_code
                   and key != space_code):
                plotter.plot(canvas_3d, poses_3d, edges)
                cv2.imwrite(os.path.join(output_dir, f"canvas_3d_2_{i:04}.jpg"), canvas_3d)
                key = cv2.waitKey(33)
            if key == esc_code:
                break
            else:
                delay = 1

if __name__ == "__main__":
    run()