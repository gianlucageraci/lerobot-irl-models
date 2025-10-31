from pathlib import Path
from simulation.base_sim import BaseSim
import logging
import numpy as np

from real_robot_env.robot.hardware_depthai import DepthAI, DAICameraType
from real_robot_env.robot.hardware_audio import AudioInterface
from real_robot_env.robot.hardware_franka import FrankaArm, ControlType
from real_robot_env.robot.hardware_frankahand import FrankaHand
from real_robot_env.robot.utils.keyboard import KeyManager
from real_robot_env.real_robot_env import RealRobotEnv

import torch
import einops
import cv2
import time

DELTA_T = 0.034

logger = logging.getLogger(__name__)


class RealRobot(BaseSim):
    def __init__(self, device: str):
        super().__init__(seed=-1, device=device)

        self.p4 = FrankaArm(
            name="202_robot",
            ip_address="141.3.53.63",
            port=50053,
            control_type=ControlType.HYBRID_JOINT_IMPEDANCE_CONTROL,
            hz=100
        )
        assert self.p4.connect(), f"Connection to {self.p4.name} failed"

        self.p4_hand = FrankaHand(name="202_gripper", ip_address="141.3.53.63", port=50054)
        assert self.p4_hand.connect(), f"Connection to {self.p4_hand.name} failed"

        self.i = 0

    def image_to_tensor(self, image):

        rgb = (
                torch.from_numpy(image.copy())
                .float()
                .permute(2, 0, 1)
                / 255.0
        )
        rgb = einops.rearrange(rgb, "c h w -> 1 1 c h w").to(self.device)

        return rgb

    def test_agent(self, agent):

        self.cam0 = DepthAI( # right cam
            device_id='1844301051D9B50F00',
            name='right_cam', # named orb due to other code dependencies
            height=126,
            width=224,
            camera_type=DAICameraType.OAK_D
        )
        #############Wrist camera 
        self.cam1 = DepthAI( # wrist cam
            device_id='1944301061BB782700',
            name='wrist_cam', # named orb due to other code dependencies
            height=126,
            width=224,
            camera_type=DAICameraType.OAK_D_SR
        )
        # self.cam2 = DepthAI( # front cam
        #     device_id='184430102111900E00',
        #     name='front_cam', # named orb due to other code dependencies
        #     height=180,
        #     width=320,
        #     camera_type=DAICameraType.OAK_D_LITE
        # )

        self.audio = AudioInterface(
            device_id="4",
            name="mic"
        )


        env = RealRobotEnv(
            robot_arm=self.p4,
            robot_hand=self.p4_hand,
            discrete_devices= [self.cam0, self.cam1],
            continuous_devices=[self.audio])

        lang_emb = torch.zeros(1, 1, 512).float().to(self.device)

        logger.info("Starting trained model evaluation on real robot")

        km = KeyManager()

        while km.key != 'q':
            print("Press 's' to start a new evaluation, or 'q' to quit")
            km.pool()

            while km.key not in ['s', 'q']:
                km.pool()
            vis=False
            if km.key == 's':
                print()

                agent.reset()
                obs, _ = env.reset()

                print("Starting evaluation. Press 'd' to stop current evaluation")

                km.pool()
                while km.key != 'd':
                    km.pool()

                    # print("Current step:", self.i)
                    # print("Current robot arm joint positions:", obs["robot_arm"].joint_pos)
                    robot_states = torch.tensor(obs["robot_arm"].joint_pos)
                    # print("Robot arm joint positions:", robot_states)
                    robot_states = robot_states.unsqueeze(0).unsqueeze(0)
                    # print("Robot states shape:", robot_states.shape)


                    # front_cam = obs['front_cam']['rgb']
                    right_cam = obs['right_cam']['rgb']
                    wrist_cam = obs['wrist_cam']['rgb']
                    if vis:
                        self.compare_images(
                                "/home/multimodallearning/data_collected/audio/eraser_task/2025_10_10-07_47_58/sensors/right_cam/0.png",
                                "/home/multimodallearning/data_collected/audio/eraser_task/2025_10_10-07_47_58/sensors/wrist_cam/0.png",
                                right_cam, wrist_cam
                            )
                        vis = False
                    # left_tactile = obs['left_tactile']['rgb']
                    # right_tactile = obs['right_tactile']['rgb']

                    # # cv2.imshow("front_cam", front_cam)
                    # # cv2.imshow("right_cam", right_cam)
                    # # cv2.imshow("wrist_cam", wrist_cam)
                    # # cv2.imshow("left_tactile", left_tactile)
                    # # cv2.imshow("right_tactile", right_tactile)
                    # # cv2.waitKey(1)

                    # # front_cam = self.image_to_tensor(front_cam)

                    # left_tactile = cv2.cvtColor(left_tactile, cv2.COLOR_RGB2BGR)
                    # right_tactile = cv2.cvtColor(right_tactile, cv2.COLOR_RGB2BGR)
                    

                    right_cam = self.image_to_tensor(right_cam)
                    wrist_cam = self.image_to_tensor(wrist_cam)
                    # left_tactile = self.image_to_tensor(left_tactile)
                    # right_tactile = self.image_to_tensor(right_tactile)
                    # print shape(front_cam)
                    # print("front_cam shape:", front_cam.shape)
                    mic = obs['mic']

                    if mic is None or (hasattr(mic, '__len__') and len(mic) == 0):
                        mic = torch.zeros(1, 1, 512).float().to(self.device)
                        print("Warning: mic data is empty, using zeros")
                    else:
                        print("Mic data shape:", mic.shape)
                        # Convert numpy array to tensor and reshape to match expected format
                        mic = torch.from_numpy(mic).float().unsqueeze(0).unsqueeze(0).to(self.device)
                        print("Mic tensor shape after conversion:", mic.shape)

                    # print("right_cam shape:", right_cam.shape)
                    # print("wrist_cam shape:", wrist_cam.shape)
                    # print("left_tactile shape:", left_tactile.shape)
                    # print("right_tactile shape:", right_tactile.shape)

                    obs_dict = {
                        # "front_cam_image": front_cam,
                        "right_cam_image": right_cam,
                        "wrist_cam_image": wrist_cam,
                        "audio": mic,
                        "lang_emb": lang_emb,
                        "robot_states": robot_states
                    }

                    pred_action = agent.predict(obs_dict).cpu().numpy()

                    pred_joint_pos = pred_action[:7]
                    # print("pred_joint_pos action:", pred_joint_pos)
                    pred_gripper_command = pred_action[-1]
                    pred_gripper_command = 1 if pred_gripper_command > 0 else -1

                    action = {'robot_arm': pred_joint_pos,
                              'robot_hand': pred_gripper_command}
                    pred_joint_pos = pred_action[:7]
                    # print("pred_joint_pos action:", pred_joint_pos)
                    pred_gripper_command = pred_action[-1]
                    pred_gripper_command = 1 if pred_gripper_command > 0 else -1

                    action = {'robot_arm': pred_joint_pos,
                              'robot_hand': pred_gripper_command}
                    obs, *_ = env.step(action)

                    time.sleep(DELTA_T)

                print()
                logger.info("Evaluation done. Resetting robots")

                env.reset()

        print()
        logger.info("Quitting evaluation")

        km.close()
        env.close()
    def compare_images(self,image_dataset1_path, image_dataset2_path, image_1, image_2):
        import matplotlib.pyplot as plt
        from PIL import Image
        # --- Load dataset images from paths (as HWC RGB uint8 numpy arrays) ---
        dataset_top_img = Image.open(image_dataset1_path).convert("RGB")
        dataset_side_img = Image.open(image_dataset2_path).convert("RGB")
        dataset_top = np.array(dataset_top_img)      # (H, W, 3), uint8
        dataset_side = np.array(dataset_side_img)    # (H, W, 3), uint8

        # --- Normalize inputs image_1 / image_2 to HWC numpy arrays ---
        # image_1 -> obs_top, image_2 -> obs_side
        if isinstance(image_1, torch.Tensor):
            obs_top = image_1.detach().cpu().numpy()
            # Accept CHW or HWC; squeeze singleton batch/channel dims
            obs_top = np.squeeze(obs_top)
            if obs_top.ndim == 3 and obs_top.shape[0] in (1, 3):  # CHW
                obs_top = np.transpose(obs_top, (1, 2, 0))        # -> HWC
        elif isinstance(image_1, Image.Image):
            obs_top = np.array(image_1.convert("RGB"))
        else:  # assume numpy
            obs_top = np.array(image_1)
            if obs_top.ndim == 3 and obs_top.shape[0] in (1, 3):  # CHW -> HWC
                obs_top = np.transpose(obs_top, (1, 2, 0))
        # Ensure 3 channels
        if obs_top.ndim == 2:
            obs_top = np.stack([obs_top]*3, axis=-1)

        if isinstance(image_2, torch.Tensor):
            obs_side = image_2.detach().cpu().numpy()
            obs_side = np.squeeze(obs_side)
            if obs_side.ndim == 3 and obs_side.shape[0] in (1, 3):  # CHW
                obs_side = np.transpose(obs_side, (1, 2, 0))
        elif isinstance(image_2, Image.Image):
            obs_side = np.array(image_2.convert("RGB"))
        else:  # assume numpy
            obs_side = np.array(image_2)
            if obs_side.ndim == 3 and obs_side.shape[0] in (1, 3):  # CHW -> HWC
                obs_side = np.transpose(obs_side, (1, 2, 0))
        if obs_side.ndim == 2:
            obs_side = np.stack([obs_side]*3, axis=-1)

        # --- (Optional) resize obs_* to dataset_* size if mismatched ---
        if obs_top.shape[:2] != dataset_top.shape[:2]:
            obs_top = np.array(Image.fromarray(obs_top.astype(np.uint8)).resize(dataset_top.shape[1::-1]))
        if obs_side.shape[:2] != dataset_side.shape[:2]:
            obs_side = np.array(Image.fromarray(obs_side.astype(np.uint8)).resize(dataset_side.shape[1::-1]))

        # --- Compute absolute difference images ---
        diff_top = np.abs(dataset_top.astype(np.float32) - obs_top.astype(np.float32))
        diff_side = np.abs(dataset_side.astype(np.float32) - obs_side.astype(np.float32))

        # --- Debug prints (first 3x3 of channel 0) ---
        print("First 3x3 of dataset_top:\n", dataset_top[:3, :3, 0])
        print("First 3x3 of obs_top:\n", obs_top[:3, :3, 0])
        print("First 3x3 of diff_top:\n", diff_top[:3, :3, 0])

        # --- Plot for visual comparison ---
        fig, axs = plt.subplots(3, 2, figsize=(8, 12))
        axs[0, 0].imshow(dataset_top)
        axs[0, 0].set_title("Dataset Top Image")
        axs[0, 1].imshow(obs_top)
        axs[0, 1].set_title("Observation Top Image")
        axs[1, 0].imshow(dataset_side)
        axs[1, 0].set_title("Dataset Side Image")
        axs[1, 1].imshow(obs_side)
        axs[1, 1].set_title("Observation Side Image")
        axs[2, 0].imshow(diff_top.astype(np.uint8))
        axs[2, 0].set_title("Abs Diff Top")
        axs[2, 1].imshow(diff_side.astype(np.uint8))
        axs[2, 1].set_title("Abs Diff Side")
        for ax in axs.flat:
            ax.axis('off')
        plt.tight_layout()
        plt.show()
    def __get_obs(self):

        img0 = self.cam0._get_sensors()["rgb"][:, :, :3]  # remove depth
        img1 = self.cam1._get_sensors()["rgb"][:, :, :3]
        img2 = self.cam2._get_sensors()["rgb"][:, :, :3]

        img0 = cv2.resize(img0, (512, 512))[:, 100:370]
        img1 = cv2.resize(img1, (512, 512))
        img2 = cv2.resize(img2, (512, 512))

        # cv2.imshow('0', img0)
        # cv2.imshow('1', img1)
        # cv2.waitKey(0)

        processed_img0 = cv2.resize(img0, (128, 256)).astype(np.float32).transpose((2, 0, 1)) / 255.0
        processed_img1 = cv2.resize(img1, (256, 256)).astype(np.float32).transpose((2, 0, 1)) / 255.0
        processed_img2 = cv2.resize(img2, (256, 256)).astype(np.float32).transpose((2, 0, 1)) / 255.0

        return (processed_img0, processed_img1, processed_img2)