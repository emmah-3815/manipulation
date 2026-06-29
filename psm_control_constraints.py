import rclpy
from rclpy.node import Node
from pathlib import Path
import pdb
from scipy.spatial.transform import Rotation as R
import numpy as np
import os
from psm_control.psm_control import utils as dvrk_utils
from psm_control.psm_control import PsmControl

from message_filters import Subscriber, ApproximateTimeSynchronizer
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

os.environ['ROS_DOMAIN_ID'] = '100'

"""
reads goal position commands from ros for psm1 and psm2
reads jaw control commands from ros for psm1 and psm2
moves to position, nothing else
all higher level movement sequences computed in simulation code


"""
class PSMControl():
    
    def __init__(self, args):
        self.psm_control = PsmControl()
        self.H_cam_base_1 = None
        self.H_cam_base_2 = None
        self.pose_cam_base_1 = None
        self.pose_cam_base_2 = None
        self.psm1_current_jaw = None
        self.psm1_current_jaw = None
        
        self.cam_base_coord_change = np.array([
            [0., 0., -1., 0.], 
            [-1., 0., 0., 0.], 
            [0., 1., 0., 0.], 
            [0., 0., 0., 1.], 
        ])
        self.cam_base_coord_change_inv = np.linalg.inv(self.cam_base_coord_change)

        self.init_cam2base(args)
        self.pose_init = False # set to true when messages arrive
        psm1_pose_sub = Subscriber(self.node, PoseStamped, '/PSM1/measured_cp',) # comes in as mm
        psm2_pose_sub = Subscriber(self.node, PoseStamped, '/PSM2/measured_cp',) # comes in as mm
        psm1_jaw_sub = Subscriber(self.node, JointState, '/PSM1/jaw/measured_cp',) # comes in as mm
        psm2_jaw_sub = Subscriber(self.node, JointState, '/PSM2/jaw/measured_cp',) # comes in as mm

        # --- DEBUG TRACKING ---
        self.latest_stamps = {
            'psm1': 0.0, 'psm2': 0.0,
            'psm1_jaw': 0.0, 'psm2_jaw': 0.0
        }

        def _track_stamp(name, msg):
            # Convert ROS header stamp to float seconds
            self.latest_stamps[name] = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Register the tracker to run whenever a message arrives on that topic individually
        psm1_pose_sub.registerCallback(lambda msg: _track_stamp('psm1', msg))
        psm2_pose_sub.registerCallback(lambda msg: _track_stamp('psm2', msg))
        psm1_jaw_sub.registerCallback(lambda msg: _track_stamp('psm1_jaw', msg))
        psm2_jaw_sub.registerCallback(lambda msg: _track_stamp('psm2_jaw', msg))
        # ---------------------------
        sync_targets = [psm1_pose_sub, psm2_pose_sub, psm1_jaw_sub, psm2_jaw_sub]

        # 2. Synchronize Topics
        sync = ApproximateTimeSynchronizer(
            sync_targets,
            queue_size=10,
            slop=0.1
        )

        # Register the callback
        sync.registerCallback(self.synced_callback)

        self.psm1_goal_sub = self.node.create_subscription(
            PoseStamped, 
            '/PSM1/goal', 
            self.psm1_goal_callback, 
            10
        )

        self.psm2_goal_sub = self.node.create_subscription(
            PoseStamped, 
            '/PSM2/goal', 
            self.psm2_goal_callback, 
            10
        )

        self.psm1_jaw_goal_sub = self.node.create_subscription(
            JointState, 
            '/PSM1/jaw_goal', 
            self.psm1_jaw_callback, 
            10
        )

        self.psm2_jaw_goal_sub = self.node.create_subscription(
            JointState, 
            '/PSM2/jaw_goal', 
            self.psm2_jaw_callback, 
            10
        )

    def init_cam2base(self, args):
        calib = args.psm_calibrate
        if Path(calib).exists():
            data = np.load(calib)

            self.H_cam_base_1 = data['PSM1'].copy()
            self.H_cam_base_1[:3, 3] *= 1000
            self.H_cam_base_2 = data['PSM2'].copy()
            self.H_cam_base_2[:3, 3] *= 1000

            self.pose_cam_base_1 = data['BASE1'] #TODO check the format and unit of this
            self.pose_cam_base_2 = data['BASE2']

            self.H_cam_base_1_inv = np.linalg.inv(self.H_cam_base_1)
            self.H_cam_base_2_inv = np.linalg.inv(self.H_cam_base_2)

    def synced_callback(self, psm1_pose_msg, psm2_pose_msg, psm1_jaw_msg, psm2_jaw_msg):
        pos = psm1_pose_msg.pose.position
        ori = psm1_pose_msg.pose.orientation
        pose = [pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w]
        T = self._pose_to_matrix(pose)
        self.psm1_current_T = self.H_cam_base_1 @ (T @ self.cam_base_coord_change)

        pos = psm2_pose_msg.pose.position
        ori = psm2_pose_msg.pose.orientation
        pose = [pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w]
        T = self._pose_to_matrix(pose)
        self.psm2_current_T = self.H_cam_base_2 @ (T @ self.cam_base_coord_change)
        
        self.psm1_current_jaw = psm1_jaw_msg.data
        self.psm2_current_jaw = psm2_jaw_msg.data
        
        if not self.pose_init:
            self.node.get_logger().info("Received first synced psm poses...")
            self.pose_init = True

    def psm1_goal_callback(self, psm1_pose_msg):
        pos = psm1_pose_msg.pose.position
        ori = psm1_pose_msg.pose.orientation
        pose = [pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w]
        T = self._pose_to_matrix(pose)
        T1 = self.H_cam_base_1_inv @ T @ self.cam_base_coord_change_inv
        self.control_PSM(psm=1, goal_H_cam_ee=T1)

    def psm2_goal_callback(self, psm2_pose_msg):
        pos = psm2_pose_msg.pose.position
        ori = psm2_pose_msg.pose.orientation
        pose = [pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w]
        T = self._pose_to_matrix(pose)
        T2 = self.H_cam_base_2_inv @ T @ self.cam_base_coord_change_inv
        self.control_PSM(psm=2, goal_H_cam_ee=T2)

    def psm1_jaw_callback(self, psm1_jaw_msg):
        rad = psm1_jaw_msg.position[0]
        degree = rad / np.pi * 180
        self.control_jaw(psm=1, degree=degree)

    def psm2_jaw_callback(self, psm2_jaw_msg):
        rad = psm2_jaw_msg.position[0]
        degree = rad / np.pi * 180
        self.control_jaw(psm=2, degree=degree)

    def _pose_to_matrix(self, pose):
        t = pose[:3]
        q = pose[3:]

        R_mat = R.from_quat(q).as_matrix()

        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = t

        return T

    def debug_sync_status(self):
        if self.images_init:
            return # Stop spamming once we successfully sync
            
        print("\n--- Synchronizer Diagnostic ---")
        stamps = []
        for name, stamp in self.latest_stamps.items():
            if stamp == 0.0:
                print(f"[MISSING] {name}: No messages received yet.")
            else:
                print(f"[OK]      {name}: Last stamp at {stamp:.3f}")
                stamps.append(stamp)
        
        # If we have received at least one message from ALL 6 topics, check the slop
        if len(stamps) == 4:
            max_diff = max(stamps) - min(stamps)
            print(f"\n=> Maxtimestamp spread across all 4 topics: {max_diff:.4f} seconds")
            if max_diff > 0.1: # Your current slop is 0.1
                print(f"=> WARNING: Spread ({max_diff:.4f}s) is LARGER than your slop (0.1s)!")
                print("=> Synchronizer is rejecting them. Increase slop or fix publishing rates.")
            else:
                print("=> Spread is within slop. Synchronizer should fire (unless queue size is too small).")
        print("-------------------------------")


    def control_PSM(self, psm, goal_H_cam_ee, pos_dist_th = 1e-3, angle_dist_th = 3 * np.pi / 180):
        # if goal_H_cam_ee[2, 3] < 0.04:
        #     print("~~~long thread~~~~~~~~~~~~~~")
        #     pdb.set_trace()
        #     self.long_thread = True # TODO move long thread logic to sim if needed
        #     goal_H_cam_ee[2, 3] = 0.055 # lift up to avoid table collision
        goal_pos_cam_ee, goal_quat_cam_ee = dvrk_utils.matrix2PosQuat(goal_H_cam_ee)
        goal_pose_cam_ee = np.concatenate([goal_quat_cam_ee, goal_pos_cam_ee])
        # self.psm_control.controlPoseReeInCam(psm, 
        #                                     goal_pose_cam_ee, 
        #                                     pos_dist_th = pos_dist_th, 
        #                                     angle_dist_th = angle_dist_th, 
        #                                     H_cam_base=self.H_cam_base_both['psm_{}'.format(psm)])
        self.psm_control.controlPoseReeInCamBypass(psm, 
                                            goal_pose_cam_ee, 
                                            get_H_cam_ree_cb=lambda: self.get_curr_H_pose_both(),
                                            pos_dist_th = pos_dist_th, 
                                            angle_dist_th = angle_dist_th, 
                                            )

        if self.use_curr_pose:
            goal_H_cam_ee = self.get_curr_H_pose_both()['psm_{}'.format(psm)]
        return goal_H_cam_ee

    def control_jaw(self, PSM, degree, steps=100):
        '''
        when psm is grasping needle, check jaw angle
        psm_1 open jaw angle 1.04703528
        psm_1 grasping needle jaw angle 0.00212347
        psm_1 closed jaw angle 0.00016334
        psm_2 open jaw angle 1.04458512
        psm_2 grasping needle jaw angle 0.00833055
        psm_2 closed jaw angle -0.00065338
        '''
        if not (-20 <= degree <= 90):
            print(f"degree goal is out of bound [-20 - 100], goal: {degree}")
            pdb.set_trace()
        if PSM in [1, 2]:
            if degree == 0:
                degree = -9
            rad = self.psm1_current_jaw if PSM==1 else self.psm2_current_jaw # get the current state of the robot
            init_degree = rad / np.pi * 180 # convert from rad to degree
            interp = np.linspace(init_degree, degree, steps)
            for step in interp:
                if self.jaw_effort['psm_{}'.format(PSM)] is not None:
                    high_effort = -0.04
                    if self.jaw_effort['psm_{}'.format(PSM)] < high_effort:
                        print(f"Jaw stuck? Jaw effort: {self.jaw_effort['psm_{}'.format(PSM)]}")
                        break
                step = int(step)
                self.psm_control.openGripperDegree(PSM, degree=step, sleep=0.01)
        else:
            self.psm_control.openGripperDegree(PSM, degree=degree, sleep=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--speedy',           action="store_true")
    # parser.add_argument('--calib',            default=None)
    parser.add_argument('--psm_calibrate',
        default=os.path.dirname(__file__) + "/../../../RaftStereo/assets/psm_calibration.npz")
    args = parser.parse_args()

    move = PSMControl(args)
    try:
        move.node.get_logger().info("Starting continuous stereo processing. Waiting for messages...")
        while rclpy.ok() and not move.pose_init:
            rclpy.spin_once(move.node, timeout_sec=0.1)
            move.debug_sync_status()
        
        if move.images_init:
            print("\n✅ Stereo pair captured and synchronized! Entering normal spin loop...")
            # Now we can block normally since initialization is done
            rclpy.spin(move.node)

        print("Stereo pair captured!")


    except KeyboardInterrupt:
        move.node.get_logger().info("Keyboard interrupt detected. Shutting down processor...")
    finally:
        # Clean up resources safely
        move.node.destroy_self.node()
        if rclpy.ok():
            rclpy.shutdown()

