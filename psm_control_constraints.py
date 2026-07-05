import argparse
import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from pathlib import Path
import pdb
from scipy.spatial.transform import Rotation as R
import numpy as np
import os
import transforms3d.quaternions as quaternions

from message_filters import Subscriber, ApproximateTimeSynchronizer
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

# os.environ['ROS_DOMAIN_ID'] = '111'


# ---------------------------------------------------------------------------
# Geometry helpers ported from psm_control.utils (quat convention: w, x, y, z)
# ---------------------------------------------------------------------------
def posquat2H(pos, quat):
    H = np.zeros([4, 4])
    H[:3, 3] = pos
    H[:3, :3] = quaternions.quat2mat(quat)
    H[3, 3] = 1
    return H


def matrix2PosQuat(H):
    return H[:3, 3], quaternions.mat2quat(H[:3, :3])  # quat: wxyz


def angleDist(q1, q2):
    if np.dot(q1, q2) < 0:
        q1 = -q1
    return np.arccos(np.clip(np.dot(q1, q2), -1, 1)), q1  # assume both unit-norm


def slerp(q_now, q_end, max_move_angle=3.):
    angle_dist, q_now = angleDist(q_now, q_end)
    move_angle = min(max_move_angle * np.pi / 180, angle_dist)
    on_q_end = q_end - np.dot(q_now, q_end) * q_now  # (q_now, on_q_end) not orthonormal
    on_q_end /= np.linalg.norm(on_q_end)
    return q_now * np.cos(move_angle) + on_q_end * np.sin(move_angle)

"""
reads goal position commands from ros for psm1 and psm2
reads jaw control commands from ros for psm1 and psm2
moves to position, nothing else
all higher level movement sequences computed in simulation code


"""
class PSMControl():
    
    def __init__(self, args):
        self.node = rclpy.create_node('psm_control_constraints')
        self._calibrationReeFee()  # sets self.cal_Hs (ree<->fee transforms)
        self.H_cam_base_1 = None
        self.H_cam_base_2 = None
        self.pose_cam_base_1 = None
        self.pose_cam_base_2 = None
        self.psm1_current_jaw = None
        self.psm2_current_jaw = None
        # latest measured_cp pose (qw,qx,qy,qz,x,y,z) in the PSM base frame
        # (base->FEE) -- the same frame/convention as /PSM*/goal and /PSM*/servo_cp
        self.pose_base_fee1 = None
        self.pose_base_fee2 = None
        self.jaw_effort = {'psm_1': None, 'psm_2': None}  # latest jaw effort (N·m)
        self.desired_jaw = {1: None, 2: None}  # latest commanded jaw angle (deg)

        # control callbacks (goal / jaw_goal) run blocking loops; put them in a
        # separate callback group so a MultiThreadedExecutor keeps servicing the
        # sensor callbacks (measured_cp) concurrently -> live closed-loop poses.
        self.control_cb_group = MutuallyExclusiveCallbackGroup()

        # only one control action (goal move or jaw move) may run at a time,
        # regardless of callback-group config. The sensor callbacks are NOT
        # gated by this, so measured_cp keeps updating for closed-loop control.
        self._control_lock = threading.Lock()

        self.init_cam2base(args)
        self.pose_init = False # set to true when messages arrive
        psm1_pose_sub = Subscriber(self.node, PoseStamped, '/PSM1/measured_cp',) # meters
        psm2_pose_sub = Subscriber(self.node, PoseStamped, '/PSM2/measured_cp',) # meters
        psm1_jaw_sub = Subscriber(self.node, JointState, '/PSM1/jaw/measured_js',) # radians
        psm2_jaw_sub = Subscriber(self.node, JointState, '/PSM2/jaw/measured_js',) # radians

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
            10,
            callback_group=self.control_cb_group,
        )

        self.psm2_goal_sub = self.node.create_subscription(
            PoseStamped,
            '/PSM2/goal',
            self.psm2_goal_callback,
            10,
            callback_group=self.control_cb_group,
        )

        self.psm1_jaw_goal_sub = self.node.create_subscription(
            JointState,
            '/PSM1/jaw_goal',
            self.psm1_jaw_callback,
            10,
            callback_group=self.control_cb_group,
        )

        self.psm2_jaw_goal_sub = self.node.create_subscription(
            JointState,
            '/PSM2/jaw_goal',
            self.psm2_jaw_callback,
            10,
            callback_group=self.control_cb_group,
        )

        # --- publishers for commanding the robot (ported from PsmControl) ---
        self.set_gripper1_pub = self.node.create_publisher(
            JointState, '/PSM1/jaw/servo_jp', 10)
        self.set_gripper2_pub = self.node.create_publisher(
            JointState, '/PSM2/jaw/servo_jp', 10)
        self.set_ee1_pub = self.node.create_publisher(
            PoseStamped, '/PSM1/servo_cp', 10)
        self.set_ee2_pub = self.node.create_publisher(
            PoseStamped, '/PSM2/servo_cp', 10)

    def init_cam2base(self, args):
        calib = args.psm_calibrate
        if not Path(calib).exists():
            raise FileNotFoundError(
                f"PSM calibration file not found: {calib}. "
                f"Pass --psm_calibrate <path to psm_calibration.npz>."
            )

        data = np.load(calib)

        # calibration, measured_cp, and /PSM*/goal are all in meters
        self.H_cam_base_1 = data['PSM1'].copy()
        self.H_cam_base_2 = data['PSM2'].copy()

        self.pose_cam_base_1 = data['BASE1'] #TODO check the format and unit of this
        self.pose_cam_base_2 = data['BASE2']

        self.H_cam_base_1_inv = np.linalg.inv(self.H_cam_base_1)
        self.H_cam_base_2_inv = np.linalg.inv(self.H_cam_base_2)

    def synced_callback(self, psm1_pose_msg, psm2_pose_msg, psm1_jaw_msg, psm2_jaw_msg):
        # measured_cp is base->FEE (same frame/convention as /PSM*/goal)
        self.pose_base_fee1 = self._measured_cp_to_posquat(psm1_pose_msg)
        self.pose_base_fee2 = self._measured_cp_to_posquat(psm2_pose_msg)

        self.psm1_current_jaw = psm1_jaw_msg.position[0]  # radians
        self.psm2_current_jaw = psm2_jaw_msg.position[0]  # radians

        # jaw effort, used by control_jaw to detect a stuck/over-torqued jaw
        self.jaw_effort['psm_1'] = psm1_jaw_msg.effort[0] if len(psm1_jaw_msg.effort) > 0 else None
        self.jaw_effort['psm_2'] = psm2_jaw_msg.effort[0] if len(psm2_jaw_msg.effort) > 0 else None

        if not self.pose_init:
            self.node.get_logger().info("Received first synced psm poses...")
            self.pose_init = True

    def psm1_goal_callback(self, psm1_pose_msg):
        pos = psm1_pose_msg.pose.position
        ori = psm1_pose_msg.pose.orientation
        # goal in PSM base frame, base->FEE: (qw, qx, qy, qz, x, y, z)
        pose = np.array([ori.w, ori.x, ori.y, ori.z, pos.x, pos.y, pos.z])
        with self._control_lock:  # serialize with all other control actions
            self.control_PSM(psm=1, goal_pose_base_fee=pose)

    def psm2_goal_callback(self, psm2_pose_msg):
        pos = psm2_pose_msg.pose.position
        ori = psm2_pose_msg.pose.orientation
        # goal in PSM base frame, base->FEE: (qw, qx, qy, qz, x, y, z)
        pose = np.array([ori.w, ori.x, ori.y, ori.z, pos.x, pos.y, pos.z])
        with self._control_lock:  # serialize with all other control actions
            self.control_PSM(psm=2, goal_pose_base_fee=pose)

    def psm1_jaw_callback(self, psm1_jaw_msg):
        rad = psm1_jaw_msg.position[0]
        degree = rad / np.pi * 180
        with self._control_lock:  # serialize with all other control actions
            self.control_jaw(psm=1, degree=degree, stop_on_effort=True)

    def psm2_jaw_callback(self, psm2_jaw_msg):
        rad = psm2_jaw_msg.position[0]
        degree = rad / np.pi * 180
        with self._control_lock:  # serialize with all other control actions
            self.control_jaw(psm=2, degree=degree, stop_on_effort=True)

    def _pose_to_matrix(self, pose):
        t = pose[:3]
        q = pose[3:]

        R_mat = R.from_quat(q).as_matrix()

        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = t

        return T

    def _measured_cp_to_posquat(self, pose_msg):
        """
        Read a /PSM*/measured_cp PoseStamped (base->FEE) as a pose
        (qw, qx, qy, qz, x, y, z) in the PSM base frame.
        """
        pos = pose_msg.pose.position
        ori = pose_msg.pose.orientation
        return np.array([ori.w, ori.x, ori.y, ori.z, pos.x, pos.y, pos.z])

    def debug_sync_status(self):
        if self.pose_init:
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


    def control_PSM(self, psm, goal_pose_base_fee, pos_dist_th = 1e-3, angle_dist_th = 3 * np.pi / 180):
        # goal_pose_base_fee: (qw, qx, qy, qz, x, y, z), FEE pose in the PSM base frame
        self.controlPoseFeeInBase(psm,
                                  goal_pose_base_fee,
                                  pos_dist_th=pos_dist_th,
                                  angle_dist_th=angle_dist_th,
                                  )

    def control_jaw(self, psm, degree, max_step_deg=2.0, sleep=0.005, stop_on_effort=False):
        '''
        when psm is grasping needle, check jaw angle
        psm_1 open jaw angle 1.04703528
        psm_1 grasping needle jaw angle 0.00212347
        psm_1 closed jaw angle 0.00016334
        psm_2 open jaw angle 1.04458512
        psm_2 grasping needle jaw angle 0.00833055
        psm_2 closed jaw angle -0.00065338

        The jaw is moved from its current angle to `degree` in small increments
        of at most `max_step_deg` degrees each so the motion stays smooth.
        '''
        if not (-20 <= degree <= 90):
            print(f"degree goal is out of bound [-20 - 100], goal: {degree}")
            pdb.set_trace()
        if psm in [1, 2]:
            if degree == 0:
                degree = -9
            # latest commanded jaw angle; controlPoseFeeInBase holds this while moving
            self.desired_jaw[psm] = degree
            rad = self.psm1_current_jaw if psm==1 else self.psm2_current_jaw # get the current state of the robot
            init_degree = rad / np.pi * 180 # convert from rad to degree
            # subdivide into steps of at most max_step_deg degrees
            n_steps = max(int(np.ceil(abs(degree - init_degree) / max_step_deg)), 1)
            interp = np.linspace(init_degree, degree, n_steps + 1)[1:]  # skip current angle
            for step in interp:
                # optional stuck detection (threshold unverified -> off by default,
                # so the jaw always drives all the way to the commanded angle)
                if stop_on_effort and self.jaw_effort['psm_{}'.format(psm)] is not None:
                    high_effort = -0.15
                    if self.jaw_effort['psm_{}'.format(psm)] < high_effort:
                        print(f"Jaw stuck? Jaw effort: {self.jaw_effort['psm_{}'.format(psm)]}")
                        break
                self.openGripperDegree(psm, degree=float(step), sleep=sleep)
        else:
            self.openGripperDegree(psm, degree=degree, sleep=1)

    # ------------------------------------------------------------------
    # Methods ported from psm_control.PsmControl so this node can command
    # the PSMs directly, without a ROSdVRK / ros_dvrk instance.
    # ------------------------------------------------------------------
    def _latest_base_fee_pose(self, psm_id):
        """Latest base->FEE pose (qw, qx, qy, qz, x, y, z) from measured_cp."""
        return self.pose_base_fee1 if psm_id == 1 else self.pose_base_fee2

    def _latest_jaw(self, psm_id):
        """Latest jaw angle in radians."""
        return self.psm1_current_jaw if psm_id == 1 else self.psm2_current_jaw

    def _calibrationReeFee(self):
        H_ree_fee = np.array([
            [0., -1., 0., 0.],
            [0., 0., 1., 0.],
            [-1., 0., 0., 0.],
            [0., 0., 0., 1.],
        ])
        self.cal_Hs = {
            'H_ree1_fee1': H_ree_fee,
            'H_ree2_fee2': H_ree_fee,
        }

    def _setGripper(self, pub, end_pos):
        """Set a gripper's jaw angle. end_pos is in degrees."""
        j_msg = JointState()
        j_msg.name = ['jaw']
        j_msg.position = [end_pos / 180 * np.pi]
        j_msg.velocity = [0.0]
        j_msg.effort = [0.0]
        pub.publish(j_msg)

    def openGripperDegree(self, psm_id, degree=60, sleep=0.1):
        """Open a gripper to a given jaw angle in degrees. psm_id in {1, 2, -1(both)}."""
        if psm_id in [1, -1]:
            self._setGripper(self.set_gripper1_pub, end_pos=degree)
        if psm_id in [2, -1]:
            self._setGripper(self.set_gripper2_pub, end_pos=degree)
        time.sleep(sleep)

    # quat: wxyz
    def _publishPoseBaseFee(self, psm_id, set_ee_pub, pos, quat, sleep_time=0.01):
        msg = PoseStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = '/PSM{}_base'.format(psm_id)

        msg.pose.position.x = pos[0]
        msg.pose.position.y = pos[1]
        msg.pose.position.z = pos[2]

        msg.pose.orientation.w = quat[0]
        msg.pose.orientation.x = quat[1]
        msg.pose.orientation.y = quat[2]
        msg.pose.orientation.z = quat[3]

        set_ee_pub.publish(msg)
        time.sleep(sleep_time)

    def controlPoseFeeInBase(
        self,
        psm_id: int,
        goal_pose_base_fee: np.ndarray,
        pos_dist_th: float = 1e-3,
        angle_dist_th: float = 5 * np.pi / 180,
        pos_step: float = 2e-3,
        angle_step_deg: float = 2.0,
        sleep: float = 0.005,
        max_iters: int = 5000,
    ):
        """
        Move the PSM's end-effector (FEE) to a goal pose in the PSM base frame.

        /PSM*/goal is published in the same frame and convention as
        /PSM*/measured_cp and /PSM*/servo_cp (PSM base frame, base->FEE), so we
        servo directly there -- no camera / hand-eye (H_cam_base) transform.

        Closed-loop: every iteration re-reads the live measured_cp pose (kept
        fresh by the sensor callback on another executor thread) and steps a
        small, bounded amount toward the goal, so each published servo_cp command
        stays close to the current pose and avoids the PSM's PID tracking-error
        fault.

        args:
            psm_id: which PSM to control (1 or 2)
            goal_pose_base_fee: (qw, qx, qy, qz, x, y, z) FEE goal in the PSM base frame
            pos_dist_th: positional distance threshold (m)
            angle_dist_th: orientational distance threshold (rad)
            pos_step: max positional increment per published command (m)
            angle_step_deg: max rotational increment per published command (deg)
            sleep: pause between published commands (s)
            max_iters: safety cap on the number of published commands
        """

        set_ee_pub = self.set_ee1_pub if psm_id == 1 else self.set_ee2_pub

        # --- Jaw angle to hold while moving ---
        # Prefer the latest angle commanded via control_jaw so a pose move never
        # fights a jaw goal; fall back to the current measured jaw if none yet.
        maintain_jaw_angle = self.desired_jaw[psm_id]
        if maintain_jaw_angle is None:
            try:
                current_jaw_rad = self._latest_jaw(psm_id)
                if isinstance(current_jaw_rad, (list, np.ndarray)):
                    current_jaw_rad = current_jaw_rad[0]
                # Convert to degrees because _setGripper divides by 180
                maintain_jaw_angle = current_jaw_rad * 180.0 / np.pi
            except Exception as e:
                self.node.get_logger().info('Could not get initial jaw angle: {}'.format(e))
                maintain_jaw_angle = 60.0  # Fallback to 60 degrees open if it fails
        print(f"maintain jaw angle: {maintain_jaw_angle}")

        goal_pos = np.asarray(goal_pose_base_fee[-3:], dtype=float)
        goal_quat = np.asarray(goal_pose_base_fee[:4], dtype=float)

        # --- DEBUG: current measured pose vs goal, both base->FEE. When the robot
        # is already at the goal this offset should be ~0.
        cur = self._latest_base_fee_pose(psm_id)
        if cur is not None:
            print(f"[goal check] PSM{psm_id} current base_fee pos: {np.asarray(cur[-3:])}, "
                  f"goal pos: {goal_pos}, offset: {goal_pos - np.asarray(cur[-3:], dtype=float)}")

        for _ in range(max_iters):
            # 1. Re-read the LIVE base->FEE pose from measured_cp (closed-loop).
            #    self.pose_base_fee* is kept fresh by the sensor callback thread.
            try:
                cur = self._latest_base_fee_pose(psm_id)
                current_pos = np.asarray(cur[-3:], dtype=float)
                current_quat = np.asarray(cur[:4], dtype=float)
            except Exception as e:
                self.node.get_logger().info('Could not get current pose: {}'.format(e))
                time.sleep(sleep)
                continue

            # 2. Distance from the current pose to the goal
            pos_dist = np.linalg.norm(goal_pos - current_pos)
            angle_dist, _ = angleDist(current_quat, goal_quat)

            if pos_dist < pos_dist_th and angle_dist < angle_dist_th:
                break

            # 3. Take one small step toward the goal from the live current pose
            if pos_dist > 1e-9:
                next_pos = current_pos + \
                    (goal_pos - current_pos) * min(pos_step, pos_dist) / pos_dist
            else:
                next_pos = current_pos
            # skip slerp when already aligned (avoids a 0/0 NaN in slerp)
            if angle_dist > 1e-6:
                next_quat = slerp(current_quat, goal_quat, max_move_angle=angle_step_deg)
            else:
                next_quat = current_quat

            # 4. Publish the base->FEE command directly to servo_cp
            # hold the jaw angle while servoing the cartesian pose
            self.openGripperDegree(psm_id=psm_id, degree=maintain_jaw_angle, sleep=0.0)
            self._publishPoseBaseFee(
                psm_id=psm_id,
                set_ee_pub=set_ee_pub,
                pos=next_pos,
                quat=next_quat,
                sleep_time=sleep,
            )
        else:
            print("max iters reached")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--speedy',           action="store_true")
    # parser.add_argument('--calib',            default=None)
    parser.add_argument('--psm_calibrate',
        default=os.path.dirname(__file__) + "/../RaftStereo/assets/psm_calibration.npz")
    args = parser.parse_args()

    rclpy.init(args=None)
    move = PSMControl(args)

    # Spin the node with a MultiThreadedExecutor on a separate thread so the
    # measured_cp sensor callbacks keep updating the live pose while the blocking
    # control loops run in their own callback group -> closed-loop tracking.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(move.node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        move.node.get_logger().info("Waiting for first synced psm poses...")
        while rclpy.ok() and not move.pose_init:
            move.debug_sync_status()
            time.sleep(0.1)

        print("\n✅ Synchronized! Control loops run on the executor threads. Waiting for goals...")
        while rclpy.ok():
            time.sleep(0.5)

    except KeyboardInterrupt:
        move.node.get_logger().info("Keyboard interrupt detected. Shutting down processor...")
    finally:
        # Clean up resources safely
        executor.shutdown()
        move.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=2.0)

