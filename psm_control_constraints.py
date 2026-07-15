import os
import sys

# ---------------------------------------------------------------------------
# Source the dVRK ROS overlay (crtk_msgs, etc.) up front. Launching this with a
# plain `python psm_control_constraints.py` from a shell that only has base ROS
# sourced fails to import crtk_msgs. We can't just extend sys.path: the rosidl
# C-extension .so files need LD_LIBRARY_PATH / AMENT_PREFIX_PATH, which the
# dynamic loader reads at process start. So re-exec this process through bash
# with the overlay sourced, then continue. Edit _ROS_OVERLAY for your setup.
# ---------------------------------------------------------------------------
_ROS_OVERLAY = "/home/arclab/ct_ws/install/setup.bash"
if os.environ.get("_PSM_ENV_SOURCED") != "1":
    _base = "/opt/ros/humble/setup.bash"
    for _p in (_base, _ROS_OVERLAY):
        if not os.path.exists(_p):
            sys.exit(f"[bootstrap] setup file not found: {_p} "
                     f"(edit _ROS_OVERLAY at the top of this script)")
    os.environ["_PSM_ENV_SOURCED"] = "1"  # guard against a re-exec loop
    os.execvp("bash", [
        "bash", "-c",
        'source "$1" && source "$2" && shift 2 && exec "$@"',
        "bash", _base, _ROS_OVERLAY,
        sys.executable, os.path.abspath(__file__), *sys.argv[1:],
    ])

import argparse
import atexit
import time
import threading
import termios
import tty
import select

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import (QoSProfile, QoSDurabilityPolicy,
                       QoSReliabilityPolicy, QoSHistoryPolicy)
from pathlib import Path
import pdb
from scipy.spatial.transform import Rotation as R
import numpy as np
import transforms3d.quaternions as quaternions

from message_filters import Subscriber, ApproximateTimeSynchronizer
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
# crtk_msgs powers the operating-state gating / FAULT recovery (e.g. after a
# teleop toggle). The overlay providing it is sourced by the bootstrap above.
from crtk_msgs.msg import OperatingState, StringStamped
# PsmState (header + int32 psm_id): published on /manipulate/psm to announce
# which PSM (1 or 2) most recently closed its jaws.
from thread_reconstruction_msgs.msg import PsmState

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


# 180 deg rotation about the tool's local z (shaft) axis, quat (w, x, y, z).
# A symmetric gripper grasps the same after this flip, so we can choose whichever
# roll representation of a goal keeps the wrist roll joint away from its limit.
Q_ROLL_FLIP = np.array([0., 0., 0., 1.])

# index of the tool-roll joint in /PSM*/measured_js.
# dVRK PSM joint order: [outer_yaw, outer_pitch, insertion, roll, wrist_pitch, wrist_yaw]
ROLL_JOINT_IDX = 3

# ---------------------------------------------------------------------------
# Collision-avoidance / homing configuration (all distances in meters).
#
# Each arm is modeled as a capsule: the line segment from its base origin (RCM)
# to its gripper tip (FEE), i.e. the instrument shaft. Two arms are kept apart
# by requiring the shaft-shaft centerline distance and the tip-tip distance to
# stay above these clearances. TUNE these to your instruments/setup.
# ---------------------------------------------------------------------------
SHAFT_CLEARANCE = 0.012   # min centerline distance between the two shafts (planning)
TIP_CLEARANCE = 0.012     # min distance between the two gripper tips (planning)
SHAFT_HARD_MIN = 0.008    # live safety guard during execution -> abort if breached

# RRT (planned in the moving arm's base frame, R^3 tip position)
RRT_STEP = 0.005          # extend distance per tree edge (m)
RRT_GOAL_BIAS = 0.10      # probability of sampling the goal
RRT_MAX_ITERS = 4000      # give up after this many samples
RRT_COLLISION_RES = 0.002 # edge collision-check spacing (m)
RRT_SAMPLE_MARGIN = 0.05  # expand the start/goal bbox by this when sampling (m)
RRT_SHORTCUT_ITERS = 150  # path-smoothing attempts

# waypoint-following tolerances (intermediate waypoints just pass through)
WAYPOINT_POS_TH = 3e-3
WAYPOINT_ANG_TH = 10 * np.pi / 180

# --- homing ('h' key) -------------------------------------------------------
# The home pose per arm is a Cartesian base->FEE target loaded from psm_home.npz
# (see init_home); the arm Cartesian-servos there. Open the jaw before moving.
# jaw angle (deg) to open to before homing, so the tool moves home open.
OPEN_JAW_DEG = 40.0
# the jaw moves this many times slower while closed (angle < 0 deg) than open.
JAW_CLOSED_SLOW_FACTOR = 5.0
# the arm servos this many times slower while its jaw is closed (below
# PSM_CLOSED_JAW_DEG), e.g. while grasping, for finer control.
PSM_CLOSED_JAW_DEG = 4.0
PSM_CLOSED_SLOW_FACTOR = 2.0


def _clamp(v, lo, hi):
    return max(lo, min(v, hi))


def seg_seg_dist(p1, q1, p2, q2):
    """
    Shortest distance between 3D segments [p1,q1] and [p2,q2]
    (Ericson, Real-Time Collision Detection, ClosestPtSegmentSegment).
    """
    p1 = np.asarray(p1, float); q1 = np.asarray(q1, float)
    p2 = np.asarray(p2, float); q2 = np.asarray(q2, float)
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = float(d1 @ d1)
    e = float(d2 @ d2)
    f = float(d2 @ r)
    EPS = 1e-12
    if a <= EPS and e <= EPS:
        return float(np.linalg.norm(p1 - p2))
    if a <= EPS:
        s, t = 0.0, _clamp(f / e, 0.0, 1.0)
    else:
        c = float(d1 @ r)
        if e <= EPS:
            t, s = 0.0, _clamp(-c / a, 0.0, 1.0)
        else:
            b = float(d1 @ d2)
            denom = a * e - b * b
            s = _clamp((b * f - c * e) / denom, 0.0, 1.0) if denom > EPS else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, _clamp(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t, s = 1.0, _clamp((b - c) / a, 0.0, 1.0)
    c1 = p1 + d1 * s
    c2 = p2 + d2 * t
    return float(np.linalg.norm(c1 - c2))


def slerp_frac(q0, q1, frac):
    """Interpolate `frac` (0..1) of the way from q0 to q1 (quats wxyz)."""
    ad, q0 = angleDist(np.asarray(q0, float), np.asarray(q1, float))
    if ad < 1e-6:
        return np.asarray(q1, float)
    move = frac * ad
    on = q1 - np.dot(q0, q1) * q0
    n = np.linalg.norm(on)
    if n < 1e-9:
        return np.asarray(q1, float)
    on /= n
    return q0 * np.cos(move) + on * np.sin(move)


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
        self.psm1_joints = None  # latest /PSM1/measured_js positions (rad/m)
        self.psm2_joints = None
        # gripper orientation (wxyz) with the tool-roll joint at 0, captured once
        # (computed from measured pose + roll joint, no motion). Goal roll is
        # chosen nearest this so the wrist roll joint stays near 0, off its limit.
        self.init_quat = {1: None, 2: None}

        # latest /PSM*/operating_state (crtk). Toggling teleop on/off in the dVRK
        # console can leave the arm FAULTed / not-ENABLED; commanding the jaw or
        # pose then triggers "arm not ready" faults. We watch this and re-enable
        # before commanding. Each entry: dict(state, is_homed, is_busy) or None.
        self.operating_state = {1: None, 2: None}
        self._warned_no_state = {1: False, 2: False}  # warn once if state absent

        # set while a homing/retract ('h' key) sequence is running, so repeated
        # presses don't stack multiple homing threads.
        self._homing = threading.Event()

        # saved terminal attributes so we can always restore cooked/echo mode on
        # exit. keyboard_listener puts the tty in cbreak (no echo); its own
        # finally does not run because it lives in a daemon thread that is killed
        # at shutdown -> without this the terminal is left with echo OFF (typed
        # text invisible). Restored via atexit and the main finally.
        self._term_fd = None
        self._term_old_attrs = None
        atexit.register(self._restore_terminal)
        # seeded RNG for the RRT planner (reproducible paths).
        self._rng = np.random.default_rng(0)

        # control callbacks (goal / jaw_goal) run blocking loops; put them in a
        # separate callback group so a MultiThreadedExecutor keeps servicing the
        # sensor callbacks (measured_cp) concurrently -> live closed-loop poses.
        self.control_cb_group = MutuallyExclusiveCallbackGroup()

        # only one control action (goal move or jaw move) may run at a time,
        # regardless of callback-group config. The sensor callbacks are NOT
        # gated by this, so measured_cp keeps updating for closed-loop control.
        self._control_lock = threading.Lock()

        # latching PAUSE toggle. Space sets it (pause) and space clears it
        # (resume). While paused, the active motion parks in place (robot holds
        # its last commanded pose) and every incoming goal/jaw call blocks at its
        # entry, so no control call is processed until space is pressed again.
        self._paused = threading.Event()

        # transient preempt, set by the 'h' homing key: aborts the current motion
        # (returns without /done) so the homing routine can take the control lock.
        # Homing clears it when it starts. Separate from _paused so homing is not
        # itself blocked by the pause.
        self._preempt = threading.Event()

        self.init_cam2base(args)
        self.init_home(args)  # sets self.home_pose_base_fee {1:.., 2:..}
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

        # Update each arm's live base->FEE pose DIRECTLY from its own measured_cp,
        # independent of the 4-way synchronizer below. The closed-loop servo needs
        # a fresh pose every few ms; if it relied only on synced_callback, a stall
        # on ANY of the 4 synced topics (the other arm's pose or either jaw) would
        # freeze this pose and the servo would settle in place without moving.
        psm1_pose_sub.registerCallback(lambda msg: self._pose_cb(msg, 1))
        psm2_pose_sub.registerCallback(lambda msg: self._pose_cb(msg, 2))
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

        # published (Bool True) when a PSM move finishes (goal reached, settled,
        # or max iters) so the sim knows it can send the next step.
        self.done_pub_1 = self.node.create_publisher(Bool, '/PSM1/done', 10)
        self.done_pub_2 = self.node.create_publisher(Bool, '/PSM2/done', 10)

        # announces which PSM (psm_id 1 or 2) most recently closed its jaws.
        self.psm_state_pub = self.node.create_publisher(PsmState, '/manipulate/psm', 10)

        # joint states, for the tool-roll joint used to compute the zero-roll ref
        self.node.create_subscription(
            JointState, '/PSM1/measured_js', lambda m: self._joint_cb(m, 1), 10)
        self.node.create_subscription(
            JointState, '/PSM2/measured_js', lambda m: self._joint_cb(m, 2), 10)

        # operating-state feedback + state_command to enable/home/clear FAULT.
        # These stay in the default callback group (NOT control_cb_group) so they
        # keep updating on another executor thread while a control loop blocks.
        #
        # The dVRK bridge publishes operating_state from a LATCHED event
        # (AddPublisherFromEventWrite, latched=true -> transient_local QoS) and
        # only ON CHANGE, not periodically. A default (volatile) subscriber never
        # receives the latched sample and, since the state hasn't changed since
        # the arm came up, gets nothing at all. Match the publisher with a
        # transient_local + reliable profile so we receive the current state
        # immediately on subscription.
        state_qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.node.create_subscription(
            OperatingState, '/PSM1/operating_state',
            lambda m: self._operating_state_cb(m, 1), state_qos)
        self.node.create_subscription(
            OperatingState, '/PSM2/operating_state',
            lambda m: self._operating_state_cb(m, 2), state_qos)
        self.state_cmd_pub = {
            1: self.node.create_publisher(StringStamped, '/PSM1/state_command', 10),
            2: self.node.create_publisher(StringStamped, '/PSM2/state_command', 10),
        }

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

    def init_home(self, args):
        """
        Load the per-arm home pose ('h' key target) from psm_home.npz. Each key
        (PSM1, PSM2) is a 4x4 base->FEE homogeneous transform with the translation
        in millimeters (same base frame as measured_cp / servo_cp, which are in
        meters), so we convert mm->m. Stored as (qw,qx,qy,qz,x,y,z), the same
        convention control_PSM / controlPoseFeeInBase expect.
        """
        home = args.psm_home
        if not Path(home).exists():
            raise FileNotFoundError(
                f"PSM home file not found: {home}. "
                f"Pass --psm_home <path to psm_home.npz>."
            )
        data = np.load(home)
        self.home_pose_base_fee = {}
        for psm_id, key in ((1, 'PSM1'), (2, 'PSM2')):
            H = np.asarray(data[key], dtype=float)
            pos_m = H[:3, 3] / 1000.0                       # mm -> m
            quat_wxyz = quaternions.mat2quat(H[:3, :3])     # wxyz
            self.home_pose_base_fee[psm_id] = np.array(
                [*quat_wxyz, *pos_m], dtype=float)
            self.node.get_logger().info(
                f"PSM{psm_id} home pos (m): {pos_m}")

    def _pose_cb(self, pose_msg, psm_id):
        """
        Update one arm's live base->FEE pose from its OWN /PSM*/measured_cp,
        independent of the 4-way synchronizer. This is the pose the closed-loop
        servo reads, so it must stay fresh even if a synced topic stalls.
        """
        pq = self._measured_cp_to_posquat(pose_msg)
        if psm_id == 1:
            self.pose_base_fee1 = pq
        else:
            self.pose_base_fee2 = pq
        self.pose_init = True

    def synced_callback(self, psm1_pose_msg, psm2_pose_msg, psm1_jaw_msg, psm2_jaw_msg):
        # NOTE: the live base->FEE poses are updated in _pose_cb (per-arm, direct
        # from measured_cp) so the servo never starves on a synced-topic stall.
        # This callback only maintains the synchronized jaw state / effort.
        self.psm1_current_jaw = psm1_jaw_msg.position[0]  # radians
        self.psm2_current_jaw = psm2_jaw_msg.position[0]  # radians

        # jaw effort, used by control_jaw to detect a stuck/over-torqued jaw
        self.jaw_effort['psm_1'] = psm1_jaw_msg.effort[0] if len(psm1_jaw_msg.effort) > 0 else None
        self.jaw_effort['psm_2'] = psm2_jaw_msg.effort[0] if len(psm2_jaw_msg.effort) > 0 else None

    def psm1_goal_callback(self, psm1_pose_msg):
        pos = psm1_pose_msg.pose.position
        ori = psm1_pose_msg.pose.orientation
        # goal in PSM base frame, base->FEE: (qw, qx, qy, qz, x, y, z)
        pose = np.array([ori.w, ori.x, ori.y, ori.z, pos.x, pos.y, pos.z])
        self._wait_if_paused()  # SPACE pause: hold this call until resumed
        with self._control_lock:  # serialize with all other control actions
            self.control_PSM(psm=1, goal_pose_base_fee=pose)

    def psm2_goal_callback(self, psm2_pose_msg):
        pos = psm2_pose_msg.pose.position
        ori = psm2_pose_msg.pose.orientation
        # goal in PSM base frame, base->FEE: (qw, qx, qy, qz, x, y, z)
        pose = np.array([ori.w, ori.x, ori.y, ori.z, pos.x, pos.y, pos.z])
        self._wait_if_paused()  # SPACE pause: hold this call until resumed
        with self._control_lock:  # serialize with all other control actions
            self.control_PSM(psm=2, goal_pose_base_fee=pose)

    def psm1_jaw_callback(self, psm1_jaw_msg):
        rad = psm1_jaw_msg.position[0]
        degree = rad / np.pi * 180
        self._wait_if_paused()  # SPACE pause: hold this call until resumed
        with self._control_lock:  # serialize with all other control actions
            self.control_jaw(psm=1, degree=degree, stop_on_effort=True)

    def psm2_jaw_callback(self, psm2_jaw_msg):
        rad = psm2_jaw_msg.position[0]
        degree = rad / np.pi * 180
        self._wait_if_paused()  # SPACE pause: hold this call until resumed
        with self._control_lock:  # serialize with all other control actions
            self.control_jaw(psm=2, degree=degree, stop_on_effort=True)

    def _wait_if_paused(self):
        """
        Block while SPACE has paused the node so the robot holds its pose and no
        control call proceeds. Returns early if a homing preempt ('h') is
        requested (homing must not be blocked by the pause) or on shutdown.
        """
        while self._paused.is_set() and not self._preempt.is_set() and rclpy.ok():
            time.sleep(0.05)

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

    def _joint_cb(self, msg, psm_id):
        """Store the latest /PSM*/measured_js joint positions."""
        if psm_id == 1:
            self.psm1_joints = np.array(msg.position)
        else:
            self.psm2_joints = np.array(msg.position)

    def _operating_state_cb(self, msg, psm_id):
        """Store the latest /PSM*/operating_state (crtk OperatingState)."""
        self.operating_state[psm_id] = {
            'state': msg.state,
            'is_homed': msg.is_homed,
            'is_busy': msg.is_busy,
        }

    def _send_state_command(self, psm_id, command):
        """Publish a crtk state_command ('enable'|'disable'|'home'|'pause'|'unpause')."""
        msg = StringStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.string = command
        self.state_cmd_pub[psm_id].publish(msg)

    def _ensure_ready(self, psm_id, timeout=5.0):
        """
        Make sure PSM `psm_id` is ENABLED and homed before we command it.

        Toggling teleop on/off in the dVRK console (or a prior tracking-error
        trip) can leave the arm in FAULT/DISABLED/PAUSED. Sending jaw or pose
        commands then faults the gripper ("arm not ready"). Here we re-enable
        (which clears a FAULT and re-powers the actuators) and, if needed,
        unpause -- but we only 'home' when the arm reports it is not homed, so a
        normal re-enable never triggers a homing motion.

        Returns True if the arm ends up ENABLED (+homed), else False.
        """
        st = self.operating_state.get(psm_id)

        # No state received (topic absent, e.g. sim, or QoS mismatch): do NOT
        # block -- returning immediately keeps jaw/pose commands responsive.
        # Warn only once per arm so we don't stall or spam every command.
        if st is None:
            if not self._warned_no_state.get(psm_id):
                self.node.get_logger().warn(
                    f"PSM{psm_id}: no operating_state received; state gating "
                    f"disabled, commanding directly")
                self._warned_no_state[psm_id] = True
            return True

        if st['state'] == 'ENABLED' and st['is_homed']:
            return True

        deadline = time.time() + timeout
        self.node.get_logger().warn(
            f"PSM{psm_id}: not ready (state={st['state']}, homed={st['is_homed']}); "
            f"attempting to enable")

        last_cmd = 0.0
        while time.time() < deadline:
            st = self.operating_state.get(psm_id)
            if st is not None and st['state'] == 'ENABLED' and st['is_homed']:
                self.node.get_logger().info(f"PSM{psm_id}: ready")
                return True
            # (re)issue the appropriate recovery command at ~2 Hz
            now = time.time()
            if now - last_cmd > 0.5 and st is not None:
                if st['state'] in ('FAULT', 'DISABLED'):
                    self._send_state_command(psm_id, 'enable')
                elif st['state'] == 'PAUSED':
                    self._send_state_command(psm_id, 'unpause')
                elif st['state'] == 'ENABLED' and not st['is_homed']:
                    self._send_state_command(psm_id, 'home')
                last_cmd = now
            time.sleep(0.05)

        st = self.operating_state.get(psm_id)
        self.node.get_logger().error(
            f"PSM{psm_id}: still not ready after {timeout}s "
            f"(state={st['state'] if st else None}); skipping command")
        return False

    def _zero_roll_ref(self, psm_id):
        """
        Orientation of the gripper with the tool-roll joint at 0, computed from
        the current measured pose and roll joint -- no motion. Used as the roll
        reference so goal-roll selection biases the roll joint toward 0 (mid
        range), away from its limit.

        Assumes the tool shaft is the FEE local z axis, so the roll joint
        rotates the tip about local z; de-rolling removes that rotation.
        Returns (qw,qx,qy,qz) or None if pose/joints not available yet.
        """
        pose = self.pose_base_fee1 if psm_id == 1 else self.pose_base_fee2
        joints = self.psm1_joints if psm_id == 1 else self.psm2_joints
        if pose is None or joints is None or len(joints) <= ROLL_JOINT_IDX:
            return None
        roll = float(joints[ROLL_JOINT_IDX])
        # de-roll: R_tip(0) = R_tip(roll) * Rz_local(-roll)
        q_deroll = np.array([np.cos(-roll / 2.0), 0.0, 0.0, np.sin(-roll / 2.0)])
        return quaternions.qmult(pose[:4], q_deroll)

    def _nearest_roll_goal(self, goal_quat, ref_quat):
        """
        A symmetric gripper grasps the same after a 180 deg roll about its shaft.
        Return whichever of {goal, goal rolled 180 deg about local z} keeps the
        gripper orientation nearest ref_quat, so the wrist roll joint stays close
        to its (in-range) startup value instead of wrapping toward a limit.
        """
        goal_quat = np.asarray(goal_quat, dtype=float)
        flipped = quaternions.qmult(goal_quat, Q_ROLL_FLIP)
        d_goal, _ = angleDist(goal_quat, ref_quat)
        d_flip, _ = angleDist(flipped, ref_quat)
        return flipped if d_flip < d_goal else goal_quat

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


    def keyboard_listener(self):
        """
        Read single keypresses from the terminal. SPACE toggles PAUSE: the first
        press pauses (the active motion parks and holds its pose, and every
        incoming goal/jaw call blocks until resumed); the next press resumes.
        H retracts both grippers to home. Runs in its own thread.
        """
        if not sys.stdin.isatty():
            self.node.get_logger().warn(
                "stdin is not a TTY; spacebar pause is disabled.")
            return

        print("[keyboard] SPACE = pause/resume (toggle), "
              "H = retract both grippers to home")
        fd = sys.stdin.fileno()
        old_attrs = termios.tcgetattr(fd)
        # record for _restore_terminal (atexit / main finally), since this
        # daemon thread's finally may not run at shutdown.
        self._term_fd = fd
        self._term_old_attrs = old_attrs
        try:
            tty.setcbreak(fd)  # read keys without waiting for Enter
            while rclpy.ok():
                # poll with a timeout so we can notice rclpy shutting down
                r, _, _ = select.select([sys.stdin], [], [], 0.1)
                if r:
                    ch = sys.stdin.read(1)
                    if ch == ' ':
                        if self._paused.is_set():
                            self._paused.clear()
                            print("\n[RESUME] space pressed - resuming; "
                                  "control calls will be processed again")
                        else:
                            self._paused.set()
                            print("\n[PAUSE] space pressed - paused; holding pose "
                                  "and ignoring all calls until space is pressed")
                    elif ch in ('h', 'H'):
                        # homing overrides a pause: clear it so homing isn't
                        # blocked, preempt any active motion so homing can grab the
                        # control lock, then retract both arms in a background
                        # thread (so this listener stays responsive during homing).
                        self._paused.clear()
                        self._preempt.set()
                        print("\n[HOME] h pressed - retracting both grippers")
                        threading.Thread(target=self.home_all, daemon=True).start()
        finally:
            self._restore_terminal()

    def _restore_terminal(self):
        """Restore the terminal to its original (cooked/echo) mode. Safe to call
        multiple times; runs from the listener's finally, atexit, and the main
        finally so a daemon-thread shutdown never leaves the tty with echo off."""
        if self._term_old_attrs is not None and self._term_fd is not None:
            try:
                termios.tcsetattr(self._term_fd, termios.TCSADRAIN,
                                  self._term_old_attrs)
            except Exception:
                pass
            self._term_old_attrs = None  # restored; don't do it again

    def control_PSM(self, psm, goal_pose_base_fee, pos_dist_th = 5e-4, angle_dist_th = 1 * np.pi / 180):
        # goal_pose_base_fee: (qw, qx, qy, qz, x, y, z), FEE pose in the PSM base frame
        if not self._ensure_ready(psm):
            self._publish_done(psm)  # can't move; don't leave the sim waiting
            return
        self.controlPoseFeeInBase(psm,
                                  goal_pose_base_fee,
                                  pos_dist_th=pos_dist_th,
                                  angle_dist_th=angle_dist_th,
                                  )

    def control_jaw(self, psm, degree, max_step_deg=0.25, sleep=0.005, stop_on_effort=False):
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
        if not (-20 <= degree <= 120):
            print(f"degree goal is out of bound [-20 - 100], goal: {degree}")
        if psm in [1, 2] and not self._ensure_ready(psm):
            self._publish_done(psm)  # arm not ready (e.g. after teleop toggle)
            return
        if psm in [1, 2]:
            if degree == 0:
                degree = -9
            # latest commanded jaw angle; controlPoseFeeInBase holds this while moving
            self.desired_jaw[psm] = degree
            rad = self.psm1_current_jaw if psm==1 else self.psm2_current_jaw # get the current state of the robot
            init_degree = rad / np.pi * 180 # convert from rad to degree
            # March from the current angle to the goal. The per-command step is
            # max_step_deg while the jaw is open (>= 0 deg) and JAW_CLOSED_SLOW_FACTOR
            # times smaller while it is closed (< 0 deg), so motion through the
            # closed range is that many times slower.
            direction = 1.0 if degree >= init_degree else -1.0
            pos = init_degree
            while abs(pos - degree) > 1e-9:
                self._wait_if_paused()  # SPACE pause: hold here, resume in place
                if self._preempt.is_set():  # 'h' homing -> abort jaw motion
                    print("[HOME] jaw motion preempted for homing")
                    return
                # optional stuck detection (threshold unverified -> off by default,
                # so the jaw always drives all the way to the commanded angle)
                if stop_on_effort and self.jaw_effort['psm_{}'.format(psm)] is not None:
                    high_effort = -0.15
                    if self.jaw_effort['psm_{}'.format(psm)] < high_effort:
                        print(f"Jaw stuck? Jaw effort: {self.jaw_effort['psm_{}'.format(psm)]}")
                        break
                step = max_step_deg / JAW_CLOSED_SLOW_FACTOR if pos < 0.0 else max_step_deg
                pos += direction * step
                if (direction > 0 and pos > degree) or (direction < 0 and pos < degree):
                    pos = degree  # clamp to the goal, don't overshoot
                self.openGripperDegree(psm, degree=float(pos), sleep=sleep)
            # a jaw CLOSE (commanded to < 0 deg) just completed -> announce which
            # PSM most recently closed its jaws on /manipulate/psm.
            if degree < 0:
                self._publish_psm_state(psm)
            # jaw move finished (reached target or effort-stopped, NOT space-stop
            # which returns early) -> tell the sim it can send the next step
            self._publish_done(psm)
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

    def _publish_done(self, psm_id):
        """Signal that a PSM move finished so the sim can send the next step."""
        pub = self.done_pub_1 if psm_id == 1 else self.done_pub_2
        pub.publish(Bool(data=True))

    def _publish_psm_state(self, psm_id):
        """Announce on /manipulate/psm which PSM (1 or 2) last closed its jaws."""
        msg = PsmState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = f'PSM{psm_id}'
        msg.psm_id = int(psm_id)
        self.psm_state_pub.publish(msg)

    # ------------------------------------------------------------------
    # Cartesian homing ('h' key): move both arms to the home poses in psm_home.npz.
    # ------------------------------------------------------------------
    def home_all(self):
        """Move both PSMs to their home Cartesian poses, one arm at a time."""
        if self._homing.is_set():
            print("[home] already homing; ignoring")
            return
        self._homing.set()
        try:
            for psm in (1, 2):
                self._home_arm(psm)
            print("[home] both grippers homed")
        finally:
            self._homing.clear()

    def _home_arm(self, psm_id):
        """
        Open the gripper, then move one PSM to its home base->FEE pose (loaded
        from psm_home.npz) using the normal Cartesian servo + collision-avoidance
        pipeline. Serialized with all other control actions by _control_lock;
        pausable with SPACE and preemptible by another 'h' press.
        """
        with self._control_lock:
            self._preempt.clear()  # consume the 'h' preempt; this homing run owns it
            if not self._ensure_ready(psm_id):
                self.node.get_logger().error(
                    f"PSM{psm_id}: not ready; skipping homing")
                return

            home_pose = self.home_pose_base_fee.get(psm_id)
            if home_pose is None:
                self.node.get_logger().error(
                    f"PSM{psm_id}: no home pose loaded; skipping homing")
                return

            # Open the jaw before moving (sets desired_jaw so it holds open).
            self._open_jaw_for_home(psm_id)
            if self._preempt.is_set():
                print(f"[HOME] homing PSM{psm_id} preempted before move")
                return

            # Cartesian-servo to the home pose (same path as a normal goal: RRT
            # around the other arm, closed-loop servo_cp, holds the open jaw, and
            # publishes /done on completion).
            print(f"[home] PSM{psm_id} moving to home pose {home_pose[-3:]}")
            self.controlPoseFeeInBase(psm_id, home_pose)
            print(f"[home] PSM{psm_id} homed")

    def _open_jaw_for_home(self, psm_id, step_deg=0.5, sleep=0.005):
        """
        Smoothly open the jaw to OPEN_JAW_DEG before the home move. Steps in small
        increments (like control_jaw) but does NOT publish /done -- the caller
        finishes the home move first. Interruptible with SPACE.
        """
        cur_rad = self._latest_jaw(psm_id)
        init_deg = (cur_rad * 180.0 / np.pi) if cur_rad is not None else 0.0
        self.desired_jaw[psm_id] = OPEN_JAW_DEG  # hold this angle through the retract
        n = max(int(np.ceil(abs(OPEN_JAW_DEG - init_deg) / step_deg)), 1)
        print(f"[home] PSM{psm_id} opening jaw {init_deg:.1f} -> {OPEN_JAW_DEG:.1f} deg")
        for step in np.linspace(init_deg, OPEN_JAW_DEG, n + 1)[1:]:
            self._wait_if_paused()  # SPACE pause: hold mid jaw-open, then resume
            if self._preempt.is_set():
                return
            self.openGripperDegree(psm_id, degree=float(step), sleep=sleep)

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

    # ------------------------------------------------------------------
    # Collision avoidance between the two arms.
    #
    # Both arms are expressed in the camera frame via the calibration
    # (p_cam = H_cam_base_i @ p_base). Each arm is a capsule from its base
    # origin (RCM) to its gripper tip. Because control actions are serialized
    # by self._control_lock, only ONE arm moves at a time -- the other is a
    # static obstacle -- so we can plan the moving arm's tip path (R^3 in its
    # own base frame) around the other arm's shaft with a simple RRT.
    # ------------------------------------------------------------------
    def _H_cam_base(self, psm_id):
        return self.H_cam_base_1 if psm_id == 1 else self.H_cam_base_2

    def _base_to_cam(self, psm_id, p_base):
        """Map a point from PSM `psm_id`'s base frame into the camera frame."""
        p = np.asarray(p_base, float)
        return (self._H_cam_base(psm_id) @ np.array([p[0], p[1], p[2], 1.0]))[:3]

    def _base_origin_cam(self, psm_id):
        """PSM base origin (RCM, proximal end of the shaft) in the camera frame."""
        return self._H_cam_base(psm_id)[:3, 3].copy()

    def _obstacle_shaft_cam(self, moving_psm_id):
        """
        The other (stationary) arm's shaft as a segment (B_o, T_o) in the camera
        frame, using its latest measured tip. None if its pose isn't available.
        """
        other = 2 if moving_psm_id == 1 else 1
        other_pose = self._latest_base_fee_pose(other)
        if other_pose is None:
            return None
        B_o = self._base_origin_cam(other)
        T_o = self._base_to_cam(other, np.asarray(other_pose[-3:], float))
        return (B_o, T_o)

    def _tip_clearance_ok(self, psm_id, tip_base, obstacle,
                          shaft_min=SHAFT_CLEARANCE, tip_min=TIP_CLEARANCE):
        """
        True if placing PSM `psm_id`'s tip at `tip_base` (its base frame) keeps
        both the shaft-shaft and tip-tip clearances against `obstacle`.
        """
        if obstacle is None:
            return True
        B_o, T_o = obstacle
        B_a = self._base_origin_cam(psm_id)
        T_a = self._base_to_cam(psm_id, tip_base)
        if seg_seg_dist(B_a, T_a, B_o, T_o) < shaft_min:
            return False
        if np.linalg.norm(T_a - T_o) < tip_min:
            return False
        return True

    def _edge_ok(self, psm_id, a_base, b_base, obstacle):
        """Collision-free straight tip motion from a_base to b_base (discretized)."""
        a = np.asarray(a_base, float)
        b = np.asarray(b_base, float)
        length = np.linalg.norm(b - a)
        n = max(int(np.ceil(length / RRT_COLLISION_RES)), 1)
        for i in range(n + 1):
            if not self._tip_clearance_ok(psm_id, a + (b - a) * (i / n), obstacle):
                return False
        return True

    def _plan_path(self, psm_id, start_base, goal_base, obstacle):
        """
        RRT in the moving arm's base frame (tip position). Returns a list of
        waypoints [start, ..., goal] whose straight segments are all collision
        free, or None if the goal is in collision / planning fails.
        """
        start = np.asarray(start_base, float)
        goal = np.asarray(goal_base, float)

        # Fast path: straight line already clear (also the no-obstacle case).
        if self._edge_ok(psm_id, start, goal, obstacle):
            return [start, goal]

        if not self._tip_clearance_ok(psm_id, goal, obstacle):
            self.node.get_logger().error(
                f"PSM{psm_id}: goal pose collides with the other arm; not moving")
            return None

        lo = np.minimum(start, goal) - RRT_SAMPLE_MARGIN
        hi = np.maximum(start, goal) + RRT_SAMPLE_MARGIN
        nodes = [start]
        parents = [-1]
        for _ in range(RRT_MAX_ITERS):
            sample = goal if self._rng.random() < RRT_GOAL_BIAS \
                else lo + self._rng.random(3) * (hi - lo)
            idx = int(np.argmin([np.linalg.norm(sample - nd) for nd in nodes]))
            near = nodes[idx]
            direction = sample - near
            dn = np.linalg.norm(direction)
            if dn < 1e-9:
                continue
            new = near + direction / dn * min(RRT_STEP, dn)
            if not self._edge_ok(psm_id, near, new, obstacle):
                continue
            nodes.append(new)
            parents.append(idx)
            if np.linalg.norm(new - goal) < RRT_STEP and \
                    self._edge_ok(psm_id, new, goal, obstacle):
                nodes.append(goal)
                parents.append(len(nodes) - 2)
                path = self._backtrace(nodes, parents)
                return self._shortcut(psm_id, path, obstacle)
        self.node.get_logger().error(
            f"PSM{psm_id}: RRT found no collision-free path in {RRT_MAX_ITERS} iters")
        return None

    def _backtrace(self, nodes, parents):
        path = []
        i = len(nodes) - 1
        while i != -1:
            path.append(nodes[i])
            i = parents[i]
        path.reverse()
        return path

    def _shortcut(self, psm_id, path, obstacle):
        """Randomized shortcut smoothing: drop waypoints whose bypass stays free."""
        path = [np.asarray(p, float) for p in path]
        for _ in range(RRT_SHORTCUT_ITERS):
            if len(path) <= 2:
                break
            i = int(self._rng.integers(0, len(path) - 1))
            j = int(self._rng.integers(0, len(path) - 1))
            a, b = min(i, j), max(i, j)
            if b - a < 2:
                continue
            if self._edge_ok(psm_id, path[a], path[b], obstacle):
                path = path[:a + 1] + path[b:]
        return path

    def controlPoseFeeInBase(
        self,
        psm_id: int,
        goal_pose_base_fee: np.ndarray,
        pos_dist_th: float = 1e-4,
        angle_dist_th: float = 5 * np.pi / 180,
        pos_step: float = 9e-4,
        angle_step_deg: float = 1.0,
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

        # Command the jaw ONCE, before the servo_cp loop, then let dVRK hold it.
        # On a dVRK PSM the jaw is joint 7 of the same arm/controller, so a
        # jaw/servo_jp command forces the arm into JOINT_SPACE while servo_cp
        # forces CARTESIAN_SPACE. Interleaving them every iteration made the arm
        # flip-flop control spaces (JOINT<->CARTESIAN), which trips a PID
        # tracking-error fault on the wrist_yaw joint. Sending the jaw once here
        # (and only servo_cp in the loop) keeps the arm in CARTESIAN_SPACE.
        self.openGripperDegree(psm_id=psm_id, degree=maintain_jaw_angle, sleep=0.05)

        goal_pos = np.asarray(goal_pose_base_fee[-3:], dtype=float)
        goal_quat = np.asarray(goal_pose_base_fee[:4], dtype=float)

        # choose the roll representation (goal or 180 deg shaft flip) nearest the
        # zero-roll orientation so the wrist roll joint stays near 0 (off its
        # limit). The reference is captured once, computed from joints (no motion).
        if self.init_quat.get(psm_id) is None:
            self.init_quat[psm_id] = self._zero_roll_ref(psm_id)
        ref_quat = self.init_quat.get(psm_id)
        if ref_quat is not None:
            goal_quat = self._nearest_roll_goal(goal_quat, ref_quat)

        # --- DEBUG: current measured pose vs goal, both base->FEE. When the robot
        # is already at the goal this offset should be ~0.
        cur = self._latest_base_fee_pose(psm_id)
        if cur is not None:
            print(f"[goal check] PSM{psm_id} current base_fee pos: {np.asarray(cur[-3:])}, "
                  f"goal pos: {goal_pos}, offset: {goal_pos - np.asarray(cur[-3:], dtype=float)}")

        # --- Plan a collision-free tip path around the other (stationary) arm ---
        if cur is None:
            self.node.get_logger().error(
                f"PSM{psm_id}: no measured pose yet; cannot plan, skipping move")
            self._publish_done(psm_id)
            return
        start_pos = np.asarray(cur[-3:], dtype=float)
        start_quat = np.asarray(cur[:4], dtype=float)

        obstacle = self._obstacle_shaft_cam(psm_id)  # other arm's shaft (snapshot)
        path = self._plan_path(psm_id, start_pos, goal_pos, obstacle)
        if path is None:
            # goal in collision or no path found -> hold, but let the sim advance
            self._publish_done(psm_id)
            return
        if len(path) > 2:
            print(f"[plan] PSM{psm_id} routing around other arm: {len(path)} waypoints")

        # --- Follow the waypoints; interpolate orientation by cumulative arc len ---
        seglen = [np.linalg.norm(path[k + 1] - path[k]) for k in range(len(path) - 1)]
        total = sum(seglen) or 1.0
        cum = 0.0
        for k in range(1, len(path)):
            cum += seglen[k - 1]
            frac = cum / total
            target_quat = slerp_frac(start_quat, goal_quat, frac)
            is_final = (k == len(path) - 1)
            status = self._servo_to_target(
                psm_id, set_ee_pub, np.asarray(path[k], float), target_quat,
                pos_dist_th=pos_dist_th if is_final else WAYPOINT_POS_TH,
                angle_dist_th=angle_dist_th if is_final else WAYPOINT_ANG_TH,
                pos_step=pos_step, angle_step_deg=angle_step_deg, sleep=sleep,
                max_iters=max_iters if is_final else 2000, settle_iters=400,
            )
            if status == 'stopped':
                return  # 'h' homing preempt -> hold here, no done (homing owns it)
            if status == 'aborted':
                # live collision guard tripped -> hold here, but tell the sim
                self._publish_done(psm_id)
                return

        # reached goal / settled / max-iters -> the move is done, tell the sim.
        self._publish_done(psm_id)

    def _servo_to_target(self, psm_id, set_ee_pub, target_pos, target_quat,
                         pos_dist_th, angle_dist_th, pos_step, angle_step_deg,
                         sleep, max_iters, settle_iters):
        """
        Closed-loop servo of one FEE target (position + quat, base frame). Steps a
        small bounded amount each iteration from the LIVE measured pose so every
        servo_cp command stays close to the current pose (no PID tracking fault),
        and re-checks clearance to the other arm live (guards against it moving).

        Returns: 'reached' | 'settled' | 'maxiters' | 'stopped' | 'aborted'.
        Does NOT publish /done or touch the jaw -- the caller owns those.
        """
        best_pos_dist = np.inf
        best_angle_dist = np.inf
        no_improve = 0
        start_meas_pos = None  # first measured pos, to report how far we actually moved
        for _ in range(max_iters):
            self._wait_if_paused()  # SPACE pause: hold pose here, resume in place
            if self._preempt.is_set():  # 'h' homing -> abort so homing takes over
                print("[HOME] motion preempted for homing")
                return 'stopped'

            try:
                cur = self._latest_base_fee_pose(psm_id)
                current_pos = np.asarray(cur[-3:], dtype=float)
                current_quat = np.asarray(cur[:4], dtype=float)
            except Exception as e:
                self.node.get_logger().info('Could not get current pose: {}'.format(e))
                time.sleep(sleep)
                continue

            if start_meas_pos is None:
                start_meas_pos = current_pos.copy()

            pos_dist = np.linalg.norm(target_pos - current_pos)
            angle_dist, _ = angleDist(current_quat, target_quat)
            if pos_dist < pos_dist_th and angle_dist < angle_dist_th:
                return 'reached'

            # move the arm 2x slower while its jaw is closed (< PSM_CLOSED_JAW_DEG)
            # -> smaller per-command increments at the same cadence.
            jaw_rad = self._latest_jaw(psm_id)
            jaw_closed = (jaw_rad is not None and
                          jaw_rad * 180.0 / np.pi < PSM_CLOSED_JAW_DEG)
            step_scale = 1.0 / PSM_CLOSED_SLOW_FACTOR if jaw_closed else 1.0
            eff_pos_step = pos_step * step_scale
            eff_angle_step = angle_step_deg * step_scale

            # scale the min-progress ("settled") thresholds with the step size:
            # slower motion makes less progress per iteration, so a fixed
            # threshold would falsely read as "no progress" and stop early.
            pos_improve_th = 5e-5 * step_scale
            angle_improve_th = 5e-4 * step_scale
            if (pos_dist < best_pos_dist - pos_improve_th or
                    angle_dist < best_angle_dist - angle_improve_th):
                best_pos_dist = min(best_pos_dist, pos_dist)
                best_angle_dist = min(best_angle_dist, angle_dist)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= settle_iters:
                    moved = float(np.linalg.norm(current_pos - start_meas_pos))
                    print(f"[settled] PSM{psm_id} no further progress "
                          f"(pos_dist={pos_dist:.4f} m, angle_dist={angle_dist:.4f} rad); "
                          f"measured moved {moved*1000:.1f} mm this move")
                    if moved < 1e-3:
                        self.node.get_logger().warn(
                            f"PSM{psm_id}: commanded servo_cp but the arm barely "
                            f"moved ({moved*1000:.1f} mm) -- not tracking (fault / "
                            f"wrong control space / joint limit) or measured_cp stale")
                    return 'settled'

            if pos_dist > 1e-9:
                next_pos = current_pos + \
                    (target_pos - current_pos) * min(eff_pos_step, pos_dist) / pos_dist
            else:
                next_pos = current_pos
            next_quat = slerp(current_quat, target_quat, max_move_angle=eff_angle_step) \
                if angle_dist > 1e-6 else current_quat

            # live safety guard: the planned path assumed a static obstacle; if the
            # other arm has moved into our way, refuse the step and hold.
            obstacle = self._obstacle_shaft_cam(psm_id)
            if obstacle is not None:
                B_o, T_o = obstacle
                B_a = self._base_origin_cam(psm_id)
                T_a = self._base_to_cam(psm_id, next_pos)
                if (seg_seg_dist(B_a, T_a, B_o, T_o) < SHAFT_HARD_MIN or
                        np.linalg.norm(T_a - T_o) < SHAFT_HARD_MIN):
                    self.node.get_logger().error(
                        f"PSM{psm_id}: live collision guard tripped; holding pose")
                    return 'aborted'

            self._publishPoseBaseFee(
                psm_id=psm_id, set_ee_pub=set_ee_pub,
                pos=next_pos, quat=next_quat, sleep_time=sleep,
            )
        print("max iters reached")
        return 'maxiters'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--speedy',           action="store_true")
    # parser.add_argument('--calib',            default=None)
    parser.add_argument('--psm_calibrate',
        default=os.path.dirname(__file__) + "/../RaftStereo/assets/psm_calibration_servo.npz")
    parser.add_argument('--psm_home',
        default=os.path.dirname(__file__) + "/../RaftStereo/assets/psm_home.npz",
        help="npz with 4x4 base->FEE home pose per arm (keys PSM1/PSM2, mm)")
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

    # keyboard listener: SPACE toggles pause/resume; H retracts both grippers
    kb_thread = threading.Thread(target=move.keyboard_listener, daemon=True)
    kb_thread.start()

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
        move._restore_terminal()  # deterministic tty restore on Ctrl-C / exit
        executor.shutdown()
        move.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=2.0)

