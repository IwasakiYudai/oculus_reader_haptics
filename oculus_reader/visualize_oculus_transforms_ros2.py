from reader  import OculusReader
from oculus_reader.coordinate_conversion import quest_pose_to_ros
import math
import rclpy
from rclpy.node import Node
from rclpy.timer import Rate
import tf2_ros
from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import Joy
from rclpy import time, clock
from std_srvs.srv import SetBool,Trigger
from std_msgs.msg import Bool, Float64MultiArray

# For spinning
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from threading import Thread

from scipy.spatial.transform import Rotation

class OculusReaderNode(Node):
    def __init__(self):
        super().__init__('oculus_reader')
        self.headset_mount_roll_deg = float(self.declare_parameter(
            'headset_mount_roll_deg', 0.0).value)
        if not math.isfinite(self.headset_mount_roll_deg):
            raise ValueError('headset_mount_roll_deg must be finite')
        self.headset_mount_rotation = Rotation.from_euler(
            'x', self.headset_mount_roll_deg, degrees=True).as_matrix()
        self.oculus_reader = OculusReader()
        self.br = tf2_ros.TransformBroadcaster(self)

        self.clock = rclpy.clock.Clock()
        self.last_time = self.clock.now()

        # subscriber
        self.create_subscription(Float64MultiArray, '/L_haptic_amplitude', self.left_haptic_callback, 10)
        self.create_subscription(Float64MultiArray, '/R_haptic_amplitude', self.right_haptic_callback, 10)

        # publisher
        self.robotiq_l = self.create_publisher(Float64MultiArray, '/L_gripper_forward_position_controller/commands', 10)
        self.robotiq_r = self.create_publisher(Float64MultiArray, '/R_gripper_forward_position_controller/commands', 10)
        self.pose_l_pub = self.create_publisher(PoseStamped, '/oculus/left/pose', 10)
        self.pose_r_pub = self.create_publisher(PoseStamped, '/oculus/right/pose', 10)
        self.input_l_pub = self.create_publisher(Joy, '/oculus/left/input', 10)
        self.input_r_pub = self.create_publisher(Joy, '/oculus/right/input', 10)
        self.valid_l_pub = self.create_publisher(Bool, '/oculus/left/pose_valid', 10)
        self.valid_r_pub = self.create_publisher(Bool, '/oculus/right/pose_valid', 10)
        # The upstream example mixes generic Quest input with application
        # services and direct gripper commands.  Keep those legacy side effects
        # opt-in so grip can safely serve as a hold-to-run signal.
        self.enable_legacy_robot_actions = self.declare_parameter(
            'enable_legacy_robot_actions', False).value
        self.enable_gripper_actions = bool(self.declare_parameter(
            'enable_gripper_actions',
            self.enable_legacy_robot_actions).value)
        self.enable_legacy_dataset_actions = self.declare_parameter(
            'enable_legacy_dataset_actions', False).value
        self.teleop_button_side = str(self.declare_parameter(
            'teleop_button_side', 'right').value).lower()
        if self.teleop_button_side not in ('left', 'right'):
            raise ValueError('teleop_button_side must be left or right')
        self.index_trigger_toggle_gripper = bool(self.declare_parameter(
            'index_trigger_toggle_gripper', False).value)
        self.gripper_toggle_closed = {'left': False, 'right': False}

        # service
        self.tele_cli = self.create_client(SetBool, '/teleop_start')
        self.dataset_record_cli = self.create_client(Trigger, '/start_recording')
        self.dataset_stop_cli = self.create_client(SetBool, '/stop_recording')
        self.phase_state_cli = self.create_client(Trigger, '/stage_up')
        self.button_count_poll= 0
        self.button_info_init = False
        self.button_triggered_dict = {}
        self.last_sample_sequence = 0
        self.running = True

        # Create a thread to run the main loop
        self.rate = self.create_rate(70)  # 90 Hz
        self.thread = Thread(target=self.loop, daemon=True)
        self.thread.start()
        self.get_logger().info(
            'Quest frame calibration: headset_mount_roll_deg=%.1f'
            % self.headset_mount_roll_deg)

        

    def loop(self):
        while self.running and rclpy.ok():
            transformations, buttons, sequence, _ = self.oculus_reader.get_snapshot()
            if sequence == 0 or sequence == self.last_sample_sequence:
                self.rate.sleep()
                continue

            # One ROS stamp per newly acquired APK/logcat sample.  In
            # particular, never refresh a cached sample merely because this
            # polling loop ran again.
            self.last_sample_sequence = sequence
            sample_stamp = self.clock.now().to_msg()
            self.last_time = self.clock.now()
            transformations = transformations or {}
            buttons = buttons or {}
            self.button_funcs(buttons)

            valid_l = False
            valid_r = False
            if 'l' in transformations:
                tf_l, valid_l = self.publish_transform(
                    transformations['l'], 'oculus_l', sample_stamp)
                if valid_l:
                    self.br.sendTransform(tf_l)
                    self.pose_l_pub.publish(self.transform_to_pose(tf_l))
            if 'r' in transformations:
                tf_r, valid_r = self.publish_transform(
                    transformations['r'], 'oculus_r', sample_stamp)
                if valid_r:
                    self.br.sendTransform(tf_r)
                    self.pose_r_pub.publish(self.transform_to_pose(tf_r))

            valid_msg = Bool()
            valid_msg.data = valid_l
            self.valid_l_pub.publish(valid_msg)
            valid_msg = Bool()
            valid_msg.data = valid_r
            self.valid_r_pub.publish(valid_msg)
            self.publish_controller_inputs(buttons, sample_stamp)

            if not valid_l or not valid_r:
                self.get_logger().warning(
                    'Missing/invalid controller transform in source sample; '
                    'affected pose was not published.',
                    throttle_duration_sec=2.0)
            self.rate.sleep()

    def stop(self):
        self.running = False
        self.oculus_reader.stop()
        self.thread.join(timeout=2.0)

    def button_funcs(self, buttons):
        # A disconnected controller may omit its half of the input fields.
        # Keep the connected controller usable and treat omitted inputs as idle.
        buttons = dict(buttons)
        for key in ('A', 'B', 'X', 'Y', 'LJ', 'LG', 'LTr',
                    'RJ', 'RG', 'RTr'):
            buttons.setdefault(key, False)
        for key, default in (('leftJS', [0.0, 0.0]),
                             ('rightJS', [0.0, 0.0]),
                             ('leftGrip', [0.0]),
                             ('rightGrip', [0.0])):
            buttons.setdefault(key, default)

        if not self.button_info_init:
            for key in buttons.keys():
                self.button_triggered_dict[key] = False
            self.button_info_init = True


        # keys: dict_keys([
        ## -- boolean values --
        # 'A',      : Button A on the right controller
        # 'B',      : Button B on the right controller
        # 'RThU',   : Right Thumbstick Up
        # 'RJ',     : Right Joystick Button
        # 'RG',     : Right Grip Button (Bottom trigger)
        # 'RTr',    : Right Trigger Button (Top trigger)
        # 'X',      : Button X on the left controller
        # 'Y',      : Button Y on the left controller
        # 'LThU',   : Left Thumbstick Up
        # 'LJ',     : Left Joystick Button
        # 'LG',     : Left Grip Button (Bottom trigger)
        # 'LTr',   : Left Trigger Button (Top trigger)
        ## -- double values --
        # 'leftJS', : Left Joystick Button (2d as [double, double] for x and y axis)
        # 'leftTrig', : Left Trigger Button (1d as [double] for pressure)
        # 'leftGrip', : Left Grip Button (1d as [double] for pressure)
        # 'rightJS', : Right Joystick Button (2d as [double, double] for x and y axis)
        # 'rightTrig', : Right Trigger Button (1d as [double] for pressure)
        # 'rightGrip' : Right Grip Button (1d as [double] for pressure)
        # ])

        if buttons['A'] and not self.button_triggered_dict['A']:
            self.button_triggered_dict['A'] = True
            if (self.enable_legacy_robot_actions
                    and self.teleop_button_side == 'right'
                    and self.tele_cli.wait_for_service(timeout_sec=1.0)):
                req = SetBool.Request()
                req.data = True
                result = self.tele_cli.call(req)
                self.get_logger().info(f'Result of service call: {result.success}, message: {result.message}')
            elif (self.enable_legacy_robot_actions
                    and self.teleop_button_side == 'right'):
                self.get_logger().error('Service not available')
        elif not buttons['A'] and self.button_triggered_dict['A']:
            self.button_triggered_dict['A'] = False

        if buttons['B'] and not self.button_triggered_dict['B']:
            self.button_triggered_dict['B'] = True
            if (self.enable_legacy_robot_actions
                    and self.teleop_button_side == 'right'
                    and self.tele_cli.wait_for_service(timeout_sec=1.0)):
                req = SetBool.Request()
                req.data = False
                result = self.tele_cli.call(req)
                self.get_logger().info(f'Result of service call: {result.success}, message: {result.message}')
            elif (self.enable_legacy_robot_actions
                    and self.teleop_button_side == 'right'):
                self.get_logger().error('Service not available')
        elif not buttons['B'] and self.button_triggered_dict['B']:
            self.button_triggered_dict['B'] = False

        if buttons['X'] and not self.button_triggered_dict['X']:
            self.button_triggered_dict['X'] = True
            if (self.enable_legacy_robot_actions
                    and self.teleop_button_side == 'left'
                    and self.tele_cli.wait_for_service(timeout_sec=1.0)):
                req = SetBool.Request()
                req.data = True
                result = self.tele_cli.call(req)
                self.get_logger().info(f'Result of service call: {result.success}, message: {result.message}')
            elif (self.enable_legacy_robot_actions
                    and self.teleop_button_side == 'left'):
                self.get_logger().error('Service not available')
            # for starting a new episode
            if (self.enable_legacy_dataset_actions
                    and self.dataset_record_cli.wait_for_service(timeout_sec=1.0)):
                req = Trigger.Request()
                result = self.dataset_record_cli.call(req)
                self.get_logger().info(f'Result of service call: {result.success}, message: {result.message}')
            elif self.enable_legacy_dataset_actions:
                self.get_logger().error('Service not available')
        elif not buttons['X'] and self.button_triggered_dict['X']:
            self.button_triggered_dict['X'] = False

        if buttons['Y'] and not self.button_triggered_dict['Y']:
            self.button_triggered_dict['Y'] = True
            if (self.enable_legacy_robot_actions
                    and self.teleop_button_side == 'left'
                    and self.tele_cli.wait_for_service(timeout_sec=1.0)):
                req = SetBool.Request()
                req.data = False
                result = self.tele_cli.call(req)
                self.get_logger().info(f'Result of service call: {result.success}, message: {result.message}')
            elif (self.enable_legacy_robot_actions
                    and self.teleop_button_side == 'left'):
                self.get_logger().error('Service not available')
            # for stopping and saving current episode
            if (self.enable_legacy_dataset_actions
                    and self.dataset_stop_cli.wait_for_service(timeout_sec=1.0)):
                req = SetBool.Request()
                req.data = True
                result = self.dataset_stop_cli.call(req)
                self.get_logger().info(f'Result of service call: {result.success}, message: {result.message}')
            elif self.enable_legacy_dataset_actions:
                self.get_logger().error('Service not available')
        elif not buttons['Y'] and self.button_triggered_dict['Y']:
            self.button_triggered_dict['Y'] = False

        if buttons['LJ'] and not self.button_triggered_dict['LJ']:
            self.button_triggered_dict['LJ'] = True
            # dispose current episode
            if (self.enable_legacy_dataset_actions
                    and self.dataset_stop_cli.wait_for_service(timeout_sec=1.0)):
                req = SetBool.Request()
                req.data = False
                result = self.dataset_stop_cli.call(req)
                self.get_logger().info(f'Result of service call: {result.success}, message: {result.message}')
            elif self.enable_legacy_dataset_actions:
                self.get_logger().error('Service not available')
        elif not buttons['LJ'] and self.button_triggered_dict['LJ']:
            self.button_triggered_dict['LJ'] = False

        # gripper control
        if buttons['LTr'] and not self.button_triggered_dict['LTr']:
            self.button_triggered_dict['LTr'] = True
            if self.enable_gripper_actions:
                msg = Float64MultiArray()
                if self.index_trigger_toggle_gripper:
                    self.gripper_toggle_closed['left'] = not self.gripper_toggle_closed['left']
                    msg.data = [0.8 if self.gripper_toggle_closed['left'] else 0.0]
                else:
                    msg.data = [0.0]
                self.robotiq_l.publish(msg)
        elif not buttons['LTr'] and self.button_triggered_dict['LTr']:
            self.button_triggered_dict['LTr'] = False

        if buttons['LG'] and not self.button_triggered_dict['LG']:
            self.button_triggered_dict['LG'] = True

        elif not buttons['LG'] and self.button_triggered_dict['LG']:
            self.button_triggered_dict['LG'] = False

        if buttons['RTr'] and not self.button_triggered_dict['RTr']:
            self.button_triggered_dict['RTr'] = True
            if self.enable_gripper_actions:
                msg = Float64MultiArray()
                if self.index_trigger_toggle_gripper:
                    self.gripper_toggle_closed['right'] = not self.gripper_toggle_closed['right']
                    msg.data = [0.8 if self.gripper_toggle_closed['right'] else 0.0]
                else:
                    msg.data = [0.0]
                self.robotiq_r.publish(msg)
        elif not buttons['RTr'] and self.button_triggered_dict['RTr']:
            self.button_triggered_dict['RTr'] = False

        if buttons['RG'] and not self.button_triggered_dict['RG']:
            self.button_triggered_dict['RG'] = True
            
        elif not buttons['RG'] and self.button_triggered_dict['RG']:
            self.button_triggered_dict['RG'] = False

        if buttons['rightJS'][0] > 0.8 and not self.button_triggered_dict['rightJS']:
            self.button_triggered_dict['rightJS'] = True
            # phase up
            if (self.enable_legacy_dataset_actions
                    and self.phase_state_cli.wait_for_service(timeout_sec=1.0)):
                req = Trigger.Request()
                result = self.phase_state_cli.call(req)
                self.get_logger().info(f'Result of service call: {result.success}, message: {result.message}')
            elif self.enable_legacy_dataset_actions:
                self.get_logger().error('Service not available')
        elif buttons['rightJS'][0] < 0.2 and self.button_triggered_dict['rightJS']:
            self.button_triggered_dict['rightJS'] = False

        # Using LeftGrasp control left gripper to  decrease accidental triggering
        if buttons['leftGrip'][0] > 0.99 and not self.button_triggered_dict['leftGrip']:
            self.button_triggered_dict['leftGrip'] = True
            if (self.enable_gripper_actions
                    and not self.index_trigger_toggle_gripper):
                msg = Float64MultiArray()
                msg.data = [0.8]  # close
                self.robotiq_l.publish(msg)
        elif buttons['leftGrip'][0] < 0.2 and self.button_triggered_dict['leftGrip']:
            self.button_triggered_dict['leftGrip'] = False
        
        if buttons['rightGrip'][0] > 0.99 and not self.button_triggered_dict['rightGrip']:
            self.button_triggered_dict['rightGrip'] = True
            if (self.enable_gripper_actions
                    and not self.index_trigger_toggle_gripper):
                msg = Float64MultiArray()
                msg.data = [0.8]  # close
                self.robotiq_r.publish(msg)
        elif buttons['rightGrip'][0] < 0.2 and self.button_triggered_dict['rightGrip']:
            self.button_triggered_dict['rightGrip'] = False


    @staticmethod
    def transform_to_pose(transform):
        pose = PoseStamped()
        pose.header = transform.header
        pose.pose.position.x = transform.transform.translation.x
        pose.pose.position.y = transform.transform.translation.y
        pose.pose.position.z = transform.transform.translation.z
        pose.pose.orientation = transform.transform.rotation
        return pose

    @staticmethod
    def _axis_values(buttons, key, size):
        values = buttons.get(key, ())
        if not isinstance(values, (tuple, list)):
            values = ()
        return [float(values[index]) if index < len(values) else 0.0
                for index in range(size)]

    def publish_controller_inputs(self, buttons, stamp):
        """Publish standard Joy messages for garment-side adapters.

        Left axes: joystick x/y, index trigger, grip trigger.
        Left buttons: X, Y, joystick click, grip click, index-trigger click.
        Right uses the same ordering with A, B, RJ, RG, RTr.
        """
        if any(key in buttons for key in ('X', 'Y', 'leftJS', 'leftGrip')):
            msg = Joy()
            msg.header.stamp = stamp
            msg.header.frame_id = 'oculus_l'
            msg.axes = (
                self._axis_values(buttons, 'leftJS', 2)
                + self._axis_values(buttons, 'leftTrig', 1)
                + self._axis_values(buttons, 'leftGrip', 1)
            )
            msg.buttons = [int(bool(buttons.get(key, False)))
                           for key in ('X', 'Y', 'LJ', 'LG', 'LTr')]
            self.input_l_pub.publish(msg)

        if any(key in buttons for key in ('A', 'B', 'rightJS', 'rightGrip')):
            msg = Joy()
            msg.header.stamp = stamp
            msg.header.frame_id = 'oculus_r'
            msg.axes = (
                self._axis_values(buttons, 'rightJS', 2)
                + self._axis_values(buttons, 'rightTrig', 1)
                + self._axis_values(buttons, 'rightGrip', 1)
            )
            msg.buttons = [int(bool(buttons.get(key, False)))
                           for key in ('A', 'B', 'RJ', 'RG', 'RTr')]
            self.input_r_pub.publish(msg)

    def publish_transform(self, transform, name, stamp=None):
        t = TransformStamped()
        t.header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()
        t.header.frame_id = 'oculus_base'
        t.child_frame_id = name
        try:
            translation, rotation = quest_pose_to_ros(
                transform, self.headset_mount_rotation)
        except ValueError:
            return t, False
        t.transform.translation.x = translation[0]
        t.transform.translation.y = translation[1]
        t.transform.translation.z = translation[2]
        quat = Rotation.from_matrix(rotation).as_quat()

        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        return t, True

    def left_haptic_callback(self, msg):
        intensity = msg.data[0]
        self.oculus_reader.set_haptic_left(intensity)
    
    def right_haptic_callback(self, msg):
        intensity = msg.data[0]
        self.oculus_reader.set_haptic_right(intensity)

def main():
    rclpy.init()
    node = OculusReaderNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
