import sys
from types import SimpleNamespace
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import Mock


REPO_ROOT = Path(__file__).resolve().parents[1]
READER_DIR = REPO_ROOT / 'oculus_reader'
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(READER_DIR))

from oculus_reader.visualize_oculus_transforms_ros2 import OculusReaderNode


def _buttons(**changes):
    values = {
        'A': False, 'B': False, 'X': False, 'Y': False,
        'LJ': False, 'LG': False, 'LTr': False,
        'RJ': False, 'RG': False, 'RTr': False,
        'leftJS': [0.0, 0.0], 'rightJS': [0.0, 0.0],
        'leftGrip': [0.0], 'rightGrip': [0.0],
    }
    values.update(changes)
    return values


class TeleopButtonMappingTest(TestCase):
    @staticmethod
    def _node(side):
        service = Mock()
        service.wait_for_service.return_value = True
        service.call.return_value = SimpleNamespace(success=True, message='ok')
        node = SimpleNamespace(
            button_info_init=False,
            button_triggered_dict={},
            enable_legacy_robot_actions=True,
            enable_gripper_actions=True,
            enable_legacy_dataset_actions=False,
            teleop_button_side=side,
            index_trigger_toggle_gripper=False,
            gripper_toggle_closed={'left': False, 'right': False},
            tele_cli=service,
            dataset_record_cli=Mock(),
            dataset_stop_cli=Mock(),
            phase_state_cli=Mock(),
            robotiq_l=Mock(),
            robotiq_r=Mock(),
            get_logger=Mock(return_value=Mock()),
        )
        return node, service

    def test_left_buttons_work_without_right_controller_fields(self):
        node, service = self._node('left')

        # Deliberately omit all right-controller fields.
        OculusReaderNode.button_funcs(node, {
            'X': False, 'Y': False, 'LJ': False, 'LG': False, 'LTr': False,
            'leftJS': [0.0, 0.0], 'leftGrip': [0.0],
        })
        OculusReaderNode.button_funcs(node, _buttons(X=True))
        OculusReaderNode.button_funcs(node, _buttons())
        OculusReaderNode.button_funcs(node, _buttons(Y=True))

        self.assertEqual(
            [call.args[0].data for call in service.call.call_args_list],
            [True, False],
        )

    def test_default_right_mapping_ignores_left_teleop_buttons(self):
        node, service = self._node('right')

        OculusReaderNode.button_funcs(node, _buttons())
        OculusReaderNode.button_funcs(node, _buttons(X=True))

        service.call.assert_not_called()

    def test_each_index_trigger_toggles_its_gripper_and_grips_are_deadman_only(self):
        node, service = self._node('right')
        node.enable_legacy_robot_actions = False
        node.index_trigger_toggle_gripper = True

        OculusReaderNode.button_funcs(node, _buttons())
        for trigger in ('LTr', 'RTr'):
            OculusReaderNode.button_funcs(node, _buttons(**{trigger: True}))
            OculusReaderNode.button_funcs(node, _buttons())
            OculusReaderNode.button_funcs(node, _buttons(**{trigger: True}))
            OculusReaderNode.button_funcs(node, _buttons())
        OculusReaderNode.button_funcs(
            node, _buttons(leftGrip=[1.0], rightGrip=[1.0])
        )

        for publisher in (node.robotiq_l, node.robotiq_r):
            self.assertEqual(
                [list(call.args[0].data)
                 for call in publisher.publish.call_args_list],
                [[0.8], [0.0]],
            )
        service.call.assert_not_called()


if __name__ == '__main__':
    main()
