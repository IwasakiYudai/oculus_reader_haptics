import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


REPO_ROOT = Path(__file__).resolve().parents[1]
READER_DIR = REPO_ROOT / "oculus_reader"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(READER_DIR))

from haptics_trigger import HapticClient
from reader import OculusReader


class Device:
    def __init__(self, serial):
        self.serial = serial


class AdbTargetingTest(unittest.TestCase):
    def test_reader_uses_environment_target_for_input_and_haptics(self):
        device = Device("QUEST-SERIAL")
        env = {
            "OCULUS_ADB_SERVER_HOST": "127.0.0.1",
            "OCULUS_ADB_SERVER_PORT": "5038",
            "OCULUS_ADB_SERIAL": device.serial,
        }
        with patch.dict(os.environ, env), \
                patch.object(OculusReader, "get_device", return_value=device), \
                patch.object(OculusReader, "install"), \
                patch("reader.HapticClient") as haptic_client:
            reader = OculusReader(run=False)

        self.assertEqual(reader.adb_server_port, 5038)
        self.assertEqual(reader.adb_serial, device.serial)
        haptic_client.assert_called_once_with(
            adb_serial=device.serial,
            adb_server_host="127.0.0.1",
            adb_server_port=5038,
        )

    def test_usb_selection_requires_exact_requested_serial(self):
        reader = OculusReader.__new__(OculusReader)
        reader.adb_serial = "QUEST-B"
        reader.adb_server_host = "127.0.0.1"
        reader.adb_server_port = 5038
        client = Mock()
        client.devices.return_value = [Device("QUEST-A"), Device("QUEST-B")]

        self.assertEqual(reader.get_usb_device(client).serial, "QUEST-B")

    def test_usb_selection_rejects_ambiguous_devices(self):
        reader = OculusReader.__new__(OculusReader)
        reader.adb_serial = None
        reader.adb_server_host = "127.0.0.1"
        reader.adb_server_port = 5037
        client = Mock()
        client.devices.return_value = [Device("QUEST-A"), Device("QUEST-B")]

        with self.assertRaisesRegex(RuntimeError, "Multiple USB ADB devices"):
            reader.get_usb_device(client)

    @patch("haptics_trigger.subprocess.run")
    def test_haptic_adb_command_uses_server_and_serial(self, run):
        client = HapticClient(
            adb_serial="QUEST-SERIAL",
            adb_server_host="127.0.0.1",
            adb_server_port=5038,
        )

        client._adb("devices")

        run.assert_called_once_with(
            [
                "adb", "-H", "127.0.0.1", "-P", "5038",
                "-s", "QUEST-SERIAL", "devices",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_haptic_close_sends_zero_without_reconnecting(self):
        client = HapticClient()
        sock = Mock()
        client.sock = sock

        client.close()

        payload = sock.sendall.call_args.args[0].decode("utf-8")
        self.assertIn('"amp_l": 0.0', payload)
        self.assertIn('"amp_r": 0.0', payload)
        sock.shutdown.assert_called_once()
        sock.close.assert_called_once()
        self.assertIsNone(client.sock)

    def test_reader_stop_is_bounded_and_closes_haptics(self):
        reader = OculusReader.__new__(OculusReader)
        reader.running = True
        reader.haptic_amp_l = 0.5
        reader.haptic_amp_r = 0.5
        reader.haptic_client = Mock()
        reader.thread = Mock()

        reader.stop()

        self.assertFalse(reader.running)
        self.assertEqual(reader.haptic_amp_l, 0.0)
        self.assertEqual(reader.haptic_amp_r, 0.0)
        reader.haptic_client.close.assert_called_once_with()
        reader.thread.join.assert_called_once_with(timeout=2.0)


if __name__ == "__main__":
    unittest.main()
