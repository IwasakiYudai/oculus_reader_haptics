from oculus_reader.FPS_counter import FPSCounter
from oculus_reader.buttons_parser import parse_buttons
import numpy as np
import threading
import time
import os
import subprocess
from ppadb.client import Client as AdbClient
import sys
from haptics_trigger import HapticClient

def eprint(*args, **kwargs):
    RED = "\033[1;31m"  
    sys.stderr.write(RED)
    print(*args, file=sys.stderr, **kwargs)
    RESET = "\033[0;0m"
    sys.stderr.write(RESET)

class OculusReader:
    def __init__(self,
            ip_address=None,
            port = 5555,
            APK_name='com.rail.oculus.teleop',
            print_FPS=False,
            run=True,
            adb_server_host=None,
            adb_server_port=None,
            adb_serial=None,
        ):
        self.running = False
        self.last_transforms = {}
        self.last_buttons = {}
        # Host-side acquisition metadata.  The APK protocol has no source
        # timestamp, so this sequence and monotonic timestamp identify actual
        # logcat samples.  Consumers must not turn repeated reads of
        # last_transforms into apparently fresh tracking data.
        self.last_sample_sequence = 0
        self.last_sample_monotonic = None
        self._lock = threading.Lock()
        self.tag = 'wE9ryARX'

        self.ip_address = ip_address
        self.port = port
        self.APK_name = APK_name
        self.print_FPS = print_FPS
        self.adb_server_host = adb_server_host or os.environ.get(
            'OCULUS_ADB_SERVER_HOST', '127.0.0.1')
        self.adb_server_port = int(
            adb_server_port or os.environ.get('OCULUS_ADB_SERVER_PORT', '5037'))
        self.adb_serial = adb_serial or os.environ.get('OCULUS_ADB_SERIAL')
        if self.print_FPS:
            self.fps_counter = FPSCounter()

        self.device = self.get_device()
        self.install(verbose=False)

        self.haptic_client = HapticClient(
            adb_serial=self.device.serial,
            adb_server_host=self.adb_server_host,
            adb_server_port=self.adb_server_port,
        )
        self.haptic_amp_l = 0.0
        self.haptic_amp_r = 0.0
        # create two watchdogs that reset the haptic amplitudes to zero after some time
        self.haptic_reset_interval = 5.0  # seconds
        self.haptic_cmd_got_l = time.time()
        self.haptic_cmd_got_r = time.time()
        
        if run:
            self.run()

    def __del__(self):
        self.stop()

    def run(self):
        self.running = True
        self.device.shell('am start -n "com.rail.oculus.teleop/com.rail.oculus.teleop.MainActivity" -a android.intent.action.MAIN -c android.intent.category.LAUNCHER')
        self.thread = threading.Thread(
            target=self.device.shell,
            args=("logcat -T 0", self.read_logcat_by_line),
            daemon=True,
        )
        self.thread.start()

    def stop(self):
        self.running = False
        self.haptic_amp_l = 0.0
        self.haptic_amp_r = 0.0
        if hasattr(self, 'haptic_client'):
            self.haptic_client.close()
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2.0)

    def _run_adb(self, *args):
        cmd = [
            'adb',
            '-H', self.adb_server_host,
            '-P', str(self.adb_server_port),
        ]
        if self.adb_serial:
            cmd += ['-s', self.adb_serial]
        return subprocess.run(cmd + list(args), check=False)

    def get_network_device(self, client, retry=0):
        try:
            client.remote_connect(self.ip_address, self.port)
        except RuntimeError:
            self._run_adb('devices')
            client.remote_connect(self.ip_address, self.port)
        device = client.device(self.ip_address + ':' + str(self.port))

        if device is None:
            if retry==1:
                self._run_adb('tcpip', str(self.port))
            if retry==2:
                eprint('Make sure that device is running and is available at the IP address specified as the OculusReader argument `ip_address`.')
                eprint('Currently provided IP address:', self.ip_address)
                eprint('Run `adb shell ip route` to verify the IP address.')
                exit(1)
            else:
                self.get_device(client=client, retry=retry+1)
        return device

    def get_usb_device(self, client):
        try:
            devices = client.devices()
        except RuntimeError:
            self._run_adb('devices')
            devices = client.devices()

        usb_devices = [device for device in devices if device.serial.count('.') < 3]
        if self.adb_serial:
            for device in usb_devices:
                if device.serial == self.adb_serial:
                    return device
            raise RuntimeError(
                f'ADB device {self.adb_serial!r} not found on '
                f'{self.adb_server_host}:{self.adb_server_port}.')

        if len(usb_devices) == 1:
            return usb_devices[0]
        if len(usb_devices) > 1:
            serials = ', '.join(device.serial for device in usb_devices)
            raise RuntimeError(
                'Multiple USB ADB devices found. Set OCULUS_ADB_SERIAL to one '
                f'of: {serials}')
        eprint('Device not found. Make sure that device is running and is connected over USB')
        eprint('Run `adb devices` to verify that the device is visible.')
        raise RuntimeError('No USB ADB device found.')

    def get_device(self):
        client = AdbClient(host=self.adb_server_host, port=self.adb_server_port)
        if self.ip_address is not None:
            return self.get_network_device(client)
        else:
            return self.get_usb_device(client)

    def install(self, APK_path=None, verbose=True, reinstall=False):
        try:
            installed = self.device.is_installed(self.APK_name)
            if not installed or reinstall:
                if APK_path is None:
                    APK_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'APK', 'teleop-debug.apk')
                success = self.device.install(APK_path, test=True, reinstall=reinstall)
                installed = self.device.is_installed(self.APK_name)
                if installed and success:
                    print('APK installed successfully.')
                else:
                    eprint('APK install failed.')
            elif verbose:
                print('APK is already installed.')
        except RuntimeError:
            eprint('Device is visible but could not be accessed.')
            eprint('Run `adb devices` to verify that the device is visible and accessible.')
            eprint('If you see "no permissions" next to the device serial, please put on the Oculus Quest and allow the access.')
            exit(1)

    def uninstall(self, verbose=True):
        try:
            installed = self.device.is_installed(self.APK_name)
            if installed:
                success = self.device.uninstall(self.APK_name)
                installed = self.device.is_installed(self.APK_name)
                if not installed and success:
                    print('APK uninstall finished.')
                    print('Please verify if the app disappeared from the list as described in "UNINSTALL.md".')
                    print('For the resolution of this issue, please follow https://github.com/Swind/pure-python-adb/issues/71.')
                else:
                    eprint('APK uninstall failed')
            elif verbose:
                print('APK is not installed.')
        except RuntimeError:
            eprint('Device is visible but could not be accessed.')
            eprint('Run `adb devices` to verify that the device is visible and accessible.')
            eprint('If you see "no permissions" next to the device serial, please put on the Oculus Quest and allow the access.')
            exit(1)

    @staticmethod
    def process_data(string):
        try:
            transforms_string, buttons_string = string.split('&')
        except ValueError:
            return None, None
        split_transform_strings = transforms_string.split('|')
        transforms = {}
        for pair_string in split_transform_strings:
            transform = np.empty((4,4))
            pair = pair_string.split(':')
            if len(pair) != 2:
                continue
            left_right_char = pair[0] # is r or l
            transform_string = pair[1]
            values = transform_string.split(' ')
            c = 0
            r = 0
            count = 0
            for value in values:
                if not value:
                    continue
                transform[r][c] = float(value)
                c += 1
                if c >= 4:
                    c = 0
                    r += 1
                count += 1
            if count == 16:
                transforms[left_right_char] = transform
        buttons = parse_buttons(buttons_string)
        return transforms, buttons

    def extract_data(self, line):
        output = ''
        if self.tag in line:
            try:
                output += line.split(self.tag + ': ')[1]
            except ValueError:
                pass
        return output

    def get_transformations_and_buttons(self):
        with self._lock:
            return self.last_transforms, self.last_buttons

    def get_snapshot(self):
        """Return the latest controller sample and host acquisition metadata.

        ``last_sample_sequence`` changes only when a new APK/logcat record is
        parsed.  This preserves the existing reader API while allowing ROS
        bridges to publish each physical sample once instead of continuously
        re-stamping stale data.
        """
        with self._lock:
            return (
                self.last_transforms,
                self.last_buttons,
                self.last_sample_sequence,
                self.last_sample_monotonic,
            )

    def read_logcat_by_line(self, connection):
        file_obj = connection.socket.makefile()
        while self.running:
            try:
                line = file_obj.readline().strip()
                if not self.running:
                    break
                data = self.extract_data(line)
                if data:
                    transforms, buttons = OculusReader.process_data(data)
                    with self._lock:
                        self.last_transforms, self.last_buttons = transforms, buttons
                        self.last_sample_sequence += 1
                        self.last_sample_monotonic = time.monotonic()
                    if self.print_FPS:
                        self.fps_counter.getAndPrintFPS()
                
                # Trigger haptics based on stored amplitudes
                self.reset_haptics() # reset if no command received recently to avoid continuous vibration
                self.haptic_client.send_haptics(amp_l=self.haptic_amp_l, amp_r=self.haptic_amp_r)

            except UnicodeDecodeError:
                pass
        file_obj.close()
        connection.close()

    def set_haptic_left(self, intensity: float):
        """Set left hand haptic intensity ∈ [0,1]"""
        self.haptic_amp_l = max(0.0, min(1.0, float(intensity)))
        self.haptic_cmd_got_l = time.time()

    def set_haptic_right(self, intensity: float):
        """Set right hand haptic intensity ∈ [0,1]"""
        self.haptic_amp_r = max(0.0, min(1.0, float(intensity)))
        self.haptic_cmd_got_r = time.time()

    def reset_haptics(self):
        """Reset both hand haptic intensities to zero"""
        if time.time() - self.haptic_cmd_got_l > self.haptic_reset_interval:
            self.haptic_amp_l = 0.0
        if time.time() - self.haptic_cmd_got_r > self.haptic_reset_interval:
            self.haptic_amp_r = 0.0


def main():
    oculus_reader = OculusReader()

    while True:
        time.sleep(0.3)
        print(oculus_reader.get_transformations_and_buttons())


if __name__ == '__main__':
    main()
