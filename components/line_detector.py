import wpilib
import time
import hal
import numpy as np

if hal.isSimulation():
    from hal_impl.serial_helpers import SerialSimBase

    class LineDetectorSerialSimulator(SerialSimBase):
        serial_string = "f0.1,0.2,0.3,0.4,0.5,0.6,0.7\nb0.9,0.2,0.3,0.4,0.5,0.6,9.8\n"

        def readString(self, count):
            num_strings = (count % len(self.serial_string))
            return ([self.serial_string]*num_strings)[:count]

        def readSerial(self, port, buffer, count, status):
            status.value = 0
            buffer[:] = [ord("c")] * count
            return count


class LineDetectorSensor:

    arduino_port: wpilib.SerialPort

    baud_rate: int = 115200
    read_char: int = 100

    num_sensors: int = 14

    readings_threshold = 15000

    reading_timeout: float = 0.4 # s

    front_char = 'f'
    back_char = 'b'

    packet_separator = '\n'
    value_separator = ','

    def __init__(self):
        self.readings = None
        self.last_reading_time = 0
        # read these to get the position
        self.position_cargo = None
        self.position_hatch = None

    def split_packet(self, packet):
        if self.arduino_port is None:
            return
        try:
            packet_split = [value for value in packet.split(self.value_separator)
                            if not value == ""]
            # todo - exclude empty strings here
            reading = [
                float(reading)
                for reading in packet_split
            ]
            if len(reading) == self.num_sensors:
                return reading
        except ValueError:
            pass
        return None

    def generate_positions(self):
        if self.readings is None:
            self.position_cargo = None
            self.position_hatch = None
            return
        indices_hatch = [i for i, reading in enumerate(self.readings[0:7])
                         if reading < self.readings_threshold]
        if indices_hatch:
            self.position_hatch = -(np.mean(indices_hatch) - 3)
        else:
            self.position_hatch = None
        indices_cargo = [i for i, reading in enumerate(self.readings[7:14])
                         if reading < self.readings_threshold]
        if indices_cargo:
            self.position_cargo = -(np.mean(indices_cargo) - 3)
        else:
            self.position_cargo = None

    def execute(self):
        total_str = self.arduino_port.readString(count=self.read_char)
        packets = total_str.split(self.packet_separator)

        for packet in reversed(packets):
            if packet == "":
                continue
            values = self.split_packet(packet)
            if values is not None:
                self.last_reading_time = time.monotonic()
                self.readings = values
                self.generate_positions()
                break
        if (time.monotonic() - self.last_reading_time) > self.reading_timeout:
            self.readings = None
            self.position_cargo = None
            self.position_hatch = None
