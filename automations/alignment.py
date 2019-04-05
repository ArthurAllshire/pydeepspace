from magicbot import tunable
from magicbot.state_machine import StateMachine, state

from automations.cargo import CargoManager, CargoManipulator
from automations.hatch import HatchAutomation
from components.vision import Vision
from components.line_detector import LineDetectorSensor
from pyswervedrive.chassis import SwerveChassis
from utilities.navx import NavX
from utilities.functions import rotate_vector

import numpy as np


class Aligner(StateMachine):
    """
    A state machine for alignment using vision systems.

    The robot will use two methods of alignment, targets above
    objectives from longer range and fine adjustment using the ground
    tape once we are able to see it.
    """

    VERBOSE_LOGGING = True

    chassis: SwerveChassis
    vision: Vision
    line_detector: LineDetectorSensor
    cargo: CargoManager
    imu: NavX

    def setup(self):
        self.successful = False
        self.last_vision = 0
        self.direction = 1
        self.tolerance = 0.1

    alignment_speed = tunable(0.5)  # m/s changed in teleop and autonomous
    alignment_kp_y = tunable(1.5)
    alignment_kp_tape = tunable(0.1)
    tape_align_speed = tunable(1)
    tape_forward_speed = tunable(1)
    hit_wall_accel_thresh = tunable(-1)
    odometry_command_comparison_factor = tunable(0.5)
    # lookahead_factor = tunable(4)

    def on_disable(self):
        self.done()

    # def get_fiducial_y(self):
    #     return self.vision.get_fiducial_position()[2]

    @state(first=True)
    def wait_for_vision(self):
        if self.cargo.cargo_component.is_contained():
            tape_offset = self.line_detector.position_cargo
        else:
            tape_offset = self.line_detector.position_hatch

        if self.vision.fiducial_in_sight or tape_offset is not None:
            self.next_state("target_tape_align")

    @state(must_finish=True)
    def target_tape_align(self, initial_call, state_tm):
        """
        Align with the objective using the vision tape above the objective.

        The robot will try to correct errors until they are within tolerance
        by strafing and moving in a hyberbolic curve towards the target.
        """
        if initial_call:
            self.successful = False
            self.last_vision = state_tm
            self.chassis.automation_running = True
            self.vision_counter = 0
            self.tape_counter = 0
            self.last_range = 2.5
            self.dist = None
            self.accel_profile = []
            # self.v = 0
        # self.u = self.chassis.speed
        # if abs(self.v - self.alignment_speed) > self.tolerance:
        #     if self.v > self.u:
        #         a = self.chassis.decceleration
        #     if self.v < self.u:
        #         a = self.chassis.acceleration
        #     self.v = self.u + a * state_tm
        self.accel_profile.append(self.imu.getAccelX())

        if self.chassis.command_speed * self.odometry_command_comparison_factor > self.chassis.speed \
                and state_tm > 0.5:
            self.logger.info("Odometry vs command speed difference high - assuming wheels stalled")
            self.next_state("success")

        if self.cargo.cargo_component.is_contained():
            tape_offset = self.line_detector.position_cargo
        else:
            tape_offset = self.line_detector.position_hatch
        fiducial_x, fiducial_y, delta_heading = self.vision.get_fiducial_position()

        if tape_offset is not None:
            if self.tape_counter < 1:
                self.logger.info("Seen tape")
                self.tape_counter += 1
            vy_tape = self.alignment_kp_tape * tape_offset
            self.chassis.set_inputs(
                0.5, vy_tape, 0, field_oriented=False
            )
            if state_tm > 2.5 or self.imu.getAccelX() < self.hit_wall_accel_threshold:
                self.next_state("success")
        elif not self.vision.fiducial_in_sight or abs(fiducial_x) > abs(self.last_range):
            # self.chassis.set_inputs(0, 0, 0)
            # self.next_state("success")
            self.chassis.set_inputs(
                self.alignment_speed * self.direction, 0, 0, field_oriented=False
            )
            if state_tm - self.last_vision > self.last_range / self.alignment_speed:
                self.chassis.set_inputs(0, 0, 0)
                self.next_state("success")
        else:
            if self.vision_counter < 1:
                self.logger.info("Seen vision")
                self.vision_counter += 1
            self.last_vision = state_tm
            self.last_range = fiducial_x
            # fiducial_x /= self.lookahead_factor
            # norm = math.hypot(fiducial_x, fiducial_y)
            # vx = fiducial_x / norm * self.alignment_speed
            # vy = fiducial_y / norm * self.alignment_speed
            if fiducial_x > 0:
                # Target in front of us means we are using the hatch camera - move forwards
                vx = self.alignment_speed * (1 - min(abs(fiducial_y), 1.5) / 1.5)
            else:
                # Target behind us means we are using the cargo camera - move backwards
                vx = -self.alignment_speed * (1 - min(abs(fiducial_y), 1.5) / 1.5)

            if tape_offset is not None:
                vy = vy_tape
            else:
                vy = self.alignment_speed * max(
                    min(fiducial_y * self.alignment_kp_y, 1), -1
                )

            vx, vy = rotate_vector(vx, vy, -delta_heading)
            self.chassis.set_inputs(vx, vy, 0, field_oriented=False)

    @state(must_finish=True)
    def success(self):
        self.logger.info(f"Accel profile {self.accel_profile}")
        self.done()

    def done(self):
        super().done()
        self.chassis.automation_running = False


class HatchDepositAligner(Aligner):

    VERBOSE_LOGGING = True
    hatch_automation: HatchAutomation

    @state(must_finish=True)
    def success(self, state_tm, initial_call):
        if initial_call:
            self.hatch_automation.outake()
            self.done()


class CargoDepositAligner(Aligner):

    VERBOSE_LOGGING = True
    cargo: CargoManager

    def setup(self):
        super().setup()
        self.direction = -1

    @state(must_finish=True)
    def success(self):
        self.done()


class HatchIntakeAligner(Aligner):
    VERBOSE_LOGGING = True
    hatch_automation: HatchAutomation

    @state(must_finish=True)
    def success(self):
        self.hatch_automation.grab()
        self.done()
