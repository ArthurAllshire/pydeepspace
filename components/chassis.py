from swervedrive.icr import Controller
from .module import SwerveModule
import numpy as np


class Chassis:
    module_a: SwerveModule
    module_b: SwerveModule
    module_c: SwerveModule
    module_d: SwerveModule

    DELTA_T = 1 / 50  # 50hz

    # TODO find values
    initial_pos = np.array([0, 0, 0]).reshape(-1,1)  # np.array
    STEERING_ANGLE_BOUNDS = [-10, 10]  # rad
    STEERING_SPEED_BOUNDS = [-5, 5]  # rad/s
    STEERING_ACCELERATION_BOUNDS = [-20, 20]  # rad/s2
    DRIVE_SPEED_BOUNDS = [-128, 128]  # rad/s
    DRIVE_ACCELERATION_BOUNDS = [-100, 100]  # rad/s2

    def __init__(self):
        self.vx = 0
        self.vy = 0
        self.vz = 0

    def setup(self):
        self.modules = [self.module_a, self.module_b, self.module_c, self.module_d]

        modules_alpha = np.array([[module.alpha] for module in self.modules])
        modules_l = np.array([[module.l] for module in self.modules])
        modules_b = np.array([[module.b] for module in self.modules])
        modules_radius = np.array([[module.WHEEL_RADIUS] for module in self.modules])

        self.controller = Controller(
            modules_alpha,
            modules_l,
            modules_b,
            modules_r=modules_radius,
            epsilon_init=self.initial_pos,
            beta_bounds=self.STEERING_ANGLE_BOUNDS,
            beta_dot_bounds=self.STEERING_SPEED_BOUNDS,
            beta_2dot_bounds=self.STEERING_ACCELERATION_BOUNDS,
            phi_dot_bounds=self.DRIVE_SPEED_BOUNDS,
            phi_2dot_bounds=self.DRIVE_ACCELERATION_BOUNDS,
        )

    def execute(self):
        modules_beta = np.array([module.get_beta_angle() for module in self.modules]).reshape(-1,1)
        modules_angular_velocity = np.array(
            [module.get_drive_angular_velocity() for module in self.modules]
        ).reshape(-1,1)
        # TODO controller mapping\
        lmda_d, mu_d = self.twist_to_icr(self.vx, self.vy, self.vz)

        desired_angles, angular_velocities, odometry = self.controller.control_step(
            modules_beta,
            modules_phi_dot=modules_angular_velocity,
            lmda_d=lmda_d,
            mu_d=mu_d,
            delta_t=self.DELTA_T,
        )
        desired_angles = desired_angles.reshape(-1)
        angular_velocities = angular_velocities.reshape(-1)
        odometry = odometry.reshape(-1)

        for module, desired_angle, angular_velocity in zip(
            self.modules, desired_angles, angular_velocities
        ):
            module.set_drive_angular_velocity(angular_velocity)
            module.set_beta_angle(desired_angle)

    @staticmethod
    def twist_to_icr(vx: float, vy: float, vz: float):
        """Convert a twist command (vx, vy, vz) to lmda and mu.

        Eta represents the motion about the ICR as represented in the projective plane.
        See eq.(1) of the control paper.
        """
        norm = np.linalg.norm([vx, vy, vz])
        if np.isclose(norm, 0, atol=0.01):
            return None, 0
        lmda = (1 / norm) * np.array([-vy, vx, vz])
        mu = norm
        return lmda.reshape(-1,1), mu

    def set_inputs(self, vx, vy, vz):
        self.vx = vx
        self.vy = vy
        self.vz = vz
