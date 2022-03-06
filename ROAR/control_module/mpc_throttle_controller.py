from pydantic import BaseModel, Field
from ROAR.control_module.controller import Controller
from ROAR.utilities_module.vehicle_models import VehicleControl, Vehicle

from ROAR.utilities_module.data_structures_models import Transform, Location
from collections import deque
import numpy as np
import logging
from ROAR.agent_module.agent import Agent
from typing import Tuple
import json
from pathlib import Path
import cvxpy as cp
import scipy
import scipy.signal
import scipy.linalg

b_motor = 7834  # dimensionless motor constant
mass = 1500  # kg
F_f = 164  # N
C_d = 3.2  # Drag coefficient
delta_lim = 0.1

# Common Optimization Constants
Tf = 2.0  # seconds, time to reach desired speed
M = 5  # optimization steps
N = 5  # number of prediction steps
Tp = Tf / M  # seconds, prediction time per optimizaion run
Ts = Tp / N 

class MPCController(Controller):
    def __init__(self, agent, steering_boundary: Tuple[float, float],
                 throttle_boundary: Tuple[float, float], **kwargs):
        super().__init__(agent, **kwargs)
        self.max_speed = self.agent.agent_settings.max_speed
        self.throttle_boundary = throttle_boundary
        self.steering_boundary = steering_boundary
        self.config = json.load(Path(agent.agent_settings.pid_config_file_path).open(mode='r'))
        self.long_pid_controller = LongMPCController(agent=agent,
                                                     throttle_boundary=throttle_boundary,
                                                     max_speed=self.max_speed,
                                                     config=self.config["longitudinal_controller"])
        self.lat_pid_controller = LatPIDController(
            agent=agent,
            config=self.config["latitudinal_controller"],
            steering_boundary=steering_boundary
        )
        self.logger = logging.getLogger(__name__)


    def run_in_series(self, next_waypoint: Transform, **kwargs) -> VehicleControl:
        throttle = self.long_pid_controller.run_in_series(next_waypoint=next_waypoint,
                                                          target_speed=kwargs.get("target_speed", self.max_speed))
        steering = self.lat_pid_controller.run_in_series(next_waypoint=next_waypoint)
        return VehicleControl(throttle=throttle, steering=steering)

    @staticmethod
    def find_k_values(vehicle: Vehicle, config: dict) -> np.array:
        current_speed = Vehicle.get_speed(vehicle=vehicle)
        k_p, k_d, k_i = 1, 0, 0
        for speed_upper_bound, kvalues in config.items():
            speed_upper_bound = float(speed_upper_bound)
            if current_speed < speed_upper_bound:
                k_p, k_d, k_i = kvalues["Kp"], kvalues["Kd"], kvalues["Ki"]
                break
        return np.array([k_p, k_d, k_i])


class LongMPCController(Controller):
    def __init__(self, agent, config: dict, throttle_boundary: Tuple[float, float], max_speed: float,
                 dt: float = 0.03, **kwargs):
        super().__init__(agent, **kwargs)
        self.config = config
        self.max_speed = max_speed
        self.throttle_boundary = throttle_boundary
        self._error_buffer = deque(maxlen=10)
        self._dt = dt
        self.A, self.B = self.get_linearized_matrices()

    def get_linearized_matrices(self):
        Ac = np.array([[-2 * C_d / mass]])
        Bc = np.array([[b_motor / mass]])

        # C and D are just for calling cont2discrete
        Cc = np.zeros((1, 2))
        Dc = np.zeros((1, 1))
        system = (Ac, Bc, Cc, Dc)
        Ad, Bd, Cd, Dd, dt = scipy.signal.cont2discrete(system, Ts)
        return Ad, Bd

    def solve_cftoc(self, target, current, speed_bounds, throttle_bounds):
        nx = 1
        nu = 1

        x = cp.Variable((nx, M + 1))
        u = cp.Variable((nu, M))

        cost = 0
        constr = []
        for m in range(M):
            cost += cp.sum_squares(x - target)
            constr += [x[:, m + 1] == self.A @ x[:, m] + self.B @ u[:, m]]

        # Concatenates model constraints
        constr += [x[:, 0] == current]
        constr += [x >= -speed_bounds[1], x <= speed_bounds[1]]
        constr += [u >= throttle_bounds[0], u <= throttle_bounds[1]]
        for i in range(M - 1):
            constr += [u[:, i + 1] - u[:, i] <= delta_lim,
                    u[:, i + 1] - u[:, i] >= -delta_lim]

        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve(solver=cp.ECOS)

        if u.value is None or u.value[0] is None or u.value[0][0] is None:
            cmd = 0.5
        else:
            cmd = u.value[0][0]
        return cmd


    def run_in_series(self, next_waypoint: Transform, **kwargs) -> float:
        target_speed = min(self.max_speed, kwargs.get("target_speed", self.max_speed))
        current_speed = Vehicle.get_speed(self.agent.vehicle)
        speed_bounds = [-5, self.max_speed * 1.25]
        motor_cmd = self.solve_cftoc(target_speed, current_speed, speed_bounds, self.throttle_boundary)
        return motor_cmd


class LatPIDController(Controller):
    def __init__(self, agent, config: dict, steering_boundary: Tuple[float, float],
                 dt: float = 0.03, **kwargs):
        super().__init__(agent, **kwargs)
        self.config = config
        self.steering_boundary = steering_boundary
        self._error_buffer = deque(maxlen=10)
        self._dt = dt

    def run_in_series(self, next_waypoint: Transform, **kwargs) -> float:
        """
        Calculates a vector that represent where you are going.
        Args:
            next_waypoint ():
            **kwargs ():

        Returns:
            lat_control
        """
        # calculate a vector that represent where you are going
        v_begin = self.agent.vehicle.transform.location.to_array()
        direction_vector = np.array([-np.sin(np.deg2rad(self.agent.vehicle.transform.rotation.yaw)),
                                     0,
                                     -np.cos(np.deg2rad(self.agent.vehicle.transform.rotation.yaw))])
        v_end = v_begin + direction_vector

        v_vec = np.array([(v_end[0] - v_begin[0]), 0, (v_end[2] - v_begin[2])])
        # calculate error projection
        w_vec = np.array(
            [
                next_waypoint.location.x - v_begin[0],
                0,
                next_waypoint.location.z - v_begin[2],
            ]
        )

        v_vec_normed = v_vec / np.linalg.norm(v_vec)
        w_vec_normed = w_vec / np.linalg.norm(w_vec)
        error = np.arccos(v_vec_normed @ w_vec_normed.T)
        _cross = np.cross(v_vec_normed, w_vec_normed)

        if _cross[1] > 0:
            error *= -1
        self._error_buffer.append(error)
        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        k_p, k_d, k_i = MPCController.find_k_values(config=self.config, vehicle=self.agent.vehicle)

        lat_control = float(
            np.clip((k_p * error) + (k_d * _de) + (k_i * _ie), self.steering_boundary[0], self.steering_boundary[1])
        )
        return lat_control
