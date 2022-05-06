from ROAR.control_module.controller import Controller
from ROAR.utilities_module.vehicle_models import VehicleControl, Vehicle
from ROAR.utilities_module.data_structures_models import Transform, Location
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

# Longitudinal Dynamics Parameters
b_motor = 9503  # dimensionless motor constant
mass = 1845  # kg
F_friction = 133  # N
C_d = .46  # Drag coefficient

# Lateral Dynamics Parameters   
B = 4.52
C = 2.16
# To match the parameter fitting optimization program for the lateral dynamics model,
# mu should be set to 1 but that appears to make steering not aggressive enough, 
# so mu is set to .75 here instead 
mu = .75
wheelbase = 3.0
Izz = 0.95 * mass / (wheelbase / 2) ** 2
Lf = 1.62
Lr = 1.38
Ff_z = 7239
Fr_z = 10859
max_angle = np.deg2rad(70.0)

# Optimization Parameters
M = 10  # optimization steps
delta_lim = np.array([50, 500])
angle_arr = np.deg2rad(np.array([0, 15, 30, 45, 60]))

class MPCController(Controller):
    def __init__(self, agent, steering_boundary: Tuple[float, float],
                 throttle_boundary: Tuple[float, float], **kwargs):
        super().__init__(agent, **kwargs)
        self.max_speed = self.agent.agent_settings.max_speed
        self.throttle_boundary = throttle_boundary
        self.steering_boundary = steering_boundary
        self.config = json.load(
            Path(agent.agent_settings.pid_config_file_path).open(mode='r'))
        self.controller = FullMPCController(agent=agent,
                                            throttle_boundary=throttle_boundary,
                                            steering_boundary=steering_boundary,
                                            max_speed=self.max_speed,
                                            config=self.config["longitudinal_controller"])
        self.logger = logging.getLogger(__name__)

    def run_in_series(self, next_waypoint: Transform, **kwargs) -> VehicleControl:
        long_control, lat_control = self.controller.run_in_series(next_waypoint=next_waypoint,
                                                           target_speed=kwargs.get("target_speed", self.max_speed))
        
        long_control = float(np.clip(long_control, *self.throttle_boundary))
        lat_control = float(np.clip(lat_control, *self.steering_boundary))

        return VehicleControl(throttle=long_control, steering=lat_control)


class FullMPCController(Controller):
    def __init__(self, agent, config: dict,
                 throttle_boundary: Tuple[float, float],
                 steering_boundary: Tuple[float, float],
                 max_speed: float,
                 dt: float = 0.03, **kwargs):
        super().__init__(agent, **kwargs)
        self.config = config
        self.max_speed = max_speed
        self.throttle_boundary = throttle_boundary
        self.steering_boundary = steering_boundary
        self._dt = dt
        self.A_matrices, self.B_matrices = self.construct_linearized_matrices()
        self.Q = np.diag([1, 1, 0, 1, 0, 0])
        self.last_steer_CMD = 0

    def get_throttle_CMD(self, Fr_x, vx):
        return (Fr_x + F_friction + C_d * vx**2) / b_motor

    def get_steer_CMD(self, Ff_y, beta, r, vx):
        arcsin_arg = np.clip(Ff_y / (-mu * Ff_z), -1, 1)
        alpha_f = np.tan(np.arcsin(arcsin_arg) / C) / B
        steer_angle = np.arctan(beta + ((r * Lf) / (vx + 10e-1))) - alpha_f 
        steer_cmd = steer_angle / max_angle
        self.last_steer_CMD = np.abs(steer_cmd)
        return steer_cmd

    def linearize_around_steer_angle(self, steer_angle_eq):
        # Linearize system state equations around a steering angle and 100km/hr
        v_mag_eq = 100
        beta_eq = np.arctan((Lr / wheelbase) * np.tan(steer_angle_eq))
        vx_eq = v_mag_eq * np.cos(beta_eq)
        r_eq = (v_mag_eq / Lr) * np.sin(beta_eq)

        alpha_f = np.arctan(beta_eq + (r_eq * Lf) / vx_eq) - steer_angle_eq
        Ff_y_eq = -mu * Ff_z * np.sin(C * np.arctan(B * alpha_f))
        Fr_y_eq = (Lf * Ff_y_eq * np.cos(steer_angle_eq)) / Lr

        # Find x,y components of velocity
        a_13 = -(Fr_y_eq + Ff_y_eq * np.cos(steer_angle_eq)) / (mass * vx_eq)
        a_31 = -vx_eq * r_eq
        # More complex a_13 term that comes from Gonzales dissertation 
        # a_31 = vx_eq * r_eq \
            # + ((Ff_y_eq * np.cos(steer_angle_eq)) / mass) \
            # * (1 /(1 + (beta_eq + ((r_eq * Lf) / vx_eq))**2))

        Ac = np.array([
            [0, -1, a_13], 
            [0, 0, 0,], 
            [a_31, 0, 0]])
        
        b_11 = np.cos(steer_angle_eq) / (mass * vx_eq)
        b_21 = np.cos(steer_angle_eq) * Lf / Izz
        b_31 = -np.sin(steer_angle_eq) / mass 

        Bc = np.array([
            [b_11, 0],
            [b_21, 0],
            [b_31, 1/mass]])

        # C and D are just for calling cont2discrete
        Cc = np.zeros((3, 3))
        Dc = np.zeros((3, 2))
        system = (Ac, Bc, Cc, Dc)
        Ad, Bd, Cd, Dd, dt = scipy.signal.cont2discrete(system, self._dt)
        return Ad, Bd


    def construct_linearized_matrices(self):
        A_matrices = {}
        B_matrices = {}
        for angle in angle_arr:
            A, B = self.linearize_around_steer_angle(angle)
            A_matrices.update({angle: A})
            B_matrices.update({angle: B})
        return A_matrices, B_matrices

    def get_linearized_matrices(self, steer_angle):
        if steer_angle < angle_arr[1]:
            angle_eq = angle_arr[0]
            return self.A_matrices.get(angle_eq), self.B_matrices.get(angle_eq)
        elif steer_angle < angle_arr[2]:
            angle_eq = angle_arr[1]
            return self.A_matrices.get(angle_eq), self.B_matrices.get(angle_eq)
        elif steer_angle < angle_arr[3]:
            angle_eq = angle_arr[2]
            return self.A_matrices.get(angle_eq), self.B_matrices.get(angle_eq)
        elif steer_angle < angle_arr[4]:
            angle_eq = angle_arr[3]
            return self.A_matrices.get(angle_eq), self.B_matrices.get(angle_eq)
        else:
            angle_eq = angle_arr[4]
            return self.A_matrices.get(angle_eq), self.B_matrices.get(angle_eq)

    def solve_cftoc(self, target_state, current_state, state_bounds, input_bounds):
        nx = 3
        nu = 2

        x = cp.Variable((nx, M + 1))
        u = cp.Variable((nu, M))

        cost = 0
        constr = []

        # Set Initial State
        constr += [x[:, 0] == current_state]

        # Get correct state matrices based on the last steering angle
        A, B = self.get_linearized_matrices(self.last_steer_CMD * max_angle)

        for m in range(M):
            cost += cp.sum_squares(x[0, m] - target_state[0])
            cost += cp.sum_squares(x[2, m] - target_state[2])

            constr += [x[:, m + 1] == A @ x[:, m] + B @ u[:, m]]

            constr += [state_bounds[2, 0] <= x[2, m]]
            constr += [state_bounds[2, 1] >= x[2, m]]

            constr += [input_bounds[:, 0] <= u[:, m]]
            constr += [input_bounds[:, 1] >= u[:, m]]
            if m < M - 1:
                constr += [u[:, m + 1] - u[:, m] <= delta_lim, u[:, m + 1] - u[:, m] >= -delta_lim]

        # Set final state constraints
        cost += cp.sum_squares(x[0, M] - target_state[0])
        cost += cp.sum_squares(x[2, M] - target_state[2])

        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve(warm_start=True)

        uOpt = u.value
        # In case optimizer doesnt return any values for u
        if uOpt is None or uOpt.size == 0 or np.isnan(uOpt[0][0]):
            Ff_y_cmd = 0.0
        else:
            Ff_y_cmd = u.value[0, 0]
        
        if uOpt is None or uOpt.size == 0 or np.isnan(uOpt[0][1]):
            Fr_x_cmd = 5000
        else:
            Fr_x_cmd = u.value[1, 0]
        
        return self.get_throttle_CMD(Fr_x_cmd, current_state[2]), self.get_steer_CMD(Ff_y_cmd, *current_state)


    def run_in_series(self, next_waypoint: Transform, **kwargs) -> float:
        current_steer = self.last_steer_CMD * max_angle
        current_beta = np.arctan((Lr / wheelbase) * np.tan(current_steer))

        current_speed = Vehicle.get_speed(self.agent.vehicle)
        current_vx = current_speed * np.cos(current_beta)

        # calculate a vector that represent where you are going
        v_begin = self.agent.vehicle.transform.location.to_array()
        current_yaw = np.deg2rad(self.agent.vehicle.transform.rotation.yaw)
        direction_vector = np.array([-np.sin(current_yaw),
                                     0,
                                     -np.cos(current_yaw)])
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
        error = np.arccos(np.dot(v_vec_normed, w_vec_normed))
        _cross = np.cross(v_vec_normed, w_vec_normed)
        if _cross[1] > 0:
            error *= -1

        # Set the target speed manually for testing
        target_speed = 100
        target_beta = -error
        target_vx = target_speed * np.cos(current_beta)

        motor_cmd, steer_cmd = self.solve_cftoc(
            target_state=np.array([target_beta, 0, target_vx]), 
            current_state=np.array([current_beta, current_yaw, current_vx]), 
            state_bounds=np.array([
                [0, 0], 
                [0, 0],
                [-100, 200]]), 
            input_bounds=np.array([[-3000, 3000], [-1000, 10000]]))

        return motor_cmd, steer_cmd

