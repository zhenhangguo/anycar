import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev, make_interp_spline
import random
import math

def generate_random_points(num_points = 10, scale = 1):
    points = []
    min_distance = int(4 * scale)
    for i in range(num_points):
        x = random.randrange(0,  int(20 * scale), 1)
        y = random.randrange(0, int(20 * scale), 1)
        distances = list(filter(lambda x: x < min_distance, [math.sqrt((p[0]-x)**2 + (p[1]-y)**2) for p in points]))
        if len(distances) == 0:
            points.append((x, y))
    return np.array(points)

def check_min_distances(og_points, min_distance):
    points = []
    for [x,y] in og_points:
        distances = list(filter(lambda x: x < min_distance, [math.sqrt((p[0]-x)**2 + (p[1]-y)**2) for p in points]))
        if len(distances) == 0:
            points.append((x, y))
    return np.array(points)

def get_track_points(hull, points):
    # get the original points from the random 
    # set that will be used as the track starting shape
    return np.array([points[hull.vertices[i]] for i in range(len(hull.vertices))])

def expand_track(points, scale):
    new_points = []
    num_points = points.shape[0]
    for i in range(-1, num_points - 1): # should start from -1
        curr_x = points[i][0]
        curr_y = points[i][1]
        next_x = points[i+1][0]
        next_y = points[i+1][1]
        new_points.append([curr_x, curr_y])

        mid_x = (curr_x + next_x) / 2
        mid_y = (curr_y + next_y) / 2

        a = next_x - curr_x
        b = next_y - curr_y
    
        # Find an orthogonal vector
        ortho_vec = np.array([-b, a])
        if np.linalg.norm(ortho_vec) > 4:
            norm_ortho_vec = ortho_vec/np.linalg.norm(ortho_vec)
            random_distance = np.random.uniform(2*scale, 5*scale)
            sign = 1 if random.random() < 0.5 else -1

            transformed_point = sign * random_distance * norm_ortho_vec + np.array([mid_x, mid_y])

            new_points.append([transformed_point[0], transformed_point[1]])
    
    return np.array(new_points)

def fit_spline(points, timesteps):
    points = np.vstack((points[-1], points))
    tck, u = splprep([points[:,0], points[:,1]], s=0, per=True)
    u_new = np.linspace(0, 1, timesteps)
    x_new, y_new = splev(u_new, tck)
    return(np.hstack((np.array(x_new).reshape(-1, 1), np.array(y_new).reshape(-1,1))))

def change_track(scale:int = 1, direction = 1):
    num_points = 5 * max(scale, 1)
    num_total_points = 100 * max(scale, 1)
    random_points = generate_random_points(num_points, scale)
    # get convex hull

    hull = None
    while hull is None:
        try:
            hull = ConvexHull(random_points)
        except:
            random_points = generate_random_points(num_points)
            pass

    track_points = get_track_points(hull, random_points)
    expanded_track_points = expand_track(track_points, scale)
    fitted_spline = fit_spline(expanded_track_points, num_total_points)
    if direction == -1:
        fitted_spline = np.flip(fitted_spline)
    # plt.plot(fitted_spline[:,0], fitted_spline[:,1])
    # plt.show()
    return fitted_spline

class CarDataset:
    def __init__(self) -> None:
        self.reset_logs()  
        self.car_params = {"wheelbase": None, 
                           "mass": None, 
                           "com": None, 
                           "friction": None,  
                           "delay": 0, 
                           'max_throttle': None, 
                           'max_steer': None,
                           'steer_bias': None,
                           "sim": None,}

    
    def reset_logs(self):
        self.data_logs = {   "steer": [],
                        "throttle": [],
                        "xpos_x": [],  # X component of position
                        "xpos_y": [],  # Y component of position
                        "xpos_z": [],  # Z component of position
                        "xori_w": [],  # W compoenent of orientation (Quaternion)
                        "xori_x": [],  # X component of orientation (Quaternion)
                        "xori_y": [],  # Y component of orientation (Quaternion)
                        "xori_z": [],  # Z component of orientation (Quaternion)
                        "xvel_x": [],  # X component of linear velocity
                        "xvel_y": [],  # Y component of linear velocity
                        "xvel_z": [],  # Z component of linear velocity
                        "xacc_x": [],  # X component of linear acceleration
                        "xacc_y": [],  # Y component of linear acceleration
                        "xacc_z": [],  # Z component of linear acceleration
                        "avel_x": [],  # X component of angular velocity
                        "avel_y": [],  # Y component of angular velocity
                        "avel_z": [],  # Z component of angular velocity
                        "traj_x": [],
                        "traj_y": [],
                        "lap_end":[],
                        "vehicle_yaw": [],
                        }   
        
    def __len__(self):
        return len(self.data_logs["xpos_x"])


class car_state:
    def __init__(self):
        self.car_pos = [0,0,0]
        self.car_orientation = [0,0,0,0]
        self.car_lin_vel = [0,0,0]
        self.car_lin_acc = [0,0,0]
        self.car_ang_vel = [0,0,0]
        self.car_yaw = [0]


class RandWalkController:
    def __init__(self, totaltime, x, y, speed, yaw, steer_angle,tstep):
        self.totaltime = totaltime

        self.lowervel = -1
        self.uppervel = 1
        self.lowersteer = -0.5
        self.uppersteer = 0.5
        
        self.tstep = tstep
        self.init_tire_steering_angle = steer_angle
        
        self.x = x
        self.y = y
        self.speed = speed
        self.yaw = yaw

        self.generate_target_velocities()
        self.generate_target_steering_rate()
        
    def generate_target_velocities(self):
        expert_speed = 15.0
        ratio = np.random.uniform(0.3 , 1)

        samples = self.totaltime // 100
        mean =  (expert_speed - self.speed) / samples * ratio
        std_dev = 0.5
        sampled_vels = np.random.normal(mean, std_dev, samples)
        # print(min(sampled_vels), max(sampled_vels ))
        sampled_vels = np.clip(sampled_vels, self.lowervel, self.uppervel)
        x = np.linspace(0, self.totaltime, samples)
        spline = make_interp_spline(x, sampled_vels)
        x_total = np.arange(0, self.totaltime)
        clipped_spline = np.clip(spline(x_total), self.lowervel, self.uppervel)
        self.target_velocities = clipped_spline
        # plt.plot(np.array(self.target_velocities) + self.speed)
        # plt.show()

    def generate_target_steering_rate(self):
        samples = self.totaltime // 20
        
        expert_speed = 10.0
        ratio_v = (expert_speed / self.speed ) ** 2
        ratio = np.random.uniform(0.01 , 0.5)
        mean = - self.init_tire_steering_angle / samples * ratio
        std_dev = 0.1 * ratio_v
        
        sampled_steers = np.random.normal(mean, std_dev, samples)
        sampled_steers = np.clip(sampled_steers, self.lowersteer, self.uppersteer)
        x = np.linspace(0, self.totaltime, samples)
        spline = make_interp_spline(x, sampled_steers)
        x_total = np.arange(0, self.totaltime)
        clipped_spline = np.clip(spline(x_total), self.lowersteer, self.uppersteer)
        self.target_steer = clipped_spline

    def get_control(self, t):
        throttle = self.target_velocities[t]
        steer = self.target_steer[t]
        return [throttle, steer]
    
    def reset_controller(self):
        self.generate_target_velocities()
        self.generate_target_steering_rate()
        
def steer_hysteresis_actuator(vehicle_speed, steer_angle, steer_offset):
    
    # idea: add deadzone around steer offset as hysteresis
    # return: front wheel angle after deadzone

    steer_hysteresis_model = {
        "hysteresis_bound":          [15.0, 15.0, 15.0, 15.0, 17.0, 18.0, 15.0, 15.0],  # deg
        "step_speed":                [0.0, 6.15, 11.1, 16.6, 19.0, 22.2, 25.0, 30.0],  # vehicle speed m/s
        "steer_ratio":               [26.4, 26.4, 26.4, 26.4, 26.4, 26.4, 26.4, 26.4],  # -
        "steer_hysteresis_ratio":    [70.0, 70.0, 70.0, 70.0, 100.0, 100.0, 100.0, 100.0],  #
    }

    gain_at_normal_steer = np.interp(vehicle_speed, steer_hysteresis_model["step_speed"], steer_hysteresis_model["steer_ratio"])
    gain_at_hysteresis = np.interp(vehicle_speed, steer_hysteresis_model["step_speed"], steer_hysteresis_model["steer_hysteresis_ratio"])
    hysteresis_bound = np.interp(vehicle_speed, steer_hysteresis_model["step_speed"], steer_hysteresis_model["hysteresis_bound"]) / 360 * 2 * math.pi * 0.5

    upper_hysteresis_bound = steer_offset + hysteresis_bound
    lower_hysteresis_bound = steer_offset - hysteresis_bound

    if lower_hysteresis_bound >= 0:
        upper_hys_out = 2 * hysteresis_bound / gain_at_hysteresis + lower_hysteresis_bound / gain_at_normal_steer
        lower_hys_out = lower_hysteresis_bound / gain_at_normal_steer
    if upper_hysteresis_bound <= 0:
        upper_hys_out = upper_hysteresis_bound / gain_at_normal_steer
        lower_hys_out = upper_hysteresis_bound / gain_at_normal_steer - 2 * hysteresis_bound / gain_at_hysteresis
    if lower_hysteresis_bound <=0 and upper_hysteresis_bound >=0:
        upper_hys_out = upper_hysteresis_bound / gain_at_hysteresis
        lower_hys_out = lower_hysteresis_bound / gain_at_hysteresis

    if steer_angle >= upper_hysteresis_bound:
        front_wheel_angle = (steer_angle - upper_hysteresis_bound) / gain_at_normal_steer + upper_hys_out
    if steer_angle <= lower_hysteresis_bound:
        front_wheel_angle = (steer_angle - lower_hysteresis_bound) / gain_at_normal_steer + lower_hys_out
    if steer_angle < upper_hysteresis_bound and steer_angle > lower_hysteresis_bound:
        front_wheel_angle = (steer_angle - lower_hysteresis_bound) / gain_at_hysteresis + lower_hys_out

    return front_wheel_angle

class backlash():
    def __init__(self, backlash_size):
        self.backlash_size = abs(backlash_size)
        self.last_value = None
        
    def calculate(self, input):
        if self.last_value is None:
            self.last_value = input
            return self.last_value
        else:
            if input >= self.last_value:
                if input > self.last_value + self.backlash_size * 0.5:
                    self.last_value = input - self.backlash_size * 0.5
            else:
                if input < self.last_value - self.backlash_size * 0.5:
                    self.last_value = input + self.backlash_size * 0.5
            
            return self.last_value
    
    def set_backlash_size(self, size):
        self.backlash_size = size
        
    def reset(self):
        self.last_value = None