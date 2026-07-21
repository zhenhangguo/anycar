import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from control import *

from sample_action_pb2 import SampleFuncParameterization, FuncParameterization, ControlActionSamplingParams
from google.protobuf import text_format

# vehicle params
Cf_ = 200000 
Cr_ = 2129608.6   #107200 
Lf_ = 3.0     #1.64
Lr_ = 0.9     #2.26
L_ = Lf_ + Lr_
Iz_ = 123611.03
Total_weight_ = 7590
Steering_tua_ = 0.4 # 0.2
Steering_ratio_ = 40 # 40


# sample params
Prediction_length_ = 50
Ts_ = 0.05

Sample_num_ = 64
Speed_select_list_ = [40 ,60, 70 , 80 ,90]
Mass_select_list_ = [15000, 20000, 25000, 30000, 35000, 40000, 45000]
perturb_value = ["ratio", "q_matrix", "Iz"]    # Cf, Cr, Lf, Iz, tua, ratio, q_matrix
# perturb_value = []

# Q, R matrix
Q_ = np.array([0.05, 1.5, 0.3, 0.3, 0.03, 0, 0])  #np.array([0.033, 0.61, 0.033, 0.33,0,0])    # np.array([0.05, 1.5, 1.0, 1.0, 0.03, 0, 0]) 
R_ = 1

class Vehicle_param:
    def __init__(self):
        self.init_cf = Cf_ 
        self.init_cr = Cr_  
        self.init_lf = Lf_
        self.init_lr = L_ - self.init_lf
        self.init_Iz = Iz_ 
        self.init_total_weight = Total_weight_
        self.init_tua = Steering_tua_
        self.init_ratio = Steering_ratio_

        self.cf = Cf_ 
        self.cr = Cr_  
        self.lf = Lf_
        self.lr = L_ - self.lf
        self.Iz = Iz_ 
        self.total_weight = Total_weight_
        self.tua = Steering_tua_
        self.ratio = Steering_ratio_
    
    def get_param(self):
        return self.cf, self.cr, self.lf, self.lr, self.Iz, self.total_weight, self.tua, self.ratio

    def get_init_param(self):
        return self.init_cf, self.init_cr, self.init_lf, self.init_lr, self.init_Iz, self.init_total_weight, self.init_tua, self.init_ratio
    
    def reset(self):
        self.cf = Cf_ 
        self.cr = Cr_  
        self.lf = Lf_
        self.lr = L_ - self.lf
        self.Iz = Iz_ 
        self.tua = Steering_tua_
        self.ratio = Steering_ratio_

def get_matrix(vehicle_param, init_speed):
    cf, cr, lf, lr, Iz, total_weight, tua, ratio = vehicle_param.get_param()

    # model matrix
    a11 = -(2 * cf + 2 * cr) / (total_weight * init_speed)
    a12 = (2 * cf + 2 * cr) / total_weight
    a13 = -(2 * cf * lf - 2 * cr * lr) / (total_weight * init_speed)
    a14 = 2 * cf / total_weight

    a31 = -(2 * lf * cf - 2 * lr * cr) / (Iz * init_speed)
    a32 = 2*(lf*cf-lr*cr)/Iz
    a33 = -(2 * lf * lf * cf + 2 * lr * lr * cr) / (Iz * init_speed)
    a34 = 2 * cf * lf / Iz

    a44 = -1 / tua
    a45 = 1 / tua / ratio

    matrix_A = np.array([[0, 1, 0, 0, 0, 0 ,0 ],[0, 0, 1, 0, 0, 0 ,0 ], [0, 0, a11, a12, a13, a14, 0], [0, 0,0,0, 1, 0, 0], [0, 0,a31,a32,a33, a34, 0], [0, 0, 0, 0, 0, a44, a45], [0, 0,0,0,0,0,0]])
    matrix_B = np.array([[0], [0],[0],[0],[0],[0],[1]])

    return matrix_A, matrix_B

def calculate_kappa_coeff(vehicle_param, speed):
    cf, cr, lf, lr, Iz, total_weight, tua, ratio = vehicle_param.get_param()
    L = lf + lr
    coeff = (L + ((total_weight / 2 / cf) * lf / L - (total_weight / 2 / cr) * lr / L) * (speed ** 2)) * ratio * tua

    init_cf, init_cr, init_lf, init_lr, init_Iz, init_total_weight, init_tua, init_ratio = vehicle_param.get_init_param()

    init_coeff = (L + ((total_weight / 2 / init_cf) * init_lf / L - (total_weight / 2 / init_cr) * init_lr / L) * (speed ** 2)) * init_ratio * init_tua
    return coeff / init_coeff

def calculate_feedback_coeff_list(A, B, K, t_step, length):
    coeff_list = []
    M = A - B @ K
    for i in range(length):
        coeff_list.append(-K @ expm(M * t_step * i))
    
    return coeff_list

def analytic_solution(coeff_list, x0_):
    u = []
    for coeff in coeff_list:
        u_t = coeff @ x0_ 
        u.append(u_t[0, 0])
    return u

def calculate_curvature_ff(kappa_list, coeff):
    ff_front_wheel_angle = []
    for kappa in kappa_list:
        ff_cmd =  coeff * kappa
        ff_front_wheel_angle.append(ff_cmd)

    return ff_front_wheel_angle


def set_sample_vehicle_params(vehicle_param, speed):
    samp_func_param =  SampleFuncParameterization()

    samp_func_param.speed = speed
    samp_func_param.mass = vehicle_param.total_weight
    samp_func_param.ratio = vehicle_param.ratio
    samp_func_param.tua = vehicle_param.tua
    samp_func_param.cf = vehicle_param.cf
    samp_func_param.cr = vehicle_param.cr
    samp_func_param.lf = vehicle_param.lf
    samp_func_param.lr = vehicle_param.lr
    samp_func_param.Iz = vehicle_param.Iz

    return samp_func_param

def add_coeff_into_proto(samp_func_param, coeff_feedback, feedforward_coeff, backlash_value, control_type= 0):
    
    function_feedback =  FuncParameterization()

    samp_func_param.bl_val.append(backlash_value)

    for coeffs in coeff_feedback:
        for coeff in coeffs[0]:
            function_feedback.f_c.append(coeff)

    samp_func_param.ff_func.append(feedforward_coeff)

    samp_func_param.feedback_func.append(function_feedback)

    samp_func_param.sample_type = control_type


sample_func_params = ControlActionSamplingParams()
perturb_value_info = ""
for perturb in perturb_value:
    if perturb_value_info != "":
        perturb_value_info += " and "
    perturb_value_info += perturb

for mass in Mass_select_list_:
    for speed in Speed_select_list_:
        vehicle_param = Vehicle_param()
        select_speed = speed
        vehicle_param.total_weight = mass

        Q = Q_
        vehicle_param.lf -= 0.3
        vehicle_param.lr = L_ - vehicle_param.lf
        # vehicle_param.ratio *= 0.5
        samp_func_param = set_sample_vehicle_params(vehicle_param, speed)
        idx = 0
        while idx < Sample_num_:
            backlash_value = 3.0 / 57.3 # 3 degree backlash value
            vehicle_param.reset()
            vehicle_param.lf -= 0.3
            vehicle_param.lr = L_ - vehicle_param.lf
            # vehicle_param.ratio *= 0.5
            Q = Q_ * 0.2
            for perturb in perturb_value:
                random_factor = np.random.uniform(0.0, 1.0)
                if perturb == "ratio":
                    vehicle_param.ratio *= (0.8 + random_factor * 0.4)
                if perturb == "q_matrix":
                    q_scale = np.random.uniform(0.5, 3)
                    Q = Q_ * q_scale
                    Q[1] *= q_scale
                if perturb == "Iz":
                    vehicle_param.Iz *= np.random.uniform(0.3, 5.0)
                if perturb == "tua":
                    vehicle_param.tua *= 1 / (0.6 + random_factor * 0.8) * np.random.uniform(0.9, 1.1)

                # update backlash compensation value
                backlash_value *= (0.6 + random_factor * 0.8)
            
            matrix_A, matrix_B = get_matrix(vehicle_param, select_speed)

            try:
                K, _, _ = lqr(matrix_A, matrix_B, np.diag(Q), R_)
                idx += 1
            except:
                continue

            coeff_feedback = calculate_feedback_coeff_list(matrix_A, matrix_B, K, Ts_, Prediction_length_)
            feedforward_coeff = calculate_kappa_coeff(vehicle_param, speed)

            add_coeff_into_proto(samp_func_param, coeff_feedback, feedforward_coeff, backlash_value)
            samp_func_param.perturb_value = perturb_value_info
            samp_func_param.sample_type = 0

        sample_func_params.sample_func_params.append(samp_func_param)

        # # 赋值steady mode 的 采样序列
        vehicle_param.lf += 0.1
        vehicle_param.lr = L_ - vehicle_param.lf
        vehicle_param.ratio *= 0.5
        samp_func_param_base = set_sample_vehicle_params(vehicle_param, speed)
        idx = 0
        Q = Q_.copy()
        while idx < Sample_num_:
            vehicle_param.reset()
            vehicle_param.lf -= 0.3
            vehicle_param.lr = L_ - vehicle_param.lf
            vehicle_param.ratio *= 0.5
            backlash_value = 0.0
            Q = Q_ * 0.2
            for perturb in perturb_value:
                random_factor = np.random.uniform(0.0, 1.0)
                if perturb == "ratio":
                    vehicle_param.ratio *= (0.8 + random_factor * 0.4)
                if perturb == "q_matrix":
                    q_scale = np.random.uniform(0.5, 3)
                    Q *= q_scale
                    Q[1] *= q_scale
                if perturb == "Iz":
                    vehicle_param.Iz *= np.random.uniform(0.3, 5.0)
                if perturb == "tua":
                    vehicle_param.tua *= 1 / (0.6 + random_factor * 0.8) * np.random.uniform(0.9, 1.1)

            matrix_A_base, matrix_B_base = get_matrix(vehicle_param, select_speed)

            try:
                K_base, _, _ = lqr(matrix_A_base, matrix_B_base, np.diag(Q), R_)
                idx += 1
            except:
                continue

            coeff_feedback_base = calculate_feedback_coeff_list(matrix_A_base, matrix_B_base, K_base, Ts_, Prediction_length_)
            feedforward_coeff_base = calculate_kappa_coeff(vehicle_param, speed)

            add_coeff_into_proto(samp_func_param_base, coeff_feedback_base, feedforward_coeff_base, backlash_value)
            samp_func_param_base.perturb_value = perturb_value_info
            samp_func_param_base.sample_type = 1

        sample_func_params.sample_func_params.append(samp_func_param_base)

# 序列化为字节流
text_data = text_format.MessageToString(sample_func_params, float_format='.3f')

# 写入文件
with open("sample_action_list.prototxt", "w") as f:
    f.write(text_data)
   