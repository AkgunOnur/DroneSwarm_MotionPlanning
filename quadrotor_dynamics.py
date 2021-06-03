import numpy as np
from numpy import array, zeros, diag, ones, sin, cos, tan, linspace, dot, pi
from numpy.random import uniform
import time
from scipy import integrate
from numpy import isnan, pi, isinf
from numpy.random import normal
import os
import math
from scipy.spatial.transform import Rotation as R


def world_to_body(state, waypoint_world):
    psi, theta, phi = state[5], state[4], state[3]
    rot = R.from_euler('zyx', [[psi, theta, phi]], degrees=False).as_dcm()
    rot = rot.reshape(3,3)
    # R = np.array([[cos(theta)*cos(psi), sin(phi)*sin(theta)*cos(psi)-cos(phi)*sin(psi), cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi)],
    #               [cos(theta)*sin(psi), sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi), cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi)],
    #               [-sin(theta), sin(phi)*cos(theta), cos(phi)*cos(theta)]])
    waypoint_body = np.dot(rot.T, waypoint_world.reshape(-1,1))
    
    return waypoint_body.ravel()


def body_to_world(state, waypoint_body):
    psi, theta, phi = state[5], state[4], state[3]
    rot = R.from_euler('zyx', [[psi, theta, phi]], degrees=False).as_dcm()
    # R = np.array([[cos(theta)*cos(psi), sin(phi)*sin(theta)*cos(psi)-cos(phi)*sin(psi), cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi)],
    #               [cos(theta)*sin(psi), sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi), cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi)],
    #               [-sin(theta), sin(phi)*cos(theta), cos(phi)*cos(theta)]])
    waypoint_world = np.dot(rot, waypoint_body.reshape(-1,1))
    
    return waypoint_world.ravel()


class Drone:
    def __init__(self, state0):
        self.state = np.array(state0)
        self.v = 4
        #self.x = 0
        #self.y = 0
        #self.z = 0
        self.psi = 0
        self.vx = 0
        self.vy = 0
        self.psi_dot = 0
        self.angle_between = 0
        self.reward = 0
        self.is_alive = True


class Bot:
    def __init__(self, state0, target_state0):
        self.state = np.array(state0)
        self.target_state = np.array(target_state0)
        self.x_targetd = None
        self.y_targetd = None
        self.vd = 1
        #self.xd = 0
        #self.yd = 0
        #self.zd = 0
        self.psid = 0
        self.vxd = 0
        self.vyd = 0
        self.psi_dotd = 0
        self.angle_betweend = 0
        self.is_alive = True


class Quadrotor:
    
    def __init__(self, state0, coeff_pos=1.0, coeff_angle = 0.25, coeff_control = 0.0, coeff_final_pos=0.0):
        
        self.state = np.array(state0)
        # self.state2 = state0
        # self.state3 = state0
        self.U = [1, 0., 0., 0.]
        # self.U2 = [1, 0., 0., 0.]
        # self.U3 = [1, 0., 0., 0.]
        self.costValue = 0.
        self.coeff_pos = coeff_pos
        self.coeff_angle = coeff_angle
        self.coeff_control = coeff_control
        self.coeff_final_pos = coeff_final_pos
        self.Controllers = ["Backstepping_1","Backstepping_2","Backstepping_3","Backstepping_4"]
        
    
    def model_parameters(self):
        g = 9.81
        m = 1.52
        Ixx, Iyy, Izz = 0.0347563, 0.0458929, 0.0977
        I1 = (Iyy - Izz) / Ixx
        I2 = (Izz - Ixx) / Iyy
        I3 = (Ixx - Iyy) / Izz
        Jr = 0.0001
        l = 0.09
        b = 8.54858e-6
        d = 1.6e-2

        return g, m, Ixx, Iyy, Izz, I1, I2, I3, Jr, l, b, d


    def model_dynamics_2(self, t, state):
        g, m, Ixx, Iyy, Izz, I1, I2, I3, Jr, l, b, d = self.model_parameters()
        #states: [x,y,z,phi,theta,psi,x_dot,y_dot,z_dot,phi_dot,theta_dot,psi_dot]

        x,y,z,phi,theta,psi,x_dot,y_dot,z_dot,phi_dot,theta_dot,psi_dot = state
        U1, U2, U3, U4 = self.U

        omega = 0.

        state_dot = np.zeros(12)
        state_dot[0] = x_dot
        state_dot[1] = y_dot
        state_dot[2] = z_dot
        state_dot[3] = phi_dot
        state_dot[4] = theta_dot
        state_dot[5] = psi_dot

        state_dot[6] = (cos(phi)*sin(theta)*cos(psi) + sin(phi)*sin(psi))*U1/m
        state_dot[7] = (cos(phi)*sin(theta)*sin(psi) - sin(phi)*cos(psi))*U1/m
        state_dot[8] = -g + cos(phi)*cos(theta)*U1/m    
        state_dot[9] = theta_dot*psi_dot*I1 - Jr / Ixx * theta_dot * omega  + l/Ixx*U2
        state_dot[10] = phi_dot*psi_dot*I2 + Jr / Iyy * phi_dot * omega + l/Iyy*U3
        state_dot[11] = phi_dot*theta_dot*I3 + 1/Izz*U4

        return state_dot


    def model_dynamics(self, state, U):
        g, m, Ixx, Iyy, Izz, I1, I2, I3, Jr, l, b, d = self.model_parameters()
        #states: [x,y,z,phi,theta,psi,x_dot,y_dot,z_dot,phi_dot,theta_dot,psi_dot]

        x,y,z,phi,theta,psi,x_dot,y_dot,z_dot,phi_dot,theta_dot,psi_dot = state
        U1, U2, U3, U4 = U

        omega = 0.

        state_dot = np.zeros(12)
        state_dot[0] = x_dot
        state_dot[1] = y_dot
        state_dot[2] = z_dot
        state_dot[3] = phi_dot
        state_dot[4] = theta_dot
        state_dot[5] = psi_dot

        state_dot[6] = (cos(phi)*sin(theta)*cos(psi) + sin(phi)*sin(psi))*U1/m
        state_dot[7] = (cos(phi)*sin(theta)*sin(psi) - sin(phi)*cos(psi))*U1/m
        state_dot[8] = -g + cos(phi)*cos(theta)*U1/m    
        state_dot[9] = theta_dot*psi_dot*I1 - Jr / Ixx * theta_dot * omega  + l/Ixx*U2
        state_dot[10] = phi_dot*psi_dot*I2 + Jr / Iyy * phi_dot * omega + l/Iyy*U3
        state_dot[11] = phi_dot*theta_dot*I3 + 1/Izz*U4

        return state_dot


    

    def backstepping(self, A1, A2, A3, A4, A5, A6, state, U_list, ref_traj):
        g, m, Ixx, Iyy, Izz, I1, I2, I3, Jr, l, b, d = self.model_parameters()

        U1, U2, U3, U4 = U_list

        #self.state: [x,y,z,phi,theta,psi,x_dot,y_dot,z_dot,phi_dot,theta_dot,psi_dot]
        x, y, z = state[0], state[1], state[2]
        phi, theta, psi = state[3], state[4], state[5]
        x_dot, y_dot, z_dot = state[6], state[7], state[8]
        phi_dot, theta_dot, psi_dot = state[9], state[10], state[11]

    #     ref_traj = [xd[i], yd[i], zd[i], xd_dot[i], yd_dot[i], zd_dot[i], 
    #                 xd_ddot[i], yd_ddot[i], zd_ddot[i], xd_dddot[i], yd_dddot[i],
    #                 xd_ddddot[i], yd_ddddot[i], psid[i], psid_dot[i], psid_ddot[i]]


        xd, yd, zd = ref_traj[0], ref_traj[1], ref_traj[2], 
        xd_dot, yd_dot, zd_dot = ref_traj[3], ref_traj[4], ref_traj[5]
        xd_ddot, yd_ddot, zd_ddot = ref_traj[6], ref_traj[7], ref_traj[8]
        xd_dddot, yd_dddot = ref_traj[9], ref_traj[10]
        xd_ddddot, yd_ddddot = ref_traj[11], ref_traj[12]
        psid, psid_dot, psid_ddot = ref_traj[13], ref_traj[14], ref_traj[15]

        x1, x2, x3 = array([[x], [y]]), array([[x_dot], [y_dot]]), array([[phi], [theta]])
        x4, x5, x6 = array([[phi_dot], [theta_dot]]), array([[psi], [z]]), array([[psi_dot], [z_dot]])

        g0 = array([[np.cos(psi), np.sin(psi)],  [np.sin(psi), -np.cos(psi)]])
        g0_inv = array([[np.cos(psi), np.sin(psi)],  [np.sin(psi), -np.cos(psi)]])

        g1 = array([[theta_dot*psi_dot*I1],  [phi_dot*psi_dot*I2]])
        g2 = array([[phi_dot*theta_dot*I3],  [-g]])

        l0 = array([[np.cos(phi)*np.sin(theta)],  [np.sin(phi)]])*U1/m 
        dl0_dx3 = array([[-np.sin(phi)*np.sin(theta), np.cos(phi)*np.cos(theta)],  [np.cos(phi), 0]])*U1/m 
        dl0_dx3_inv = array([[0, 1/np.cos(phi)],  [1/np.cos(theta)*1/np.cos(phi), 1/np.cos(phi)*np.tan(theta)*np.tan(phi)]])*m/U1 
        dl0_dx3_inv_dot = array([[0, 1/np.cos(phi)*np.tan(phi)*phi_dot], 
                                 [1/np.cos(theta)*1/np.cos(phi)*(np.tan(theta)*theta_dot + np.tan(phi)*phi_dot), 1/np.cos(phi)*((1/np.cos(theta))**2*np.tan(phi)*theta_dot + (-1+2*(1/np.cos(phi))**2)*np.tan(theta)*phi_dot)]])*m/U1 

    #     Omega_square = Omega_coef_inv * abs([U1/b  U2/b  U3/b  U4/d]) 
    #     Omega_param = sqrt(Omega_square) 
    #     omega = Omega_param(2) + Omega_param[3] - Omega_param(1) - Omega_param(3) 

    #     h1 = [-Jr/Ixx*theta_dot*omega  Jr/Iyy*phi_dot*omega] 
        h1 = 0 
        k1 = diag([l/Ixx, l/Iyy]) 
        k1_inv = diag([Ixx/l, Iyy/l]) 
        k2 = diag([1/Izz, np.cos(phi)*np.cos(theta)/m]) 
        k2_inv = diag([Izz, m/(np.cos(phi)*np.cos(theta))]) 

        x1d = array([[xd], [yd]])  
        x1d_dot = array([[xd_dot], [yd_dot]]) 
        x1d_ddot = array([[xd_ddot], [yd_ddot]]) 
        x1d_dddot = array([[xd_dddot], [yd_dddot]]) 
        x1d_ddddot = array([[xd_ddddot], [yd_ddddot]]) 

        x5d = array([[psid], [zd]])
        x5d_dot = array([[psid_dot], [zd_dot]]) 
        x5d_ddot = array([[psid_ddot], [zd_ddot]]) 

        z1 = x1d - x1 
        v1 = x1d_dot + dot(A1,z1) 
        z2 = v1 - x2 
        z1_dot = -dot(A1,z1) + z2 
        v1_dot = x1d_ddot + dot(A1,z1_dot) 
        v2 = dot(g0_inv, z1 + v1_dot + dot(A2,z2)) 
        z3 = v2 - l0  
        z2_dot = -z1 - dot(A2,z2) + dot(g0,z3) 
        z1_ddot = -dot(A1,z1_dot) + z2_dot 
        v1_ddot = x1d_dddot + dot(A1, z1_ddot) 
        v2_dot = dot(g0_inv, z1_dot + v1_ddot + dot(A2,z2_dot)) 
        v3 = dot(dl0_dx3_inv, dot(g0.T,z2) + v2_dot + dot(A3, z3)) 
        z4 = v3 - x4 
        z3_dot = -dot(g0.T, z2) - dot(A3,z3) + dot(dl0_dx3, z4) 
        z2_ddot = - z1_dot - dot(A2, z2_dot) + dot(g0, z3_dot) 
        z1_dddot = -dot(A1, z1_ddot) + z2_ddot 
        v1_dddot = x1d_ddddot + dot(A1, z1_dddot) 
        v2_ddot = dot(g0_inv, z1_ddot + v1_dddot + dot(A2, z2_ddot)) 
        v3_dot = dot(dl0_dx3_inv, dot(g0.T, z2_dot) + v2_ddot + dot(A3, z3_dot)) + dot(dl0_dx3_inv_dot, dot(g0.T, z2) + v2_dot + dot(A3, z3))
        l1 = dot(k1_inv, dot(dl0_dx3.T, z3) + v3_dot - g1 - h1 + dot(A4, z4)).ravel()

        z5 = x5d - x5 
        v5 = x5d_dot + dot(A5, z5) 
        z6 = v5 - x6 
        z5_dot = - dot(A5, z5) + z6 
        v5_dot = x5d_ddot + dot(A5, z5_dot) 
        l2 = dot(k2_inv, z5 + v5_dot - g2 + dot(A6, z6)).ravel()

        U1, U2, U3, U4 = l2[1], l1[0], l1[1], l2[0]

        U1 = np.clip(U1, 0.0, 1e5)
        U2 = np.clip(U2, -1e5, 1e5)
        U3 = np.clip(U3, -1e5, 1e5)
        U4 = np.clip(U4, -1e5, 1e5)

        U = np.array([U1, U2, U3, U4])

        return U

    def get_control_input(self, state, U0, cont, current_traj):
        if (cont == self.Controllers[0]): #Backstepping_1
            A1, A2, A3 = 15*diag([1,1]), 10*diag([1,1]), 15*diag([1,1]) 
            A4, A5, A6 = 10*diag([1,1]), 15*diag([1,1]), 10*diag([1,1]) 
            U = self.backstepping(A1, A2, A3, A4, A5, A6, state, U0, current_traj) 
        elif (cont == self.Controllers[1]): #Backstepping_2
            A1, A2, A3 = 10*diag([1,1]), 5*diag([1,1]), 10*diag([1,1]) 
            A4, A5, A6 = 5*diag([1,1]), 10*diag([1,1]), 5*diag([1,1])
            U = self.backstepping(A1, A2, A3, A4, A5, A6, state, U0, current_traj) 
        elif (cont == self.Controllers[2]): #Backstepping_3
            A1, A2, A3 = 5*diag([1,1]), 3*diag([1,1]), 10*diag([1,1]) 
            A4, A5, A6 = 7*diag([1,1]), 1*diag([1,1]), 1*diag([1,1])  
            U = self.backstepping(A1, A2, A3, A4, A5, A6, state, U0, current_traj)
        elif (cont == self.Controllers[3]): #Backstepping_4
            A1, A2, A3 = 2*diag([1,1]), 5*diag([1,1]), 2*diag([1,1]) 
            A4, A5, A6 = 5*diag([1,1]), 2*diag([1,1]), 5*diag([1,1]) 
            U = self.backstepping(A1, A2, A3, A4, A5, A6, state, U0, current_traj)
        return U

    def euler(self, state, U):
        state_dot = self.model_dynamics(state, U)
        return state_dot
    
    def runge_kutta(self, state, U, dtau):
        k1 = self.euler(state, U)
        k2 = self.euler(state + k1/2., U + dtau/2.)
        k3 = self.euler(state + k2/2., U + dtau/2.)
        k4 = self.euler(state + k3, U + dtau)

        state_dot = (k1 + 2*k2 + 2*k3 + k4) / 6.
        return state_dot 


    def simulate(self, current_traj, dtau=1e-3, method="Backstepping_3"):
    #     states: [x,y,z,phi,theta,psi,x_dot,y_dot,z_dot,phi_dot,theta_dot,psi_dot]
        std_list = [0, 0, 0, 0]
        r_std, phi_std, theta_std, psi_std = std_list[0], std_list[1], std_list[2], std_list[3]
        fail_check = False
       
        ## Add noise, if you wish ##
        self.state[6] = normal(self.state[6], 0*r_std / 3.0)
        self.state[7] = normal(self.state[7], 0*r_std / 3.0)
        self.state[8] = normal(self.state[8], 0*r_std / 3.0)
        self.state[9] = normal(self.state[9], 0*phi_std)
        self.state[10] = normal(self.state[10], 0*theta_std)
        self.state[11] = normal(self.state[11], 0*psi_std)

        # Scipy diff solver
        # U = self.get_control_input(self.state, self.U, method, current_traj)
        # self.U = U

        # sol = integrate.solve_ivp(fun=self.model_dynamics_2, t_span=(0, dtau), y0=self.state)
        # self.state = sol.y[:,-1]

        # Euler Solver
        self.U = self.get_control_input(self.state, self.U, method, current_traj)

        state_dot = self.euler(self.state, self.U)
        self.state = self.state + state_dot * dtau

        # Runge-Kutta Solver
        # U3 = self.get_control_input(self.state3, self.U3, method, current_traj)
        # self.U3 = U3

        # state_dot = self.runge_kutta(self.state3, self.U3, dtau)
        # self.state3 = self.state3 + state_dot * dtau
        
        # diff12 = np.sum((self.state - self.state2)**2)
        # diff13 = np.sum((self.state - self.state3)**2)
        # diff23 = np.sum((self.state2 - self.state3)**2)
        # diff = np.array([diff12, diff13, diff23])

        if (np.abs(self.state[3]) > np.pi/2)  | (np.abs(self.state[4]) > np.pi/2):
            self.costValue = 1e12
            fail_check = True

        
        return fail_check