#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import math 
import rospy
import message_filters
import std_msgs
#from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
import geometry_msgs.msg
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import WrenchStamped
from franka_msgs.msg import StampedFloat32

#from scipy.optimize import curve_fit

import numpy as np
from collections import OrderedDict
from pytictoc import TicToc
#import quaternion
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

import skfuzzy as fuzz
from skfuzzy import control as ctrl

#Import libraries. Presenti sia quelle necessarie per la costruzione della rete neurale che quelle per ROS


""" My code for calculating manipulability in the desired direction """
# Definition of the model neural network class
class NN_model(nn.Module):
    
    # Size: The number of nodes in the model.
    # Width: The number of nodes in a specific layer.
    # Depth: The number of layers in a neural network.
    def __init__(self, input_size, hidden_depth, hidden_size, output_size, print_NN=False):
        super(NN_model, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_depth = hidden_depth
        self.output_size = output_size
        self.print_NN = print_NN
        
        self.layers = OrderedDict()
        # first layer linear part:
        # Applies a linear transformation to the incoming data: y=xAT+b
        self.layers["lin" + str(1)] = nn.Linear(self.input_size, self.hidden_size)
        # Layers appears to be a dictionary and we associate to the key lin1 the nn.Linear value
        # first layer ReLU part:
        self.layers["relu" + str(1)] = nn.ReLU()
        # Applies the rectified linear unit function element-wise:
        
        # other inner layers linear part:
        for i in range(2, self.hidden_depth + 1):
            # During training, randomly zeroes some of the elements of the input tensor with probability p
            self.layers["drop"+ str(i)] = nn.Dropout(p=0.2)
            self.layers["lin" + str(i)] = nn.Linear(self.hidden_size, self.hidden_size)
            self.layers["relu" + str(i)] = nn.ReLU()
            
        # last layer just linear:
        self.layers["drop"+ str(i)] = nn.Dropout(p=0.1)
        self.layers["lin" + str(self.hidden_depth +1)] = nn.Linear(self.hidden_size, self.output_size)
        
        self.pipe = nn.Sequential(self.layers)
        # A sequential container. Modules will be added to it in the order they are passed in the constructor.
        # Alternatively, an ordered dict of modules can also be passed in.
        
        if self.print_NN:
            print(self.pipe)
        
    
    def get_parameters(self):
        return self.pipe.parameters()
        
    
    def forward(self, x):
        return self.pipe(x)
    
# Define actor model
class ActorNN(nn.Module):
    
    def __init__(self):
        super(ActorNN, self).__init__()

        self.layers = OrderedDict()
        self.layers["lin" + str(1)] = nn.Linear(4, 64)
        self.layers["relu" + str(1)] = nn.ReLU()
        self.layers["drop"+ str(2)] = nn.Dropout(p=0.1)
        self.layers["lin" + str(2)] = nn.Linear(64,64)
        self.layers["relu" + str(2)] = nn.ReLU()
        self.layers["drop"+ str(3)] = nn.Dropout(p=0.1)
        self.layers["lin" + str(3)] = nn.Linear(64,2)
        self.layers["tanh" + str(3)] = nn.Tanh()
        
        self.pipe = nn.Sequential(self.layers)

    def get_parameters(self):
        return self.pipe.parameters()
        
    
    def forward(self, x):
        return self.pipe(x)

# Define critic model
class CriticNN(nn.Module):
    
    def __init__(self):
        super(CriticNN, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = F.dropout(F.relu(self.fc1(  torch.abs(x)  )), p=0.5)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.abs(x)

class Q_LMPC():
    
    def __init__(self, Device_, ensemble_NN_, actor_NN_, critic_NN_, ensemble_size_, prediction_horizon_, buffer_size_, samples_num_):
        
        self.Device = Device_
        self.NN_ensemble = ensemble_NN_
        self.actor_NN = actor_NN_
        self.critic_NN = critic_NN_
        self.ensemble_size = ensemble_size_
        self.prediction_horizon = prediction_horizon_
        self.buffer_size = buffer_size_
        self.samples_num = samples_num_
        
        cartesian_position_sub = message_filters.Subscriber("/franka_ee_pose", PoseStamped, queue_size = 1, buff_size = 2**20) # buffer size = 10, queue = 1
        cartesian_velocity_sub = message_filters.Subscriber("/franka_ee_velocity", TwistStamped, queue_size = 1, buff_size = 2**20)
        wrench_sub = message_filters.Subscriber("/simulated_wrench", WrenchStamped, queue_size = 1, buff_size = 2**20)
        pbo_sub = message_filters.Subscriber("/PBO_index",StampedFloat32, queue_size = 1, buff_size = 2**20)
        # Subscription to a certain topic and message type

        sync = message_filters.ApproximateTimeSynchronizer([cartesian_position_sub, cartesian_velocity_sub, wrench_sub, pbo_sub], queue_size = 1, slop = 0.1 )
        #policy used by message_filters::sync::Synchronizer to match messages coming on a set of topics
        sync.registerCallback(self.measurements_callback)
        #In the ROS setting a callback in most cases is a message handler. You define the message handler function and give it to subscribe.
        #You never call it yourself, but whenever a message arrives ROS will call you message handler and pass it the new message,
        #so you can deal with that.
        self.u_pub = rospy.Publisher("/equilibirum_pose", PoseStamped, queue_size = 1)
        self.D_pub = rospy.Publisher("/D_information", Float64, queue_size = 1)
        # Publishing a message on a certain topic
        self.tictoc = TicToc()
        
        # load data
        a_file = open("/home/vascomelo/work_space/src/Laboratorio/Updating strategies/Normalization/3dataDict_robot_py2.pkl", "rb")
        self.training_dict = pickle.load(a_file)
        #self.training_dict = torch.load('tensors.pt')
        self.x_mean_v, self.x_std_v, self.y_mean_v, self.y_std_v = self.training_dict['xy_norm']
        self.xn_train = self.training_dict['xn_train']
        self.yn_train = self.training_dict['yn_train']
        
        # Definition of the limits
        self.us = 0.7
        self.ul = 0.3
        self.action_mean = (self.us + self.ul)/2
        self.action_std = (self.us - self.ul)/2
        self.action_norm = (self.action_mean, self.action_std)

        self.Drs = 5
        self.Drl = 0.1
        self.damping_mean = (self.Drs + self.Drl)/2
        self.damping_std = (self.Drs - self.Drl)/2
        self.damping_norm = (self.damping_mean, self.damping_std)

        self.fh_mean = np.mean(self.x_mean_v[2:3])
        self.fh_std  = np.std(self.x_std_v[2:3])
        self.fh_norm = (self.fh_mean, self.fh_std)
        
        self.Cost_norm = np.load("/home/vascomelo/work_space/src/Laboratorio/Updating strategies/Normalization/3Cost_robot.npy")
        self.media_costo, self.stad_cost = self.Cost_norm

        # Fuzzy Network Data
        self.v_max = 0.1
        self.wrench_max = 150
        self.dwrench_max = 50
        self.fuzzy_limit = [ self.v_max, self.wrench_max, self.dwrench_max]
        

        self.v11=0 
        self.v12=0.01 
        self.v21=0.005 
        self.v22=0.025 
        self.v31=0.02
        self.v32=0.04 
        self.v41=0.04
        self.v42=0.06

        self.velocity_ranges = np.array([[self.v11,self.v12],
                                        [ self.v21,self.v22], 
                                        [self.v31,self.v32],
                                        [self.v41,self.v42]])
        
        
        self.fuzzy_parameters = np.array([[110,130,30,40],
                                        [100,120,20,40],
                                        [90,110,15,35],
                                        [65,90,10,20]])
        
        self.filter_alpha = 0.05

        self.loss_history_critic=[]
        self.loss_history_actor=[]
        # putting it in delta form for learning
        ### New data (st, at)--> (st+1 - st)
        self.yn_train_d = self.yn_train[:,0:3] - self.xn_train[:,0:3]
       
        self.z_record_norm = np.zeros((self.buffer_size + 1, 1))
        self.x_record_norm = np.zeros((self.buffer_size + 1, 1))
        self.fh_record_norm = np.zeros((self.buffer_size + 1, 1))
        self.fh_record_not_norm = np.ones((self.buffer_size + 1, 1))
        self.u_record_norm = np.zeros((self.buffer_size, 1))
        self.D_record_norm = np.zeros((self.buffer_size, 1))
        self.Upper_limit_u_norm = np.zeros((self.buffer_size, 1))
        self.Lower_limit_u_norm = np.zeros((self.buffer_size, 1))
        self.Upper_limit_D_norm = np.zeros((self.buffer_size, 1))
        self.Lower_limit_D_norm = np.zeros((self.buffer_size, 1))
        self.PBO_record = np.zeros((self.buffer_size+1, 1))
        self.time_record = np.zeros((self.buffer_size+1, 1))
        
        self.iter_buffer_ = 0
        self.SIZEN = 100
        self.iter_init = 0

        self.rate = rospy.Rate(6)  # 6 Hz
        
        self.recording_index = 0
        # Define the directory where you want to save the file
        directory = os.path.join(os.path.expanduser('~'), 'work_space', 'src','Laboratorio', 'Updating strategies', 'Data')

        # Ensure the directory exists; if it doesn't, create it
        os.makedirs(directory, exist_ok=True)

        # Define the full path to the file
        self.file_name = os.path.join(directory, 'recorded_data_med_freq_PBO_test.npy')
        self.cost_file_name = os.path.join(directory, 'cost_std.npy')

        self.critic_norm = os.path.join(directory, 'critic_norm.npy')
        self.critic_mean, self.critic_std = np.load(self.critic_norm)

        self.velocity_record_=np.zeros((self.buffer_size +1, 1))
        self.pose_record_=np.zeros((self.buffer_size+1, 1))
        self.force_record_=np.zeros((self.buffer_size+1, 1))
        self.PBO_record_=np.zeros((self.buffer_size+1, 1))
        self.setpoint_record_=np.zeros((self.buffer_size+1, 1))
        self.damping_record_=np.zeros((self.buffer_size+1, 1))
        self.time_record_=np.zeros((self.buffer_size+1, 1))

        self.filter_alpha = 0.05
        #rospy.on_shutdown(self.on_shutdown)  # Register shutdown handler

    
    def low_pass_filter(self, current_value, filtered_value):

        filtered_value = self.filter_alpha * current_value + (1 - self.filter_alpha) * filtered_value
        return filtered_value
       
    def fuzzy_logic(self, cartesian_velocity_, wrench_, dwrench_, PBO_index_,range_index):
        
        if abs(cartesian_velocity_) >= self.velocity_ranges[range_index][0] and abs(cartesian_velocity_) < self.velocity_ranges[range_index][1]:
            shaping_factor=(1+math.sin(2*math.pi*( (abs(cartesian_velocity_)-self.velocity_ranges[range_index][0]))/(self.velocity_ranges[range_index][1]-self.velocity_ranges[range_index][0] ) + self.velocity_ranges[range_index][0] + (np.average(self.velocity_ranges[range_index])-self.velocity_ranges[range_index][0])/2))/2
        
            AL_=0
            f_1,f_2,df_1,df_2 = self.fuzzy_parameters[range_index,:]
            self.Al_med_min = 0.025 + 0.01*PBO_index_
            self.Al_med_max = 0.045 + 0.01*PBO_index_
            self.Al_high_min = 0.040 + 0.01*PBO_index_
            self.Al_high_max = 0.070 + 0.01*PBO_index_
            #print(f"self.Al_med_min shape: {self.Al_med_min.shape}")
            # Step 1: Define the fuzzy sets for input variables (cost and benefit)
            vel = ctrl.Antecedent(np.arange(0, self.fuzzy_limit[0], 0.001), 'Velocity')
            wrench = ctrl.Antecedent(np.arange(0, self.fuzzy_limit[1], 0.1), 'Wrench')
            dwrench = ctrl.Antecedent(np.arange(0, self.fuzzy_limit[2], 0.1), 'dWrench')
            AL = ctrl.Consequent(np.arange(0, self.Al_high_max, 0.01), 'Assistance Level')

            # Membership functions for input
            vel['stop'] = fuzz.trimf(vel.universe,[0,self.velocity_ranges[0][0],self.velocity_ranges[0][1]])
            vel['slow'] = fuzz.trimf(vel.universe, [self.velocity_ranges[1][0], np.mean([self.v21, self.v22]), self.velocity_ranges[1][1]])
            vel['move'] = fuzz.trimf(vel.universe, [self.velocity_ranges[2][0], np.mean([self.v31, self.v32]), self.velocity_ranges[2][1]])
            vel['fast'] = fuzz.trapmf(vel.universe,[self.velocity_ranges[3][0],self.velocity_ranges[3][1],self.fuzzy_limit[0],self.fuzzy_limit[0]])


            wrench['safe'] = fuzz.trapmf(wrench.universe,[0,0,f_1,f_2])
            wrench['non_safe'] = fuzz.trapmf(wrench.universe,[f_1,f_2,self.wrench_max,self.wrench_max])

            dwrench['no_var'] = fuzz.trapmf(dwrench.universe,[0,0,df_1,df_2])
            dwrench['var'] = fuzz.trapmf(dwrench.universe,[df_1,df_2,self.dwrench_max,self.dwrench_max])

            # Membership functions for output
            AL['none'] = fuzz.trimf(AL.universe, [0, 0, self.Al_med_min])
            AL['medium'] = fuzz.trimf(AL.universe, [self.Al_med_min, np.mean([self.Al_med_min,self.Al_med_max]), self.Al_med_max])
            AL['high'] = fuzz.trimf(AL.universe, [self.Al_high_min, np.mean([self.Al_high_min,self.Al_high_max]), self.Al_high_max])

            # Step 3: Define the fuzzy rules
            rule1 = ctrl.Rule(vel['stop'] & wrench['safe'] & dwrench['no_var'] , AL['medium'])
            rule2 = ctrl.Rule(vel['stop'] & wrench['safe'] & dwrench['var'] , AL['medium'])

            rule3 = ctrl.Rule(vel['slow'] & wrench['safe'] & dwrench['no_var'] , AL['medium'])
            rule4 = ctrl.Rule(vel['slow'] & wrench['safe'] & dwrench['var'] , AL['high'])

            rule5 = ctrl.Rule(vel['move'] & wrench['safe'] & dwrench['no_var'] , AL['medium'])
            rule6 = ctrl.Rule(vel['move'] & wrench['safe'] & dwrench['var'] , AL['high'])

            rule7 = ctrl.Rule(vel['fast'] , AL['none'])
            rule8 = ctrl.Rule(wrench['non_safe'], AL['none'])

            # Step 4: Implement the fuzzy inference system
            AL_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8])
            AL_sim = ctrl.ControlSystemSimulation(AL_ctrl)

            # Step 5: Test the fuzzy logic system with sample inputs
            AL_sim.input['Velocity'] = cartesian_velocity_
            AL_sim.input['Wrench'] = wrench_  
            AL_sim.input['dWrench'] = dwrench_ 

            AL_sim.compute()
            AL_= AL_sim.output['Assistance Level']*shaping_factor
        else: 
            AL_ = 0

        return AL_ 
    
    def cost_func_p(self, fh_, delta_u_): 
      
        N_  = fh_.shape[1]   # The samples are the values extracted  
        ense_ = fh_.shape[0] # 5
        
        # weights
        Q_= 1
        R = 5
        
        cost_fh = (np.sum(np.sum(np.multiply(fh_,fh_)*Q_, axis=2).reshape(ense_, N_), axis=0)/ense_).reshape(N_)
        cost_du = (np.sum(np.sum(np.multiply(delta_u_,delta_u_)*R, axis=2).reshape(ense_, N_), axis=0)/ense_).reshape(N_)

        #print(f"cost_fh: {cost_fh}")
        #print(f"cost_du: {cost_du}")

        return cost_fh+cost_du
    
    def model_approximator_train(self, state_force_norm_train, u_D_norm_train, NN_model, Nh_, learning_rate):
        
        x_data_ = np.copy(state_force_norm_train[0:Nh_, :])
        x_data_ = np.append(x_data_, u_D_norm_train, axis = 1)

        y_data_ = state_force_norm_train[1:Nh_+1, :] - state_force_norm_train[0:Nh_, :]
        
        
        NN_model_    = NN_model
        learning_rate_   = learning_rate
        
        
        NN_model_.to(self.Device)  # putting the model into GPU (but not available now)
        cost_ = nn.MSELoss()  # Mean squared loss
        # Creates a criterion that measures the mean squared error (squared L2 norm) between each element 
        # in the input xxx and target yyy 
        optimizer_ = torch.optim.Adam(NN_model_.get_parameters(), lr =learning_rate_)
        NN_model_.train()
        
        # clear grads
        optimizer_.zero_grad()
    
        # Forward propagation
        x_ = torch.tensor(x_data_, dtype= torch.float32, device=self.Device)
        y_ = torch.tensor(y_data_, dtype= torch.float32, device=self.Device)
    
        y_hat_ = NN_model_.forward(x_)
        loss_ = cost_.forward(y_hat_, y_)             # calculating the loss
        loss_.backward()                              # backprop
        optimizer_.step()                             # updating the parameters
        #torch.cuda.empty_cache()
               
            
        return

        # Input data are normalized
    
    def Critic_train(self, state_force_train, u_record_train_, timestamp, PBO_record, Critic_NN, xy_norm, Cost_norm, Nh, learning_rate):
    
        x_mean_v, x_std_v, y_mean_v, y_std_v = xy_norm

        #Normalization parameters to/from Model Approximator and Actor:
        vel_mean_ = x_mean_v[0]
        vel_std_ = x_std_v[0]
        pose_mean_ = x_mean_v[1]
        pose_std_ = x_std_v[1]   
        fh_mean_ = x_mean_v[2]
        fh_std_ = x_std_v[2]
        u_mean_ = x_mean_v[3]
        u_std_ = x_std_v[3]

        #Normalization parameters to/from Critic:
        critic_delta_u_mean = self.critic_mean[0]
        critic_delta_u_std = self.critic_std[0]
        critic_fh_mean = self.critic_mean[1]
        critic_fh_std = self.critic_std[1]
        

        Critic_NN_ = Critic_NN
        Nh_ = Nh
        learning_rate_ = learning_rate
        PBO_record_ = PBO_record

        #Exctract Velocity
        velocity_norm_ = state_force_train[0:Nh_,0:1] #Shape (Nh_,1)
        velocity_unorm_ = (velocity_norm_ * vel_std_) + vel_mean_

        #Exctract Pose
        pose_norm_ = state_force_train[0:Nh_,1:2] #Shape (Nh_,1)
        pose_unorm_ = (pose_norm_ * pose_std_) + pose_mean_

        #Exctract Force
        fh_norm_ = state_force_train[0:Nh_,2:3] #Shape (Nh_,1)
        fh_unorm_ = (fh_norm_*fh_std_)+fh_mean_

        #Exctract Set-point
        u_norm = u_record_train_ #Shape (Nh_,1)
        u_unorm_ = (u_norm*u_std_)+u_mean_

        #Compute the dwrench
        dfh_raw = np.zeros((Nh_,1))
        dft_filtered = np.zeros((Nh_,1))

        for i in range(1, Nh_):
            dfh_raw[i, 0] = (fh_unorm_[i, 0] - fh_unorm_[i - 1, 0]) / (timestamp[i, 0] - timestamp[i - 1, 0])
            dft_filtered[i,0] = self.low_pass_filter(dfh_raw[i, 0],dft_filtered[i - 1, 0])

        #Compute the set-point
        fuzzy_u_=np.zeros((Nh_,1))
        fuzzy_delta_u_= np.zeros((Nh_,1))
        for i in range(1, Nh_):
            u_range_ = np.zeros(4)
            for j in range(4):
                u_range_[j] = self.fuzzy_logic(abs(velocity_unorm_[i,0]),abs(fh_unorm_[i,0]),abs(dft_filtered[i,0]),PBO_record_[i,0],j)
            fuzzy_delta_u_[i,0] = np.sum(u_range_)*np.sign(dft_filtered[i,0])
            fuzzy_u_[i,0] = fuzzy_delta_u_[i,0]+ pose_unorm_[i,0]

        # At this point, fuzzy_u_ has size (Nh_,1) but the first line is 0

        #Compute the delta_u, normalize it and make it a tensor
        delta_setpoint_unorm_ = fuzzy_u_[1:Nh_,0] - u_unorm_[1:Nh_,0]
        delta_setpoint_norm_ = (delta_setpoint_unorm_-critic_delta_u_mean)/critic_delta_u_std
        delta_setpoint_norm_torch_ = torch.tensor(delta_setpoint_norm_, dtype=torch.float32, device=self.Device).unsqueeze(1)

        # Normalize fh and make it a tensor
        fh_norm_ = (fh_unorm_ - critic_fh_mean)/critic_fh_std
        fh_norm_torch = torch.tensor(fh_norm_[1:Nh_], dtype=torch.float32, device=self.Device) 

        # Both Tensors are of length (Nh_-1)
        # Concatenate the tensors
        combined_input_torch = torch.cat(( fh_norm_torch[:-1], delta_setpoint_norm_torch_[:-1]), dim=1)
        combined_input_p1_torch = torch.cat(( fh_norm_torch[1:], delta_setpoint_norm_torch_[1:]), dim=1)
        
        #This code is used to collect the unormalized data for the Critic Input
        """
        delta_setpoint_unorm_torch_array = delta_setpoint_unorm_torch.reshape(1, -1)
        fh_unorm_1d = fh_unorm_[1:]).reshape(1, -1)
        
        # Stack them together vertically (along rows)
        new_data_1 = np.vstack([delta_setpoint_unorm_torch_array, fh_unorm_1d])
        print(f"new_data_1.shape: {new_data_1.shape}")

        # Define the filename
        file_name_1 = self.cost_file_name

        # Check if the file exists
        if os.path.exists(file_name_1):
            # Load the existing data
            existing_data = np.load(file_name_1)
            
            # Append the new data to the existing data
            updated_data = np.hstack((existing_data, new_data_1))
            print(f"updated_data.shape: {updated_data.shape}")
            # Save the updated data back to the file
            np.save(file_name_1, updated_data)
            print("DATA UPDATED")
        else:
            # If the file doesn't exist, create it by saving the new data
            np.save(file_name_1, new_data_1)
            print("DATA CREATED")

        """

        Critic_NN_.to(self.Device)
        optimizer_C = torch.optim.SGD(Critic_NN_.parameters(), lr =learning_rate_)
        errore_f = nn.MSELoss()
      
        for j in range(0, Nh_-2):
            
            Critic_NN_.train()
            optimizer_C.zero_grad()
            
            # Q represent the cost function integrated between t and infinite
            Q_n = Critic_NN_.forward(combined_input_torch[j,:])
            Q_npiu1 = Critic_NN_.forward(combined_input_p1_torch[j,:]) 
            
            Cost_fh = ((combined_input_torch[j:j+1,0].unsqueeze(0)).detach().numpy()*critic_fh_std + critic_fh_mean).reshape(1,1,1)
            Cost_du = ((combined_input_torch[j:j+1,1].unsqueeze(0)).detach().numpy()*critic_delta_u_std + critic_delta_u_mean).reshape(1,1,1)
            
            COSTO = self.cost_func_p(Cost_fh,Cost_du) 
            del Cost_fh, Cost_du
            
            Q_Bellman = Q_npiu1 + torch.from_numpy(COSTO)

            Error_c = errore_f.forward(Q_n, Q_Bellman)
            
            Error_c.backward(retain_graph=True)                          # backprop
            optimizer_C.step()                             # updating the parameters        
        
        return
    
    def ComputationUP(self, xy_norm, state_action_norm_torch_lie, model_approximator, num_ensembles):
    
        x_mean_v, x_std_v, y_mean_v, y_std_v = xy_norm
        cost_mean, cost_std_dev = self.Cost_norm
    
        state_action_norm_torch_lie_f = torch.clone(state_action_norm_torch_lie)
        state_action_norm_torch_lie_f[:,3:4] = 0
        state_action_norm_torch_lie_f[:,4:5] = 0
        Dati_lie_f = torch.zeros(2,5)
        for k in range(num_ensembles):
            model_approximator["NN"+str(k)].to(self.Device)
            model_approximator["NN"+str(k)].eval()
            output_f_lie = model_approximator["NN"+str(k)].forward(state_action_norm_torch_lie_f)
            Dati_lie_f[:,k] = output_f_lie[:,0:2]
        #Risultato_lie_f = (torch.sum(Dati_lie_f, dim=1)/num_ensembles)
        Risultato_lie_f = (torch.sum(Dati_lie_f, dim=1)/num_ensembles)*torch.FloatTensor(x_std_v[0:2]) + torch.FloatTensor(x_mean_v[0:2])
        #print( Risultato_lie_f )
    
        state_action_norm_torch_lie_g1 = torch.clone(state_action_norm_torch_lie)
        state_action_norm_torch_lie_g1[:,3:4] = 1
        state_action_norm_torch_lie_g1[:,4:5] = 0
        Dati_lie_g1 = torch.zeros(2,5)
        for k in range(num_ensembles):
            model_approximator["NN"+str(k)].to(self.Device)
            model_approximator["NN"+str(k)].eval()
            output_g1_lie = model_approximator["NN"+str(k)].forward(state_action_norm_torch_lie_g1)
            Dati_lie_g1[:,k] = output_g1_lie[:,0:2]
        #Risultato_lie_g1 = (torch.sum(Dati_lie_g1, dim=1)/num_ensembles)
        Risultato_lie_g1 = (torch.sum(Dati_lie_g1, dim=1)/num_ensembles)*torch.FloatTensor(x_std_v[0:2]) + torch.FloatTensor(x_mean_v[0:2]) - Risultato_lie_f
        #print("Risultato_lie_g1", Risultato_lie_g1 )
        
        state_action_norm_torch_lie_g2 = torch.clone(state_action_norm_torch_lie)
        state_action_norm_torch_lie_g2[:,3:4] = 0
        state_action_norm_torch_lie_g2[:,4:5] = 1
        Dati_lie_g2 = torch.zeros(2,5)
        for k in range(num_ensembles):
            model_approximator["NN"+str(k)].to(self.Device)
            model_approximator["NN"+str(k)].eval()
            output_g2_lie = model_approximator["NN"+str(k)].forward(state_action_norm_torch_lie_g2)
            Dati_lie_g2[:,k] = output_g2_lie[:,0:2]
        #Risultato_lie_g2 = (torch.sum(Dati_lie_g2, dim=1)/num_ensembles)
        Risultato_lie_g2 = (torch.sum(Dati_lie_g2, dim=1)/num_ensembles)*torch.FloatTensor(x_std_v[0:2]) + torch.FloatTensor(x_mean_v[0:2]) - Risultato_lie_f
        #print("Risultato_lie_g2", Risultato_lie_g2 )
        
        # Vl = 1/2 z P1 z + 1/2 (xd-x)P2(xd-x)
        P1 = 1e-5 #1e-5 peso presentazione     #7
        P2 = 4e-4 #1e-4 peso presentazione  #-700
        gradient_P_z_lie = P1*(state_action_norm_torch_lie[:,0:1]*torch.FloatTensor(x_std_v[0:1]) + torch.FloatTensor(x_mean_v[0:1]))
        gradient_P_x_lie = -P2*((state_action_norm_torch_lie[:,2:3])*torch.FloatTensor(x_std_v[2:3]) + torch.FloatTensor(x_mean_v[2:3]))
        #print("gradient_P_z_lie: ", gradient_P_z_lie)
        #print("gradient_P_x_lie: ", gradient_P_x_lie)
    
        gradient_P = torch.cat((gradient_P_z_lie, gradient_P_x_lie), 0)
        Lie_F = Risultato_lie_f.unsqueeze_(0).T
        Lie_G1 = Risultato_lie_g1.unsqueeze_(0).T
        Lie_G2 = Risultato_lie_g2.unsqueeze_(0).T
        #print("gradient_P: ", gradient_P[0,:], gradient_P.shape)
        #print("Lie_g: ", Lie_F[0,:], Lie_F.shape)
        Lf_P = torch.sum(gradient_P[0,:]*Lie_F[0,:] + gradient_P[1,:]*(state_action_norm_torch_lie[:,0:1]*torch.FloatTensor(x_std_v[0:1]) + torch.FloatTensor(x_mean_v[0:1])))
        Lg1_P = torch.sum(gradient_P[0,:]*Lie_G1[0,:] + gradient_P[1,:]*torch.abs(Lie_G1[1,:])) # gradient_P[0,:]*Lie_G1[0,:]
        Lg2_P = torch.sum(gradient_P[:,:]*Lie_G2[:,:])
        a = Lf_P
        b1 = Lg1_P
        b2 = Lg2_P
        beta = b1**2 + b2**2
        #print("Lg_P: ", Lg_P)
        #print("Lf_P: ", Lf_P)
    
    
        h1 = ((-(a + torch.sqrt(a**2 + beta**2))/beta)*b1).item()
        h2 = ((-(a + torch.sqrt(a**2 + beta**2))/beta)*b2).item()
    
        if h1 > 0.52: # 0.25
            h1 = 0.52
        elif h1 < 0.48: # -0.25 #-0.1
            h1 = 0.48
    
        if h2 > 3: #18
            h2 = 3 
        elif h2 < 2: #12
            h2 = 2
        return h1, Lg1_P.item(), h2, Lg2_P.item()
    
    def CEM_critic(self, x_initial_, action_dim_, time_horizon_, num_samples_, xy_norm_, U_l_U_, L_l_U_, U_l_D_, L_l_D_,
               NN_model_, Actor_NN_, PBO_index, num_ensembles_cem_):
        
        assert(x_initial_.shape[0] == 1)  # state should be a row vector z, x and fh (no actions)
        # if this condition is not true, an assertion error is returned
        
        x_mean_v_, x_std_v_, y_mean_v_, y_std_v_ = xy_norm_ # normalization variables
        
        state_dim_  = x_initial_.shape[1]
        state_action_dim_ = action_dim_ + state_dim_ + 1
        smoothing_rate_ = 0.9 #0.9
        iteration_      = 3 #10
        num_elites_ = 4 #32
        # 16-32 elites are enough with 64-128 samples for: action_dim * time_horizon <= 100
        num_ensembles_ = num_ensembles_cem_
        PBO_index_ = PBO_index

        #Normalization parameters to/from CriticNN:
        critic_delta_u_mean = self.critic_mean[0]
        critic_delta_u_std = self.critic_std[0]
        critic_fh_mean = self.critic_mean[1]
        critic_fh_std = self.critic_std[1]

        for k in range(num_ensembles_):
            NN_model_["NN"+str(k)].to(self.Device) 
        
        # Initialize the samples:
        mu_matrix_u_  = np.zeros((action_dim_, time_horizon_))
        std_matrix_u_ = np.ones((action_dim_, time_horizon_))
        
        mu_matrix_D_  = np.zeros((1, time_horizon_))
        std_matrix_D_ = np.ones((1, time_horizon_))

        PBO_array = np.full((num_ensembles_, num_samples_, 1), PBO_index_)

        #Start the iterations
        for _ in range(iteration_):

            #Broadcast the state to be used for all ensembles and samples:
            state_t_broadcasted_ = np.ones((num_ensembles_, num_samples_, state_dim_)) * x_initial_
    
            if 'action_samples_' in locals(): 
                del action_samples_
                del damping_samples_
    
            # Draw random samples from a normal (Gaussian) distribution of the optimizing variables:
            action_samples_ = np.random.normal(loc=mu_matrix_u_, scale=std_matrix_u_,
                                               size=(num_samples_, action_dim_, time_horizon_))
            
            damping_samples_ = np.random.normal(loc=mu_matrix_D_, scale=std_matrix_D_,
                                               size=(num_samples_, 1, time_horizon_))

            # Clip the samples in order to be coherent with the Stability Guarentees:
            action_samples_.clip(L_l_U_, U_l_U_, out=action_samples_)
            damping_samples_.clip(L_l_D_, U_l_D_, out=damping_samples_)
    
            costs_ = np.zeros(num_samples_-1)
            dforce_filt = np.zeros((num_ensembles_,num_samples_,time_horizon_))

            # Evaluate the trajectories and find the elites:
            for t in range(time_horizon_):

                # Take the action and damping sample at time t and broadcast it to all ensembles. (It is already normalized)
                action_t_norm_ = action_samples_[:,:,t].reshape(num_samples_, action_dim_)
                # 2 dimensions
                action_t_broadcasted_norm_ = np.ones((num_ensembles_, num_samples_, action_dim_)) * action_t_norm_
                # 3 dimensions
                
                damping_t_norm_ = damping_samples_[:,:,t].reshape(num_samples_, 1)
                # 2 dimensions
                damping_t_broadcasted_norm_ = np.ones((num_ensembles_, num_samples_, 1)) * damping_t_norm_
                # 3 dimensions
                
                #Normalize the state, append both the action and damping (already normalized) to be forwarded through the Model Approximator:

                state_t_broadcasted_norm_ = (state_t_broadcasted_ - x_mean_v_[0:state_dim_])/x_std_v_[0:state_dim_]
                state_action_norm_ = np.append(state_t_broadcasted_norm_, action_t_broadcasted_norm_, axis=2)
                state_action_damping_norm_ = np.append(state_action_norm_, damping_t_broadcasted_norm_, axis=2)
                # at this point the dimension is: (5 ensembles, 64 samples, 5 numbers (z, x, fh, u and Dr))
                
                state_action_damping_norm_torch_ = torch.tensor(state_action_damping_norm_,
                                                                dtype=torch.float32, device=self.Device)
                state_t_broadcasted_norm_torch_ = torch.tensor(state_t_broadcasted_norm_, 
                                                               dtype=torch.float32, device=self.Device)
                                                                                                                 
                NN_model_["NN0"].eval()
                state_tt_norm_torch_ = NN_model_["NN0"].forward(state_action_damping_norm_torch_[0,:,:].view(num_samples_, state_action_dim_)) + state_t_broadcasted_norm_torch_[0,:,:].view(num_samples_, state_dim_).view(1,num_samples_, state_dim_ )

                state_tt_norm_ = np.asarray(state_tt_norm_torch_.detach()).reshape(num_ensembles_, num_samples_,state_dim_)

                # Take the state_tt and concenate PBO to get the action that the actor would take:
                state_tt_PBO_norm_ = np.concatenate((state_tt_norm_, PBO_array), axis=2)
                state_tt_PBO_norm_torch_ = torch.tensor(state_tt_PBO_norm_,dtype=torch.float32, device=self.Device)
                Actor_NN_.eval()
                action_tt_norm_torch = Actor_NN_.forward(state_tt_PBO_norm_torch_[:,:,:].view(num_samples_,4))
                output_tt_norm_ = np.asarray(action_tt_norm_torch.detach()).reshape(num_ensembles_, num_samples_,2)
                action_tt_norm_ = output_tt_norm_[:,:,0]
                action_tt_ = action_tt_norm_*x_std_v_[3] + x_mean_v_[3]

                #Denormalize the state_tt and predict the action it would come from the fuzzy:
                state_tt_ = state_tt_norm_[0:3]*y_std_v_ + y_mean_v_ 

                velocity_tt = state_tt_[:,:,0].reshape(-1)
                position_tt = state_tt_[:,:,1].reshape(-1)
                force_tt = state_tt_[:,:,2].reshape(-1)
                force_t = state_t_broadcasted_[:,:,2].reshape(-1)

                fh_critic_norm = (force_tt-critic_fh_mean)/critic_fh_std

                #Compute df and filter it
                dforce_raw = (force_tt-force_t)*6
                dforce_filt[:, :, t] = np.where(t == 0, 0, self.low_pass_filter(dforce_raw, dforce_filt[:, :, t-1]))
                
                step_cost_ = np.empty_like(np.zeros(num_samples_-1))
                costo_rete_= np.zeros((num_ensembles_cem_,num_samples_-1))
                self.critic_NN.eval()
                fuzzy_action = np.zeros[1,num_samples_]
                costi_rete_ = []
                for j1 in range(0,num_samples_-1):

                    action_range = np.zeros(4)
                    for i in range(4):
                        action_range[i]=self.fuzzy_logic(velocity_tt[j1],force_tt[j1],dforce_filt[0,j1,t],PBO_index_[0],i)
                    fuzzy_action[0,j1] = np.sum(action_range)*np.sign(dforce_filt[0,j1,t]) + position_tt[j1]

                    # Compute delta_u, and normalize it to be forwared thorugh the Critic:
                    delta_action_unorm_ = fuzzy_action[0,j1]-action_tt_[0,j1]                        
                    delta_critic_norm = (delta_action_unorm_-critic_delta_u_mean)/critic_delta_u_std

                    critic_input_norm_ = np.column_stack((fh_critic_norm[j1], delta_critic_norm))
                    critic_input_norm_torch_ = torch.tensor(critic_input_norm_,dtype=torch.float32,device=self.Device)

                    # Pass the combined input to the CriticNN
                    costo_rete_[0, j1] = self.critic_NN.forward(critic_input_norm_torch_)
                    #print(f"Cost of input: {critic_input_norm_torch_} was {costo_rete_[0, j1]}")
                    costi_rete_.append(costo_rete_[0,j1].item())

                step_cost_[j1] = (np.sum(costi_rete_)/num_ensembles_cem_)
                #rospy.loginfo("After loop")
    
                STEPPO = step_cost_.reshape(num_samples_-1)
    
                # the input of the critic network is normalized, the output is assumed not normalized
                state_t_broadcasted_ = state_tt_
                del state_action_damping_norm_torch_; del state_t_broadcasted_norm_torch_; del step_cost_; 
                #torch.cuda.empty_cache()
    
                costs_ += (STEPPO)
            top_elites_index_ = costs_.argsort()[::1][:num_elites_]  # sorting index with min cost first

            elites_u_  = action_samples_[top_elites_index_,:,:].reshape(num_elites_, action_dim_, time_horizon_)
            mu_matrix_new_u_  = np.sum(elites_u_, axis=0)/num_elites_
            std_matrix_new_u_ = np.sqrt( np.sum( np.square(elites_u_ - mu_matrix_new_u_), axis=0)/num_elites_) 
            
            elites_D_ = damping_samples_[top_elites_index_,:,:].reshape(num_elites_, 1, time_horizon_)
            mu_matrix_new_D_  = np.sum(elites_D_, axis=0)/num_elites_
            std_matrix_new_D_ = np.sqrt( np.sum( np.square(elites_D_ - mu_matrix_new_D_), axis=0)/num_elites_) 
            # mu_new should broadcast to size of elites_ then subtract and then elementwise square 
    
            # Update the mu_ and std_
            mu_matrix_u_  = smoothing_rate_*mu_matrix_new_u_  + (1-smoothing_rate_)*mu_matrix_u_
            std_matrix_u_ = smoothing_rate_*std_matrix_new_u_ + (1-smoothing_rate_)*std_matrix_u_
            best_action_n_seq_ = elites_u_[0,:,:].reshape(action_dim_, time_horizon_)
            
            mu_matrix_D_  = smoothing_rate_*mu_matrix_new_D_  + (1-smoothing_rate_)*mu_matrix_D_
            std_matrix_D_ = smoothing_rate_*std_matrix_new_D_ + (1-smoothing_rate_)*std_matrix_D_
            best_damping_n_seq_ = elites_D_[0,:,:].reshape( 1, time_horizon_)
        
            # mu is the average of the source from which samples are generated, but the real normalization for
            # the action applied and used in the cost function is made respect to the previous average function
            
        
        return best_action_n_seq_, best_damping_n_seq_
    
    def Actor_train(self, state_force_train, U_l_U_, L_l_U_, U_l_D_, L_l_D_,
                Actor_NN, Model_NN, xy_norm, Nh, PBO_index, learning_rate ):

        #print("Actor train was called")
        Actor_NN_ = Actor_NN
        PBO_index_ = PBO_index
        Model_NN_ = Model_NN
        Nh_ = Nh
        learning_rate_ = learning_rate
        xy_norm_ = xy_norm
        
        # variables to normalize
        x_mean_v_, x_std_v_, y_mean_v_, y_std_v_ = xy_norm_
        # denormalization
        state_force_train_not_norm_ = state_force_train*x_std_v_[0:3] + x_mean_v_[0:3]
        
        # normalized quantities
        state_force_trainp1_ = np.copy(state_force_train[1:Nh_+1, :])
        PBO_index_trainp1_ = np.copy(PBO_index_[1:Nh_+1, :])
        actor_input_ = np.append(state_force_trainp1_,PBO_index_trainp1_,axis=1)

        
        Actor_NN_.to(self.Device)  # putting the model into GPU (but not available now)
        #optimizer_A = torch.optim.Adam(Actor_NN_.layers["lin" + str(2)].parameters(), lr =learning_rate_)
        optimizer_A = torch.optim.Adam(Actor_NN_.get_parameters(), lr =learning_rate_)
        errore_f = nn.MSELoss()
        
        actor_input_torch_ = torch.tensor(actor_input_, dtype=torch.float32, device=self.Device)
        
        for j in range(0, Nh_):
            #rospy.loginfo("Nh")
            Actor_NN_.train()
            action_damping_train = Actor_NN_.forward(actor_input_torch_[j,:])
            optimizer_A.zero_grad()
            
            # Actions from minimization of CEM
            U_npiu1, D_npiu1 = self.CEM_critic(state_force_train_not_norm_[j+1:j+2], 1, self.prediction_horizon, 
                                               self.samples_num, xy_norm_, U_l_U_[j:j+1].item(),
                                               L_l_U_[j:j+1].item(), U_l_D_[j:j+1].item(), L_l_D_[j:j+1].item(), 
                                               Model_NN_,Actor_NN_,PBO_index_[j:j+1,0], 1)
            
            U_D_npiu1 = torch.Tensor([U_npiu1[0,0],D_npiu1[0,0]])

            action_damping_from_NN = action_damping_train
            action_damping_from_CEM = U_D_npiu1
            
            # Cost function
            Error_a = errore_f.forward(action_damping_from_NN, action_damping_from_CEM)
            Error_a.backward(retain_graph=True)                           # backprop
            optimizer_A.step()                             # updating the parameters

        return
    
    def poly2d(self, xy, *params):
        #print(f"params length: {len(params)}")  # Debugging line
        c0, c1, c2, c3, c4, c5 = params
        x, y = xy
        result = c0 + c1 * x + c2 * y + c3 * x**2 + c4 * y**2 + c5 * x * y
        # Ensure that the return value is a numpy array of floats
        return result
        
    def on_shutdown(self):
        """This function will be called when the ROS node shuts down."""
        #print("Shutting down the controller, plotting the losses...")
        torch.save(self.actor_NN.state_dict(), "/home/vascomelo/catkin_ws/src/Laboratorio/Updating strategies/Training/actor_vasco")
        torch.save(self.critic_NN.state_dict(), "/home/vascomelo/catkin_ws/src/Laboratorio/Updating strategies/Training/critic_vasco")              
        torch.save(self.NN_ensemble["NN" + str(0)], "/home/vascomelo/catkin_ws/src/Laboratorio/Updating strategies/Training/NN_model0_vasco")
        torch.save(self.NN_ensemble["NN" + str(1)], "/home/vascomelo/catkin_ws/src/Laboratorio/Updating strategies/Training/NN_model1_vasco")
        #self.plot_losses()
        print("Models saved successfully!")

    def measurements_callback(self, pose, velocity, wrench, pbo):
        # the three inputs should represent three messages
        #print("Function is being triggered")
        if (self.iter_buffer_ < self.buffer_size):
            print("Iter buffer is: ", self.iter_buffer_ ," and buffer size is: ", self.buffer_size )
            # Collection of measured state
            
            EXTERNAL_FORCES   = np.array([wrench.wrench.force.x, wrench.wrench.force.y, wrench.wrench.force.z])
            # -wrench.wrench.force.z is used since the Robot measures positive forces when directed backwards, but for this script they are directed upwards
            # now in the impedence node the force has the right convention so the sign remains +
            #print("msg.Wrench: ", EXTERNAL_FORCES)
            
            CARTESIAN_POSE = np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z, pose.pose.orientation.x, pose.pose.orientation.y,
                                       pose.pose.orientation.z, pose.pose.orientation.w])
            CARTESIAN_VEL = np.array([velocity.twist.linear.x, velocity.twist.linear.y, velocity.twist.linear.z, velocity.twist.angular.x, velocity.twist.angular.y,
                                      velocity.twist.angular.z])            
            
            PBO_index = np.array([pbo.data])

            Timestamp = rospy.Time.now().to_sec()

            if self.iter_init == 0:
                
                self.initial_tra_x = pose.pose.position.x
                self.initial_tra_y = pose.pose.position.y
                self.initial_rot_x = pose.pose.orientation.x
                self.initial_rot_y = pose.pose.orientation.y
                self.initial_rot_z = pose.pose.orientation.z
                self.initial_rot_w = pose.pose.orientation.w
                
                self.u_old = pose.pose.position.z
                self.D_old = 1.5

                self.iter_init += 1

            # Normalization of measured state
            z_norm = np.array([ (CARTESIAN_VEL[2]-self.x_mean_v[0])/self.x_std_v[0] ])
            x_norm = np.array([ (CARTESIAN_POSE[2]-self.x_mean_v[1])/self.x_std_v[1] ])
            fh_norm = np.array([ (EXTERNAL_FORCES[2]-self.x_mean_v[2])/self.x_std_v[2] ])

            "Set point generation throught Neural Network"

            actor_input_norm = np.array([z_norm, x_norm, fh_norm, PBO_index], dtype=np.float32).T
            self.actor_NN.to(self.Device)
            self.actor_NN.eval()
            # normalization of the input of the actor network
            actor_data_torch_ = torch.tensor(actor_input_norm, dtype=torch.float32, device=self.Device)
            u_D_3_norm = self.actor_NN.forward(actor_data_torch_)

            # denormalization
            self.u_3 = (u_D_3_norm[0,0]*self.x_std_v[3] + self.x_mean_v[3]).detach().numpy()
            self.Dr_3 = (u_D_3_norm[0,1]*self.x_std_v[4] + self.x_mean_v[4]).detach().numpy()
            
            if(wrench.wrench.force.z==0):
                self.u_3 = self.u_old
            else:
                if (self.u_3 - self.u_old) > 0.08:
                    self.u_3 = self.u_old + 0.08
                elif (self.u_3 - self.u_old) < -0.08:
                    self.u_3 = self.u_old - 0.08
            
            if (self.Dr_3 - self.D_old) > 0.3 + 0.2*PBO_index:
                self.Dr_3 =  self.D_old + 0.3 + 0.2*PBO_index 
            elif (self.Dr_3 - self.D_old) < -0.3 - 0.2*PBO_index:
                self.Dr_3 =  self.D_old - 0.3 - 0.2*PBO_index 
            
            if (self.u_3 ) > 0.65:
                    self.u_3 = 0.65
            elif (self.u_3 ) < 0.35:
                    self.u_3 = 0.35

            print(f"vel: {CARTESIAN_VEL[2]}, pos: {CARTESIAN_POSE[2]} , force: {EXTERNAL_FORCES[2]}")
            print("u_3: ", self.u_3)
            print("D_informatiom:", self.Dr_3)

            state_action_norm = np.append(actor_input_norm[:,0:3], u_D_3_norm[0,0].detach().numpy())
            state_action_damping_norm = np.append(state_action_norm, u_D_3_norm[0,1].detach().numpy())
            
            state_action_damping_norm_torch_lie = torch.tensor(state_action_damping_norm, dtype=torch.float32, device=self.Device).unsqueeze(0)
            #print("state_action_damping_norm_torch_lie", state_action_damping_norm_torch_lie)
            h1, c1, h2, c2 = self.ComputationUP(self.training_dict['xy_norm'], state_action_damping_norm_torch_lie, model_approximator, num_ensembles = 2)
    
            if c1 > 0:
                self.Upper_limit_u_norm[self.iter_buffer_,0] = (h1 - x_mean_v[3])/x_std_v[3]
                self.Lower_limit_u_norm[self.iter_buffer_,0] = -1
            elif c1 < 0:
                self.Upper_limit_u_norm[self.iter_buffer_,0] = 1
                self.Lower_limit_u_norm[self.iter_buffer_,0] = (h1 - x_mean_v[3])/x_std_v[3]
    
            if c2 > 0:
                self.Upper_limit_D_norm[self.iter_buffer_,0] = (h2 - x_mean_v[4])/x_std_v[4]
                self.Lower_limit_D_norm[self.iter_buffer_,0] = -1
            elif c2 < 0:
                self.Upper_limit_D_norm[self.iter_buffer_,0] = 1
                self.Lower_limit_D_norm[self.iter_buffer_,0] = (h2 - x_mean_v[4])/x_std_v[4]
            
            #print("Action generated", self.u_3)
            #print("Damping generated", self.Dr_3)
            #print(u_D_3_norm[0,0])
            self.u_record_norm[self.iter_buffer_,0] = u_D_3_norm[0,0]
            self.D_record_norm[self.iter_buffer_,0] = u_D_3_norm[0,1]
            self.z_record_norm[self.iter_buffer_,0] = z_norm
            self.x_record_norm[self.iter_buffer_,0] = x_norm
            self.fh_record_norm[self.iter_buffer_,0] = fh_norm
            self.fh_record_not_norm[self.iter_buffer_,0] = wrench.wrench.force.z
            self.PBO_record[self.iter_buffer_,0] = PBO_index
            self.time_record[self.iter_buffer_,0]= Timestamp
                
            u_message = geometry_msgs.msg.PoseStamped()

            u_message.pose.position.x = self.initial_tra_x
            u_message.pose.position.y = self.initial_tra_y                   
            u_message.pose.position.z = self.u_3
            u_message.pose.orientation.x = self.initial_rot_x                           
            u_message.pose.orientation.y = self.initial_rot_y            
            u_message.pose.orientation.z = self.initial_rot_z  
            u_message.pose.orientation.w = self.initial_rot_w 
            
            self.u_pub.publish(u_message)
            
            D_message = std_msgs.msg.Float64
            
            D_message = self.Dr_3

            self.D_pub.publish(D_message)
                
            self.iter_buffer_ += 1
            self.u_old = self.u_3
        
        if (self.iter_buffer_ % self.buffer_size == 0):
            #print("Iter buffer is: ", self.iter_buffer_ ," and buffer size is: ", self.buffer_size )
            EXTERNAL_FORCES   = np.array([wrench.wrench.force.x, wrench.wrench.force.y, wrench.wrench.force.z])
 
            CARTESIAN_POSE = np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z, pose.pose.orientation.x, pose.pose.orientation.y,
                                       pose.pose.orientation.z, pose.pose.orientation.w])
            CARTESIAN_VEL = np.array([velocity.twist.linear.x, velocity.twist.linear.y, velocity.twist.linear.z, velocity.twist.angular.x, velocity.twist.angular.y,
                                      velocity.twist.angular.z])
                                      
            PBO_index = pbo.data

            Timestamp = rospy.Time.now().to_sec() 
            
            z_norm = np.array([ (CARTESIAN_VEL[2]-self.x_mean_v[0])/self.x_std_v[0] ])
            x_norm = np.array([ (CARTESIAN_POSE[2]-self.x_mean_v[1])/self.x_std_v[1] ])
            fh_norm = np.array([ (EXTERNAL_FORCES[2]-self.x_mean_v[2])/self.x_std_v[2] ])
            self.z_record_norm[self.iter_buffer_,0] = z_norm
            self.x_record_norm[self.iter_buffer_,0] = x_norm
            self.fh_record_norm[self.iter_buffer_,0] = fh_norm
            self.PBO_record[self.iter_buffer_,0] = PBO_index
            self.time_record[self.iter_buffer_,0]= Timestamp
            #print("Values were recorded")
            z_tra = self.z_record_norm
            x_tra = self.x_record_norm
            fh_record_tra = self.fh_record_norm
            state_norm_tra = np.append(z_tra, x_tra, axis=1)
            state_force_norm_tra = np.append(state_norm_tra, fh_record_tra, axis=1)

            u_record_tra = self.u_record_norm
            u_D_record_tra = np.append(u_record_tra,self.D_record_norm, axis=1)

            #Recording values for data analysis
            self.velocity_record_[self.recording_index,0] =  np.array([CARTESIAN_VEL[2]])
            self.pose_record_[self.recording_index,0] =  np.array([CARTESIAN_POSE[2]])
            self.force_record_[self.recording_index,0] =  np.array([EXTERNAL_FORCES[2]])
            self.PBO_record_[self.recording_index,0] = np.array([PBO_index])
            self.setpoint_record_[self.recording_index,0] = np.array([self.u_3])
            self.damping_record_[self.recording_index,0] = np.array([self.Dr_3])
            self.time_record_[self.recording_index,0]= np.array([Timestamp])

            #Code used to save the variables in a .npy file. It records at 6Hz
            """
            new_data = np.column_stack((self.velocity_record_, self.pose_record_, self.force_record_, self.PBO_record_,
                                self.setpoint_record_, self.damping_record_, self.time_record_))

            # Define the filename
            file_name = self.data_path

            # Check if the file exists
            if os.path.exists(file_name):
                # Load the existing data
                existing_data = np.load(file_name)
                
                # Append the new data to the existing data
                updated_data = np.vstack((existing_data, new_data))
                
                # Save the updated data back to the file
                np.save(file_name, updated_data)
                print("DATA UPDATED")
            else:
                # If the file doesn't exist, create it by saving the new data
                np.save(file_name, new_data)
                print("DATA CREATED")
            """
            fh_threshold = np.mean(fh_record_tra)
            #print(fh_threshold)
            #print(fh_record_3[i-Nh:i+1])
            
            if abs(fh_threshold) < 0.1:
                lr_actor = 5e-5
                #print("lr_actor", lr_actor)
            elif abs(fh_threshold) < 0.5:
                lr_actor = 8e-5
                #print("lr_actor", lr_actor)
            else:
                lr_actor = 1e-4  
                #print("lr_actor", lr_actor)
            rospy.loginfo("Before Model Training")

        
            for n_rete in range(self.ensemble_size):
                self.model_approximator_train(state_force_norm_tra, u_D_record_tra, self.NN_ensemble["NN" + str(n_rete)], self.buffer_size,
                                              learning_rate = 1e-3)
            #rospy.loginfo("After Model Training and before Critic Training")
            self.Critic_train(state_force_norm_tra, u_record_tra, self.time_record, self.PBO_record, self.critic_NN, self.training_dict['xy_norm'], self.Cost_norm, self.buffer_size, 
                         learning_rate = 1e-4)

            #rospy.loginfo("After Critic Training")
            #print("Critic was trained")
            """
            # Your data preparation steps
            AAA1 = np.linspace(-100., 100., self.SIZEN)  # Fh
            AAA2 = np.linspace(-1., 1., self.SIZEN)  # Delta set-point

            AAA = np.column_stack((AAA1, AAA2))  # Shape: (N, 2)
            BBB = np.zeros(AAA.shape[0])  # Cost
            InputAAA = torch.tensor(AAA, dtype=torch.float32)

            # Compute all outputs at once
            BBB = critic.forward(InputAAA).detach().numpy()
            BBB = BBB.flatten()

            try:
                Q_interp, covariance = curve_fit(self.poly2d, AAA.T, BBB, p0=[0, 0, 0, 0, 0, 0])
                #print("Fitting Critic parameters:", Q_interp)
                #print("Covariance:", covariance)
            except Exception as e:
                print(f"Error during curve fitting: {e}")
            """
            self.Actor_train(state_force_norm_tra, self.Upper_limit_u_norm,
                             self.Lower_limit_u_norm, self.Upper_limit_D_norm, self.Lower_limit_D_norm,
                             self.actor_NN, self.NN_ensemble, self.training_dict['xy_norm'], 
                             self.buffer_size, self.PBO_record, lr_actor)
            rospy.loginfo("Actor was trained")
        
            self.z_record_norm = np.zeros((self.buffer_size + 1, 1))
            self.x_record_norm = np.zeros((self.buffer_size + 1, 1))
            self.fh_record_norm = np.zeros((self.buffer_size + 1, 1))
            self.u_record_norm = np.zeros((self.buffer_size, 1))
            self.D_record_norm = np.zeros((self.buffer_size, 1))
            self.Upper_limit_u_norm = np.zeros((self.buffer_size, 1))
            self.Lower_limit_u_norm = np.zeros((self.buffer_size, 1))
            self.Upper_limit_D_norm = np.zeros((self.buffer_size, 1))
            self.Lower_limit_D_norm = np.zeros((self.buffer_size, 1))
            self.PBO_record = np.zeros((self.buffer_size+ 1, 1))
            self.time_record = np.zeros((self.buffer_size+ 1, 1))
            self.iter_buffer_ = 0
            self.recording_index = 0
            #print("Iter buffer was reset")

        self.rate.sleep()
        
    
# Main function.

if __name__=='__main__':

    print("Main function is started")
    # Node initialization
    rospy.init_node('Q_LMPC', anonymous=True)
    
    Device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    print("Using {} Device".format(Device))
    
    #torch.cuda.device(0)
    #print(torch.cuda.get_device_name(0))
    #print(torch.cuda.get_device_properties(0))
    # Loading the Data
    a_file = open("/home/vascomelo/work_space/src/Laboratorio/Updating strategies/Normalization/3dataDict_robot_py2.pkl", "rb")
    training_dict = pickle.load(a_file)
    #training_dict = torch.jit.load('tensors.pt')
    x_mean_v, x_std_v, y_mean_v, y_std_v = training_dict['xy_norm']
    xn_train = training_dict['xn_train']
    yn_train = training_dict['yn_train']

    # Required data for NN models
    n_x = xn_train.shape[1]                         # input_size
    n_y = yn_train.shape[1]                         # output_size
    n_d = 5                                         # depth of the hidden layers
    n_h = 512                                       # size of the hidden layers
    num_ensembles = 2                               # number of NN in the ensemble
    T=3                                             # prediction horizon
    BS=6                                            # buffer size 
    N=64                                            # number of samples
    print_loss = True
    dx = yn_train.shape[1]
    model_approximator = OrderedDict()

    # Initializing the NN models
    for i in range(num_ensembles):
        model_approximator["NN" + str(i)] = NN_model(n_x, n_d, n_h, n_y, print_NN=False)
    
    actor = ActorNN()
    critic = CriticNN()
    
    # Loading the NN models (comment these lines in case you want to perform your own training)
    PATH_model0 = "/home/vascomelo/work_space//src/Laboratorio/Updating strategies/Training/NN/model_0"
    PATH_model1 = "/home/vascomelo/work_space/src/Laboratorio/Updating strategies/Training/NN/model_1"
    #PATH_actor = "/home/vascomelo/work_space/src/Laboratorio/Updating strategies/Training/Actor/actor_vasco"
    #PATH_critic = "/home/vascomelo/work_space/src/Laboratorio/Updating strategies/Training/Critic/critic_vasco"
    model_approximator["NN0"].load_state_dict(torch.load(PATH_model0))
    model_approximator["NN1"].load_state_dict(torch.load(PATH_model1))

    for i in range(num_ensembles):
        model_approximator["NN" + str(i)].to(Device)
        model_approximator["NN" + str(i)].eval()
    
    actor = ActorNN().to(Device)
    critic = CriticNN().to(Device)

    #actor.load_state_dict(torch.load(PATH_actor)) #comment in case you want to perform your own training
    #critic.load_state_dict(torch.load(PATH_critic)) #comment in case you want to perform your own training
    
    print("QLMPC controller is starting")
    myController = Q_LMPC(Device, model_approximator, actor, critic, num_ensembles, T, BS, N)
    rospy.spin()