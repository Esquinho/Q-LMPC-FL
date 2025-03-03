#!/usr/bin/env python3

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import csv
import os

class FuzzyLogic:
    def __init__(self):
        # Initialize fuzzy system parameters
        self.v_max = 0.1
        self.wrench_max = 130
        self.dwrench_max = 40
        self.fuzzy_limit = [self.v_max, self.wrench_max, self.dwrench_max]

        self.velocity_ranges = np.array([
            [0, 0.01],
            [0.005, 0.025],
            [0.02, 0.04],
            [0.04, 0.06]
        ])

        self.fuzzy_parameters = np.array([
            [110, 130, 30, 40],
            [100, 120, 20, 40],
            [90, 110, 15, 35],
            [65, 90, 10, 20]
        ])

    def fuzzy_logic(self, cartesian_velocity_, wrench_, dwrench_, PBO_index_,range_index):
        
        if abs(cartesian_velocity_) >= self.velocity_ranges[range_index][0] and abs(cartesian_velocity_) < self.velocity_ranges[range_index][1]:
            shaping_factor=(1+math.sin(2*math.pi*( (abs(cartesian_velocity_)-self.velocity_ranges[range_index][0]))/(self.velocity_ranges[range_index][1]-self.velocity_ranges[range_index][0] ) + self.velocity_ranges[range_index][0] + (np.average(self.velocity_ranges[range_index])-self.velocity_ranges[range_index][0])/2))/2
        
            AL_=0
            f_1,f_2,df_1,df_2 = self.fuzzy_parameters[range_index,:]
            Al_med_min = 0.025 + 0.01 * PBO_index_
            Al_med_max = 0.045 + 0.01 * PBO_index_
            Al_med_mid = np.mean((Al_med_min,Al_med_max))
            Al_high_min = 0.040 + 0.02 * PBO_index_
            Al_high_max = 0.070 + 0.02* PBO_index_
            Al_high_mid = np.mean((Al_high_min,Al_high_max))

            #print(f"self.Al_med_min shape: {self.Al_med_min.shape}")
            # Step 1: Define the fuzzy sets for input variables (cost and benefit)
            vel = ctrl.Antecedent(np.arange(0, self.fuzzy_limit[0], 0.001), 'Velocity')
            wrench = ctrl.Antecedent(np.arange(0, self.fuzzy_limit[1], 0.1), 'Wrench')
            dwrench = ctrl.Antecedent(np.arange(0, self.fuzzy_limit[2], 0.1), 'dWrench')
            AL = ctrl.Consequent(np.arange(0, Al_high_max, 0.01), 'Assistance Level')

            # Membership functions for input
            vel['stop'] = fuzz.trimf(vel.universe,[0,self.velocity_ranges[0][0],self.velocity_ranges[0][1]])
            vel['slow'] = fuzz.trimf(vel.universe, [self.velocity_ranges[1][0], np.mean([self.velocity_ranges[1][0], self.velocity_ranges[1][1]]), self.velocity_ranges[1][1]])
            vel['move'] = fuzz.trimf(vel.universe, [self.velocity_ranges[2][0], np.mean([self.velocity_ranges[2][0], self.velocity_ranges[2][1]]), self.velocity_ranges[2][1]])
            vel['fast'] = fuzz.trapmf(vel.universe,[self.velocity_ranges[3][0],self.velocity_ranges[3][1],self.fuzzy_limit[0],self.fuzzy_limit[0]])


            wrench['safe'] = fuzz.trapmf(wrench.universe,[0,0,f_1,f_2])
            wrench['non_safe'] = fuzz.trapmf(wrench.universe,[f_1,f_2,self.fuzzy_limit[1],self.fuzzy_limit[1]])

            dwrench['no_var'] = fuzz.trapmf(dwrench.universe,[0,0,df_1,df_2])
            dwrench['var'] = fuzz.trapmf(dwrench.universe,[df_1,df_2,self.fuzzy_limit[2],self.fuzzy_limit[2]])

            # Membership functions for output
            AL['none'] = fuzz.trimf(AL.universe, [0, 0, Al_med_min])
            AL['medium'] = fuzz.trimf(AL.universe, [Al_med_min, Al_med_mid, Al_med_max])
            AL['high'] = fuzz.trimf(AL.universe, [Al_high_min, Al_high_mid, Al_high_max])

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
    def generate_samples(self):
        PBO_range = np.linspace(-1, 1, 10)  # For PBO_index_
        
        # Generate evenly spaced sample inputs
        v_array = np.linspace(-self.v_max, self.v_max, 15)
        f_array = np.linspace(-15, 15, 15)
        f_dot_array = np.linspace(-40, 40, 15)

        save_path = os.path.expanduser('~/work_space_robot/src/Q-LMPC-FL/Laboratorio/Updating strategies/Training/fuzzy_samples_3.csv')

        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write header row
            writer.writerow(['Velocity', 'Wrench', 'dWrench', 'PBO_index', 'Assistance Level'])

            # Iterate over sample combinations
            print("Generating fuzzy logic samples...\n")
            i = 0
            for PBO_index in PBO_range:
                for v in v_array:
                    for f in f_array:
                        for f_dot in f_dot_array:
                            total_AL = 0  # Initialize total assistance level
                            i += 1
                            #print("\nTotal samples generated: ", i)
                            for range_index in range(len(self.velocity_ranges)):
                                # Calculate fuzzy logic output and shaping factor
                                assistance_level = self.fuzzy_logic(abs(v), abs(f), abs(f_dot), PBO_index, range_index)
                                
                                # Weighted contribution to total assistance level
                                total_AL += assistance_level 
                            
                            # Write sample data to file
                            AL = total_AL * np.sign(f_dot)
                            
                            if i%10==0: 
                                print("\nTotal samples generated: ", i)
                                print(f"vel: {v}, for: {f}, dfor: {f_dot}, PBO: {PBO_index} ")
                                print(f"AL: {AL} ")

                            writer.writerow([v, f, f_dot, PBO_index, AL])

        print("\nSamples generation complete.")

                            
if __name__ == '__main__':
    fl = FuzzyLogic()
    fl.generate_samples()