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
        self.wrench_max = 150
        self.dwrench_max = 50
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

    def fuzzy_logic(self, cartesian_velocity_, wrench_, dwrench_, PBO_index_, range_index):
        # Extract fuzzy parameters for the given range
        f_1, f_2, df_1, df_2 = self.fuzzy_parameters[range_index, :]
        Al_med_min = 0.025 + 0.005 * PBO_index_
        Al_med_max = 0.045 + 0.005 * PBO_index_
        Al_high_min = 0.040 + 0.005 * PBO_index_
        Al_high_max = 0.070 + 0.005 * PBO_index_

        # Define fuzzy variables
        vel = ctrl.Antecedent(np.arange(0, self.fuzzy_limit[0], 0.001), 'Velocity')
        wrench = ctrl.Antecedent(np.arange(0, self.fuzzy_limit[1], 0.1), 'Wrench')
        dwrench = ctrl.Antecedent(np.arange(0, self.fuzzy_limit[2], 0.1), 'dWrench')
        AL = ctrl.Consequent(np.arange(0, Al_high_max, 0.01), 'Assistance Level')

        # Define membership functions
        vel['stop'] = fuzz.trimf(vel.universe, [0, self.velocity_ranges[0][0], self.velocity_ranges[0][1]])
        vel['slow'] = fuzz.trimf(vel.universe, [self.velocity_ranges[1][0], 0.015, self.velocity_ranges[1][1]])
        vel['move'] = fuzz.trimf(vel.universe, [self.velocity_ranges[2][0], 0.03, self.velocity_ranges[2][1]])
        vel['fast'] = fuzz.trapmf(vel.universe, [self.velocity_ranges[3][0], self.velocity_ranges[3][1], self.v_max, self.v_max])

        wrench['safe'] = fuzz.trapmf(wrench.universe, [0, 0, f_1, f_2])
        wrench['non_safe'] = fuzz.trapmf(wrench.universe, [f_1, f_2, self.wrench_max, self.wrench_max])

        dwrench['no_var'] = fuzz.trapmf(dwrench.universe, [0, 0, df_1, df_2])
        dwrench['var'] = fuzz.trapmf(dwrench.universe, [df_1, df_2, self.dwrench_max, self.dwrench_max])

        AL['none'] = fuzz.trimf(AL.universe, [0, 0, Al_med_min])
        AL['medium'] = fuzz.trimf(AL.universe, [Al_med_min, 0.035, Al_med_max])
        AL['high'] = fuzz.trimf(AL.universe, [Al_high_min, 0.055, Al_high_max])

        # Define fuzzy rules
        rules = [
            ctrl.Rule(vel['stop'] & wrench['safe'] & dwrench['no_var'], AL['medium']),
            ctrl.Rule(vel['stop'] & wrench['safe'] & dwrench['var'], AL['medium']),
            ctrl.Rule(vel['slow'] & wrench['safe'] & dwrench['no_var'], AL['medium']),
            ctrl.Rule(vel['slow'] & wrench['safe'] & dwrench['var'], AL['high']),
            ctrl.Rule(vel['move'] & wrench['safe'] & dwrench['no_var'], AL['medium']),
            ctrl.Rule(vel['move'] & wrench['safe'] & dwrench['var'], AL['high']),
            ctrl.Rule(vel['fast'], AL['none']),
            ctrl.Rule(wrench['non_safe'], AL['none']),
        ]

        # Fuzzy inference system
        AL_ctrl = ctrl.ControlSystem(rules)
        AL_sim = ctrl.ControlSystemSimulation(AL_ctrl)

        # Input values
        AL_sim.input['Velocity'] = abs(cartesian_velocity_)
        AL_sim.input['Wrench'] = abs(wrench_)
        AL_sim.input['dWrench'] = abs(dwrench_)

        # Compute output
        AL_sim.compute()
        return AL_sim.output['Assistance Level']

    def shaping_function(self, cartesian_velocity_, range_index):
        if abs(cartesian_velocity_) >= self.velocity_ranges[range_index][0] and abs(cartesian_velocity_) < self.velocity_ranges[range_index][1]:
            shaping_factor=(1+math.sin(2*math.pi*( (abs(cartesian_velocity_)-self.velocity_ranges[range_index][0]))/(self.velocity_ranges[range_index][1]-self.velocity_ranges[range_index][0] ) + self.velocity_ranges[range_index][0] + (np.average(self.velocity_ranges[range_index])-self.velocity_ranges[range_index][0])/2))/2
        elif abs(cartesian_velocity_)<self.velocity_ranges[0][1]:
            shaping_factor = 1
        else: 
            shaping_factor = 0
        #rospy.loginfo(f"Shapping factor: {shaping_factor}")    
        return shaping_factor 
    
    def generate_samples(self):
        PBO_range = np.linspace(-1, 1, 10)  # For PBO_index_
        
        # Generate evenly spaced sample inputs
        v_array = np.linspace(-self.v_max, self.v_max, 15)
        f_array = np.linspace(-self.wrench_max, self.wrench_max, 15)
        f_dot_array = np.linspace(-self.dwrench_max, self.dwrench_max, 15)

        save_path = os.path.expanduser('~/catkin_ws/src/Laboratorio/Updating strategies/Training/fuzzy_samples.csv')

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
                            print("\nTotal samples generated: ", i)
                            for range_index in range(len(self.velocity_ranges)):
                                # Calculate fuzzy logic output and shaping factor
                                assistance_level = self.fuzzy_logic(v, f, f_dot, PBO_index, range_index)
                                shaping_factor = self.shaping_function(v, range_index)
                                
                                # Weighted contribution to total assistance level
                                total_AL += assistance_level * shaping_factor
                            
                            # Write sample data to file
                            AL = total_AL * np.sign(f_dot)
                            writer.writerow([v, f, f_dot, PBO_index, AL])

        print("\nSamples generation complete.")

                            
if __name__ == '__main__':
    fl = FuzzyLogic()
    fl.generate_samples()