#!/usr/bin/env python3
import rospy
import math
import random
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Header

def simulated_wrench_function(amplitude, frequency, phase_1, phase_2,random_amplitude,random_freq):
    # Publisher initialization, done once to avoid re-initializing in the loop
    wrench_pub = rospy.Publisher("/simulated_wrench", WrenchStamped, queue_size=1)

    rospy.init_node('wrench_simulator', anonymous=False)
    
    # Set the rate at which to publish the message (1000 Hz)
    publish_rate = 100  # Hz
    rate = rospy.Rate(publish_rate)
    time_step = 1.0 / publish_rate  # Time step for updates
    
    # Simulation time variable
    t = 0.0
    
    # Initialize the time variable for sinusoidal function
    t = 0 
    
    while not rospy.is_shutdown():
        # Create a WrenchStamped message
        wrench_msg = WrenchStamped()

        # Set the header for the message
        
        # Sinusoidal force components
        wrench_msg.wrench.force.x = 1 * math.sin(2*math.pi*t+phase_1)  # Sinusoidal force in x
        wrench_msg.wrench.force.y = 1 * math.sin(2*math.pi*t+phase_2)  # Sinusoidal force in y
        wrench_msg.wrench.force.z = amplitude * math.sin(2*math.pi*frequency * t) 
        #+ random_amplitude * math.sin(2*math.pi*random_freq * t) + (2/3)*random_amplitude * math.sin(random_freq * t)  # Sinusoidal force in z
  
        # No torque in this example, but you can add torque if needed
        wrench_msg.wrench.torque.x = 0
        wrench_msg.wrench.torque.y = 0
        wrench_msg.wrench.torque.z = 0

        # Publish the wrench message
        wrench_pub.publish(wrench_msg)
        
        # Increment the time variable (for the next cycle)
        t += time_step  # Control the frequency of updates (1 ms time step)
        
        # Sleep to maintain the desired loop rate
        rate.sleep()

if __name__ == '__main__':
    try:
        # Set amplitude and frequency
        amplitude = 10
        frequency = 0.4
        phase_1 = random.uniform(0 ,2*math.pi)
        phase_2 = random.uniform(0 ,2*math.pi)
        random_amplitude = random.uniform(0, 0.2)
        random_freq = random.uniform(30.0,80.0)
        rospy.loginfo("Wrench simulator node started successfully.")

        # Call the function to start publishing
        simulated_wrench_function(amplitude, frequency, phase_1,phase_2,random_amplitude,random_freq)
    except rospy.ROSInterruptException:
        pass
