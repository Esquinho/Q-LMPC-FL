#!/usr/bin/env python

import rospy
from geometry_msgs.msg import WrenchStamped

class StairPublisherNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('stair_publisher_node', anonymous=True)
        
        # Define the publisher
        self.pub = rospy.Publisher('/simulated_wrench', WrenchStamped, queue_size=10)
        
        # Get parameters or use defaults
        self.time_intervals = rospy.get_param('~time_intervals', [4, 4, 4, 4])  # Intervals in seconds
        self.values = rospy.get_param('~values', [10 , -20, 30,0 ])  # Values to publish
        
        # Ensure both lists have the same length
        if len(self.time_intervals) != len(self.values):
            rospy.logerr("time_intervals and values must have the same length!")
            rospy.signal_shutdown("Invalid parameters.")
            return
        
        # Internal state variables
        self.current_index = 0  # Index of the current value
        self.start_time = rospy.Time.now()  # Record the start time
        
        # Publishing rate
        self.rate = rospy.Rate(20)  # 10 Hz
        
    def run(self):
        while not rospy.is_shutdown():
            # Compute elapsed time
            elapsed_time = (rospy.Time.now() - self.start_time).to_sec()
            
            # Check if it's time to switch to the next value
            if elapsed_time >= self.time_intervals[self.current_index]:
                # Update the index and reset the start time
                self.current_index = (self.current_index + 1) % len(self.values)
                self.start_time = rospy.Time.now()
                rospy.loginfo(f"Switched to value: {self.values[self.current_index]}")
            
            # Publish the current value
            msg = WrenchStamped()
            msg.wrench.force.z = self.values[self.current_index]
            self.pub.publish(msg)
            
            # Sleep to maintain the rate
            self.rate.sleep()

if __name__ == '__main__':
    try:
        node = StairPublisherNode()
        rospy.loginfo("Stair publisher node started.")
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Stair publisher node terminated.")
