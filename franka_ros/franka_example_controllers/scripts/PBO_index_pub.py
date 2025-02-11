#!/usr/bin/env python3

import rospy
from franka_msgs.msg import StampedFloat32
import threading

# Shared variable for PBO value
pbo_value = StampedFloat32()

def update_pbo():
    """Thread function to handle user input."""
    global pbo_value
    while not rospy.is_shutdown():
        try:
            new_value = float(input("Enter a new PBO value (-1 to 1): "))
            if -1 <= new_value <= 1:
                pbo_value.data= new_value
            else:
                rospy.logwarn("Invalid input. Please enter a number between -1 and 1.")
        except ValueError:
            rospy.logwarn("Invalid input. Please enter a valid float number.")

def pbo_publisher():
    """Main function to publish the PBO index."""
    global pbo_value

    # Initialize the ROS node
    rospy.init_node('pbo_index_node', anonymous=True)

    # Create a publisher for the "PBO_index" topic
    pub = rospy.Publisher('PBO_index', StampedFloat32, queue_size=10)

    # Set the publishing rate (e.g., 60 Hz)
    rate = rospy.Rate(100)

    # Start the input thread
    input_thread = threading.Thread(target=update_pbo)
    input_thread.daemon = True
    input_thread.start()

    rospy.loginfo("PBO Publisher is running.")

    # Publishing loop
    while not rospy.is_shutdown():
        # Publish the current PBO value
        pub.publish(pbo_value)
        rate.sleep()

if __name__ == '__main__':
    try:
        pbo_publisher()
    except rospy.ROSInterruptException:
        rospy.loginfo("PBO Publisher node terminated.")