#!/usr/bin/env python3
"""
This script provides diagnostic tools for ROS 2 topics.
It can be used to check topic availability, message types, and publish test messages.
"""

import rclpy
from rclpy.node import Node
import time
import sys
import threading
import numpy as np
sys.path.insert(0, "/home/npfitzer/robomaster_ws/install/freyja_msgs/lib/python3.10/site-packages")
from freyja_msgs.msg import CurrentState

class ROSTopicDiagnostics(Node):
    """
    Tools to diagnose ROS 2 topic issues
    """

    def __init__(self):
        super().__init__('ros_topic_diagnostics')
        
        # Default values
        self.robot_id = 0
        
        # Parse command line arguments
        if len(sys.argv) > 1:
            self.robot_id = int(sys.argv[1])
            
        self.get_logger().info(f"Diagnostics for robot ID: {self.robot_id}")
        
    def list_all_topics(self):
        """List all available topics in the ROS system"""
        topics = self.get_topic_names_and_types()
        
        print("\n=== All Available Topics ===")
        for topic_name, topic_types in sorted(topics):
            print(f"Topic: {topic_name}")
            print(f"  Types: {', '.join(topic_types)}")
        print("===========================\n")
    
    def check_specific_topic(self, topic_name, msg_type):
        """Check if a specific topic exists and listen for messages"""
        print(f"\n=== Checking Topic: {topic_name} ===")
        
        # Check if topic exists
        topics = self.get_topic_names_and_types()
        found = False
        for name, types in topics:
            if name == topic_name:
                found = True
                print(f"✓ Topic exists with types: {', '.join(types)}")
                break
        
        if not found:
            print(f"✗ Topic {topic_name} does not exist!")
            return False
        
        # Listen for messages
        self.msg_received = False
        self.msg_count = 0
        
        def callback(msg):
            self.msg_received = True
            self.msg_count += 1
            print(f"✓ Message #{self.msg_count} received!")
            print(f"  Content: {msg}")
            
        subscription = self.create_subscription(
            msg_type,
            topic_name,
            callback,
            10
        )
        
        print(f"Listening for messages on {topic_name}... Press Ctrl+C to stop")
        
        return subscription
    
    def publish_test_message(self, topic_name):
        """Publish a test message to the /robomaster_X/current_state topic"""
        publisher = self.create_publisher(
            CurrentState,
            topic_name,
            10
        )
        
        msg = CurrentState()
        # Create a dummy state vector with 6 elements
        # [north, east, down, vn, ve, vd]
        msg.state_vector = [0.5, 0.5, 0.0, 0.0, 0.0, 0.0]
        
        # Publish message every second
        def publish_loop():
            count = 0
            try:
                while rclpy.ok():
                    # Update position slightly to show movement
                    msg.state_vector[0] = 0.5 + 0.1 * np.sin(count * 0.1)  # north
                    msg.state_vector[1] = 0.5 + 0.1 * np.cos(count * 0.1)  # east
                    msg.header.stamp = self.get_clock().now().to_msg()
                    
                    publisher.publish(msg)
                    print(f"Published test message #{count+1}: {msg.state_vector}")
                    count += 1
                    time.sleep(1.0)
            except KeyboardInterrupt:
                pass
        
        # Start publishing in a separate thread
        thread = threading.Thread(target=publish_loop)
        thread.daemon = True
        thread.start()
        
        return thread, publisher

def main():
    rclpy.init()
    node = ROSTopicDiagnostics()
    
    print("\nROS 2 Topic Diagnostics Tool")
    print("==========================")
    print("1. List all topics")
    print("2. Check robot state topic")
    print("3. Publish test messages to state topic")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ")
    
    try:
        if choice == '1':
            node.list_all_topics()
            input("Press Enter to continue...")
            
        elif choice == '2':
            robot_id = node.robot_id
            topic_name = f"/robomaster_{robot_id}/current_state"
            subscription = node.check_specific_topic(topic_name, CurrentState)
            
            try:
                rclpy.spin(node)
            except KeyboardInterrupt:
                print("\nStopping topic monitoring")
                
        elif choice == '3':
            robot_id = node.robot_id
            topic_name = f"/robomaster_{robot_id}/current_state"
            thread, publisher = node.publish_test_message(topic_name)
            
            print(f"\nPublishing test messages to {topic_name}")
            print("Press Ctrl+C to stop")
            
            try:
                rclpy.spin(node)
            except KeyboardInterrupt:
                print("\nStopping publisher")
                
        elif choice == '4':
            print("Exiting...")
            
        else:
            print("Invalid choice. Exiting...")
    
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()