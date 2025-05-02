import rclpy
from rclpy.node import Node
from freyja_msgs.msg import CurrentState
import time

class TopicDebugger(Node):
    """
    A simple node to debug topic issues by checking if topics are available
    and messages are being published.
    """

    def __init__(self):
        super().__init__('topic_debugger')
        
        # Check how many robot IDs to monitor (adjust as needed)
        self.num_robots = 1  # Change as needed
        
        # Subscribe to all possible robot topics
        self.subscriptions = []
        for i in range(self.num_robots):
            topic_name = f"/robomaster_{i}/current_state"
            self.get_logger().info(f"Subscribing to: {topic_name}")
            sub = self.create_subscription(
                CurrentState,
                topic_name,
                lambda msg, robot_id=i: self.callback(msg, robot_id),
                10
            )
            self.subscriptions.append(sub)
        
        # Create timer to check topic list periodically
        self.timer = self.create_timer(5.0, self.check_topics)
        self.last_received = {}
        
    def callback(self, msg, robot_id):
        """Called when a message is received"""
        now = self.get_clock().now()
        self.last_received[robot_id] = now
        self.get_logger().info(f"✓ Received message on topic /robomaster_{robot_id}/current_state")
        self.get_logger().info(f"  Message content: state_vector={msg.state_vector}")
    
    def check_topics(self):
        """Periodically check for active topics"""
        self.get_logger().info("Checking for active topics...")
        
        # Get all topics being published
        topic_names_and_types = self.get_topic_names_and_types()
        
        # Print out all topics
        self.get_logger().info("Available topics:")
        for topic_name, topic_types in topic_names_and_types:
            self.get_logger().info(f"  {topic_name} ({', '.join(topic_types)})")
        
        # Check for our specific topics
        for i in range(self.num_robots):
            topic_name = f"/robomaster_{i}/current_state"
            found = False
            for name, types in topic_names_and_types:
                if name == topic_name:
                    found = True
                    last_time = self.last_received.get(i, None)
                    if last_time:
                        now = self.get_clock().now()
                        delta = now - last_time
                        delta_seconds = delta.nanoseconds / 1e9
                        self.get_logger().info(f"  ✓ Topic {topic_name} exists and last message received {delta_seconds:.2f} seconds ago")
                    else:
                        self.get_logger().warn(f"  ⚠ Topic {topic_name} exists but no messages received yet")
                    break
            if not found:
                self.get_logger().error(f"  ✗ Topic {topic_name} does not exist!")

def main():
    rclpy.init()
    node = TopicDebugger()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()