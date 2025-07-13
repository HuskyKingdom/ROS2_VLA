#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from .vla_inferencer import VLAController

class VlaControlNode(Node):
    def __init__(self):
        super().__init__('vla_control_node')
        self.get_logger().info("VLA Control Node On")
        self.pub = self.create_publisher(Float32MultiArray, 'vla_action', 10)
        self.timer = self.create_timer(0.05, self.timer_callback)  # 20 Hz
        self.controller = VLAController()

    def timer_callback(self):
        # retrive obervations
        obs = self.get_observation()  
        action = self.controller.predict(obs)
        msg = Float32MultiArray(data=list(action))
        self.pub.publish(msg)
        self.get_logger().debug(f"Published action: {action}")

    def get_observation(self):
        # TODO: subscrib scratch 3 topic
        return None

def main(args=None):
    rclpy.init(args=args)
    node = VlaControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
