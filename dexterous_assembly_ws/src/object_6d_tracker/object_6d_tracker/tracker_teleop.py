import sys
import select
import termios
import tty
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


class TrackerTeleopNode(Node):
    def __init__(self) -> None:
        super().__init__("tracker_teleop_node")
        self.reset_cli = self.create_client(Trigger, "/object_6d_tracker_node/reset")

    def send_reset_request(self) -> None:
        """非阻塞发送重启请求"""
        if not self.reset_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().error(
                "Failed to connect to reset service '/object_6d_tracker_node/reset'"
            )
            return

        req = Trigger.Request()
        future = self.reset_cli.call_async(req)
        future.add_done_callback(self.response_callback)

    def response_callback(self, future: rclpy.task.Future) -> None:
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f"Success: {response.message}")
            else:
                self.get_logger().warn(f"Failed: {response.message}")
        except Exception as e:
            self.get_logger().error(f"Unexpected error during reset request: {e}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrackerTeleopNode()

    fd = sys.stdin.fileno()
    try:
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
    except termios.error:
        print(
            "[ERROR] Unable to access terminal settings. This node must be run in a terminal"
        )
        return

    print("\n" + "=" * 60)
    print("  [Object 6D Tracker][Keyboard Listener]  ")
    print("    Press [ESC] -> Reset the tracker")
    print("    Press [Ctrl+C] -> Exit this control node")
    print("=" * 60 + "\n")

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)

            # 监听按键
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.read(1)
                if key == "\x1b":  # ESC
                    node.get_logger().info(
                        "Detection of ESC key press, sending reset request..."
                    )
                    node.send_reset_request()
                elif key == "\x03":  # Ctrl+C
                    break

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
