"""Keep the ROS context alive until the final motor stop is delivered."""

import signal


def run_node(factory, args=None):
    import rclpy
    from rclpy.duration import Duration
    from rclpy.signals import SignalHandlerOptions

    stopping = False

    def request_stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    node = None
    try:
        node = factory()
        while rclpy.ok() and not stopping:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        if node is not None:
            if rclpy.ok():
                stop = getattr(node, "stop", None)
                if stop is not None:
                    stop()
                publisher = getattr(node, "_publisher", None)
                if publisher is not None:
                    publisher.wait_for_all_acked(Duration(seconds=0.3))
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def run_motion_node(factory, args=None):
    run_node(factory, args)
