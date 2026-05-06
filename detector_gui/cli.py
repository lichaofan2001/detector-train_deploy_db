"""
CLI entry point for the Detector Training GUI.

Usage:
    detector-gui                  # Start with default settings
    detector-gui --port 8080      # Custom port
    detector-gui --host 127.0.0.1 # Bind to localhost only
"""

import argparse


def gui_command():
    """Command line entry point for the training GUI."""
    parser = argparse.ArgumentParser(description='Detector 训练管理界面')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='服务监听地址 (默认: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000,
                        help='服务端口 (默认: 5000)')
    parser.add_argument('--debug', action='store_true',
                        help='启用调试模式')
    args = parser.parse_args()

    from detector_gui.app import run_gui
    run_gui(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    gui_command()
