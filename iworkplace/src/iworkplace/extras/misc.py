import socket


def get_device_count() -> int:
    import torch
    return torch.cuda.device_count() if torch.cuda.is_available() else 1

def find_available_port() -> int:
    r"""Find an available port on the local machine."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port