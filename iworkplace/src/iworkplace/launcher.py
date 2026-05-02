import os
import sys
import subprocess

VERSION = "0.1.0"

USAGE = (
    "-" * 70
    + "\n"
    + "| Usage:                                                               |\n"
    + "|   i-cli train -h: launch model training                              |\n"
    + "-" * 70
)



def launch():
    command = sys.argv.pop(1) if len(sys.argv) > 1 else "help"

    if command == "train":
        from .extras.misc import find_available_port, get_device_count
        
        device_count = get_device_count()
        is_distributed = os.environ.get("LOCAL_RANK") is not None

        if device_count > 1 and not is_distributed:
            master_port = os.getenv("MASTER_PORT", str(find_available_port()))
            print(f"Detected {device_count} GPUs. Automatically wrapping with torchrun on port {master_port}...")
            
            args_str = " ".join(sys.argv[1:])
            cmd = (
                f"torchrun --nproc_per_node {device_count} "
                f"--master_port {master_port} "
                f"-m iworkplace.cli train {args_str}"
            )
            
            process = subprocess.run(cmd.split(), env=os.environ, check=True)
            sys.exit(process.returncode)
            
        else:
            from .train.tuner import run_train
            import tyro
            tyro.cli(run_train)

    else:
        print(f"Unknown or missing command: {command}.\n{USAGE}")