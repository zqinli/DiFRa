import sys
from multiprocessing import freeze_support

def main():
    from iworkplace import launcher
    launcher.launch()

if __name__ == "__main__":
    freeze_support()
    main()