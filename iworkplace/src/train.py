import tyro
from iworkplace.train.tuner import run_train

def main():
    tyro.cli(run_train)

if __name__ == "__main__":
    main()