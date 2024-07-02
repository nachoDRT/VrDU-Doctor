import argparse
import debugpy

if __name__ == "__main__":

    # Define parsing values
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=str)
    args = parser.parse_args()

    # Debug
    if eval(args.debug):
        debugpy.listen(("0.0.0.0", 5678))
        print("Waiting for debugger to connect...")
        debugpy.wait_for_client()
