import debugpy
from datasets import load_dataset

if __name__ == "__main__":
    debugpy.listen(("0.0.0.0", 5678))
    print("Waiting for debugger to connect...")
    debugpy.wait_for_client()

    print("Downloading dataset")
    dataset = load_dataset("naver-clova-ix/cord-v2")
    print("hola")
