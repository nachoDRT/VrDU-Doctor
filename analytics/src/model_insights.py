import os
import argparse
import debugpy
from pathlib import Path
from transformers import LayoutLMv2ForTokenClassification


def access_weights(model):
    for name, param in model.named_parameters():
        msg = f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n"
        breakpoint()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", type=str)
    parser.add_argument("--model_name", type=str)
    args = parser.parse_args()

    # Connect remote debugger
    debugpy.listen(("0.0.0.0", 5678))
    print("Waiting for debugger to connect...")
    debugpy.wait_for_client()

    model_path = os.path.join("/app/models", args.model_folder, args.model_name)
    model = LayoutLMv2ForTokenClassification.from_pretrained(model_path)
    print("hola")
    access_weights(model)
