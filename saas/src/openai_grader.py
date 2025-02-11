from openai import OpenAI
from donut import JSONParseEvaluator
import numpy as np
from utils import *
import argparse
from tqdm import tqdm


SAMPLES_LIMIT = 100


def get_output_seq(base64_image, client, list_ft_models: bool = False):

    if list_ft_models:
        list_fine_tunes(client)

    try:
        model = get_model()
    except:
        model = "gpt-4o"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an assistant that extracts grades from students' transcripts of records.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Look at the image and extract:\n"
                            "- The subjects and their grades.\n"
                            "- The level (9, 10, 11, or 12) they correspond to.\n\n"
                            "You must return a SINGLE JSON object in the exact following format:\n\n"
                            "{\n"
                            '  "year_9": [  # Or year_10, year_11, or year_12 as appropriate\n'
                            '    {"subject": "...", "grade": "..."},\n'
                            '    {"subject": "...", "grade": "..."}\n'
                            "  ]\n"
                            "}\n\n"
                            "DO NOT include any additional text, explanations, or comments. "
                            "Use the key 'year_9', 'year_10', 'year_11', or 'year_12' based on what can be inferred from the image."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            },
        ],
    )

    grades = detect_json(str(response.choices[0]))
    grades = clean_json(grades)

    return grades


def process_dataset(dataset_iterator):

    client = OpenAI()

    evaluator = JSONParseEvaluator()
    accs = []
    output_list = []

    for i, sample in tqdm(enumerate(dataset_iterator)):

        # print(f"Processing img {i}")
        image, gt = get_sample_data(sample)

        base64_image = encode_image(image)

        seq = get_output_seq(base64_image, client)
        score = evaluator.cal_acc(seq, gt)

        accs.append(score)
        output_list.append(seq)
        # print(gt)
        # print(seq, score)
        # print("")
        # print(score)

        # if i + 1 >= SAMPLES_LIMIT:
        #     break

    return np.mean(accs)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()

    dataset_name = args.dataset

    init_apis()
    subsets = get_dataset_config_names(dataset_name)

    for subset_name in subsets:
        print(f"Processing {subset_name}")
        dataset_iterator = get_dataset_iterator(dataset_name, subset_name)
        mean_acc = process_dataset(dataset_iterator)
        print(f"Mean accuracy {subset_name}: {mean_acc}")


if __name__ == "__main__":

    main()
