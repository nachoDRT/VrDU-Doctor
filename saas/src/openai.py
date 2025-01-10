from openai import OpenAI
from donut import JSONParseEvaluator
import numpy as np
from utils import *

SAMPLES_LIMIT = 1


def get_ouput_seq(base64_image, client):

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
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
            }
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

    for i, sample in enumerate(dataset_iterator):

        image, gt = get_sample_data(sample)

        base64_image = encode_image(image)

        seq = get_ouput_seq(base64_image, client)
        score = evaluator.cal_acc(seq, gt)

        accs.append(score)
        output_list.append(seq)
        print(gt)
        print(seq, score)
        print("")

        if i + 1 >= SAMPLES_LIMIT:
            break

    print(f"Mean accuracy: {np.mean(accs)}")


def main():

    init_apis()
    dataset_iterator = get_dataset_iterator()
    process_dataset(dataset_iterator)


if __name__ == "__main__":

    main()
