from openai import OpenAI
from donut import JSONParseEvaluator
import numpy as np
from utils import *
import argparse
from tqdm import tqdm


# SAMPLES_LIMIT = 100


def get_output_seq(base64_image, client, model, lang):

    if lang == "english":
        years = "'year_9', 'year_10', 'year_11', or 'year_12'"
        levels = "(9, 10, 11, or 12)"
        example_year = "year_9"
        rest_of_the_years = "year_10, year_11, or year_12"

    elif lang == "spanish":
        years = "'3_de_la_eso', '4_de_la_eso', '1_de_bachillerato', or '2_de_bachillerato'"
        levels = "(3º, 4º, 1º, or 2º)"
        example_year = "3_de_la_eso"
        rest_of_the_years = "4_de_la_eso, 1_de_bachillerato, or 2_de_bachillerato"

    else:
        years = [""]
        levels = ""
        example_year = ""
        rest_of_the_years = ""

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
                            f"- The level {levels} they correspond to.\n\n"
                            "You must return a SINGLE JSON object in the exact following format:\n\n"
                            "{\n"
                            f"  {example_year}: [  # Or {rest_of_the_years} as appropriate\n"
                            '    {"subject": "...", "grade": "..."},\n'
                            '    {"subject": "...", "grade": "..."}\n'
                            "  ]\n"
                            "}\n\n"
                            "DO NOT include any additional text, explanations, or comments. "
                            f"Use the key {years} based on what can be inferred from the image."
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

    grades = detect_json(str(response.choices[0].message.content))
    grades = clean_json(grades)

    return grades


def process_dataset(client, dataset_iterator, model, lang):

    evaluator = JSONParseEvaluator()
    accs = []
    output_list = []

    for i, sample in tqdm(enumerate(dataset_iterator)):

        # print(f"Processing img {i}")
        image, gt = get_sample_data(sample)

        base64_image = encode_image(image)

        seq = get_output_seq(base64_image, client, model, lang)
        score = evaluator.cal_acc(seq, gt)

        accs.append(score)
        output_list.append(seq)

        # if i + 1 >= SAMPLES_LIMIT:
        #     break

    return np.mean(accs)


def get_model(client, list_ft_models: bool = False):

    if list_ft_models:
        list_fine_tunes(client)

    try:
        model = input("Paste the model to use: ")

    except:
        model = "gpt-4o"

    return model


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--subset", type=str, required=True)
    parser.add_argument("--finetuned", type=lambda x: x.lower() in ["true", "1"], default=False, required=True)
    parser.add_argument("--language", type=str, required=True)
    args = parser.parse_args()

    dataset_name = args.dataset
    subset_name = args.subset
    finetuned = args.finetuned
    language = args.language

    init_apis()

    client = OpenAI()

    if subset_name == "all":
        subsets = get_dataset_config_names(dataset_name)
    else:
        subsets = [subset_name]

    model = get_model(client, finetuned)

    for subset_name in subsets:
        print(f"Processing {subset_name}")
        dataset_iterator = get_dataset_iterator(dataset_name, subset_name)
        mean_acc = process_dataset(client, dataset_iterator, model, language)
        print(f"Mean accuracy {subset_name}: {mean_acc}")


if __name__ == "__main__":

    main()
