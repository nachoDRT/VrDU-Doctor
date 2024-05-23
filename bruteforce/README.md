# Bruteforce Trainings :muscle::hammer:

### Introduction
In this folder, you can find everything you need to bruteforce train a model. Lets say we have the following dataset partition :open_file_folder:

```
model/
└── data/
  ├── train-val/
  │ └── language/
  │   ├── subset_train_A
  │   ├── subset_train_B
  │   └── subset_train_C
  │
  └── test/
    └── language
        ├── subset_test_A
        └── subset_test_B
```
### What you can do :eyes: :eyes:
When you train a model using this Dockerized pipeline, you can decide the number of subsets you want to combine. The number of trainings you will launch depend on the number of subsets you include in the `train-val` folder and the number of `elements` in every combination. You can change the content in `train-val` as you want (follow the format!). You can change the number of `elements` tuning the `--combinations_length` argument in the `start_brutefroce.sh` file:

```bash
python src/manage_bruteforce.py \
    --combinations_length 2
```

If you choose `--combinations_length 2`, you will launch three training sessions:
+ One trained on subsets A and B.
+ One trained on subsets A and C.
+ One trained on subsets B and C.

If you choose `--combinations_length 3`, you will launch one training session:
+ Just one trained on subsets A, B and C


### Why this is interesting :question:

**This is a good approach when you want to explore the effect your samples have in your model:** This is interesting because you might have different subsets with different features . By combining these subsets in a controlled way you can observe how they affect the trained model. You could this way have a bunch of trained models and observe how they perform when you combine certain features in the training process.