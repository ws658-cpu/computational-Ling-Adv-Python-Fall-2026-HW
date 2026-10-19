# Naive Bayes Text Classifier

This project implements a multinomial Naive Bayes classifier to predict
whether a presidential speech was written by John F. Kennedy or
Lyndon B. Johnson.

The classifier uses whitespace tokenization, additive smoothing, and
log probabilities. Its NaiveBayes class provides train() and test()
methods.

## Project files

- nb/nb.py: classifier implementation.
- nb/utils/load_data.py: loads speeches into pandas DataFrames.
- nb/bin/main.py: command-line interface.
- Test/test_naive_bayes.py: unit tests.
- environment.yml: course environment.
- setup.py: package installation settings.
- discussion.md: assignment responses and test output.

## Environment and installation

Create and activate the course environment:

```bash
conda env create -f environment.yml
```

Activate the environment using the name specified in environment.yml:

```bash
conda activate ENVIRONMENT_NAME
```

From inside the Naive_Bayes folder, install the project:

```bash
python -m pip install -e ".[dev]"
```

## Data

Place the speech files in these folders:

- Data/kennedy/
- Data/johnson/
- Data/unlabeled/

Each speech should be a .txt file. The main dataset is excluded from
Git through .gitignore and must be obtained separately from the course
materials.

## Running the classifier

From inside the Naive_Bayes folder, run:

```bash
python -m nb.bin.main -f Data
```

The program trains on the Kennedy and Johnson speeches and prints a
predicted author for each unlabeled document.

## Running tests

From inside the Naive_Bayes folder, run:

```bash
python -m pytest
```

The three tests use the textbook's worked example to check class
priors, word likelihoods, and whether each class's likelihoods sum
to one. The tests use alpha=1.0 for add-one smoothing.
