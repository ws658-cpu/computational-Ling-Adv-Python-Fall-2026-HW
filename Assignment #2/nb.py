import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Problem #1
def build_dataframe(folder):
    """
    Takes as input a directory containing presidential speeches and returns two
    DataFrames storing the text from those files, one for the training data
    and one for the test data (unlabeled)
    :param folder: a path to a directory containing presidential speeches
    :return: a tuple of pandas DataFrames
    """
    path = Path(folder)
    df_train = pd.DataFrame(columns=["author"])
    df_test = pd.DataFrame(columns=["author"])
    author_to_id_map = {"kennedy": 0, "johnson": 1}

    def make_df_from_dir(dir_name, df):
        """
        Takes as input directory to construct df from and returns updated df
        :param dir_name: a Path to a directory
        :param df: an empty pandas DataFrame
        :return: updated pandas DataFrames
        """
        for f in path.glob(f"./{dir_name}/*.txt"):
            with open(f, encoding="utf-8") as fp:
                text = fp.read()
                # TODO If the directory name is either kennedy or johnson,
                #  create a pandas DataFrame where the column "authors" is
                #  the directory name and there is a field "text" which
                #  contains the text from the opened file. Note that you want
                #  a single DataFrame, but you loop over numerous files.
                if dir_name in ("kennedy", "johnson"):
                    new_row = pd.DataFrame({
                        "author": [dir_name],
                        "text": [text]
                    })
                    df = pd.concat([df, new_row], ignore_index=True)
                else:
                    # TODO Otherwise, we want to create a DataFrame for the
                    #  unlabeled data in a similar fashion. But this is a
                    #  little different because we don't get the label from
                    #  the directory but instead from the file name. Again,
                    #  the field "author" should have the author's name and
                    #  the field "text" should contain the text.
                    author = f.stem.split("_")[-1] # Extract author name from file name
                    new_row = pd.DataFrame({
                        "author": [author],
                        "text": [text]
                    })
                    df = pd.concat([df, new_row], ignore_index=True)
        return df

    for p in path.iterdir():
        if p.name in ("kennedy", "johnson"):
            df_train = make_df_from_dir(p.name, df_train)
        elif p.name == "unlabeled":
            df_test = make_df_from_dir(p.name, df_test)
    # replace the strings for the author names with numeric codes (0, 1)
    df_train["author"] = df_train["author"].apply(lambda x: author_to_id_map.get(x))
    # do the same for the test data
    df_test["author"] = df_test["author"].apply(lambda x: author_to_id_map.get(x))
    return df_train, df_test


def train_nb(df, alpha=0.1):
 # TODO Create a dictionary that maps whitespace-separated tokens in the (complete)
    #  file to a unique index. Also, create variables for the number of
    #  documents and the number of classes. Use df.shape for the vocabulary
    #  and the nunique() method for the number of classes

    """
    Takes as input a pandas DataFrame containing Federalist
    files text to determine priors and likelihoods
    :param df: a pandas DataFrame
    :return: two numpy arrays for the priors and likelihoods
    """
    vocabulary = {
        word: index
        for index, word in enumerate(
             sorted({word for text in df["text"] for word in text.split()})
         )
    }
    n_docs = df.shape[0]
    n_classes = df["author"].nunique()

    # TODO Compute the priors (complete)
    priors = (
        df["author"]
        .value_counts(normalize=True)
        .reindex(range(n_classes))
        .to_numpy()
)
    # TODO Create a matrix containing all 0s called training_matrix of size
    #  (n_docs, len(vocabulary)), then fill it with the counts of each word
    #  for each document. This is the bag-of-words matrix for all the documents (complete)
    training_matrix = np.zeros((n_docs, len(vocabulary)), dtype=int)

    for row, text in enumerate(df["text"]):
        for word in text.split():
            training_matrix[row, vocabulary[word]] += 1

    # TODO Get word counts for both classes (complete)
    word_counts_per_class = np.zeros((n_classes, len(vocabulary)), dtype=int)

    for author_id in range(n_classes):
        word_counts_per_class[author_id] = training_matrix[
            df["author"].to_numpy() == author_id
        ].sum(axis=0)
    # TODO Initialize a matrix to store the likelihoods (complete)
    likelihoods = np.zeros((n_classes, len(vocabulary)), dtype=float)

    # TODO Then fill it in using Lidstone smoothing (complete)
    for author_id in range(n_classes):
        likelihoods[author_id] = (
            (word_counts_per_class[author_id] + alpha)
            / (word_counts_per_class[author_id].sum() + alpha * len(vocabulary))
        )
    return vocabulary, priors, likelihoods

def test(df, vocabulary, priors, likelihoods):
    """
    Takes as input a pandas DataFrame representing the disputed Federalist
    Papers and returns predictions for every text document
    :param df: a pandas DataFrame
    :return: a numpy array of predictions
    """
    class_predictions = []
    for text in df["text"]:
        test_vector = np.zeros(shape=(len(vocabulary)))
        # TODO Fill test_vector with counts for the words that appear in the
        #  vocabulary
        for word in text.split():
          if word in vocabulary:
            test_vector[vocabulary[word]] += 1
        # TODO Compute predictions p(y|text) (complete)
        preds = np.log(priors)+ np.log(likelihoods)@test_vector
        # TODO Then get your predictions, yhat (complete)
        yhat = int(np.argmax(preds))
        class_predictions.append(yhat)
    return class_predictions


def sklearn_nb(training_df, test_df):
    """
    Performs Naive Bayes classification using scikit-learn implementation
    :param training_df: training data
    :param test_df: test data
    :return: predictions
    """
    vectorizer = CountVectorizer()

    # TODO Fit the vectorizer on the training set text (complete)
    vectorizer.fit(training_df["text"])

    # TODO Then transform the text using the vectorizer (complete)
    training_data = vectorizer.transform(training_df["text"])
    training_data.toarray()

    # Do the same for the test data
    test_data = vectorizer.transform(test_df["text"])
    test_data.toarray()

    nb_classifier = MultinomialNB()
    # TODO Fit the Naive Bayes classifier (complete)
    nb_classifier.fit(training_data, training_df["author"])

    pred_nb = nb_classifier.predict(test_data)
    return pred_nb


def get_metrics(true, preds):
    """
    Takes gold labels and predictions to compute performance metrics
    :param true: array-like object
    :param preds: array-like object
    :return: a tuple of various performance metrics
    """
    # TODO Compute performance measures (complete)
    accuracy = metrics.accuracy_score(true, preds)
    f1_score = metrics.f1_score(true, preds, pos_label=1)
    conf_matrix = metrics.confusion_matrix(true, preds, labels=[0,1])

    return accuracy, f1_score, conf_matrix


def plot_confusion_matrix(conf_matrix_data, labels):
    """
    Takes as input confusion matrix data from get_metrics() and prints out a
    confusion matrix
    :param conf_matrix_data:
    :return: None
    """
    plt.title("Confusion matrix")
    axis = sns.heatmap(conf_matrix_data, annot=True, fmt= "d")
    axis.set_xticklabels(labels)
    axis.set_yticklabels(labels)
    axis.set(xlabel="Predicted", ylabel="True")
    plt.show()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Naive Bayes Algorithm")
    parser.add_argument("-f", "--indir", required=True, help="Data directory")
    args = parser.parse_args()


    training_df, test_df = build_dataframe(args.indir)  
    for author_id, author_name in [(0, "Kennedy"), (1, "Johnson")]:
        texts = training_df.loc[training_df["author"] == author_id, "text"]
        lengths = texts.str.split().str.len()
        print(author_name, "mean words:", lengths.mean())
        print(author_name, "median words:", lengths.median())
        print(author_name, "most common words:")
        print(texts.str.lower().str.split().explode().value_counts().head(10))
        print(author_name, "document openings:")
        print(texts.str.split().str[:12].str.join(" ").head(5).to_string(index=False))
        print(author_name, "document endings:")
        print(texts.str.split().str[-12:].str.join(" ").head(5).to_string(index=False))

    vocabulary, priors, likelihoods = train_nb(training_df)
    print("Prior probabilities (0=Kennedy, 1=Johnson):", priors)
    print("Likelihood matrix shape:", likelihoods.shape)

    class_predictions = test(test_df, vocabulary, priors, likelihoods)
    print("Predictions (0=Kennedy, 1=Johnson):", class_predictions)

    author_names = {0: "Kennedy", 1: "Johnson"}
    
    for work_number, prediction in enumerate(class_predictions, start=1):
        print(
            f"Unlabeled work {work_number} -> {author_names[prediction]}"
        )
    
    acc, f1, conf = get_metrics(test_df["author"], class_predictions)
    #plot_confusion_matrix(conf, [0, 1])#
    
    sklearn_preds = sklearn_nb(training_df, test_df)
    sklearn_metrics = get_metrics(test_df["author"], sklearn_preds)
    
    print("My Naive Bayes — accuracy:", acc, "F1:", f1)
    print(
        "Scikit-learn — accuracy:",
        sklearn_metrics[0],
        "F1:",
        sklearn_metrics[1],
    )
    # Problem #2
    print("My Naive Bayes predictions:", class_predictions)
    print("Scikit-learn predictions:", sklearn_preds)
    
    
    # Problem #3
    acc, f1, conf = get_metrics(test_df["author"], class_predictions)
    
    sklearn_preds = sklearn_nb(training_df, test_df)
    sklearn_acc, sklearn_f1, sklearn_conf = get_metrics(
        test_df["author"], sklearn_preds
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    
    sns.heatmap(
        conf,
        annot=True,
        fmt="d",
        cmap="Reds",
        xticklabels=["Kennedy", "Johnson"],
        yticklabels=["Kennedy", "Johnson"],
        ax=axes[0],
    )
    axes[0].set_title("My Naive Bayes")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    
    sns.heatmap(
        sklearn_conf,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Kennedy", "Johnson"],
        yticklabels=["Kennedy", "Johnson"],
        ax=axes[1],
    )
    axes[1].set_title("Scikit-learn Naive Bayes")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    
    plt.tight_layout()
    plt.savefig("conf.jpg", dpi=300, bbox_inches="tight")
    plt.show()
    
    print("My Naive Bayes — accuracy:", acc, "F1:", f1)
    print(
        "Scikit-learn — accuracy:",
        sklearn_acc,
        "F1:",
        sklearn_f1,
    )
