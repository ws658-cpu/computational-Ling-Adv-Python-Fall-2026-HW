import argparse
import numpy as np
class NaiveBayes:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.vocabulary = None
        self.priors = None
        self.likelihoods = None
    def train(self, df):
        vocabulary = {
            word: index
            for index, word in enumerate(
                sorted({
                    word
                    for text in df["text"]
                    for word in text.split()
                })
            )
        }

        n_docs = df.shape[0]
        n_classes = df["author"].nunique()

        priors = (
            df["author"]
            .value_counts(normalize=True)
            .reindex(range(n_classes))
            .to_numpy()
        )

        training_matrix = np.zeros(
            (n_docs, len(vocabulary)),
            dtype=int,
        )

        for row, text in enumerate(df["text"]):
            for word in text.split():
                training_matrix[row, vocabulary[word]] += 1

        word_counts_per_class = np.zeros(
            (n_classes, len(vocabulary)),
            dtype=int,
        )

        for author_id in range(n_classes):
            word_counts_per_class[author_id] = (
                training_matrix[
                    df["author"].to_numpy() == author_id
                ].sum(axis=0)
            )

        likelihoods = np.zeros(
            (n_classes, len(vocabulary)),
            dtype=float,
        )

        for author_id in range(n_classes):
            likelihoods[author_id] = (
                (word_counts_per_class[author_id] + self.alpha)
                / (
                    word_counts_per_class[author_id].sum()
                    + self.alpha * len(vocabulary)
                )
            )

        self.vocabulary = vocabulary
        self.priors = priors
        self.likelihoods = likelihoods

    def test(self, df):
        if self.vocabulary is None:
            raise RuntimeError("Train the model before testing.")

        vocabulary = self.vocabulary
        priors = self.priors
        likelihoods = self.likelihoods

        class_predictions = []

        for text in df["text"]:
            test_vector = np.zeros(len(vocabulary))

            for word in text.split():
                if word in vocabulary:
                    test_vector[vocabulary[word]] += 1

            scores = (
                np.log(priors)
                + np.log(likelihoods) @ test_vector
            )

            prediction = int(np.argmax(scores))
            class_predictions.append(prediction)

        return class_predictions
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


def build_dataframe(folder):
    """
    Takes as input a directory containing presidential speeches and returns two
    DataFrames storing the text from those files, one for the training data
    and one for the test data (unlabeled)
    :param folder: a path to a directory containing presidential speeches
    :return: a tuple of pandas DataFrames
    """
    path = Path(folder)
    df_train = pd.DataFrame(columns=['author'])
    df_test = pd.DataFrame(columns=['author'])
    author_to_id_map = {'kennedy': 0, 'johnson': 1}

    def make_df_from_dir(dir_name, df):
        """
        Takes as input directory to construct df from and returns updated df
        :param dir_name: a Path to a directory
        :param df: an empty pandas DataFrame
        :return: updated pandas DataFrames
        """
        for f in path.glob(f'./{dir_name}/*.txt'):
            with open(f) as fp:
                text = fp.read()
                if dir_name in ('kennedy', 'johnson'):
                    temp_df = pd.DataFrame({'author': dir_name, 'text': [text]})
                    df = pd.concat([df, temp_df], ignore_index=True)
                else:
                    temp_df = pd.DataFrame({'author': str(f).split('_')[-1][
                                                      :-4],
                                                      'text': [text]})
                    df = pd.concat([df, temp_df], ignore_index=True)
        return df
    for p in path.iterdir():
        if p.name in ('kennedy', 'johnson'):
            df_train = make_df_from_dir(p.name, df_train)
        elif p.name == 'unlabeled':
            df_test = make_df_from_dir(p.name, df_test)
    # replace the strings for the author names with numeric codes (0, 1)
    df_train['author'] = df_train['author'].apply(lambda x:
                                                  author_to_id_map.get(x))
    # do the same for the test data
    df_test['author'] = df_test['author'].apply(lambda x:
                                                  author_to_id_map.get(x))
    return df_train, df_test




def sklearn_nb(training_df, test_df):
    vectorizer = CountVectorizer()
    vectorizer.fit(training_df['text'])

    training_data = vectorizer.transform(training_df['text'])
    training_data.toarray()

    test_data = vectorizer.transform(test_df['text'])
    test_data.toarray()

    nb_classifier = MultinomialNB()
    nb_classifier.fit(training_data, training_df['author'])

    pred_nb = nb_classifier.predict(test_data)
    return pred_nb

def get_metrics(true, preds):
    """
    Takes gold labels and predictions to compute performance metrics
    :param true: array-like object
    :param preds: array-like object
    :return: a tuple of various performance metrics
    """
    accuracy = metrics.accuracy_score(true['author'], preds)
    f1_score = metrics.f1_score(true['author'], preds)
    conf_matrix = metrics.confusion_matrix(true['author'], preds)

    return accuracy, f1_score, conf_matrix

def plot_confusion_matrix(conf_matrix_data, labels):
    """
    Takes as input confusion matrix data from get_metrics() and prints out a
    confusion matrix
    :param conf_matrix_data:
    :return: None
    """
    plt.title("Confusion matrix")
    axis = sns.heatmap(conf_matrix_data, annot=True, cmap="crest")
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
    model = NaiveBayes(alpha=0.1)
    model.train(training_df)
    class_predictions = model.test(test_df)
    # I removed old textcode of "vocabulary, priors, likelihoods = train_nb(training_df)" and "predictions = test(test_df, vocabulary, priors, likelihoods)" because I implemented the NaiveBayes class and its methods train and test. The old code was redundant.
    acc, f1, conf = get_metrics(test_df, class_predictions)
    plot_confusion_matrix(conf, [0, 1])
    sklearn_preds = sklearn_nb(training_df, test_df)
    sklearn_metrics = get_metrics(test_df, sklearn_preds)

