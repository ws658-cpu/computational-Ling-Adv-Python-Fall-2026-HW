import numpy as np
import pandas as pd

from nb.nb import NaiveBayes


def make_model():
    # Replace these lists with the Appendix B training data.
    documents =  [
        "just plain boring",
        "entirely predictable and lacks energy",
        "no surprises and very few laughs",
        "very powerful",
        "the most fun film of the summer",
    ]
    labels = [0,0,0,1,1]

    df = pd.DataFrame({
        "text": documents,
        "author": labels,
    })

    # Match the textbook's smoothing and vocabulary convention.
    model = NaiveBayes(alpha=1.0)
    model.train(df)
    return model

#Test 1: checking the probabilities of each class before considersing the words.
def test_nb_class():
    model = make_model()

    expected_priors = np.array([3/5,2/5])

    np.testing.assert_allclose(
        model.priors,
        expected_priors,
    )

#Test 2: checking the probabilies of each vocabulary word given in a class
def test_likelihoods():
    model = make_model()

 # Build this matrix in model.vocabulary column order.
    expected_likelihoods = np.array([
        [
            3/34, 2/34, 2/34, 2/34, 2/34, #negative-class probabilities
            1/34, 1/34, 2/34, 2/34, 2/34, 
            1/34, 2/34, 1/34, 2/34, 1/34,
            2/34, 1/34, 2/34, 1/34, 2/34,
        ],
        [
            1/29, 1/29, 1/29, 1/29, 1/29, #positive-class probabilities
            2/29, 2/29, 1/29, 1/29, 1/29,
            2/29, 1/29, 2/29, 1/29, 2/29,
            1/29, 2/29, 1/29, 3/29, 2/29,
        ],
    ])
# the column order in likelihood matrix is :
#and, boring, energy, entirely, few, film, fun, just,
#lacks, laughs, most, no, of, plain, powerful,
#predictable, summer, surprises, the, very
    
    np.testing.assert_allclose(
        model.likelihoods,
        expected_likelihoods,
    )


def test_likelihoods_sum_to_one():
    model = make_model()

    np.testing.assert_allclose(
        model.likelihoods.sum(axis=1),
        # replace np.ones(len(model.classes)) to [1.0, 1.0]
        [1.0, 1.0],
    )
