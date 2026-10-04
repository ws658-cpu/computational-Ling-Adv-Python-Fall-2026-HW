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


def test_nb_class():
    model = make_model()

    expected_priors = np.array([...])

    np.testing.assert_allclose(
        model.priors,
        expected_priors,
    )


def test_likelihoods():
    model = make_model()

    # Build this matrix in model.vocabulary column order.
    expected_likelihoods = np.array([
        [...],
        [...],
    ])

    np.testing.assert_allclose(
        model.likelihoods,
        expected_likelihoods,
    )


def test_likelihoods_sum_to_one():
    model = make_model()

    np.testing.assert_allclose(
        model.likelihoods.sum(axis=1),
        np.ones(len(model.classes)),
    )
