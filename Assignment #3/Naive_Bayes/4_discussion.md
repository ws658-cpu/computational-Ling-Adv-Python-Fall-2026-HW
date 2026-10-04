Assignment #3: Whitney Syriaque

**Problem 1**
A) **How is it different from yours? What could you have done differently? What do you think I could have done better?**
My implementation follows the same overall approach as the professor’s: both load documents into DataFrames, use whitespace tokenization, estimate class priors and word likelihoods, and predict authors using logarithmic scores. However, I sort my vocabulary to make its column order reproducible, use pandas to calculate class proportions, and store word counts in a class-by-word matrix. This lets me calculate the likelihoods for all words in a class together. My smoothing denominator also matches the number of vocabulary columns, so the likelihoods for each class sum to one. The professor’s denominator includes an additional category without a corresponding matrix column. I could improve my implementation by removing repeated calculations, cleaning up comments and documentation, adding unit tests, and organizing training and prediction into a class. The professor’s implementation could benefit from consistent smoothing, a sorted vocabulary, and explicit file encoding.


B) **Organizing the Naive Bayes Class**
I reorganized my training and prediction code into a NaiveBayes class with train() and test() methods. The initialization method stores the smoothing parameter, alpha. The train() method builds the vocabulary and calculates the priors and likelihoods, storing
them as instance attributes. The test() method uses those stored values to predict classes for unseen documents.

C) **Unit Tests**
I used the five training documents from Appendix B.3 of
Jurafsky and Martin’s Speech and Language Processing.
I encoded negative documents as 0 and positive documents as 1 and used add-one smoothing (alpha = 1.0).

**Test 1 — Prior probabilities**
Three of the five documents are negative and two are positive. The expected priors are therefore P(negative) = 3/5 and P(positive) = 2/5. The test compares the model’s priors with these expected values.

**Test 2 — Word likelihoods**
The training vocabulary contains 20 unique words. The negative class contains 14 word occurrences, and the positive class contains 9. With add-one smoothing, the denominators are 14 + 20 = 34 and 9 + 20 = 29.

For example, “predictable” occurs once in the negative class and zero times in the positive class. Its expected likelihoods are (1 + 1)/34 = 2/34 and (0 + 1)/29 = 1/29.

The test compares all 20 word probabilities for both classes with the expected values, using the same alphabetical column order as the model.

**Test 3 — Probability sums**
The test adds the word likelihoods across each class’s row. The expected sum is 1 for the negative class and 1 for the positive class.

**Pytest output**
collected 3 items
Test\test_naive_bayes.py ... [100%]
3 passed in 67.31s (0:01:07)

All three tests passed: the class priors match the expected values, the word likelihoods match the Appendix B worked example, and each class’s word probabilities sum to one.

D) D) My main.py uses argparse to accept the data directory, loads the training and test DataFrames, trains the NaiveBayes model, and prints its priors and predictions. The predictions correspond to the test documents in their loading order.

Command: py -m nb.bin.main -f
Data Output:
Class labels: 0 = Kennedy, 1 = Johnson
Priors: [0.35294118 0.64705882]
Predictions: [1, 0, 0, 0, 0, 1, 0, 1, 0, 0]

