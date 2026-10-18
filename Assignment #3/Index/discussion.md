**Problem 2**

**A)** I used Python’s xml.etree.ElementTree to parse 20newsgroups-initial.xml. For each <doc> element, I extracted the document ID from <docid> and the text from <msgtext>. The initial collection contained 20 documents, stored as (document ID, text) pairs.

**B)what are the universal properties of a tokenizer that you should include in your interface?**
The universal properties of a tokenizer interface are a consistent method name, input type, and output format. Each tokenizer should provide a tokenize(self, text) method that accepts a string and returns a list of string tokens. The base class establishes this interface, while each subclass implements its own tokenization strategy: whitespace, off-the-shelf, or character n-grams. Lowercasing and punctuation stripping are applied afterward through a shared normalization function.

**C)** I used NLTK’s PorterStemmer to stem normalized tokens when -s yes is selected. For example, organization becomes organ, and lines becomes line. When -s no is selected, stemming is skipped.

**D)** I built an inverted index using a dictionary. For each document, I tokenized and normalized its text, optionally applied stemming, and used Counter to calculate token frequencies. Each dictionary key is a token, and its postings list contains (document ID, term frequency) pairs. For each tokenizer, I was able to generate a certain number of unique indexed tokens. For Whitespace, I received 1,686; off the shelf is 1,687; and character 4-grams is 10,795 unique indexed tokens. 

**E)what document ids are returned for the query “system”?**
With stemming enables, all three tokenizers returned these documents IDs for "system" : 3, 1127, 2552, 3319, 5653, 8772, 8935, 14398, 16515, 16638. Queries and documents undergo the same processing. Lowercasing allows both system and System to match. Although the tokenizers produced different vocabularies, their results were identical for the query. 

**F)what document ids are returned for the query “compatibility”?**
For the work tokenizers, stemming maps "compatibility" to "compat". The character tokenizer searches using the query's processed character n-grams

**G)what document ids are returned for the query terms “system” and “compatibility”?**
All three tokenizers returned theses documents IDs for documents matching both "system and "compatibility", with stemming enabled: 8772, 14398. I intersected the two query-results sets to retain only documents matching both terms. Both documents matching " compatibility" also matched "system".  For the character n-gram index, matching requires every processed query gram to occur somewhere in the document; it does not verify their order or adjacency. Therefore, identical results for these queries do not imply that the three tokenizers will always return identical results.

