import xml.etree.ElementTree as ET

import argparse

import string

from nltk.tokenize import TreebankWordTokenizer

from nltk.stem import PorterStemmer

from collections import Counter

class Tokenizer:
    def tokenize(self, text):
        raise NotImplementedError(
            "Each tokenizer subclass must implement tokenize()."
        )

class WhitespaceTokenizer(Tokenizer):
    def tokenize(self, text):
        return text.split()

class CharacterNGramTokenizer(Tokenizer):
    def __init__(self, n=4):
        if n < 1:
            raise ValueError("n must be at least 1.")

        self.n = n

    def tokenize(self, text):
        return [
            text[i:i + self.n]
            for i in range(len(text) - self.n + 1)
        ]

class OffTheShelfTokenizer(Tokenizer):
    def __init__(self):
        self.tokenizer = TreebankWordTokenizer()

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)
      
def normalize(tokens):
    normalized_tokens = []

    for token in tokens:
        cleaned_token = token.lower().strip(string.punctuation)

        if cleaned_token:
            normalized_tokens.append(cleaned_token)

    return normalized_tokens    
    
parser = argparse.ArgumentParser(description="Build an inverted index.")

parser.add_argument(
    "-f",
    required=True,
    help="Path to the XML collection",
)

parser.add_argument(
    "-t",
    required=True,
    choices=["whitespace", "offtheshelf", "ngram"],
    help="Tokenizer to use",
)

parser.add_argument(
    "-s",
    required=True,
    choices=["yes", "no"],
    help="Whether to apply stemming",
)

args = parser.parse_args()

tree = ET.parse(args.f)
root = tree.getroot()

print("Root tag:", root.tag)

# Inspect the first document's structure.
#for child in list(root)[:1]:
 #   print(ET.tostring(child, encoding="unicode"))

#replace inspected first document structure to exact all documents
documents = []

for doc in root.findall("doc"):
    doc_id = doc.findtext("docid")
    text = doc.findtext("msgtext", default="")

    documents.append((doc_id, text))

print("Number of documents:", len(documents))

# Check the first extracted document.
if documents:
    print("First document ID:", documents[0][0])
    print("First 200 characters:", documents[0][1][:200])

print("File:", args.f)
print("Tokenizer:", args.t)
print("Stemming:", args.s)

# Being replaced with the following code to include normalization of tokens
#tokenizer = WhitespaceTokenizer()
#if args.t == "whitespace":
#    tokenizer = WhitespaceTokenizer()
#elif args.t == "ngram":
#    tokenizer = CharacterNGramTokenizer()
#else:
 #   raise NotImplementedError(
#        "The off-the-shelf tokenizer has not been added yet."
 #   )

#Create with off-the-shelf tokenizer
if args.t == "whitespace":
    tokenizer = WhitespaceTokenizer()
elif args.t == "ngram":
    tokenizer = CharacterNGramTokenizer()
else:
    tokenizer = OffTheShelfTokenizer()


# information prior to update 
#tokenizer = WhitespaceTokenizer()
#if documents:
    #first_doc_id, first_text = documents[0]
    #tokens = tokenizer.tokenize(first_text)
    #print("First 20 tokens:", tokens[:20])

# information prior to update: #tokenizer = WhitespaceTokenizer()
if documents:
    first_doc_id, first_text = documents[0]
    tokens = tokenizer.tokenize(first_text)
    tokens = normalize(tokens)

    print("First 20 normalized tokens:", tokens[:20])

stemmer = PorterStemmer()

if documents:
    first_doc_id, first_text = documents[0]
    tokens = tokenizer.tokenize(first_text)
    tokens = normalize(tokens)

    print("Before stemming:", tokens[:20])

    if args.s == "yes":
        tokens = [stemmer.stem(token) for token in tokens]

    print("After optional stemming:", tokens[:20])    

index = dict()

for docid, text_body in documents:
    list_of_tokens = tokenizer.tokenize(text_body)
    list_of_tokens = normalize(list_of_tokens)

    if args.s == "yes":
        list_of_stemmed_tokens = [
            stemmer.stem(token) for token in list_of_tokens
        ]
    else:
        list_of_stemmed_tokens = list_of_tokens

    term_frequency = Counter(list_of_stemmed_tokens)

    for token in term_frequency:
        if token in index:
            index[token] += [(docid, term_frequency[token])]
        else:
            index[token] = [(docid, term_frequency[token])]

print("Number of unique indexed tokens:", len(index))

#index.get(token, []) keeps documents matching every query token. 
# For character n-grams, means every n-gram must occur somewhere in the document; 
# it does not verify their order or adjacency.
def search(query):
    query_tokens = tokenizer.tokenize(query)
    query_tokens = normalize(query_tokens)

    if args.s == "yes":
        query_tokens = [
            stemmer.stem(token) for token in query_tokens
        ]

    if not query_tokens:
        return set()

    matching_sets = []

    for token in query_tokens:
        postings = index.get(token, [])
        doc_ids = {docid for docid, frequency in postings}
        matching_sets.append(doc_ids)

    return set.intersection(*matching_sets)


system_docs = search("system")
compatibility_docs = search("compatibility")
both_docs = system_docs & compatibility_docs

print("E — system:", sorted(system_docs, key=int))
print("F — compatibility:", sorted(compatibility_docs, key=int))
print("G — both terms:", sorted(both_docs, key=int))
