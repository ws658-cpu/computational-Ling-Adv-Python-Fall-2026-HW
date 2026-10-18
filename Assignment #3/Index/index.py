import xml.etree.ElementTree as ET

import argparse

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
