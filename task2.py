import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
print("=" * 70)
print("TASK 2 - EMBEDDINGS AND SEMANTIC SEARCH")
print("=" * 70)
#CREATE FAQ DATASET

faq_data = [
{
"category": "HR",
"question": "What are the rules for working from home?"
},
{
"category": "HR",
"question": "How many days can employees work remotely?"
},
{
"category": "HR",
"question": "What is the company's leave policy?"
},
{
"category": "HR",
"question": "How do I apply for annual leave?"
},
{
"category": "IT",
"question": "How can I reset my company account password?"
},
{
"category": "IT",
"question": "What should I do if I cannot log in to my account?"
},
{
"category": "IT",
"question": "How do I connect to the company VPN?"
},
{
"category": "IT",
"question": "Who should I contact if my laptop is not working?"
},
{
"category": "Payroll",
"question": "When is the monthly salary credited?"
},
{
"category": "Payroll",
"question": "What day is the salary payment processed?"
},
{
"category": "Payroll",
"question": "How can I download my salary payslip?"
},
{
"category": "Payroll",
"question": "Why has my salary payment been delayed?"
}
]
df = pd.DataFrame(faq_data)
print("\nSTEP 1: FAQ DATA")
print("-" * 70)
print(df.to_string(index=False))
print("\nNumber of FAQs:", len(df))

# LOAD EMBEDDING MODEL

print("\n")
print("=" * 70)
print("STEP 2: LOAD EMBEDDING MODEL")
print("=" * 70)
print("\nLoading sentence-transformer model...")
model = SentenceTransformer(
"all-MiniLM-L6-v2"
)
print("Embedding model loaded successfully.")
# CREATE EMBEDDINGS

print("\n")
print("=" * 70)
print("STEP 3: CREATE EMBEDDINGS")
print("=" * 70)
questions = df["question"].tolist()
embeddings = model.encode(
questions
)
print("\nNumber of embeddings:", len(embeddings))
print("Embedding dimension:", embeddings.shape[1])
print("\nFirst embedding:")
print(embeddings[0])

# CREATE COSINE SIMILARITY FUNCTION

print("\n")
print("=" * 70)
print("STEP 4: COSINE SIMILARITY")
print("=" * 70)

def cosine_similarity(a, b):
    """
        Calculate cosine similarity between two vectors.
    """
    a = np.array(a)
    b = np.array(b)
    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    similarity =dot_product / (magnitude_a * magnitude_b)
    return similarity

print("\nCosine similarity function created.")
test_similarity = cosine_similarity(
embeddings[0],
embeddings[1]
)
print(
"\nExample similarity:",
round(test_similarity, 4)
)
#SEMANTIC SEARCH
print("\n")
print("=" * 70)
print("STEP 5: SEMANTIC SEARCH")
print("=" * 70)
def semantic_search(query, top_k=3):
    query_embedding = model.encode(query)

    similarities = []

    for i, faq_embedding in enumerate(embeddings):
        score = cosine_similarity(query_embedding, faq_embedding)
        similarities.append((i, score))
    similarities.sort(key=lambda x: x[1], reverse=True)

    results = []

    for index, score in similarities[:top_k]:
        results.append({
            "question": df.iloc[index]["question"],
            "category": df.iloc[index]["category"],
            "score": round(float(score), 4)
        })

    return results


print("\n")
print("=" * 70)
print("STEP 6: KEYWORD SEARCH")
print("=" * 70)
def keyword_search(query, top_k=3):
    query_words = set(query.lower().split())
    results = []

    for index, question in enumerate(questions):
        question_words = set(question.lower().split())
        common_words = query_words & question_words
        score = len(common_words)

        if score > 0:
            results.append({
                "question": question,
                "category": df.iloc[index]["category"],
                "score": score
            })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


print("\n")
print("=" * 70)
print("STEP 7: TEST QUERIES")
print("=" * 70)
queries = [
"I cannot log in",
"when is salary credited",
"work from home rules"
]

for query in queries:
    print("\n")
    print("-" * 70)
    print("QUERY:", query)
    print("-" * 70)
    print("\nKeyword Search:")
    keyword_results = keyword_search(
query
)
if keyword_results:
    for result in keyword_results:
        print(
f"- {result['question']} "
f"({result['category']}) "
f"Score: {result['score']}"
)

else:
    print("No keyword results found.")
    print("\nSemantic Search:")

semantic_results = semantic_search(
query,
top_k=3
)
for result in semantic_results:
    print(
f"- {result['question']} "
f"({result['category']}) "
f"Score: {result['score']}"
)
# COMPARISON TABLE

print("\n")
print("=" * 70)
print("STEP 8: COMPARISON TABLE")
print("=" * 70)
comparison = []
for query in queries:
    keyword_results = keyword_search(
query
)
semantic_results = semantic_search(
query,
top_k=1
)
if keyword_results:
    keyword_result = (
keyword_results[0]["question"]
)
else:
    keyword_result = "No result"
semantic_result = (
semantic_results[0]["question"]
)
comparison.append({
"Query": query,
"Keyword Result":
keyword_result,
"Semantic Result":
semantic_result
})
comparison_df = pd.DataFrame(
comparison
)
print(
comparison_df.to_string(
index=False
)
)

#FINAL SUMMARY

print("\n")
print("=" * 70)
print("STEP 9: FINAL SUMMARY")
print("=" * 70)
print("""
Semantic search understands the meaning of a query
rather than only matching exact words.
For example:
"I cannot log in"
can match:
"What should I do if I cannot log in to my account?"
even though the wording is not exactly the same.
Semantic search is useful when users ask questions
using different words with the same meaning.
""")
print("\nTASK 2 COMPLETED SUCCESSFULLY!")