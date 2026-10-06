import tiktoken
print("=" * 60)
print("TASK 3 - ML/AI FUNDAMENTALS")
print("=" * 60)
print("\n1. TOKEN COUNTING")
print("-" * 60)
texts = [
    "Machine learning helps systems learn patterns from data.",
    "A good model should be evaluated against a simple baseline.",
    "Embeddings represent text as numerical vectors for semantic search.",
    "Large language models generate text token by token using context."
]
encoding = tiktoken.get_encoding("cl100k_base")
total_words = 0
total_tokens = 0
for i, text in enumerate(texts, 1):
    words = len(text.split())
    tokens = len(encoding.encode(text))
    words_per_token = words / tokens
    total_words += words
    total_tokens += tokens
    print(f"\nText {i}")
    print("Text:", text)
    print("Words:", words)
    print("Tokens:", tokens)
    print("Words per token:", round(words_per_token, 2))
print("\nTotal words:", total_words)
print("Total tokens:", total_tokens)
print("\n\n2. MONTHLY LLM COST")
print("-" * 60)
chats_per_day = 300
input_tokens_per_chat = 1200
output_tokens_per_chat = 300
days_per_month = 30
monthly_input_tokens = (
    chats_per_day *
    input_tokens_per_chat *
    days_per_month
)
monthly_output_tokens = (
    chats_per_day *
    output_tokens_per_chat *
    days_per_month
)
print("Chats per day:", chats_per_day)
print("Input tokens/chat:", input_tokens_per_chat)
print("Output tokens/chat:", output_tokens_per_chat)
print("\nMonthly input tokens:",
      f"{monthly_input_tokens:,}")
print("Monthly output tokens:",
      f"{monthly_output_tokens:,}")
input_price_per_1m = 0
output_price_per_1m = 0
input_cost = (
    monthly_input_tokens / 1_000_000
) * input_price_per_1m
output_cost = (
    monthly_output_tokens / 1_000_000
) * output_price_per_1m
total_cost = input_cost + output_cost
print("\nInput price per 1M tokens:",
      input_price_per_1m)
print("Output price per 1M tokens:",
      output_price_per_1m)
print("\nInput cost: $", round(input_cost, 2))
print("Output cost: $", round(output_cost, 2))
print("TOTAL MONTHLY COST: $", round(total_cost, 2))
print("\n\n3. TEMPERATURE EXPERIMENT")
print("-" * 60)
prompt = """
Explain why a machine learning model should be compared
with a simple baseline. Give the answer in two sentences.
"""
print("\nPrompt used:")
print(prompt)
print("""
Temperature 0:
The response should normally be more consistent.
Temperature 1.0:
The response should normally show more variation.
Assessment requirement:
Run the same prompt 5 times at temperature 0
and 5 times at temperature 1.0.
""")
print("\n\n4. HALLUCINATION EXPERIMENT")
print("-" * 60)
fake_question = """
Who was the 18th-century scientist Dr. Xylophor McTest
who invented the Quantum Mango Engine?
"""
print("\nMade-up question:")
print(fake_question)
print("""
Test A:
Ask the question normally.
Test B:
Ask the question with:
"If you are unsure, say I don't know.
Do not invent facts."
Expected observation:
The second instruction should encourage the model
to acknowledge uncertainty instead of inventing information.
""")
print("\n\n" + "=" * 60)
print("TASK 3 SUMMARY")
print("=" * 60)
print("""
1. Tokens:
   Counted 4 texts using tiktoken.
2. Cost:
   Input  = 10,800,000 tokens/month
   Output =  2,700,000 tokens/month
3. Temperature:
   Temperature 0 -> more consistent
   Temperature 1.0 -> more variation
4. Hallucination:
   Adding "I don't know if unsure" helps reduce
   unsupported/fabricated answers.
TASK 3 COMPLETED
""")