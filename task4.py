print("=" * 70)
print("TASK 4 - SOLUTION DECISION FRAMEWORK")
print("=" * 70)
problems = [
    {
        "problem": "GST calculation on invoices",
        "input": "Invoice values, items and GST rates",
        "output": "Exact GST amount",
        "constraints": "High accuracy, low latency, explainability",
        "tool": "Rules engine / calculator",
        "human": "Review exceptions and high-value invoices",
        "ai": "NO - do not use generative AI for the actual calculation"
    },
    {
        "problem": "Current leave balance of an employee",
        "input": "Employee HR database records",
        "output": "Current leave balance",
        "constraints": "Real-time data, security, accuracy",
        "tool": "HR API + deterministic calculation",
        "human": "Review disputes",
        "ai": "NO - AI should not calculate the official balance"
    },
    {
        "problem": "Ranking resumes for a role",
        "input": "Resumes + job description",
        "output": "Candidate ranking",
        "constraints": "Bias, privacy, explainability",
        "tool": "ML / embeddings + defined scoring criteria",
        "human": "YES - recruiter must review",
        "ai": "NO - do not allow AI to make final hiring decision"
    },
    {
        "problem": "Answering questions from a 60-page contract",
        "input": "Contract PDF",
        "output": "Answer with page/section citation",
        "constraints": "Accuracy, privacy, latency, explainability",
        "tool": "RAG + embeddings + LLM",
        "human": "YES - review legal/high-impact answers",
        "ai": "NO - AI must not make autonomous legal decisions"
    },
    {
        "problem": "Forecasting next month's sales",
        "input": "Historical sales + business factors",
        "output": "Sales forecast/range",
        "constraints": "Seasonality, data quality, uncertainty",
        "tool": "Time-series / ML forecasting",
        "human": "YES - business review",
        "ai": "NO - do not rely on an LLM alone for numerical forecasting"
    }
]
for i, item in enumerate(problems, 1):
    print("\n" + "-" * 70)
    print("PROBLEM", i)
    print("-" * 70)
    print("Problem     :", item["problem"])
    print("Input       :", item["input"])
    print("Output      :", item["output"])
    print("Constraints :", item["constraints"])
    print("Tool choice :", item["tool"])
    print("Human review:", item["human"])
    print("AI decision :", item["ai"])
print("\n\n" + "=" * 70)
print("TECHNICAL PROPOSAL")
print("=" * 70)
proposal = """
TITLE:
Contract Question Answering Assistant
1. OBJECTIVE
------------------------------
Build an internal AI assistant that allows employees to ask
questions about a 60-page contract.
The system should provide answers based only on the contract
and show the relevant page or section as evidence.
2. RECOMMENDED APPROACH
------------------------------
Use Retrieval-Augmented Generation (RAG).
Architecture:
Contract PDF
     |
     v
Document Parser
     |
     v
Text Chunks + Page Metadata
     |
     v
Embeddings
     |
     v
Vector Database
     |
     v
User Question
     |
     v
Semantic Retrieval
     |
     v
LLM
     |
     v
Answer + Citation
The system should retrieve the most relevant contract sections
before asking the LLM to generate the answer.
The LLM must answer only from the retrieved contract information.
If sufficient information is not available, it should say:
"I don't know based on the provided contract."
3. DATA REQUIREMENTS
------------------------------
Required data:
Original 60-page contract PDF
Extracted contract text
Page numbers
Section/title information
Document metadata
40-50 test questions
Expected answers for evaluation
Questions should cover:
Definitions
Payment terms
Dates
Obligations
Exclusions
Termination
Exceptions
4. EVALUATION METRIC
------------------------------
Primary metric:
Answer correctness / groundedness.
Target:
At least 90% of test questions should receive a
correct answer supported by the contract.
Citation target:
At least 95% of answers should contain the correct
page or section citation.
For questions that are not answered by the contract,
the system should say that it does not know instead
of inventing information.
5. ESTIMATED BUILD TIME
------------------------------
Estimated proof-of-concept time: 3-5 working days.
Day 1:
PDF parsing, text extraction and chunking.
Day 2:
Embeddings and vector database.
Day 3:
RAG pipeline and LLM integration.
Day 4:
Testing and evaluation.
Day 5:
Security, improvements and final demonstration.
6. KEY TRADE-OFFS
------------------------------
Accuracy vs Cost:
Retrieving more document chunks can improve accuracy,
but increases token usage and cost.
Chunk Size:
Small chunks improve retrieval precision.
Large chunks preserve more context.
Model Quality vs Latency:
A larger model may provide better answers but can
increase cost and response time.
RAG vs Direct ChatGPT:
RAG requires additional development, but provides
document grounding, citations, controlled data access
and repeatable evaluation.
7. HUMAN REVIEW
------------------------------
Human review is required for:
Legal interpretation
Contract disputes
Exceptions
Financial/legal decisions
High-impact contractual decisions
The AI should assist the employee and should not act
as an autonomous legal authority.
8. WHY NOT JUST ASK CHATGPT?
------------------------------
A normal ChatGPT conversation is not enough for a
production contract workflow.
The company needs:
Controlled document grounding
Citations
Data privacy
Access control
Repeatable evaluation
Auditability
Human review
RAG allows the system to answer using the approved
contract rather than relying only on general model knowledge.
9. RECOMMENDATION
------------------------------
Build a small RAG proof of concept first.
Move to production only after the system reaches the
evaluation targets, security requirements are approved,
citations are reliable and human review procedures are defined.
"""
print(proposal)
print("\n\n" + "=" * 70)
print("3-MINUTE PRESENTATION")
print("=" * 70)
presentation = """
Our recommendation is to build a RAG-based contract
question-answering assistant.
The company has a 60-page contract and employees need
to ask questions about it.
Instead of sending the whole contract directly to an LLM,
we first extract and divide the contract into smaller
sections.
We then create embeddings and store them in a vector
database.
When an employee asks a question, the system searches
for the most relevant contract sections.
Those sections are then provided to the LLM so it can
generate an answer based on the actual contract.
The answer should also include the page or section citation.
Our target is at least 90 percent correct grounded answers
and 95 percent correct citations.
Human review is required for legal interpretation,
contract disputes and high-impact decisions.
The main trade-offs are cost, latency and accuracy.
We recommend RAG instead of simply asking ChatGPT because
a production system requires document grounding, citations,
privacy, access control, evaluation and auditability.
Therefore, we recommend starting with a 3-to-5-day proof
of concept and moving to production only after evaluation
and security requirements are satisfied.
"""
print(presentation)
print("\n" + "=" * 70)
print("TASK 4 COMPLETED")
print("=" * 70)