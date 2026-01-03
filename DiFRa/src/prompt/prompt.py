system_prompt = """
Generate {num_qa_pairs} high-quality question-answer pairs and obey the following rules:
(1) Concepts (optional): If concepts are provided, integrate them with the context to encourage wider semantic exploration in the generated question-answer pairs.
(2) Knowledge Graph (optional): If knowledge graph are provided, use them only to check for conflicts.
(3) Output format: Return a JSON array with {num_qa_pairs} objects, each object have two keys: "question" and "answer".
Here is an example of the exact format required (using 2 pairs):
{output_example}
Output ONLY JSON array and do not output any other words.
"""

# system_prompt = """
# Generate {num_qa_pairs} high-quality question-answer pairs and obey the following rules:
# (1) Output format: Return a JSON array with {num_qa_pairs} objects, each object have two keys: "question" and "answer".
# Here is an example of the exact format required (using 2 pairs):
# {output_example}
# Output ONLY JSON array and do not output any other words.
# """

user_prompt = """
{given_context}
{given_concepts}
{given_kgs}
"""
