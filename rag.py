from vector_store.faiss_store import FaissStore
import google.generativeai as genai
import os  
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-flash')

store = FaissStore()

# --- Full system prompt for the LLM ---
SYSTEM_PROMPT = """SYSTEM:
You are an assistant that answers only using the provided documentation excerpts. 
Always cite the exact attribute name(s) used in your answer in square brackets like [attribute: workspace_id]. 
If the documentation does not contain an answer, respond: "I don't know — please check the source."

DOCUMENTS:
{documents}

USER QUESTION:
{query}

INSTRUCTIONS:
- Use only the DOCUMENTS to answer.
- If you generate example code (curl/JSON), include only fields present in the docs and mark optional fields.
- Provide a confidence label (High/Medium/Low) based on how directly the docs cover the question.
- After the answer, list the sources used with their attribute names and source_url.
"""

def build_prompt(query, retrieved):
    docs_text = ""
    for i, r in enumerate(retrieved, start=1):
        m = r["meta"]
        docs_text += f"{i}) attribute: {m['attribute']}\n   type: {m['type']}\n   desc: {m['description']}\n\n"
    prompt = SYSTEM_PROMPT.format(documents=docs_text, query=query)
    return prompt


def answer_query(query):
    retrieved = store.search(query, k=6)
    prompt = build_prompt(query, retrieved)

    # --- Real LLM call ---
    response = model.generate_content(
    prompt,
    generation_config=genai.types.GenerationConfig(
        temperature=0.1,
        max_output_tokens=1000,
        )  
    )
    text = response.text

    # compute simple confidence
    avg_score = sum([r['score'] for r in retrieved]) / max(1, len(retrieved))
    if avg_score < 0.3:
        confidence = "High"
    elif avg_score < 1.0:
        confidence = "Medium"
    else:
        confidence = "Low"

    # build sources summary
    sources = []
    for r in retrieved:
        m = r["meta"]
        sources.append({
            "attribute": m["attribute"],
            "url": m["source_url"],
            "position": m["position"]
        })

    return {"text": text, "sources": sources, "confidence": confidence}
