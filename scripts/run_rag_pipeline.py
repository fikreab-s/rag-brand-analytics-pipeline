"""RAG pipeline: chunk, embed, retrieve, and validate citations."""
import json, random, hashlib, argparse, numpy as np
from pathlib import Path
random.seed(42); np.random.seed(42)

DOCUMENTS = [
    {"title": "Q3 2024 Brand Performance Report", "content": "Cardivex achieved 12.3% market share in Q3, up 1.2pp YoY. Digital channel ROI reached 4.2x, highest across all channels. Rep visit frequency remained stable at 2.3 visits/HCP/quarter."},
    {"title": "Promotional ROI Analysis - FY2024", "content": "Total promotional spend of $45M across 6 channels. TV delivered highest absolute impact ($18M incremental revenue) but lowest ROI (1.8x). Email showed highest ROI (5.1x) but limited scalability."},
    {"title": "HCP Engagement Summary", "content": "Tier 1 HCPs (15% of base) generated 42% of TRx volume. Digital engagement increased 35% YoY among HCPs under 40. Conference attendance remained the strongest driver of new prescriber activation."},
    {"title": "Competitive Landscape Q3", "content": "Market leader holds 28% share, stable QoQ. Two new entrants in the immunology space. Pricing pressure in cardiology segment intensifying. Generic competition expected in 18 months."},
    {"title": "MMM Model Results - Cardivex", "content": "Bayesian MMM (PyMC-Marketing) estimated TV adstock half-life of 4.2 weeks. Saturation parameter (Hill): EC50=$3.2M, slope=1.8. Optimal budget allocation shifts 15% from TV to Digital."},
]

def chunk_documents(docs, chunk_size=100):
    chunks = []
    for doc in docs:
        words = doc["content"].split()
        for i in range(0, len(words), chunk_size):
            chunk_text = " ".join(words[i:i+chunk_size])
            chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()[:8]
            chunks.append({"chunk_id": chunk_id, "source": doc["title"],
                          "text": chunk_text, "word_count": len(chunk_text.split())})
    return chunks

def simulate_embedding(text):
    np.random.seed(hash(text) % 2**31)
    return np.random.randn(384).tolist()

def retrieve(query, chunks, top_k=3):
    q_emb = np.array(simulate_embedding(query))
    scores = []
    for c in chunks:
        c_emb = np.array(simulate_embedding(c["text"]))
        sim = float(np.dot(q_emb, c_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(c_emb)))
        scores.append((sim, c))
    scores.sort(reverse=True)
    return [(s, c) for s, c in scores[:top_k]]

def main():
    p = argparse.ArgumentParser(); p.add_argument("--output_dir", default="data"); a = p.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    chunks = chunk_documents(DOCUMENTS)
    with open(out / "chunks.jsonl", "w") as f:
        for c in chunks: f.write(json.dumps(c) + "\n")

    queries = ["What is Cardivex market share?", "Which channel has best ROI?",
               "How are Tier 1 HCPs performing?", "What does the MMM say about TV?",
               "What is the competitive outlook?"]
    results = []
    for q in queries:
        retrieved = retrieve(q, chunks)
        results.append({"query": q, "retrieved": [{"score": round(s,4), "source": c["source"],
                        "text": c["text"][:80]+"..."} for s, c in retrieved]})

    with open(out / "retrieval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ RAG Pipeline")
    print(f"   Documents: {len(DOCUMENTS)}")
    print(f"   Chunks: {len(chunks)}")
    print(f"   Queries tested: {len(queries)}")
    for r in results:
        print(f"   Q: {r['query'][:50]}...")
        print(f"     → {r['retrieved'][0]['source']} (score: {r['retrieved'][0]['score']})")

if __name__ == "__main__": main()
