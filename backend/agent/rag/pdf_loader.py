"""
Direct PDF Document Ingestion and Parser for Jithendra's Portfolio.
Uses PyMuPDF (fitz) to extract text, page numbers, and structural sections from all
original publications, reports, project proposals, and resume PDFs in backend/documents/.
"""

import os
from pathlib import Path
from typing import Any

try:
    import pymupdf as fitz  # Modern PyMuPDF API
except ImportError:
    import fitz  # Legacy fallback


from .chunker import clean_text, chunk_text
from .types import KnowledgeChunk

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
DOCUMENTS_DIR = BACKEND_DIR / "documents"

PDF_CATALOG: list[dict[str, Any]] = [
    {
        "filename": "multi-agent-governance-graph-cvar-eaai.pdf",
        "relative_path": "adaptive-portfolio-governance/multi-agent-governance-graph-cvar-eaai.pdf",
        "source": "Elsevier EAAI Paper (PDF)",
        "title": "Multi-Agent Governance for Graph-Regularized CVaR with Adaptive Contagion Penalization",
        "category": "research_eaai",
        "keywords": [
            "G-CVaR", "contagion", "fire sale", "bipartite", "SEC 13-F", "eigenvector centrality",
            "sigmoid trust", "Wilcoxon", "Sharpe", "5-agent blackboard", "Elsevier EAAI",
        ],
    },
    {
        "filename": "supervisory-portfolio-framework-xai.pdf",
        "relative_path": "supervisory-portfolio-xai-governance/supervisory-portfolio-framework-xai.pdf",
        "source": "Elsevier COR Paper (PDF)",
        "title": "A Supervisory Portfolio Governance Framework: Composite Instability Detection, Deterministic Regime Switching, and Conversational Explainability",
        "category": "research_cor",
        "keywords": [
            "CLARABEL", "SOCP", "Frobenius norm", "covariance drift", "7-agent DAG",
            "100% numerical grounding", "Mistral-7B", "Regime Operator", "drawdown protection", "MiFID II", "Elsevier COR",
        ],
    },
    {
        "filename": "regime-adaptive-supervisory-governance.pdf",
        "relative_path": "regime-adaptive-supervisory-governance/regime-adaptive-supervisory-governance.pdf",
        "source": "Springer Nature LNCS Paper (PDF)",
        "title": "Regime-Adaptive Supervisory Governance for Instability-Aware Portfolio Stabilization",
        "category": "research_lncs",
        "keywords": [
            "Instability Index", "Ledoit-Wolf", "shrinkage alpha=0.42", "pairwise correlation",
            "cross-sectional volatility", "IJCACI 2026", "regime switching", "Springer Nature LNCS",
        ],
    },
    {
        "filename": "mtech-miniproject-aqi-forecasting-kandula-subramanyam.pdf",
        "relative_path": "personalised-aqi-system/mtech-miniproject-aqi-forecasting-kandula-subramanyam.pdf",
        "source": "M.Tech AQI Forecasting Report (PDF)",
        "title": "Personalised Air Quality Index Forecasting with Contemporary and Historical Data",
        "category": "project_aqi",
        "keywords": [
            "AQI", "PM2.5", "PM10", "Delhi", "CPCB", "XGBoost", "R2 0.912", "RMSE 18.4",
            "Markov Chains", "ARIMA", "Somaiya", "air quality",
        ],
    },
    {
        "filename": "swarm-robotics-kscst-proposal.pdf",
        "relative_path": "swarm-robots-agriculture/swarm-robotics-kscst-proposal.pdf",
        "source": "KSCST Swarm Robotics Proposal (PDF)",
        "title": "Autonomous Swarm Robots for Precision Agriculture and Plant Pathology",
        "category": "project_swarm",
        "keywords": [
            "Swarm Robotics", "ESP32", "ESP-NOW", "DenseNet121", "plant disease",
            "KSCST 46th Series", "Presidency University", "precision agriculture", "mesh networking",
        ],
    },
    {
        "filename": "kandula_jithendra_subramanyam_resume.pdf",
        "relative_path": "resume/kandula_jithendra_subramanyam_resume.pdf",
        "source": "Official Resume (PDF)",
        "title": "Kandula Jithendra Subramanyam - Professional Resume",
        "category": "candidate_resume",
        "keywords": [
            "Resume", "Jithendra", "M.Tech", "Somaiya", "Presidency", "GATE DA", "GATE CS",
            "CVXPY", "CLARABEL", "PyTorch", "LiveKit", "LangGraph", "Next.js",
        ],
    },
]


def load_all_pdfs() -> list[KnowledgeChunk]:
    """
    Parses all 6 PDF documents in backend/documents/ page-by-page.
    Extracts text with page citations and chunks into overlapping semantic windows.
    """
    chunks: list[KnowledgeChunk] = []

    for item in PDF_CATALOG:
        pdf_path = DOCUMENTS_DIR / item["relative_path"]
        if not pdf_path.exists():
            # Fallback search by filename if directory nesting differs
            matches = list(DOCUMENTS_DIR.glob(f"**/{item['filename']}"))
            if matches:
                pdf_path = matches[0]
            else:
                print(f"[PDF Ingestion Warning] File not found: {item['filename']}")
                continue

        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            doc_chunks = 0

            for page_index in range(total_pages):
                page = doc[page_index]
                page_text = page.get_text()
                if not page_text or len(page_text.strip()) < 50:
                    continue

                cleaned_page = clean_text(page_text)
                # Chunk page content with 700 chars and 140 char overlap
                page_chunks = chunk_text(cleaned_page, chunk_size=750, overlap=150)

                for chunk_idx, text_chunk in enumerate(page_chunks):
                    chunk_id = f"pdf_{item['category']}_p{page_index + 1}_c{chunk_idx}"
                    section_citation = f"Page {page_index + 1} of {total_pages}"

                    chunks.append(
                        KnowledgeChunk(
                            id=chunk_id,
                            source=item["source"],
                            title=item["title"],
                            section=section_citation,
                            category=item["category"],
                            text=f"[{item['source']} - {section_citation}]: {text_chunk}",
                            keywords=item["keywords"] + [f"page {page_index + 1}", item["category"]],
                        )
                    )
                    doc_chunks += 1

            doc.close()
            print(f"--> [PDF Loader] Ingested {doc_chunks} chunks from '{item['filename']}' ({total_pages} pages).")

        except Exception as e:
            print(f"[PDF Loader Error] Failed parsing '{item['filename']}': {e}")

    return chunks
