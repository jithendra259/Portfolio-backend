"""
Corpus ingestion and parsing pipeline.
Collects and structures research papers, case studies, page metadata, and profile knowledge.
"""

from pathlib import Path
from .chunker import clean_text, chunk_text
from .types import KnowledgeChunk

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
TRANSCRIPTS_DIR = BACKEND_DIR / "documents" / "transcripts"


def load_research_papers() -> list[KnowledgeChunk]:
    """Ingests peer-reviewed publications from extracted transcripts."""
    paper_configs = [
        (
            "eaai_extracted.txt",
            "Elsevier EAAI (2026)",
            "Multi-Agent Governance for Graph-Regularized CVaR with Adaptive Contagion Penalization",
            "research_eaai",
            [
                "G-CVaR", "contagion", "fire sale", "bipartite", "SEC 13-F",
                "eigenvector centrality", "sigmoid trust", "Wilcoxon", "Sharpe", "5-agent blackboard",
            ],
        ),
        (
            "cas_xai_extracted.txt",
            "Elsevier COR (2026)",
            "A Supervisory Portfolio Governance Framework: Composite Instability Detection, Deterministic Regime Switching, and Conversational Explainability",
            "research_cor",
            [
                "CLARABEL", "SOCP", "Frobenius norm", "covariance drift", "7-agent DAG",
                "100% numerical grounding", "Mistral-7B", "Regime Operator", "drawdown protection", "MiFID II",
            ],
        ),
        (
            "conference_extracted.txt",
            "Springer Nature LNCS (2026)",
            "Dynamic Regime-Adaptive Portfolio Governance with Composite Instability and Shrinkage Estimation",
            "research_lncs",
            [
                "Instability Index", "Ledoit-Wolf", "shrinkage alpha=0.42", "pairwise correlation",
                "cross-sectional volatility", "IJCACI 2026", "regime switching",
            ],
        ),
    ]

    chunks: list[KnowledgeChunk] = []
    for fname, source_name, paper_title, cat, base_keywords in paper_configs:
        fpath = TRANSCRIPTS_DIR / fname
        if not fpath.exists():
            continue
        try:
            raw_text = fpath.read_text(encoding="utf-8", errors="ignore")
            cleaned = clean_text(raw_text)
            text_chunks = chunk_text(cleaned, chunk_size=800, overlap=160)
            for i, chunk in enumerate(text_chunks):
                chunks.append(
                    KnowledgeChunk(
                        id=f"{cat}_{i}",
                        source=source_name,
                        title=paper_title,
                        section=f"Passage {i+1}",
                        category=cat,
                        text=chunk,
                        keywords=base_keywords,
                    )
                )
        except Exception as e:
            print(f"[Corpus Ingestion Warning] Failed reading {fname}: {e}")

    return chunks


def load_page_and_profile_knowledge() -> list[KnowledgeChunk]:
    """Loads structured page context and candidate credentials from prompts.knowledge."""
    chunks: list[KnowledgeChunk] = []

    try:
        from prompts.knowledge import (
            BIOGRAPHY,
            COMPETITIVE_EXAMS,
            EDUCATION,
            PAGE_KNOWLEDGE,
            PROJECTS,
            TECHNICAL_SKILLS,
            WORK_EXPERIENCE,
        )

        # 1. Page Knowledge
        for path, pdata in PAGE_KNOWLEDGE.items():
            title = pdata.get("title", path)
            summary = pdata.get("summary", "")
            math_rigor = pdata.get("mathematical_rigor", "")
            results = " ".join(pdata.get("empirical_results", []))
            agents = " ".join(pdata.get("architecture_agents", []))
            takeaways = " ".join(pdata.get("takeaways", []))

            combined_text = (
                f"{title}. Overview: {summary} Mathematical Rigor: {math_rigor} "
                f"Architecture Agents: {agents} Empirical Results: {results} Key Takeaways: {takeaways}"
            ).strip()

            chunks.append(
                KnowledgeChunk(
                    id=f"page_{path.replace('/', '_')}",
                    source="Portfolio Page Knowledge",
                    title=title,
                    section=f"Page: {path}",
                    category="page_knowledge",
                    text=combined_text,
                    keywords=["screen", "page", path] + list(pdata.get("technical_keywords", [])),
                )
            )

        # 2. Candidate Bio & Contact
        bio_text = (
            f"Candidate: {BIOGRAPHY['name']} ({BIOGRAPHY['preferred_name']}). Roles: {', '.join(BIOGRAPHY['roles'])}. "
            f"Mission: {BIOGRAPHY['mission']} Location: {BIOGRAPHY['location']}. Contact: {BIOGRAPHY['email']}, {BIOGRAPHY['phone']}. "
            f"Portfolio: {BIOGRAPHY['portfolio_url']}, GitHub: {BIOGRAPHY['github']}, LinkedIn: {BIOGRAPHY['linkedin']}."
        )
        chunks.append(
            KnowledgeChunk(
                id="bio_profile",
                source="Candidate Profile",
                title="Biography and Contact",
                section="Profile",
                category="profile",
                text=bio_text,
                keywords=["bio", "contact", "email", "phone", "location", "github", "linkedin"],
            )
        )

        # 3. Education
        for edu in EDUCATION:
            edu_text = (
                f"Education: {edu['degree']} at {edu['institution']} ({edu['period']}), CGPA: {edu['cgpa']}. "
                f"Thesis / Focus: {edu.get('thesis') or edu.get('capstone', '')}"
            )
            chunks.append(
                KnowledgeChunk(
                    id=f"edu_{edu['degree'][:10].replace(' ', '_')}",
                    source="Candidate Profile",
                    title=edu["degree"],
                    section="Education",
                    category="education",
                    text=edu_text,
                    keywords=["education", "degree", "college", "cgpa", "somaiya", "presidency"],
                )
            )

        # 4. Technical Skills
        skills_text = "Technical Skills: " + "; ".join(
            f"{group}: {', '.join(skills)}" for group, skills in TECHNICAL_SKILLS.items()
        )
        chunks.append(
            KnowledgeChunk(
                id="tech_skills",
                source="Candidate Profile",
                title="Technical Skills",
                section="Skills",
                category="skills",
                text=skills_text,
                keywords=["skills", "python", "cvxpy", "pytorch", "nextjs", "livekit", "langgraph", "clarabel"],
            )
        )

        # 5. Work Experience
        for exp in WORK_EXPERIENCE:
            exp_text = (
                f"Experience: {exp['role']} at {exp['organization']} ({exp['period']}). "
                f"Details: {exp.get('details', '')}"
            )
            chunks.append(
                KnowledgeChunk(
                    id=f"exp_{exp['role'][:10].replace(' ', '_')}",
                    source="Candidate Profile",
                    title=f"{exp['role']} - {exp['organization']}",
                    section="Experience",
                    category="experience",
                    text=exp_text,
                    keywords=["experience", "internship", "somaiya", "research"],
                )
            )

        # 6. Projects
        for proj in PROJECTS:
            proj_text = f"Project: {proj['name']}. Description: {proj['description']}"
            chunks.append(
                KnowledgeChunk(
                    id=f"proj_{proj['id']}",
                    source="Portfolio Projects",
                    title=proj["name"],
                    section="Projects",
                    category="projects",
                    text=proj_text,
                    keywords=["project", proj["name"], proj["id"]],
                )
            )

        # 7. Competitive Exams / GATE
        if COMPETITIVE_EXAMS:
            exams_text = "Competitive Exam Qualifications: " + "; ".join(COMPETITIVE_EXAMS)
            chunks.append(
                KnowledgeChunk(
                    id="competitive_exams",
                    source="Candidate Profile",
                    title="Competitive Exams & GATE Qualifications",
                    section="Qualifications",
                    category="qualifications",
                    text=exams_text,
                    keywords=["gate", "exam", "qualification", "cs", "da", "score"],
                )
            )

    except Exception as e:
        print(f"[Corpus Ingestion Warning] Failed loading knowledge prompts: {e}")

    return chunks


from .pdf_loader import load_all_pdfs


def build_full_corpus() -> list[KnowledgeChunk]:
    """Combines original PDF publications, project reports, case studies, and candidate profile knowledge."""
    corpus: list[KnowledgeChunk] = []
    # 1. Primary Original PDF Documents (Papers, AQI Report, Swarm Robotics Proposal, Resume)
    pdf_chunks = load_all_pdfs()
    corpus.extend(pdf_chunks)

    # 2. Structured Web Page Knowledge & Profile Facts
    page_chunks = load_page_and_profile_knowledge()
    corpus.extend(page_chunks)

    print(f"--> [Corpus Builder] Built corpus with {len(corpus)} chunks ({len(pdf_chunks)} from PDFs, {len(page_chunks)} from Web/Profile).")
    return corpus
