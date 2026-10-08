"""
Bước 3 — RAGAS Evaluation
===========================
NHIỆM VỤ:
  1. Chạy 50 QA pairs qua CẢ 2 prompt version, lưu answers + contexts
  2. Tạo EvaluationDataset với các SingleTurnSample object
  3. Đánh giá với 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. In bảng so sánh V1 vs V2
  5. Lưu kết quả vào data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 cho ít nhất 1 prompt version
             + file data/ragas_report.json được tạo ra

⏰ LƯU Ý: Bước này mất ~15-30 phút. Hãy bắt đầu sớm!
"""
import sys
import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.run_config import RunConfig

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import QA_PAIRS


# ── 1. Prompt Templates (copy từ Bước 2) ──────────────────────────────────
# TODO: Copy SYSTEM_V1 và SYSTEM_V2 mà bạn đã viết ở file 02_prompt_hub_ab_routing.py
# ⚠️ Cả 2 phải chứa {context}, ví dụ kết thúc bằng "...\n\nContext:\n{context}"
#    Thiếu {context} → LLM không thấy tài liệu, không báo lỗi, faithfulness/context_* rất thấp.
SYSTEM_V1 = (
    "Bạn là trợ lý AI thân thiện. Trả lời ngắn gọn trong 2-4 câu và chỉ dựa "
    "trên context được cung cấp. Nếu context không chứa câu trả lời, hãy nói "
    "rõ rằng bạn không biết.\n\nContext:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = (
    "Bạn là chuyên gia phân tích thông tin. Đọc kỹ context, xác định các dữ kiện "
    "liên quan, rồi trả lời rõ ràng, có tổ chức trong 3-5 câu. Không suy đoán "
    "hoặc bổ sung thông tin ngoài context.\n\nContext:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}
METRICS = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_recall": context_recall,
    "context_precision": context_precision,
}


# ── 2. Setup Vectorstore ───────────────────────────────────────────────────
def setup_vectorstore():
    """Tái sử dụng — tạo FAISS vectorstore từ knowledge base."""
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 3. Chạy RAG và thu thập kết quả ───────────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Chạy RAG chain cho 1 câu hỏi.

    ⚠️ QUAN TRỌNG: trả về contexts là LIST of strings, KHÔNG phải string đã ghép!
    RAGAS cần từng đoạn riêng để tính context_recall và context_precision.

    Trả về: {"answer": str, "contexts": list[str]}
    """
    # TODO: Retrieve documents từ retriever
    docs = retriever.invoke(question)

    # TODO: Tạo contexts là danh sách page_content (KHÔNG ghép chuỗi ở đây)
    # Gợi ý: contexts = [doc.page_content for doc in docs]
    contexts = [doc.page_content for doc in docs]

    # TODO: Ghép contexts thành 1 string để truyền vào {context} của prompt
    ctx_str = "\n\n".join(contexts)

    # TODO: Chạy chain (prompt | llm | StrOutputParser()).invoke(...)
    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": ctx_str,
        "question": question,
    })

    # TODO: Trả về dict với answer và contexts (list)
    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """
    Chạy tất cả 50 QA pairs qua prompt version được chỉ định.
    Trả về: list of dict với keys: question, reference, answer, contexts
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm       = get_llm()
    prompt    = PROMPTS[prompt_version]

    results = []
    print(f"\n🚀 Đang chạy 50 câu hỏi với prompt {prompt_version} ...")

    for i, qa in enumerate(QA_PAIRS, 1):
        # TODO: Gọi run_rag() cho câu hỏi hiện tại
        out = run_rag(retriever, llm, prompt, qa["question"])

        # TODO: Append vào results dict với 4 keys
        results.append({
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],
            "contexts":  out["contexts"],
        })
        print(f"  [{i:02d}/50] {qa['question'][:60]}")

    return results


# ── 4. Tạo RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """
    Chuyển đổi kết quả RAG thành RAGAS EvaluationDataset.

    Mỗi SingleTurnSample cần 4 trường:
      user_input         → câu hỏi
      response           → câu trả lời đã tạo
      retrieved_contexts → list[str] các đoạn đã retrieve
      reference          → đáp án chuẩn (ground truth)
    """
    # TODO: Tạo list các SingleTurnSample từ rag_results
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]

    # TODO: Wrap thành EvaluationDataset và trả về
    return EvaluationDataset(samples=samples)


# ── 5. Chạy RAGAS Evaluation ──────────────────────────────────────────────
def run_ragas_eval(
    rag_results: list,
    version: str,
    initial_scores: dict = None,
    on_progress=None,
) -> dict:
    """
    Đánh giá kết quả RAG với 4 RAGAS metrics.
    Trả về: dict {metric_name: mean_score}

    Lưu ý: evaluate() thực hiện rất nhiều lần gọi LLM → mất 5-10 phút / version.
    """
    print(f"\n📐 Đang đánh giá RAGAS cho prompt {version} ...")

    dataset = build_ragas_dataset(rag_results)
    llm_eval = get_llm(temperature=0)
    emb_eval = get_embeddings()
    run_config = RunConfig(timeout=180, max_retries=10, max_wait=60, max_workers=4)
    scores = dict(initial_scores or {})

    # Chấm và lưu từng metric riêng để có thể tiếp tục sau khi bị gián đoạn.
    for key, metric in METRICS.items():
        if key in scores and np.isfinite(scores[key]):
            print(f"  ↪ {key:30s}: {scores[key]:.4f} (đã có, bỏ qua)")
            continue

        print(f"\n  Đang chấm {key} ({len(rag_results)} samples) ...")
        result = evaluate(
            dataset,
            metrics=[metric],
            llm=llm_eval,
            embeddings=emb_eval,
            run_config=run_config,
        )
        raw = np.asarray(result[key], dtype=float)
        valid_scores = raw[np.isfinite(raw)]
        if not valid_scores.size:
            raise RuntimeError(f"RAGAS không trả về điểm hợp lệ cho {version}/{key}")

        scores[key] = float(np.mean(valid_scores))
        print(f"  ✓ {key:30s}: {scores[key]:.4f} ({valid_scores.size}/{len(raw)} hợp lệ)")
        if on_progress:
            on_progress(scores)

    # In kết quả
    print(f"\n📊 Kết quả RAGAS — Prompt {version.upper()}:")
    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")

    return scores


# ── 6. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 3: RAGAS Evaluation")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    data_dir = Path(__file__).parent.parent / "data"
    outputs_path = data_dir / "ragas_outputs.json"
    progress_path = data_dir / "ragas_progress.json"
    report_path = data_dir / "ragas_report.json"

    # Tạo vectorstore
    vectorstore = setup_vectorstore()

    # Cache 100 RAG outputs vì bước sinh câu trả lời không cần chạy lại khi
    # evaluation bị gián đoạn.
    cached_outputs = None
    if outputs_path.exists():
        try:
            payload = json.loads(outputs_path.read_text(encoding="utf-8"))
            if (
                payload.get("system_v1") == SYSTEM_V1
                and payload.get("system_v2") == SYSTEM_V2
                and len(payload.get("v1", [])) == len(QA_PAIRS)
                and len(payload.get("v2", [])) == len(QA_PAIRS)
            ):
                cached_outputs = payload
                print("♻️  Đã tải 100 RAG outputs từ cache.")
        except (json.JSONDecodeError, OSError):
            pass

    if cached_outputs:
        v1_results = cached_outputs["v1"]
        v2_results = cached_outputs["v2"]
    else:
        v1_results = collect_rag_outputs(vectorstore, "v1")
        v2_results = collect_rag_outputs(vectorstore, "v2")
        outputs_path.write_text(json.dumps({
            "system_v1": SYSTEM_V1,
            "system_v2": SYSTEM_V2,
            "v1": v1_results,
            "v2": v2_results,
        }, ensure_ascii=False), encoding="utf-8")
        print(f"💾 Đã cache RAG outputs tại {outputs_path}")

    # Khôi phục các metric hợp lệ từ lần chạy trước, kể cả report cũ.
    scores_by_version = {"v1": {}, "v2": {}}
    for saved_path in (report_path, progress_path):
        if not saved_path.exists():
            continue
        try:
            saved = json.loads(saved_path.read_text(encoding="utf-8"))
            for version in ("v1", "v2"):
                key = f"prompt_{version}_scores"
                for metric, score in saved.get(key, {}).items():
                    if isinstance(score, (int, float)) and np.isfinite(score):
                        scores_by_version[version][metric] = float(score)
        except (json.JSONDecodeError, OSError):
            pass

    def save_progress(version: str, scores: dict):
        scores_by_version[version] = dict(scores)
        progress_path.write_text(json.dumps({
            "prompt_v1_scores": scores_by_version["v1"],
            "prompt_v2_scores": scores_by_version["v2"],
        }, indent=2, allow_nan=False), encoding="utf-8")

    # Chạy RAGAS evaluation
    v1_scores = run_ragas_eval(
        v1_results,
        "v1",
        scores_by_version["v1"],
        lambda scores: save_progress("v1", scores),
    )
    save_progress("v1", v1_scores)
    v2_scores = run_ragas_eval(
        v2_results,
        "v2",
        scores_by_version["v2"],
        lambda scores: save_progress("v2", scores),
    )
    save_progress("v2", v2_scores)

    # In bảng so sánh
    print("\n" + "=" * 65)
    print(f"  {'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 65)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2  = v1_scores[metric], v2_scores[metric]
        if np.isclose(s1, s2):
            winner = "Tie"
        else:
            winner = "← V1" if s1 > s2 else "← V2"
        print(f"  {metric:30s}  {s1:>8.4f}  {s2:>8.4f}  {winner}")

    # Kiểm tra mục tiêu
    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\n✅ Đạt mục tiêu: faithfulness = {best_faith:.4f} ≥ 0.8")
    else:
        print(f"\n⚠️  Chưa đạt mục tiêu ({best_faith:.4f} < 0.8).")
        print("   Gợi ý: giảm chunk_size, tăng k, hoặc điều chỉnh prompt.")

    # TODO: Lưu báo cáo vào data/ragas_report.json
    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
    }
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    progress_path.unlink(missing_ok=True)
    print(f"💾 Đã lưu báo cáo vào {report_path}")


if __name__ == "__main__":
    main()
