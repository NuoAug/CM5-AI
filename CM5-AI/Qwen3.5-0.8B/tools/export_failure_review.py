"""从运行器对照导出人工失败审查模板；不训练、不自动判质量通过。"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def export_rows(source):
    config = source.get("configuration", {})
    for variant in source.get("variants", []):
        for case in variant.get("cases", []):
            if not case.get("case_id") or not case.get("input_sha256"):
                raise ValueError("案例缺少身份或输入摘要，拒绝建立不可追溯审查项")
            truncated = case.get("finish_reason") == "length"
            version = variant.get("version") or {"runner_tag": config.get("runner_tag"),
                       "model_revision": variant.get("model_revision"),
                       "application_version": variant.get("application_version"),
                       "note": "缺失版本为null；该实验仅运行器调用，不能填后来版本冒充实测"}
            yield {"case_id": case["case_id"], "input_sha256": case["input_sha256"],
                   "corpus_sha256": source.get("corpus_sha256"),
                   "model": variant.get("model_id"), "version": version,
                   "finish_reason": case.get("finish_reason"), "max_tokens": case.get("max_tokens"),
                   "answer": case.get("answer"),
                   "automatic_observations": ["finish_reason_length"] if truncated else [],
                   "failure_categories": ["output_truncated"] if truncated else [],
                   "human_review": {"state": "pending", "reviewer": None, "reviewed_at": None,
                       "quality_pass": None, "fact_checks": [], "human_failure_categories": [],
                       "proposed_fix": None, "engineering_fix_checked": False,
                       "higher_precision_comparison_needed": None, "training_candidate": None},
                   "data_partition": {"origin": "constructed_development_runner_smoke",
                       "is_independent_test": False, "split": "unassigned", "group_id": None,
                       "license_review": "pending", "leakage_review": "pending",
                       "approved_for_training": False}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source_path, target = Path(args.input).resolve(), Path(args.output).resolve()
    if source_path == target:
        parser.error("不能覆盖原始对照结果")
    if not target.is_relative_to((ROOT / "test-runs").resolve()):
        parser.error("人工审查模板只能写本候选test-runs，避免公开未经审核素材")
    if target.exists():
        parser.error("审查文件已存在，拒绝覆盖人工标注")
    rows = list(export_rows(json.loads(source_path.read_text(encoding="utf-8"))))
    if not rows:
        parser.error("输入没有可追溯案例")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(rows), "automatically_marked_truncation": sum("output_truncated" in row["failure_categories"] for row in rows), "human_quality_review": "pending", "training_started": False}))


if __name__ == "__main__": main()
