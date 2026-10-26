"""功能概述：验证新研究数据合同的结构边界、哈希前缀与失败关闭规则。

所有样例为CPU合成身份，无真实封存查询、模型或外部服务。测试纯选择函数，
覆盖旧验收不能改名、未知历史排除、跨核结构保护及三理论源完整性。
"""
import copy
import unittest

from .h_research_cohorts import build_cohorts, structural_split, unique_structures, row_key, review_training_rows


def row(number, split="train", domain="experimental", source="WanJuan", nucleus="1H"):
    bucket = {"train": 1, "val": 81, "test": 91}[split] + number * 100
    return dict(group=f"{bucket:08x}" + f"{number:056x}", smiles="s" + str(number),
                record_id=str(number), domain=domain, source=source, split=split,
                nucleus=nucleus, review_spectral_weight=1.0)


def fixture():
    training = [row(i) for i in range(12)]
    old = {"train": unique_structures(training, "global_train_317")[:3],
           "original_val": [row(21, "val")], "tune": [row(22, "val")], "verify": [row(23, "val")]}
    theory = [row(31 + i, domain="theory", source=s) for i, s in enumerate(("QME14S", "FG26", "QM9-NMR"))]
    rows = training + [r for n in ("original_val", "tune", "verify") for r in old[n]] + theory
    test = [row(41, "test")]
    history = {"schema": "research_history_v1", "exposed_groups": [r["group"] for r in rows],
               "evaluation_groups": [r["group"] for n in ("original_val", "tune", "verify") for r in old[n]],
               "uncertain_groups": [], "cleared_independent_groups": []}
    return rows, old, {r["group"] for r in rows + test}, test, history


class CohortTests(unittest.TestCase):
    def test_review_only_receives_training_rows_and_keeps_holdouts(self):
        rows, old, _, _, _ = fixture()
        labels = {"raw": True}
        def apply(received_labels, received_rows, artifact, mode):
            self.assertIs(received_labels, labels)
            self.assertTrue(all(r["domain"] == "experimental" and r["split"] == "train" for r in received_rows))
            self.assertEqual(mode, "domain")
            return [dict(r, review_spectral_weight=1.0) for r in received_rows], {"mode": mode}
        result, audit = review_training_rows(rows, labels, {}, apply)
        self.assertEqual(audit, {"mode": "domain"})
        self.assertEqual(len(result), len(rows))
        for source, derived in zip(rows, result):
            if source["domain"] != "experimental" or source["split"] != "train":
                self.assertIs(source, derived)

    def test_expansion_is_original_hash_prefix(self):
        rows, old, candidates, test, hist = fixture()
        cohorts, contract = build_cohorts(rows, old, candidates, test, hist, 8, 4)
        self.assertEqual(len(cohorts["train"]), 8)
        self.assertEqual([row_key(r) for r in cohorts["train"][:3]], [row_key(r) for r in old["train"]])
        self.assertEqual(cohorts["original_val"], old["original_val"])
        self.assertEqual(contract["independent_verify"]["state"], "unavailable")
        self.assertFalse(contract["independent_verify"]["old_verify_reused"])

    def test_unknown_never_becomes_independent(self):
        rows, old, candidates, test, hist = fixture()
        fresh = row(70, "test")
        rows.append(fresh); candidates.add(fresh["group"])
        hist["uncertain_groups"].append(fresh["group"])
        cohorts, _ = build_cohorts(rows, old, candidates, test, hist, 8, 4)
        self.assertEqual(cohorts["verify"], [])

    def test_explicitly_cleared_fresh_holdout_can_be_used(self):
        rows, old, candidates, test, hist = fixture()
        fresh = row(70, "test")
        rows.append(fresh); candidates.add(fresh["group"])
        hist["cleared_independent_groups"] = [fresh["group"]]
        cohorts, contract = build_cohorts(rows, old, candidates, test, hist, 8, 4)
        self.assertEqual(cohorts["verify"], [fresh])
        self.assertEqual(contract["independent_verify"]["queries"], 1)
        self.assertIn(fresh["group"], contract["forbidden_groups"])

    def test_historical_verify_cannot_be_relabelled(self):
        rows, old, candidates, test, hist = fixture()
        g = old["verify"][0]["group"]
        hist["exposed_groups"].remove(g)
        hist["cleared_independent_groups"] = [g]
        cohorts, _ = build_cohorts(rows, old, candidates, test, hist, 8, 4)
        self.assertEqual(cohorts["verify"], [])

    def test_exposed_clearance_conflict_rejected(self):
        rows, old, candidates, test, hist = fixture()
        hist["cleared_independent_groups"] = [rows[0]["group"]]
        with self.assertRaisesRegex(ValueError, "eligibility_conflict"):
            build_cohorts(rows, old, candidates, test, hist, 8, 4)

    def test_altered_control_prefix_rejected(self):
        rows, old, candidates, test, hist = fixture()
        old["train"] = list(reversed(old["train"]))
        with self.assertRaisesRegex(ValueError, "prefix_not_preserved"):
            build_cohorts(rows, old, candidates, test, hist, 8, 4)

    def test_declared_split_must_match_hash(self):
        rows, old, candidates, test, hist = fixture()
        rows[0]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "split_conflict"):
            build_cohorts(rows, old, candidates, test, hist, 8, 4)

    def test_all_three_theory_sources_required(self):
        rows, old, candidates, test, hist = fixture()
        rows = [r for r in rows if r["source"] != "QM9-NMR"]
        with self.assertRaisesRegex(ValueError, "theory_source_empty"):
            build_cohorts(rows, old, candidates, test, hist, 8, 4)

    def test_other_nucleus_does_not_add_experimental_training(self):
        rows, old, candidates, test, hist = fixture()
        rows += [row(99, nucleus="13C")]
        cohorts, _ = build_cohorts(rows, old, candidates, test, hist, 20, 4)
        self.assertEqual(len(cohorts["train"]), 12)

    def test_shared_validation_structure_protects_theory(self):
        rows, old, candidates, test, hist = fixture()
        x = copy.deepcopy(next(r for r in rows if r["source"] == "QM9-NMR"))
        hist["evaluation_groups"].append(x["group"])
        with self.assertRaisesRegex(ValueError, "theory_source_empty"):
            build_cohorts(rows, old, candidates, test, hist, 8, 4)

    def test_positive_and_negative_training_use_same_blacklist(self):
        rows, old, candidates, test, hist = fixture()
        cohorts, contract = build_cohorts(rows, old, candidates, test, hist, 8, 4)
        heldout = {r["group"] for name in ("original_val", "tune", "verify") for r in old[name]} | {test[0]["group"]}
        self.assertTrue(heldout.issubset(set(contract["training_excluded_groups"])))
        self.assertFalse(heldout & set(contract["eligible_training_groups"]))
        self.assertFalse({r["group"] for r in cohorts["train"]} & set(contract["training_excluded_groups"]))

    def test_group_hash_validation(self):
        with self.assertRaisesRegex(ValueError, "invalid_connectivity"):
            structural_split("unknown")


if __name__ == "__main__":
    unittest.main()
