from pathlib import Path
import unittest

from pls.common import load_yaml
from pls_plan_jobs import build_train_jobs
from pls_plan_v2 import build_contexts, build_target_anchored_contexts, load_design, projected_checkpoint_count


ROOT = Path(__file__).resolve().parents[1]


class TestStudyPlanning(unittest.TestCase):
    def test_training_grid_has_180_jobs(self):
        config = load_yaml(ROOT / "configs" / "study_180.yaml")
        jobs = build_train_jobs(config)
        self.assertEqual(len(jobs), 180)
        self.assertEqual(len({job["run_id"] for job in jobs}), 180)

    def test_serving_design_projects_180_checkpoints(self):
        design = load_design(str(ROOT / "configs" / "serving_context.yaml"))
        self.assertEqual(projected_checkpoint_count(design), 180)
        self.assertGreater(len(build_contexts(design)), 1)
        self.assertGreater(len(build_target_anchored_contexts(design)), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
