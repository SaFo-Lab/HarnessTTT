import tempfile
import unittest
from pathlib import Path

from outer.rounds import campaign_config


class PairedCampaignConfigTest(unittest.TestCase):
    def test_paired_reward_routes_control_and_effect_threshold(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "campaign.yaml"
            path.write_text(
                "task: eft__math__second_autocorr_ineq\n"
                "round_base: 9000\n"
                "reward:\n"
                "  impl: paired\n"
                "  min_paired_effect: 0.002\n"
                "  paired_repeats: 3\n"
            )
            env = campaign_config.load(path).env_overrides()
        self.assertEqual(env["SAH_ADV"], "paired")
        self.assertEqual(env["SAH_MIN_PAIRED_EFFECT"], "0.002")
        self.assertEqual(env["SAH_PAIRED_REPEATS"], "3")


if __name__ == "__main__":
    unittest.main()
