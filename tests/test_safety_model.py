import unittest

from safety_model import (
    CLASS_NAMES,
    FEATURE_COLUMNS,
    build_model,
    classify_reading,
    generate_reading,
    generate_training_data,
)


class SafetyModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.metrics, _ = build_model()

    def test_training_data_has_expected_features_and_three_classes(self):
        data = generate_training_data(rows_per_class=12)
        self.assertEqual(len(data), 36)
        self.assertTrue(set(FEATURE_COLUMNS).issubset(data.columns))
        self.assertEqual(set(data["severity"].unique()), {0, 1, 2})

    def test_model_returns_known_class_and_scores(self):
        reading = generate_reading("Routine activity", seed=5)
        result = classify_reading(self.model, reading)
        self.assertIn(result["status"], CLASS_NAMES.values())
        self.assertAlmostEqual(sum(result["scores"].values()), 1.0, places=6)

    def test_training_evaluation_has_expected_shape(self):
        self.assertGreater(self.metrics["accuracy"], 0.5)
        matrix = self.metrics["confusion_matrix"]
        self.assertEqual(matrix.shape, (3, 3))

    def test_fire_scenario_carries_flame_signal(self):
        reading = generate_reading("Fire emergency", seed=7)
        self.assertEqual(reading["flame_detected"], 1)
        self.assertGreaterEqual(reading["temperature_c"], 65)
        self.assertTrue(set(FEATURE_COLUMNS).issubset(reading))
        self.assertEqual(classify_reading(self.model, reading)["status"], "High-Priority Emergency")

    def test_routine_scenario_is_classified_safe(self):
        reading = generate_reading("Routine activity", seed=8)
        self.assertEqual(classify_reading(self.model, reading)["status"], "Safe")


if __name__ == "__main__":
    unittest.main()
