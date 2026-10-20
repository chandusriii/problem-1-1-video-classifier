import unittest

from src.intent_classifier import engineer_query_features, load_jsonl_dataset, predict_intent

class IntentClassifierTests(unittest.TestCase):
    def test_empty_query_rejected(self):
        with self.assertRaises(ValueError):
            predict_intent("   ")

    def test_feature_extraction_timestamp(self):
        features = engineer_query_features("What color is the car at 00:15?")
        self.assertTrue(features["has_timestamp"])
        self.assertTrue(features["asks_visual_attribute"])

    def test_frame_query(self):
        result = predict_intent("What color is the car at 00:15?")
        self.assertEqual(result["intent"], "FRAME_RECOGNITION")
        self.assertEqual(result["scope"], "FRAME")
        self.assertEqual(result["complexity"], "L1_DIRECT")

    def test_ocr_query(self):
        result = predict_intent("Read the text on the sign at 01:10.")
        self.assertEqual(result["intent"], "OCR_FRAME_QUERY")
        self.assertEqual(result["scope"], "FRAME")

    def test_transcript_query(self):
        result = predict_intent("What does the speaker say at 02:30?")
        self.assertEqual(result["intent"], "TRANSCRIPT_LOOKUP")
        self.assertEqual(result["scope"], "CLIP")

    def test_global_summary(self):
        result = predict_intent("Summarize the entire video.")
        self.assertEqual(result["intent"], "VIDEO_SUMMARY")
        self.assertEqual(result["scope"], "WHOLE_VIDEO")

    def test_complex_cross_section(self):
        result = predict_intent("How does the speaker's opinion change from the beginning to the end?")
        self.assertEqual(result["intent"], "CROSS_SECTION_REASONING")
        self.assertEqual(result["scope"], "MULTI_SECTION")
        self.assertEqual(result["complexity"], "L3_GRAPH")

    def test_count_query(self):
        result = predict_intent("How many times does the player score?")
        self.assertEqual(result["intent"], "EVENT_COUNTING")
        self.assertEqual(result["scope"], "MULTI_SECTION")

    def test_routing_output(self):
        result = predict_intent("Why did the person leave after the argument?")
        self.assertEqual(result["retrieval_strategy"], "GRAPH_RETRIEVAL")
        self.assertIn("confidence", result)

    def test_training_dataset_loads(self):
        rows = load_jsonl_dataset("data/intent_examples.jsonl")
        self.assertGreaterEqual(len(rows), 20)

if __name__ == "__main__":
    unittest.main()
