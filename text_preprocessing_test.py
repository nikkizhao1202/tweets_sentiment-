import unittest
import text_preprocessing

class Test(unittest.TestCase):
    def setup(self):
        return
    def test_divide(self):

        result = text_preprocessing.TextPreprocessing('I <user> am hewei Yang! :),, !!!')
        expected_result = [11, 1, 226, 0, 185, 10, 0, 5, 5, 10, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0]
        result_1 = result.pad_sequence()
        self.assertEqual(result_1, expected_result)