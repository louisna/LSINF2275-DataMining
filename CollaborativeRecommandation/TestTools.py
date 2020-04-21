import unittest
from Tools import *


class TestTools(unittest.TestCase):

    def test_file_tool(self):
        filepath = "test_file.XXXX"
        val = (0.95, 0.76)
        save_result(filepath, 0, val, sd=False)

        res = get_result(filepath, 0, sd=False)
        self.assertEqual(val, res)


if __name__ == '__main__':
    unittest.main()
