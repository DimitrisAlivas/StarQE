import unittest

from mphrqe.data import query_executor


class Test_assert_query_validity(unittest.TestCase):
    def test_underscore(self):
        with self.assertRaises(AssertionError):
            query_executor.assert_query_validity(["_"])

    def test_underscore_2(self):
        with self.assertRaises(AssertionError):
            query_executor.assert_query_validity(["o1_var"])

    def test_underscore_var_target(self):
        with self.assertRaises(AssertionError):
            query_executor.assert_query_validity(["_var_target"])

    def test_share_with_s_and_p(self):
        with self.assertRaises(AssertionError):
            query_executor.assert_query_validity(["s0_p_0", "o_0_target"])

    def test_one_hop(self):
        query_executor.assert_query_validity(["s0", "p0", "o0_target", "diameter"])

    def test_three_in_shared_qual(self):
        query_executor.assert_query_validity(["s0", "p0", "o0_o1_o2_target",
                                              "s1", "p1",
                                              "s2", "p2",
                                              "qr0i0_qr1i1",
                                              "qv0i0_qv1i1_var",
                                              "diameter"])


if __name__ == '__main__':
    unittest.main()
