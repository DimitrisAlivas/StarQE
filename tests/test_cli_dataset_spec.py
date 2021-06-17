"""Tests for the CLI dataset specification."""

import unittest

from mphrqe.data.loader import resolve_sample


class ResolveSampleTest(unittest.TestCase):

    def test_garbage(self):
        self.assertRaises(Exception, resolve_sample, "blabla")
        self.assertRaises(Exception, resolve_sample, "/*:::")
        self.assertRaises(Exception, resolve_sample, "/a/b/")
        self.assertRaises(Exception, resolve_sample, "bla:atleast10000")
        self.assertRaises(Exception, resolve_sample, "bla:atmost10000:nonsense")

    def test_simple(self):
        s = resolve_sample("/*/test/*:1000")
        assert s.selector == "/*/test/*"
        assert s.amount(10000) == 1000
        self.assertRaises(Exception, s.amount, 200)
        assert not s.reify
        assert not s.remove_qualifiers

    def test_atmost(self):
        s = resolve_sample("/*/test/*:atmost1000")
        assert s.selector == "/*/test/*"
        assert s.amount(10000) == 1000
        assert s.amount(500) == 500
        assert not s.reify
        assert not s.remove_qualifiers

    def test_star(self):
        s = resolve_sample("/*/test/*:*")
        assert s.selector == "/*/test/*"
        assert s.amount(10000) == 10000
        assert s.amount(500) == 500
        assert not s.reify
        assert not s.remove_qualifiers

    def test_options(self):
        s = resolve_sample("/*/test/*:*:reify")
        assert s.selector == "/*/test/*"
        assert s.amount(10000) == 10000
        assert s.amount(500) == 500
        assert s.reify
        assert not s.remove_qualifiers

        s = resolve_sample("/*/test/*:atmost1000:remove_qualifiers")
        assert s.selector == "/*/test/*"
        assert s.amount(10000) == 1000
        assert s.amount(500) == 500
        assert not s.reify
        assert s.remove_qualifiers

        self.assertRaises(Exception, resolve_sample, "/*/test/*:1000:reify_remove_qualifiers")
