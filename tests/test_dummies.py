import unittest
from PyRep.tests.core import TestCore
from PyRep.pyrep.objects.dummy import Dummy


class TestDummies(TestCore):

    def test_get_dummy(self):
        Dummy('dummy')

    def test_create_dummy(self):
        d = Dummy.create(0.01)
        self.assertIsInstance(d, Dummy)


if __name__ == '__main__':
    unittest.main()
