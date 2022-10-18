import unittest


class PSPipeTest(unittest.TestCase):
    def test_dependencies(self):
        def _assert_import(pkg_name):
            try:
                import importlib
                importlib.import_module(pkg_name)
            except:
                assert False, "Import of '{}' fails".format(pkg_name)

        _assert_import("camb")
        _assert_import("mflike")
        _assert_import("pixell")
        _assert_import("pspy")
        _assert_import("pymaster")
