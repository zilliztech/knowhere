#!/usr/bin/env python3
# Copyright (C) 2026 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under the License.

"""Exercise packaging with real ELF libraries; these are not Knowhere engine tests."""

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bundle_native.py"
PLATFORM = {"aarch64": "linux-aarch64", "x86_64": "linux-x86_64"}.get(platform.machine())


def clean_environment():
    env = os.environ.copy()
    for name in ("LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT"):
        env.pop(name, None)
    env["LC_ALL"] = "C"
    return env


def run(command, **kwargs):
    return subprocess.run(command, check=True, text=True, capture_output=True, env=clean_environment(), **kwargs)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BundleNativeTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="knowhere-bundle-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        licenses = self.source / "licenses"
        licenses.mkdir()
        (licenses / "LICENSE.txt").write_text("Fixture library license\n", encoding="utf-8")
        self.output = self.root / "resources"
        self.destination = self.output / "native" / "knowhere" / "1" / PLATFORM
        self.dependency = self.compile_library(
            self.source, "libfixture_dep.so.7", "int fixture_value(void) { return 41; }"
        )
        self.library = self.compile_library(
            self.source,
            "libknowhere_jni.so",
            "extern int fixture_value(void); int fixture_answer(void) { return fixture_value() + 1; }",
            ["-L" + str(self.source), "-l:libfixture_dep.so.7", "-Wl,-rpath,$ORIGIN"],
            soname="libknowhere_jni.so.1",
        )

    def compile_library(self, directory, name, code, linker_options=None, soname=None):
        source = directory / (name + ".c")
        source.write_text(code, encoding="utf-8")
        target = directory / name
        run(["gcc", "-shared", "-fPIC", str(source), "-Wl,-soname," + (soname or name),
             "-o", str(target)] + (linker_options or []))
        return target

    def package(self, *extra, success=True):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--library", str(self.library), "--output", str(self.output),
             "--platform", PLATFORM] + list(extra),
            text=True, capture_output=True, env=clean_environment(),
        )
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def manifest(self):
        return dict(line.split("=", 1) for line in
                    (self.destination / "manifest.properties").read_text(encoding="utf-8").splitlines()
                    if line and not line.startswith("#"))

    def test_isolated_load_checksums_soname_and_reproducibility(self):
        original = {p.name: digest(p) for p in (self.library, self.dependency)}
        self.output.mkdir()
        unrelated = self.output / "unrelated.txt"
        unrelated.write_text("keep me", encoding="utf-8")
        self.package()
        manifest = self.manifest()
        self.assertEqual(manifest["cAbiVersion"], "1")
        self.assertEqual(manifest["entryLibrary"], self.library.name)
        self.assertIn(manifest["entryLibrary"], manifest["libraries"].split(","))
        self.assertEqual(manifest["libraries"], "libfixture_dep.so.7,libknowhere_jni.so")
        self.assertEqual(manifest["missingLicenses"], "")
        license_index = json.loads((self.destination / "license-files.json").read_text(encoding="utf-8"))
        self.assertEqual(set(license_index), {
            "licenses/libfixture_dep.so.7/LICENSE.txt", "licenses/libknowhere_jni.so/LICENSE.txt"
        })
        for relative, checksum in license_index.items():
            self.assertEqual(digest(self.destination / relative), checksum)
            self.assertEqual((self.destination / relative).read_text(encoding="utf-8"), "Fixture library license\n")
        first_manifest = (self.destination / "manifest.properties").read_bytes()
        for name in manifest["libraries"].split(","):
            copied = self.destination / name
            self.assertEqual(manifest["sha256." + name], digest(copied))
            self.assertEqual(run(["patchelf", "--print-rpath", str(copied)]).stdout.strip(), "$ORIGIN")
            self.assertEqual(digest(self.source / name), original[name])
        self.assertEqual(run(["patchelf", "--print-soname", str(self.destination / self.library.name)]).stdout.strip(),
                         "libknowhere_jni.so.1")
        self.package()
        self.assertEqual((self.destination / "manifest.properties").read_bytes(), first_manifest)
        self.assertEqual(unrelated.read_text(encoding="utf-8"), "keep me")
        self.source.rename(self.root / "source-unavailable")
        result = run([sys.executable, "-c", "import ctypes,sys; "
                      "library=ctypes.CDLL(sys.argv[1]); "
                      "library.fixture_answer.restype=ctypes.c_int; "
                      "print(library.fixture_answer())", str(self.destination / manifest["entryLibrary"])], cwd="/")
        self.assertEqual(result.stdout.strip(), "42")

    def test_rejects_wrong_architecture(self):
        wrong = "linux-x86_64" if PLATFORM == "linux-aarch64" else "linux-aarch64"
        result = self.package("--platform", wrong, success=False)
        self.assertIn("architecture", result.stderr.lower())
        self.assertFalse(self.output.exists())

    def test_rejects_missing_dependency(self):
        self.dependency.rename(self.dependency.with_name("hidden-library"))
        result = self.package(success=False)
        self.assertIn("libfixture_dep.so.7", result.stderr)
        self.assertIn("unresolved", result.stderr.lower())
        self.assertFalse(self.output.exists())

    def test_rejects_missing_platform_library(self):
        run(["patchelf", "--add-needed", "libc.so.999", str(self.library)])
        result = self.package(success=False)
        self.assertIn("libc.so.999", result.stderr)
        self.assertIn("unresolved", result.stderr.lower())
        self.assertFalse(self.output.exists())

    def test_includes_runtime_dependency_and_excludes_glibc(self):
        self.library = self.compile_library(
            self.source, "libknowhere_jni.so",
            "extern int fixture_value(void); int fixture_answer(void) { return fixture_value() + 1; }",
            ["-L" + str(self.source), "-l:libfixture_dep.so.7", "-Wl,-rpath,$ORIGIN",
             "-Wl,--no-as-needed", "-l:libgcc_s.so.1", "-lm", "-Wl,--as-needed"],
        )
        self.package()
        names = self.manifest()["libraries"].split(",")
        self.assertIn("libgcc_s.so.1", names)
        self.assertNotIn("libc.so.6", names)
        self.assertNotIn("libm.so.6", names)
        self.assertLess(names.index("libgcc_s.so.1"), names.index("libknowhere_jni.so"))
        self.assertNotIn("libgcc_s.so.1", self.manifest()["missingLicenses"].split(","))
        source = Path(run(["gcc", "-print-file-name=libgcc_s.so.1"]).stdout.strip()).resolve()
        # dpkg may register /lib even when the file resolves below /usr/lib.
        records = run(["dpkg-query", "-S", "*/" + source.name]).stdout.splitlines()
        owners = set()
        for record in records:
            owner, separator, filename = record.partition(": ")
            if separator and Path(filename).resolve() == source:
                owners.add(owner)
        self.assertEqual(len(owners), 1, "Expected one package owning the resolved runtime library")
        owner = owners.pop()
        copyright_source = Path("/usr/share/doc") / owner.split(":", 1)[0] / "copyright"
        copyright_copy = self.destination / "licenses" / "libgcc_s.so.1" / "dpkg" / owner / "copyright"
        self.assertEqual(copyright_copy.read_bytes(), copyright_source.read_bytes())
        common = self.destination / "licenses" / "libgcc_s.so.1" / "common-licenses" / "GPL-3"
        self.assertEqual(common.read_bytes(), Path("/usr/share/common-licenses/GPL-3").read_bytes())

    def test_copies_source_root_license_and_notice(self):
        shutil.rmtree(self.source / "licenses")
        (self.root / "LICENSE").write_text("Source root license\n", encoding="utf-8")
        (self.root / "NOTICE").write_text("Source root notice\n", encoding="utf-8")
        self.package()
        self.assertEqual(self.manifest()["missingLicenses"], "")
        for name in ("libfixture_dep.so.7", "libknowhere_jni.so"):
            destination = self.destination / "licenses" / name
            self.assertEqual((destination / "LICENSE").read_bytes(), (self.root / "LICENSE").read_bytes())
            self.assertEqual((destination / "NOTICE").read_bytes(), (self.root / "NOTICE").read_bytes())

    def test_copies_build_licenses_for_static_and_header_dependencies(self):
        repository = SCRIPT.parents[2]
        fixture = self.root / "license-project"
        fixture.mkdir()
        (fixture / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.20)\nproject(LicenseFixture NONE)\n'
            'include("' + str(repository / "cmake" / "binding_licenses.cmake") + '")\n'
            'knowhere_collect_binding_licenses("' + str(repository) + '" "${CMAKE_BINARY_DIR}" ON)\n',
            encoding="utf-8",
        )
        env = clean_environment()
        env["CONAN_HOME"] = str(self.root / "conan-home")
        def conan(*arguments):
            result = subprocess.run(["conan"] + list(arguments), env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return result
        conan("profile", "detect")
        for name, kind in (("fixture-headers", "header-library"), ("fixture-unused", "header-library"),
                           ("fixture-static", "static-library"), ("fixture-missing", "header-library")):
            recipe = self.root / name
            recipe.mkdir()
            (recipe / "LICENSE").write_text(name + " license\n", encoding="utf-8")
            (recipe / "conanfile.py").write_text(
                'from conan import ConanFile\nfrom conan.tools.files import copy\nimport os\n'
                'class Fixture(ConanFile):\n'
                '    name = "' + name + '"\n    version = "1.0"\n'
                '    package_type = "' + kind + '"\n    exports_sources = "LICENSE"\n'
                + ('    def requirements(self):\n'
                   '        self.requires("fixture-headers/1.0")\n'
                   '        self.requires("fixture-unused/1.0", headers=False, libs=False, run=False)\n'
                   if name == "fixture-static" else "")
                + '    def package(self):\n'
                + ('        pass\n' if name in ("fixture-missing", "fixture-unused") else
                   '        copy(self, "LICENSE", self.source_folder, os.path.join(self.package_folder, "licenses"))\n'),
                encoding="utf-8",
            )
            conan("create", str(recipe), "--no-remote")
        (fixture / "conanfile.py").write_text(
            'from conan import ConanFile\nimport importlib.util\n'
            'spec = importlib.util.spec_from_file_location("knowhere_recipe", ' + repr(str(repository / "conanfile.py")) + ')\n'
            'module = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(module)\n'
            'class FixtureConsumer(ConanFile):\n'
            '    requires = "fixture-static/1.0", "fixture-missing/1.0"\n'
            '    def generate(self):\n        module.KnowhereConan._collect_host_licenses(self)\n',
            encoding="utf-8",
        )
        result = conan("install", str(fixture), "--output-folder", str(self.source), "--no-remote")
        self.assertIn("License documents require follow-up for host dependency: fixture-missing/1.0", result.stderr)
        run(["cmake", "-S", str(fixture), "-B", str(self.source)])
        packaged = self.package()
        self.assertIn("License files require follow-up", packaged.stdout)
        self.assertIn(self.library.name, self.manifest()["missingLicenses"].split(","))
        root_missing = (self.destination / "missing-licenses.txt").read_text(encoding="utf-8")
        self.assertIn("build dependency: fixture-missing/1.0#", root_missing)
        self.assertNotIn("fixture-unused", root_missing)
        prefix = self.destination / "licenses" / self.library.name
        for relative in ("LICENSE", "thirdparty/faiss/LICENSE", "thirdparty/faiss/THIRD_PARTY_NOTICES",
                         "thirdparty/hnswlib/LICENSE", "thirdparty/DiskANN/LICENSE", "thirdparty/DiskANN/NOTICE.txt"):
            self.assertEqual((prefix / "source" / relative).read_bytes(), (repository / relative).read_bytes())
        inventory = json.loads((prefix / "conan" / "dependencies.json").read_text(encoding="utf-8"))
        self.assertEqual({entry["reference"].split("/", 1)[0] for entry in inventory},
                         {"fixture-static", "fixture-headers", "fixture-missing"})
        for entry in inventory:
            name = entry["reference"].split("/", 1)[0]
            if name == "fixture-missing":
                self.assertEqual(entry["files"], [])
                continue
            self.assertEqual((prefix / "conan" / entry["directory"] / "LICENSE").read_text(encoding="utf-8"),
                             name + " license\n")
        missing = (prefix / "conan" / "missing-licenses.txt").read_text(encoding="utf-8")
        self.assertTrue(missing.startswith("fixture-missing/1.0#"), missing)
        self.assertEqual(len(missing.splitlines()), 1)
        self.assertFalse((prefix / "source" / "missing-licenses.txt").exists())

    def test_reports_missing_license_files(self):
        shutil.rmtree(self.source / "licenses")
        result = self.package()
        self.assertIn("License files require follow-up", result.stdout)
        self.assertEqual(set(self.manifest()["missingLicenses"].split(",")),
                         {"libfixture_dep.so.7", "libknowhere_jni.so"})
        missing = (self.destination / "missing-licenses.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual(set(missing), {"libfixture_dep.so.7", "libknowhere_jni.so"})

    def test_preserves_unowned_destination(self):
        self.destination.mkdir(parents=True)
        unrelated = self.destination / "important.txt"
        unrelated.write_text("keep me", encoding="utf-8")
        result = self.package(success=False)
        self.assertIn("owned", result.stderr.lower())
        self.assertEqual(unrelated.read_text(encoding="utf-8"), "keep me")

    def test_preserves_modified_bundle(self):
        self.package()
        changed = self.destination / "libfixture_dep.so.7"
        with changed.open("ab") as stream:
            stream.write(b"user change")
        changed_digest = digest(changed)
        result = self.package(success=False)
        self.assertIn("modified", result.stderr.lower())
        self.assertEqual(digest(changed), changed_digest)

    def test_rejects_missing_or_unknown_entry_library(self):
        self.package()
        path = self.destination / "manifest.properties"
        original = path.read_text(encoding="utf-8")
        for entry in (None, "unlisted.so", "../outside.so"):
            with self.subTest(entry=entry):
                lines = [line for line in original.splitlines() if not line.startswith("entryLibrary=")]
                if entry is not None:
                    lines.append("entryLibrary=" + entry)
                modified = "\n".join(lines) + "\n"
                path.write_text(modified, encoding="utf-8")
                result = self.package(success=False)
                self.assertIn("entry", result.stderr.lower())
                self.assertEqual(path.read_text(encoding="utf-8"), modified)

    def test_rejects_same_name_with_different_content(self):
        branches = []
        for letter, value in (("a", 10), ("b", 20)):
            directory = self.source / letter
            directory.mkdir()
            self.compile_library(directory, "libcollision.so.1", "int collision(void) { return %d; }" % value)
            branches.append(self.compile_library(
                directory, "libbranch_%s.so" % letter,
                "extern int collision(void); int branch_%s(void) { return collision(); }" % letter,
                ["-L" + str(directory), "-l:libcollision.so.1", "-Wl,-rpath,$ORIGIN"],
            ))
        self.library = self.compile_library(
            self.source, "libknowhere_jni.so",
            "extern int branch_a(void); extern int branch_b(void); "
            "int fixture_answer(void) { return branch_a() + branch_b(); }",
            ["-L" + str(branches[0].parent), "-L" + str(branches[1].parent), "-lbranch_a", "-lbranch_b",
             "-Wl,-rpath,$ORIGIN/a:$ORIGIN/b"],
        )
        result = self.package(success=False)
        self.assertIn("collision", result.stderr.lower())
        self.assertFalse(self.output.exists())

    def test_rejects_unsafe_library_name(self):
        renamed = self.source / "bad,name.so"
        shutil.copyfile(self.library, renamed)
        self.library = renamed
        result = self.package(success=False)
        self.assertIn("name", result.stderr.lower())
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    if sys.platform != "linux" or PLATFORM is None:
        raise SystemExit("Tests require Linux aarch64 or x86_64 with gcc, readelf, ldd, and patchelf.")
    for tool in ("gcc", "readelf", "ldd", "patchelf", "cmake", "conan"):
        if shutil.which(tool) is None:
            raise SystemExit("Missing required tool: " + tool)
    unittest.main(verbosity=2)
