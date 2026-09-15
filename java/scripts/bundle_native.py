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

"""Bundle a local Linux JNI library and its actual ELF runtime dependency closure.

Example:
    python3 java/scripts/bundle_native.py --library /build/libknowhere_jni.so \
        --output /build/native-resources --platform linux-aarch64

Requires readelf, ldd, and patchelf on a compatible Linux host. LD_LIBRARY_PATH
may locate build dependencies; no source file is modified. The output has a
flat directory, a checksum manifest, and a private ownership marker. Replacing
an existing bundle requires an intact bundle previously written by this tool.
Replacement uses Linux renameat2(RENAME_EXCHANGE), never a delete-then-copy.
Conan-style licenses directories, source-root LICENSE/NOTICE documents, and
dpkg-owned copyright/common-license files are copied under licenses/<library-name>.
Missing license documents are reported in missing-licenses.txt and on stdout;
they must be supplied before distributing the resulting resources.
"""

import argparse
import ctypes
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, Optional, Set, Tuple


ABI_VERSION = "1"
PLATFORMS = {
    "linux-aarch64": "AArch64",
    "linux-x86_64": "Advanced Micro Devices X86-64",
}
OWNER_FILE = ".knowhere-native-bundle"
OWNER_CONTENT = "knowhere-jni-native-bundle-v1\n"
MANIFEST = "manifest.properties"
LICENSE_INDEX = "license-files.json"
MISSING_LICENSES = "missing-licenses.txt"
SAFE_NAME = re.compile(r"[A-Za-z0-9_+.-]+\Z")
SOURCE_LICENSE_FILES = {
    "license", "license.txt", "license.md", "license.rst", "license-apache", "license-mit",
    "licence", "licence.txt", "licence.md", "copying", "copying.txt", "copying.lesser", "copying.lib",
    "copyright", "copyright.txt", "unlicense", "notice", "notice.txt", "notice.md",
}
DPKG_PACKAGE = re.compile(r"[a-z0-9][a-z0-9+.-]*(?::[a-z0-9-]+)?\Z")
COMMON_LICENSE_REFERENCE = re.compile(r"/usr/share/common-licenses/([A-Za-z0-9][A-Za-z0-9_.+-]*)")
# These are provided by the target's glibc installation, not bundled. In
# particular libstdc++, libgcc_s, libgomp, libomp, libcrypt and libnsl are NOT
# excluded: they are independently versioned runtime dependencies.
GLIBC_LIBRARY = re.compile(
    r"(?:lib(?:c|m|mvec|pthread|dl|rt|resolv|util|anl|BrokenLocale|thread_db|"
    r"nss_(?:files|dns|compat|hesiod))\.so(?:\.[0-9]+)*|"
    r"ld-linux-(?:aarch64|x86-64)\.so\.[0-9]+|linux-vdso\.so\.[0-9]+)\Z"
)


class BundleError(Exception):
    """An invalid or incomplete bundle must not be published."""


def validate_name(name: str) -> str:
    if not SAFE_NAME.fullmatch(name) or name in (".", "..", OWNER_FILE, MANIFEST, LICENSE_INDEX, MISSING_LICENSES, "licenses"):
        raise BundleError("Unsafe library name: %r" % name)
    return name


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def environment() -> Dict[str, str]:
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    # Resolve declared DT_NEEDED entries, without injecting unrelated objects.
    env.pop("LD_PRELOAD", None)
    env.pop("LD_AUDIT", None)
    return env


def command(arguments, allow_failure=False):
    result = subprocess.run(arguments, text=True, capture_output=True, env=environment())
    if result.returncode != 0 and not allow_failure:
        raise BundleError("Command failed: %s\n%s%s" % (" ".join(map(str, arguments)), result.stdout, result.stderr))
    return result


@dataclass(frozen=True)
class ElfInfo:
    needed: Tuple[str, ...]
    soname: Optional[str]


def inspect_elf(path: Path, target_platform: str) -> ElfInfo:
    if not path.is_file():
        raise BundleError("Library is not a regular file: %s" % path)
    output = command(["readelf", "--wide", "--file-header", "--dynamic", str(path)]).stdout
    machine = re.search(r"^\s*Machine:\s*(.*?)\s*$", output, re.MULTILINE)
    elf_class = re.search(r"^\s*Class:\s*(.*?)\s*$", output, re.MULTILINE)
    if machine is None or machine.group(1) != PLATFORMS[target_platform] or elf_class is None or elf_class.group(1) != "ELF64":
        raise BundleError("ELF architecture mismatch for %s: expected %s ELF64, found %s" %
                          (path, target_platform, machine.group(1) if machine else "unknown"))
    if re.search(r"^\s*Type:\s+DYN\b", output, re.MULTILINE) is None:
        raise BundleError("Expected an ELF shared object: %s" % path)
    needed = tuple(sorted(set(re.findall(r"\(NEEDED\).*?Shared library: \[(.*?)\]", output))))
    sonames = re.findall(r"\(SONAME\).*?Library soname: \[(.*?)\]", output)
    if len(sonames) > 1:
        raise BundleError("Multiple ELF SONAME entries in %s" % path)
    for name in needed + tuple(sonames):
        validate_name(name)
    return ElfInfo(needed, sonames[0] if sonames else None)


def resolved_dependencies(path: Path, info: ElfInfo) -> Dict[str, Optional[Path]]:
    if not info.needed:
        return {}
    result = command(["ldd", str(path)], allow_failure=True)
    resolved = {}
    for line in result.stdout.splitlines():
        missing = re.fullmatch(r"\s*(\S+)\s+=>\s+not found\s*", line)
        if missing:
            resolved[missing.group(1)] = None
            continue
        entry = re.fullmatch(r"\s*(\S+)\s+=>\s+(.*?)\s+\(0x[0-9a-fA-F]+\)\s*", line)
        if entry:
            resolved[entry.group(1)] = Path(entry.group(2)).resolve()
            continue
        loader = re.fullmatch(r"\s*(/.*?)\s+\(0x[0-9a-fA-F]+\)\s*", line)
        if loader:
            loader_path = Path(loader.group(1)).resolve()
            resolved[Path(loader.group(1)).name] = loader_path
    if result.returncode != 0:
        raise BundleError("ldd failed for %s:\n%s%s" % (path, result.stdout, result.stderr))
    return resolved


@dataclass(frozen=True)
class Library:
    source: Path
    source_sha256: str
    elf: ElfInfo


class Closure:
    def __init__(self, root: Path, target_platform: str):
        self.platform = target_platform
        self.root_name = validate_name(root.name)
        self.libraries: Dict[str, Library] = {}
        self.sonames: Dict[str, Library] = {}
        self.visited: Set[Tuple[str, Path]] = set()
        root_source = root.resolve(strict=True)
        root_info = inspect_elf(root_source, target_platform)
        self.root_resolution = resolved_dependencies(root_source, root_info)
        self.visit(self.root_name, root_source)

    def visit(self, name: str, source: Path) -> None:
        validate_name(name)
        source = source.resolve(strict=True)
        pair = (name, source)
        if pair in self.visited:
            return
        self.visited.add(pair)
        info = inspect_elf(source, self.platform)
        source_sha256 = digest(source)
        previous = self.libraries.get(name)
        if previous is not None and previous.source_sha256 != source_sha256:
            raise BundleError("Library name collision with different content: %s (%s and %s)" %
                              (name, previous.source, source))
        if previous is None:
            self.libraries[name] = Library(source, source_sha256, info)
        if info.soname:
            previous_soname = self.sonames.get(info.soname)
            if previous_soname is not None and previous_soname.source_sha256 != source_sha256:
                raise BundleError("ELF SONAME collision with different content: %s (%s and %s)" %
                                  (info.soname, previous_soname.source, source))
            self.sonames[info.soname] = self.libraries[name]
        # Inspect each source in its own loader context. This detects two
        # branches with the same DT_NEEDED name but different $ORIGIN content.
        # Root resolutions also cover inherited DT_RPATH dependencies which
        # cannot be resolved when ldd examines a child in isolation.
        resolutions = resolved_dependencies(source, info)
        for dependency in info.needed:
            resolved = resolutions.get(dependency) or self.root_resolution.get(dependency)
            if resolved is None or not resolved.is_file():
                raise BundleError("Unresolved dependency %s required by %s" % (dependency, source))
            if GLIBC_LIBRARY.fullmatch(dependency):
                inspect_elf(resolved, self.platform)
                continue
            self.visit(dependency, resolved)

    def load_order(self):
        ordered = []
        active = set()
        completed = set()

        def visit(name):
            if name in completed:
                return
            if name in active:
                raise BundleError("Dependency cycle prevents dependency-first load order at %s" % name)
            active.add(name)
            for dependency in self.libraries[name].elf.needed:
                if not GLIBC_LIBRARY.fullmatch(dependency):
                    visit(dependency)
            active.remove(name)
            completed.add(name)
            ordered.append(name)

        visit(self.root_name)
        return ordered


def read_manifest(path: Path) -> Dict[str, str]:
    properties = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or key in properties:
            raise BundleError("Existing bundle has an invalid manifest: %s" % path)
        properties[key] = value
    return properties


def file_inventory(directory: Path):
    """Return every regular file and directory, rejecting links and special files."""
    files = {}
    directories = set()
    for root, child_directories, child_files in os.walk(directory, followlinks=False):
        for name in child_directories + child_files:
            path = Path(root) / name
            relative = path.relative_to(directory).as_posix()
            if path.is_symlink():
                raise BundleError("Symlinks are not permitted inside owned bundle/license directories: %s" % path)
            if path.is_dir():
                directories.add(relative)
            elif path.is_file():
                files[relative] = path
            else:
                raise BundleError("Special files are not permitted in a bundle: %s" % path)
    return files, directories


def dpkg_license_sources(source: Path):
    if shutil.which("dpkg-query") is None:
        return {}
    candidates = [source]
    # On merged-/usr systems dpkg may record /lib/... while the ELF source
    # resolves to /usr/lib/.... Query only aliases verified to be the same file.
    if str(source).startswith("/usr/"):
        alias = Path("/") / source.relative_to("/usr")
        if alias.is_file() and alias.resolve() == source:
            candidates.append(alias)
    owners = set()
    for candidate in candidates:
        result = command(["dpkg-query", "-S", str(candidate)], allow_failure=True)
        if result.returncode != 0:
            continue
        for line in result.stdout.splitlines():
            packages, separator, filename = line.partition(": ")
            if not separator or Path(filename).resolve() != source:
                continue
            for package in packages.split(", "):
                if DPKG_PACKAGE.fullmatch(package):
                    owners.add(package)
    files = {}
    for package in sorted(owners):
        # Debian documentation paths use the binary package name without its
        # multiarch qualifier. Follow distribution-owned documentation links,
        # such as libgcc-s1 -> gcc-14-base, and copy the referenced bytes.
        copyright_file = Path("/usr/share/doc") / package.split(":", 1)[0] / "copyright"
        if copyright_file.is_file():
            files["dpkg/" + package + "/copyright"] = copyright_file.resolve()
    return files


def license_sources(source: Path):
    # Conan packages place runtime libraries in <package>/lib and license
    # documents in <package>/licenses. Search nearest ancestors first so a
    # package's own licenses take precedence over any wider source checkout.
    for ancestor in source.parents:
        if ancestor == ancestor.parent:
            break
        candidate = ancestor / "licenses"
        if candidate.is_symlink():
            raise BundleError("License directory must not be a symlink: %s" % candidate)
        if candidate.is_dir():
            files, _ = file_inventory(candidate)
            if files:
                return files
        documents = {entry.name: entry for entry in ancestor.iterdir()
                     if entry.name.lower() in SOURCE_LICENSE_FILES}
        # NOTICE alone does not identify a project's license root. Include it
        # alongside LICENSE/COPYING/etc. from the nearest documented root.
        if any(not name.lower().startswith("notice") for name in documents):
            for document in documents.values():
                if document.is_symlink() or not document.is_file():
                    raise BundleError("Source license must be a regular file: %s" % document)
            return documents
    return dpkg_license_sources(source)


def copy_licenses(source: Path, name: str, staging: Path, epoch: int):
    files = license_sources(source)
    missing = []
    # Build-time inventories cover static and header-only dependencies which
    # cannot appear in DT_NEEDED. Preserve and surface their missing documents.
    for relative, document in files.items():
        if Path(relative).name == MISSING_LICENSES:
            missing.extend("build dependency: " + line.strip() for line in
                           document.read_text(encoding="utf-8").splitlines() if line.strip())
    pending = list(files.values())
    examined = set()
    while pending:
        document = pending.pop()
        if document in examined:
            continue
        examined.add(document)
        content = document.read_text(encoding="utf-8", errors="replace")
        for reference in COMMON_LICENSE_REFERENCE.findall(content):
            reference = reference.rstrip(".")
            common = Path("/usr/share/common-licenses") / reference
            if not common.is_file():
                missing.append("missing referenced license " + str(common))
                continue
            files["common-licenses/" + reference] = common.resolve()
            pending.append(common.resolve())
    copied = {}
    for relative, original in sorted(files.items()):
        destination = staging / "licenses" / name / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
        destination.chmod(0o644)
        os.utime(destination, (epoch, epoch))
        copied[destination.relative_to(staging).as_posix()] = digest(destination)
    return copied, sorted(set(missing))


def validate_owned_directory(destination: Path, target_platform: str) -> None:
    if destination.is_symlink() or not destination.is_dir():
        raise BundleError("Destination is not an owned bundle directory: %s" % destination)
    marker = destination / OWNER_FILE
    manifest_path = destination / MANIFEST
    if marker.is_symlink() or not marker.is_file() or marker.read_text(encoding="utf-8") != OWNER_CONTENT:
        raise BundleError("Refusing to replace a directory not owned by this bundler: %s" % destination)
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise BundleError("Owned bundle manifest is missing: %s" % destination)
    manifest = read_manifest(manifest_path)
    if manifest.get("cAbiVersion") != ABI_VERSION or manifest.get("platform") != target_platform:
        raise BundleError("Existing bundle ABI or platform differs: %s" % destination)
    names = manifest.get("libraries", "").split(",")
    if not names or len(set(names)) != len(names):
        raise BundleError("Existing bundle has an invalid library list: %s" % destination)
    for name in names:
        validate_name(name)
    expected = set(names) | {MANIFEST, OWNER_FILE, LICENSE_INDEX, MISSING_LICENSES}
    files, directories = file_inventory(destination)
    license_index = destination / LICENSE_INDEX
    if LICENSE_INDEX not in files or digest(license_index) != manifest.get("licenseFilesSha256"):
        raise BundleError("Existing bundle license index was modified or is missing: %s" % destination)
    try:
        license_files = json.loads(license_index.read_text(encoding="utf-8"))
    except ValueError as error:
        raise BundleError("Existing bundle has an invalid license index: %s" % destination) from error
    if not isinstance(license_files, dict):
        raise BundleError("Existing bundle has an invalid license index: %s" % destination)
    expected_directories = set()
    for relative, checksum in license_files.items():
        path = Path(relative)
        if (path.is_absolute() or ".." in path.parts or len(path.parts) < 3 or
                path.parts[0] != "licenses" or path.parts[1] not in names or not isinstance(checksum, str)):
            raise BundleError("Existing bundle has an unsafe license path: %r" % relative)
        for parent in path.parents:
            if parent != Path("."):
                expected_directories.add(parent.as_posix())
        if relative not in files or digest(files[relative]) != checksum:
            raise BundleError("Existing bundle license was modified or is missing: %s" % relative)
    expected.update(license_files)
    if set(files) != expected or directories != expected_directories:
        raise BundleError("Existing bundle contains unowned or missing files: %s" % destination)
    if digest(destination / MISSING_LICENSES) != manifest.get("missingLicensesSha256"):
        raise BundleError("Existing bundle missing-license list was modified: %s" % destination)
    for name in names:
        library = destination / name
        if library.is_symlink() or not library.is_file() or digest(library) != manifest.get("sha256." + name):
            raise BundleError("Existing bundle library was modified or is missing: %s" % library)


def create_directory(path: Path) -> None:
    if path.is_symlink():
        raise BundleError("Refusing to use a symlink as an output directory: %s" % path)
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise BundleError("Output path is not a directory: %s" % path)


def rename_atomic(source: Path, destination: Path, exchange: bool) -> None:
    # Linux supports atomically exchanging nonempty directories. A replacement
    # therefore never exposes an absent or half-populated bundle to readers.
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise BundleError("Atomic publication requires Linux renameat2 support")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    flag = 2 if exchange else 1  # RENAME_EXCHANGE or RENAME_NOREPLACE
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), flag) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), str(destination))


def publish(closure: Closure, output: Path, target_platform: str) -> Path:
    order = closure.load_order()
    epoch_text = os.environ.get("SOURCE_DATE_EPOCH", "0")
    try:
        epoch = int(epoch_text)
    except ValueError as error:
        raise BundleError("SOURCE_DATE_EPOCH must be a nonnegative integer") from error
    if epoch < 0:
        raise BundleError("SOURCE_DATE_EPOCH must be a nonnegative integer")
    parent = output
    create_directory(parent)
    for component in ("native", "knowhere", ABI_VERSION):
        parent = parent / component
        create_directory(parent)
    destination = parent / target_platform
    lock_path = parent / ("." + target_platform + ".bundle.lock")
    # O_NOFOLLOW prevents a preexisting lock symlink from modifying another file.
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(lock_fd, "r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        exists = destination.exists() or destination.is_symlink()
        if exists:
            validate_owned_directory(destination, target_platform)
        with tempfile.TemporaryDirectory(prefix="." + target_platform + ".bundle-", dir=parent) as temporary:
            staging = Path(temporary)
            properties = ["cAbiVersion=" + ABI_VERSION, "platform=" + target_platform, "libraries=" + ",".join(order)]
            license_files = {}
            missing_licenses = []
            missing_license_details = []
            for name in order:
                library = closure.libraries[name]
                copied = staging / name
                shutil.copyfile(library.source, copied)
                if digest(copied) != library.source_sha256:
                    raise BundleError("Source library changed during packaging: %s" % library.source)
                command(["patchelf", "--set-rpath", "$ORIGIN", str(copied)])
                after = inspect_elf(copied, target_platform)
                if after != library.elf:
                    raise BundleError("Patchelf changed DT_NEEDED or SONAME for %s" % name)
                properties.append("sha256." + name + "=" + digest(copied))
                properties.append("sourceSha256." + name + "=" + library.source_sha256)
                copied.chmod(0o755)
                os.utime(copied, (epoch, epoch))
                licenses, missing_references = copy_licenses(library.source, name, staging, epoch)
                if licenses:
                    license_files.update(licenses)
                if not licenses or missing_references:
                    missing_licenses.append(name)
                    detail = ": " + "; ".join(missing_references) if missing_references else ""
                    missing_license_details.append(name + detail)
            (staging / LICENSE_INDEX).write_text(json.dumps(license_files, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            (staging / MISSING_LICENSES).write_text("".join(line + "\n" for line in missing_license_details), encoding="utf-8")
            properties.append("licenseFilesSha256=" + digest(staging / LICENSE_INDEX))
            properties.append("missingLicensesSha256=" + digest(staging / MISSING_LICENSES))
            properties.append("missingLicenses=" + ",".join(missing_licenses))
            (staging / MANIFEST).write_text("\n".join(properties) + "\n", encoding="utf-8")
            (staging / OWNER_FILE).write_text(OWNER_CONTENT, encoding="utf-8")
            for name in (MANIFEST, OWNER_FILE, LICENSE_INDEX, MISSING_LICENSES):
                (staging / name).chmod(0o644)
                os.utime(staging / name, (epoch, epoch))
            _, license_directories = file_inventory(staging / "licenses")
            for relative in license_directories:
                directory = staging / "licenses" / relative
                directory.chmod(0o755)
                os.utime(directory, (epoch, epoch))
            if (staging / "licenses").is_dir():
                (staging / "licenses").chmod(0o755)
                os.utime(staging / "licenses", (epoch, epoch))
            staging.chmod(0o755)
            os.utime(staging, (epoch, epoch))
            rename_atomic(staging, destination, exchange=exists)
            # On exchange, staging now names the verified old bundle. The
            # TemporaryDirectory context removes only that owned directory.
        if missing_licenses:
            print("License files require follow-up: " + ", ".join(missing_licenses))
            print("Missing-license list: " + str(destination / MISSING_LICENSES))
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--library", required=True, type=Path, help="Absolute path to the JNI ELF shared library")
    parser.add_argument("--output", required=True, type=Path, help="Absolute native resources root")
    parser.add_argument("--platform", required=True, choices=sorted(PLATFORMS))
    arguments = parser.parse_args()
    try:
        if sys.platform != "linux":
            raise BundleError("ELF native bundling requires Linux")
        if not arguments.library.is_absolute() or not arguments.output.is_absolute():
            raise BundleError("--library and --output must be absolute paths")
        for tool in ("readelf", "ldd", "patchelf"):
            if shutil.which(tool) is None:
                raise BundleError("Missing required tool: " + tool)
        closure = Closure(arguments.library, arguments.platform)
        destination = publish(closure, arguments.output, arguments.platform)
        print("Bundled %d libraries into %s" % (len(closure.libraries), destination))
        return 0
    except (BundleError, OSError, UnicodeError) as error:
        print("Native bundle failed: %s" % error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
