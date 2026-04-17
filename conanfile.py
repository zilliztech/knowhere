required_conan_version = ">=2.0"

from conan.tools.microsoft import is_msvc, msvc_runtime_flag
from conan.tools.build import check_min_cppstd
from conan.tools.scm import Version
from conan.tools import files
from conan import ConanFile
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.gnu import PkgConfigDeps
from conan.errors import ConanInvalidConfiguration
import os


class KnowhereConan(ConanFile):
    name = "knowhere"
    description = "Knowhere is written in C++. It is an independent project that act as Milvus's internal core"
    topics = ("vector", "simd", "ann")
    url = "https://github.com/milvus-io/knowhere"
    homepage = "https://github.com/milvus-io/knowhere"
    license = "Apache-2.0"

    settings = "os", "arch", "compiler", "build_type"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "with_cuvs": [True, False],
        "with_asan": [True, False],
        "with_diskann": [True, False],
        "with_svs": [True, False],
        "with_cardinal": [True, False],
        "with_profiler": [True, False],
        "with_ut": [True, False],
        "with_benchmark": [True, False],
        "with_coverage": [True, False],
        "with_faiss_tests": [True, False],
        "with_light": [True, False],
        "with_compile_prune": [True, False],
    }
    default_options = {
        "shared": True,
        "fPIC": False,
        "with_cuvs": False,
        "with_asan": False,
        "with_diskann": False,
        "with_svs": False,
        "with_cardinal": False,
        "with_profiler": False,
        "with_ut": False,
        "glog/*:shared": True,
        "glog/*:with_gflags": True,
        "gtest/*:build_gmock": True,
        "prometheus-cpp/*:with_pull": False,
        "with_benchmark": False,
        "with_coverage": False,
        "boost/*:without_locale": False,
        "boost/*:without_test": True,
        "boost/*:without_stacktrace": True,
        "openssl/*:shared": True,
        "openssl/*:no_apps": True,
        "gflags/*:shared": True,
        "fmt/*:header_only": False,
        "with_faiss_tests": False,
        "opentelemetry-cpp/*:with_stl": True,
        "libcurl/*:with_ssl": False,
        "with_light": False,
        "with_compile_prune": True,
        "folly/*:shared": True,
    }

    exports_sources = (
        "src/*",
        "thirdparty/*",
        "tests/ut/*",
        "include/*",
        "CMakeLists.txt",
        "*.cmake",
        "conanfile.py",
    )

    @property
    def _minimum_cpp_standard(self):
        return 20

    @property
    def _minimum_compilers_version(self):
        return {
            "gcc": "10",
            "Visual Studio": "17",
            "clang": "10",
            "apple-clang": "13",
        }

    def config_options(self):
        if self.settings.os == "Windows":
            self.options.rm_safe("fPIC")
        if self.options.with_light:
            self.options["boost"].without_locale = True
        if self.settings.os == "Macos":
            self.options["libcurl"].with_ssl = "openssl"

    def configure(self):
        if self.options.shared:
            self.options.rm_safe("fPIC")

    def requirements(self):
        self.requires("abseil/20250127.0#481edcc75deb0efb16500f511f0f0a1c")
        self.requires("boost/1.83.0#4e8a94ac1b88312af95eded83cd81ca8")
        self.requires("gflags/2.2.2#7671803f1dc19354cc90bd32874dcfda")
        self.requires("glog/0.7.1#a306e61d7b8311db8cb148ad62c48030")
        self.requires("nlohmann_json/3.11.3#ffb9e9236619f1c883e36662f944345d", force=True)
        self.requires("openssl/3.3.2#9f9f130d58e7c13e76bb8a559f0a6a8b", force=True, override=True)
        self.requires("prometheus-cpp/1.2.4#0918d66c13f97acb7809759f9de49b3f")
        self.requires("zlib/1.3.1#8045430172a5f8d56ba001b14561b4ea")
        self.requires("double-conversion/3.3.0#640e35791a4bac95b0545e2f54b7aceb")
        self.requires("xz_utils/5.4.5#fc4e36861e0a47ecd4a40a00e6d29ac8")
        self.requires("protobuf/5.27.0@milvus/dev#42f031a96d21c230a6e05bcac4bdd633", force=True, override=True)
        self.requires("lz4/1.9.4#7f0b5851453198536c14354ee30ca9ae", force=True, override=True)
        if self.settings.os == "Linux":
            self.requires("liburing/2.8", force=True, override=True)
        self.requires("fmt/11.2.0#eb98daa559c7c59d591f4720dde4cd5c", force=True, override=True)
        self.requires("libevent/2.1.12#95065aaefcd58d3956d6dfbfc5631d97")
        self.requires("grpc/1.67.1@milvus/dev#efeaa484b59bffaa579004d5e82ec4fd")
        self.requires("folly/2024.08.12.00@milvus/dev#f9b2bdf162c0ec47cb4e5404097b340d")
        self.requires("libcurl/8.10.1#a3113369c86086b0e84231844e7ed0a9", force=True, override=True)
        self.requires("simde/0.8.2#5e1edfd5cba92f25d79bf6ef4616b972")
        self.requires("xxhash/0.8.3#caa6d0af1b951c247922e38fbcebdbe6")
        if self.settings.os == "Android":
            self.requires("openblas/0.3.27")
        if not self.options.with_light:
            self.requires("opentelemetry-cpp/1.23.0@milvus/dev#11bc565ec6e82910ae8f7471da756720")
        if self.settings.os not in ["Macos", "Android"]:
            self.requires("libunwind/1.8.1#748a981ace010b80163a08867b732e71")
        if self.options.with_ut:
            self.requires("catch2/3.7.1")
        if self.options.with_benchmark:
            self.requires("gtest/1.15.0")
            self.requires("hdf5/1.14.5")
        if self.options.with_faiss_tests:
            self.requires("gtest/1.15.0")

    @property
    def _required_boost_components(self):
        return ["program_options"]

    def validate(self):
        if self.settings.compiler.get_safe("cppstd"):
            check_min_cppstd(self, self._minimum_cpp_standard)
        min_version = self._minimum_compilers_version.get(str(self.settings.compiler))
        if not min_version:
            self.output.warn(
                "{} recipe lacks information about the {} compiler support.".format(
                    self.name, self.settings.compiler
                )
            )
        else:
            if Version(self.settings.compiler.version) < min_version:
                raise ConanInvalidConfiguration(
                    "{} requires C++{} support. The current compiler {} {} does not support it.".format(
                        self.name,
                        self._minimum_cpp_standard,
                        self.settings.compiler,
                        self.settings.compiler.version,
                    )
                )

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["CMAKE_POSITION_INDEPENDENT_CODE"] = self.options.get_safe(
            "fPIC", True
        )
        # Relocatable shared lib on Macos
        tc.cache_variables["CMAKE_POLICY_DEFAULT_CMP0042"] = "NEW"
        # Honor BUILD_SHARED_LIBS from conan_toolchain (see https://github.com/conan-io/conan/issues/11840)
        tc.cache_variables["CMAKE_POLICY_DEFAULT_CMP0077"] = "NEW"

        cppstd = self.settings.compiler.get_safe("cppstd")
        if cppstd:
            if cppstd.startswith("gnu"):
                cxx_std_value = "gnu++{}".format(cppstd[3:])
            else:
                cxx_std_value = "c++{}".format(cppstd)
        else:
            cxx_std_value = "c++{}".format(self._minimum_cpp_standard)
        tc.variables["CXX_STD"] = cxx_std_value
        if is_msvc(self):
            tc.variables["MSVC_LANGUAGE_VERSION"] = cxx_std_value
            tc.variables["MSVC_ENABLE_ALL_WARNINGS"] = False
            tc.variables["MSVC_USE_STATIC_RUNTIME"] = "MT" in msvc_runtime_flag(self)
        tc.variables["WITH_ASAN"] = self.options.with_asan
        tc.variables["WITH_DISKANN"] = self.options.with_diskann
        tc.variables["WITH_SVS"] = self.options.with_svs
        tc.variables["WITH_CARDINAL"] = self.options.with_cardinal
        tc.variables["WITH_CUVS"] = self.options.with_cuvs
        tc.variables["WITH_PROFILER"] = self.options.with_profiler
        tc.variables["WITH_UT"] = self.options.with_ut
        tc.variables["WITH_BENCHMARK"] = self.options.with_benchmark
        tc.variables["WITH_COVERAGE"] = self.options.with_coverage
        tc.variables["WITH_FAISS_TESTS"] = self.options.with_faiss_tests
        tc.variables["WITH_LIGHT"] = self.options.with_light
        tc.variables["WITH_COMPILE_PRUNE"] = self.options.with_compile_prune

        # CMake 4.x removed compatibility with cmake_minimum_required < 3.5
        tc.cache_variables["CMAKE_POLICY_VERSION_MINIMUM"] = "3.5"

        # macOS: Apple Clang lacks OpenMP; point CMake to Homebrew's libomp
        if self.settings.os == "Macos":
            import subprocess
            result = subprocess.run(
                ["brew", "--prefix", "libomp"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                libomp_prefix = result.stdout.strip()
            elif os.path.isdir("/opt/homebrew/opt/libomp"):
                libomp_prefix = "/opt/homebrew/opt/libomp"
            else:
                libomp_prefix = "/usr/local/opt/libomp"
            omp_inc = f"-I{libomp_prefix}/include"
            tc.variables["OpenMP_C_FLAGS"] = f"-Xpreprocessor -fopenmp {omp_inc}"
            tc.variables["OpenMP_CXX_FLAGS"] = f"-Xpreprocessor -fopenmp {omp_inc}"
            tc.variables["OpenMP_C_LIB_NAMES"] = "omp"
            tc.variables["OpenMP_CXX_LIB_NAMES"] = "omp"
            tc.variables["OpenMP_omp_LIBRARY"] = f"{libomp_prefix}/lib/libomp.dylib"
            tc.variables["CMAKE_C_FLAGS"] = omp_inc
            tc.variables["CMAKE_CXX_FLAGS"] = omp_inc

        # Configure ccache
        tc.variables["CMAKE_CXX_COMPILER_LAUNCHER"] = "ccache"
        tc.variables["CMAKE_C_COMPILER_LAUNCHER"] = "ccache"

        tc.generate()

        deps = CMakeDeps(self)
        deps.generate()

        pc = PkgConfigDeps(self)
        pc.generate()

    def build(self):
        # files.apply_conandata_patches(self)
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()
        files.rmdir(self, os.path.join(self.package_folder, "lib", "cmake"))
        files.rmdir(self, os.path.join(self.package_folder, "lib", "pkgconfig"))

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "knowhere")
        self.cpp_info.set_property("cmake_target_name", "Knowhere::knowhere")
        self.cpp_info.set_property("pkg_config_name", "libknowhere")

        self.cpp_info.components["libknowhere"].libs = ["knowhere"]

        self.cpp_info.components["libknowhere"].requires = [
            "boost::program_options",
            "glog::glog",
            "prometheus-cpp::core",
            "prometheus-cpp::push",
        ]

        self.cpp_info.components["libknowhere"].set_property(
            "cmake_target_name", "Knowhere::knowhere"
        )
        self.cpp_info.components["libknowhere"].set_property(
            "pkg_config_name", "libknowhere"
        )
