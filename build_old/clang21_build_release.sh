conan install .. --build=missing -o with_diskann=True -o with_cardinal=False -o with_ut=True -o with_benchmark=True -s compiler.libcxx=libstdc++11 -s compiler.cppstd=17 -s build_type=Release --profile=clang21_release --install-folder=clang21 --output-folder=clang21
conan build .. --build-folder=clang21

