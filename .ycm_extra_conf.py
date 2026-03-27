# This file is NOT licensed under the GPLv3, which is the license for the rest
# of YouCompleteMe.
#
# Here's the license text for this file:
#
# This is free and unencumbered software released into the public domain.
#
# Anyone is free to copy, modify, publish, use, compile, sell, or
# distribute this software, either in source code form or as a compiled
# binary, for any purpose, commercial or non-commercial, and by any
# means.
#
# In jurisdictions that recognize copyright laws, the author or authors
# of this software dedicate any and all copyright interest in the
# software to the public domain. We make this dedication for the benefit
# of the public at large and to the detriment of our heirs and
# successors. We intend this dedication to be an overt act of
# relinquishment in perpetuity of all present and future rights to this
# software under copyright law.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#
# For more information, please refer to <http://unlicense.org/>

import os
import ycm_core


DIR_OF_THIS_SCRIPT = os.path.abspath(os.path.dirname(__file__))

flags = [
    "-Wall",
    "-Wextra",
    "-std=c++20",
    "-x",
    "c++",
    "-isystem",
    "/usr/include/c++/10",
    "-isystem",
    "/usr/include",
    "-isystem",
    "/usr/local/include",
    "-I",
    "include/",
    "-I",
    "src/",
    "-I",
    "thirdparty/",
    "-I",
    "thirdparty/faiss",
]


compilation_database_folder = DIR_OF_THIS_SCRIPT


def IsHeaderFile(filename):
    extension = os.path.splitext(filename)[1]
    return extension in [".h", ".hxx", ".hpp", ".hh"]


if os.path.exists(compilation_database_folder):
    database = ycm_core.CompilationDatabase(compilation_database_folder)
else:
    database = None


def Settings(**kwargs):
    if kwargs["language"] == "cfamily":
        # If the file is a header, try to find the corresponding source file and
        # retrieve its flags from the compilation database if using one. This is
        # necessary since compilation databases don't have entries for header files.
        # In addition, use this source file as the translation unit. This makes it
        # possible to jump from a declaration in the header file to its definition
        # in the corresponding source file.
        filename = kwargs["filename"]

        if IsHeaderFile(filename) or not database:
            return {
                "flags": flags,
                "include_paths_relative_to_dir": DIR_OF_THIS_SCRIPT,
                "override_filename": filename,
            }

        compilation_info = database.GetCompilationInfoForFile(filename)
        if not compilation_info.compiler_flags_:
            return {
                "flags": flags,
                "include_paths_relative_to_dir": DIR_OF_THIS_SCRIPT,
                "override_filename": filename,
            }

        # Bear in mind that compilation_info.compiler_flags_ does NOT return a
        # python list, but a "list-like" StringVec object.
        final_flags = list(compilation_info.compiler_flags_)
        return {
            "flags": final_flags,
            "include_paths_relative_to_dir": compilation_info.compiler_working_dir_,
            "override_filename": filename,
        }
    return {}
