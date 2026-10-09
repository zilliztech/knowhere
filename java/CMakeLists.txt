find_package(Java 11 REQUIRED COMPONENTS Development)
find_package(JNI REQUIRED COMPONENTS JVM)
file(GLOB JAVA_BINDING_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/main/java/io/knowhere/*.java")
set(JNI_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/io_knowhere_NativeBindings.h")
add_custom_command(OUTPUT "${JNI_HEADER}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated" "${CMAKE_CURRENT_BINARY_DIR}/classes"
  COMMAND ${Java_JAVAC_EXECUTABLE} --release 11 -h "${CMAKE_CURRENT_BINARY_DIR}/generated"
          -d "${CMAKE_CURRENT_BINARY_DIR}/classes" ${JAVA_BINDING_SOURCES}
  DEPENDS ${JAVA_BINDING_SOURCES}
  VERBATIM)
add_custom_target(knowhere_jni_headers DEPENDS "${JNI_HEADER}")
add_library(knowhere_jni SHARED src/main/cpp/knowhere_jni.cc "${JNI_HEADER}")
add_dependencies(knowhere_jni knowhere_jni_headers)
target_include_directories(knowhere_jni PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated" ${JNI_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(knowhere_jni PRIVATE knowhere_c)
set_target_properties(knowhere_jni PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)
if(APPLE)
  set_target_properties(knowhere_jni PROPERTIES INSTALL_RPATH "@loader_path")
else()
  set_target_properties(knowhere_jni PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()
install(TARGETS knowhere_jni LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
