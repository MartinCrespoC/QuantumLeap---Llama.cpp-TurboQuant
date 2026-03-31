# ggml-expertflow backend

if (NOT LLAMA_EXPERTFLOW)
    return()
endif()

add_library(ggml-expertflow STATIC
    ggml-expertflow.cpp
    ggml-expertflow.h
)

target_include_directories(ggml-expertflow PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../core/include
)

target_link_libraries(ggml-expertflow PUBLIC
    ggml
)

# Link ExpertFlow library if available
if (TARGET expertflow)
    target_link_libraries(ggml-expertflow PUBLIC expertflow)
endif()

target_compile_definitions(ggml-expertflow PUBLIC
    LLAMA_EXPERTFLOW=1
)

set_target_properties(ggml-expertflow PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)
