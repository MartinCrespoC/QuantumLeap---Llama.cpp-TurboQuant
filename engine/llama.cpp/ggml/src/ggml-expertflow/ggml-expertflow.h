#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// ExpertFlow custom backend for MoE-aware inference
// Intercepts GGML_OP_MUL_MAT_ID operations and replaces with cache-aware dispatch

#define GGML_EXPERTFLOW_NAME "ExpertFlow"

// Backend initialization
GGML_API ggml_backend_t ggml_backend_expertflow_init(void);

// Check if ExpertFlow backend is available
GGML_API bool ggml_backend_is_expertflow(ggml_backend_t backend);

// Buffer type for ExpertFlow (wraps CPU/GPU buffers)
GGML_API ggml_backend_buffer_type_t ggml_backend_expertflow_buffer_type(void);

// Register ExpertFlow backend globally
GGML_API void ggml_backend_expertflow_register(void);

#ifdef  __cplusplus
}
#endif
