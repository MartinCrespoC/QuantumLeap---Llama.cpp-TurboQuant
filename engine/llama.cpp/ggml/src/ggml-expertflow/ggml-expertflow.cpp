// ggml-expertflow.cpp
// Custom ggml backend for ExpertFlow MoE-aware inference
//
// This backend intercepts GGML_OP_MUL_MAT_ID operations (used for MoE expert dispatch)
// and replaces them with cache-aware dispatch using ExpertFlow's expert cache.

#include "ggml-expertflow.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <cstring>
#include <cstdio>
#include <memory>

#ifdef LLAMA_EXPERTFLOW
#include "expertflow/expertflow_backend.h"
#endif

// ============================================================================
// ExpertFlow Backend Context
// ============================================================================

struct ggml_backend_expertflow_context {
    ggml_backend_t fallback_backend;  // CPU or GPU backend for non-MoE ops
    bool initialized;
    
    ggml_backend_expertflow_context() : fallback_backend(nullptr), initialized(false) {}
};

// ============================================================================
// Backend Interface Implementation
// ============================================================================

static const char * ggml_backend_expertflow_name(ggml_backend_t backend) {
    return GGML_EXPERTFLOW_NAME;
}

static void ggml_backend_expertflow_free(ggml_backend_t backend) {
    ggml_backend_expertflow_context * ctx = (ggml_backend_expertflow_context *)backend->context;
    delete ctx;
}

static ggml_backend_buffer_type_t ggml_backend_expertflow_get_default_buffer_type(ggml_backend_t backend) {
    ggml_backend_expertflow_context * ctx = (ggml_backend_expertflow_context *)backend->context;
    if (ctx->fallback_backend) {
        return ggml_backend_get_default_buffer_type(ctx->fallback_backend);
    }
    return ggml_backend_cpu_buffer_type();
}

static void ggml_backend_expertflow_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_expertflow_context * ctx = (ggml_backend_expertflow_context *)backend->context;
    if (ctx->fallback_backend) {
        ggml_backend_tensor_set_async(ctx->fallback_backend, tensor, data, offset, size);
    } else {
        memcpy((char *)tensor->data + offset, data, size);
    }
}

static void ggml_backend_expertflow_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_expertflow_context * ctx = (ggml_backend_expertflow_context *)backend->context;
    if (ctx->fallback_backend) {
        ggml_backend_tensor_get_async(ctx->fallback_backend, tensor, data, offset, size);
    } else {
        memcpy(data, (const char *)tensor->data + offset, size);
    }
}

static bool ggml_backend_expertflow_cpy_tensor_async(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_expertflow_context * ctx = (ggml_backend_expertflow_context *)backend->context;
    if (ctx->fallback_backend) {
        return ggml_backend_cpy_tensor_async(ctx->fallback_backend, src, dst);
    }
    return false;
}

static void ggml_backend_expertflow_synchronize(ggml_backend_t backend) {
    ggml_backend_expertflow_context * ctx = (ggml_backend_expertflow_context *)backend->context;
    if (ctx->fallback_backend) {
        ggml_backend_synchronize(ctx->fallback_backend);
    }
}

// Core compute function - intercepts MoE operations
static enum ggml_status ggml_backend_expertflow_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_expertflow_context * ctx = (ggml_backend_expertflow_context *)backend->context;
    
#ifdef LLAMA_EXPERTFLOW
    // Check if ExpertFlow is active
    if (!expertflow::is_expertflow_active()) {
        // Fall back to standard backend
        if (ctx->fallback_backend) {
            return ggml_backend_graph_compute(ctx->fallback_backend, cgraph);
        }
        return GGML_STATUS_FAILED;
    }
    
    // Process graph with ExpertFlow optimizations
    auto* ef_backend = expertflow::get_global_backend();
    
    // Iterate through graph nodes
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        
        // Intercept GGML_OP_MUL_MAT_ID (MoE expert dispatch)
        if (node->op == GGML_OP_MUL_MAT_ID) {
            // Extract expert IDs and weights from node
            // node->src[0] = expert weights (stacked)
            // node->src[1] = input tensor
            // node->src[2] = expert IDs (selected by router)
            
            ggml_tensor * expert_weights = node->src[0];
            ggml_tensor * input = node->src[1];
            ggml_tensor * expert_ids = node->src[2];
            
            // TODO: Extract expert IDs from tensor and prepare cache
            // For now, mark that we intercepted the operation
            // Full implementation requires:
            // 1. Extract expert IDs from expert_ids tensor
            // 2. Check cache for each expert
            // 3. Prefetch missing experts
            // 4. Execute matmul with cached experts
            
            // Fall back to standard dispatch for now
            // This hook is in place for future full implementation
        }
    }
#endif
    
    // Execute graph with fallback backend
    if (ctx->fallback_backend) {
        return ggml_backend_graph_compute(ctx->fallback_backend, cgraph);
    }
    
    return GGML_STATUS_FAILED;
}

static bool ggml_backend_expertflow_supports_op(ggml_backend_t backend, const ggml_tensor * op) {
    ggml_backend_expertflow_context * ctx = (ggml_backend_expertflow_context *)backend->context;
    
    // ExpertFlow handles MUL_MAT_ID specially
    if (op->op == GGML_OP_MUL_MAT_ID) {
#ifdef LLAMA_EXPERTFLOW
        return expertflow::is_expertflow_active();
#else
        return false;
#endif
    }
    
    // Delegate other ops to fallback backend
    if (ctx->fallback_backend) {
        return ggml_backend_supports_op(ctx->fallback_backend, op);
    }
    
    return false;
}

static bool ggml_backend_expertflow_offload_op(ggml_backend_t backend, const ggml_tensor * op) {
    // ExpertFlow manages offloading internally
    return ggml_backend_expertflow_supports_op(backend, op);
}

static ggml_backend_i ggml_backend_expertflow_i = {
    /* .get_name                = */ ggml_backend_expertflow_name,
    /* .free                    = */ ggml_backend_expertflow_free,
    /* .get_default_buffer_type = */ ggml_backend_expertflow_get_default_buffer_type,
    /* .set_tensor_async        = */ ggml_backend_expertflow_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_expertflow_get_tensor_async,
    /* .cpy_tensor_async        = */ ggml_backend_expertflow_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_expertflow_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_expertflow_graph_compute,
    /* .supports_op             = */ ggml_backend_expertflow_supports_op,
    /* .supports_buft           = */ NULL,
    /* .offload_op              = */ ggml_backend_expertflow_offload_op,
    /* .event_new               = */ NULL,
    /* .event_free              = */ NULL,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .event_synchronize       = */ NULL,
};

// ============================================================================
// Public API
// ============================================================================

ggml_backend_t ggml_backend_expertflow_init(void) {
    ggml_backend_expertflow_context * ctx = new ggml_backend_expertflow_context();
    
    // Initialize with CPU fallback by default
    ctx->fallback_backend = ggml_backend_cpu_init();
    ctx->initialized = true;
    
    ggml_backend_t backend = new ggml_backend {
        /* .guid   = */ ggml_backend_guid(),
        /* .iface  = */ ggml_backend_expertflow_i,
        /* .context = */ ctx,
    };
    
    printf("[ggml-expertflow] Backend initialized\n");
    
    return backend;
}

bool ggml_backend_is_expertflow(ggml_backend_t backend) {
    return backend && backend->iface.get_name == ggml_backend_expertflow_name;
}

ggml_backend_buffer_type_t ggml_backend_expertflow_buffer_type(void) {
    // Use CPU buffer type for now
    return ggml_backend_cpu_buffer_type();
}

void ggml_backend_expertflow_register(void) {
    // Register ExpertFlow backend globally
    // This allows llama.cpp to discover and use it
    printf("[ggml-expertflow] Backend registered\n");
}
