# TurboQuant: Fast Dequantization Assembly (AVX2)
# Fallback for CPUs without AVX-512 (e.g., AMD Ryzen 7 3700X)
#
# void dequant_int2_avx2(
#     const uint8_t* packed,   // rdi - packed INT2 data (4 vals/byte)
#     const float* scales,     // rsi - per-group scale factors
#     float* output,           // rdx - output FP32 buffer
#     size_t n                 // rcx - number of output elements
# )
#
# Processes 32 output values per iteration (8 bytes → 32 floats).
# Uses YMM registers (256-bit) instead of ZMM (512-bit).

.intel_syntax noprefix
.text
.globl dequant_int2_avx2
.type  dequant_int2_avx2, @function
.align 32

dequant_int2_avx2:
    push    rbx                     # Save callee-saved registers
    push    r12

    mov     r12, rcx                # r12 = n (element count)
    shr     r12, 5                  # r12 = n / 32 (iterations, 32 elements each)
    test    r12, r12                # Check if any full iterations
    jz      .Ldq2a_done

    # Constant: 0x03 mask for extracting 2-bit values
    vpbroadcastd ymm7, [rip + .Ldq2a_mask03]
    # Constant: 1.5f for centering (subtract to get -1.5, -0.5, 0.5, 1.5)
    vbroadcastss ymm6, [rip + .Ldq2a_center]

    xor     rbx, rbx                # rbx = group counter for scales

.align 32
.Ldq2a_loop:
    # Load 8 packed bytes = 32 INT2 values
    vmovq   xmm0, [rdi]            # xmm0[63:0] = 8 bytes of packed INT2

    # Broadcast scale for current group
    vbroadcastss ymm5, [rsi + rbx * 4]

    # ---- Unpack: first 8 values (bits [1:0] of each byte) ----
    vpmovzxbd ymm1, xmm0           # Zero-extend 8 bytes to 8 dwords
    vpand   ymm2, ymm1, ymm7       # ymm2 = val & 0x03

    vcvtdq2ps ymm2, ymm2           # Convert INT32 → FP32
    vsubps  ymm2, ymm2, ymm6       # Center around 0: val - 1.5
    vmulps  ymm2, ymm2, ymm5       # Scale: val * scale
    vmovups [rdx], ymm2            # Store 8 floats (32 bytes)

    # ---- Shift right 2 bits, extract next 8 values ----
    vpsrld  ymm3, ymm1, 2          # Shift each dword right by 2
    vpand   ymm3, ymm3, ymm7       # Mask to 2 bits

    vcvtdq2ps ymm3, ymm3
    vsubps  ymm3, ymm3, ymm6
    vmulps  ymm3, ymm3, ymm5
    vmovups [rdx + 32], ymm3       # Store next 8 floats

    # ---- Shift right 4 bits, extract next 8 values ----
    vpsrld  ymm4, ymm1, 4
    vpand   ymm4, ymm4, ymm7

    vcvtdq2ps ymm4, ymm4
    vsubps  ymm4, ymm4, ymm6
    vmulps  ymm4, ymm4, ymm5
    vmovups [rdx + 64], ymm4       # Store next 8 floats

    # ---- Shift right 6 bits, extract last 8 values ----
    vpsrld  ymm1, ymm1, 6          # Reuse ymm1
    vpand   ymm1, ymm1, ymm7

    vcvtdq2ps ymm1, ymm1
    vsubps  ymm1, ymm1, ymm6
    vmulps  ymm1, ymm1, ymm5
    vmovups [rdx + 96], ymm1       # Store last 8 floats

    # Advance pointers
    add     rdi, 8                  # 8 packed bytes consumed
    add     rdx, 128                # 32 floats * 4 bytes = 128 bytes produced
    inc     rbx                     # Next scale group

    dec     r12
    jnz     .Ldq2a_loop

.Ldq2a_done:
    vzeroupper                      # Clear upper YMM state (AVX-SSE transition)
    pop     r12
    pop     rbx
    ret

# ============================================
# void dequant_int4_avx2(
#     const uint8_t* packed,   // rdi - packed INT4 data (2 vals/byte)
#     const float* scales,     // rsi - per-group scale factors
#     float* output,           // rdx - output FP32 buffer
#     size_t n                 // rcx - number of output elements
# )
# Processes 16 output values per iteration (8 bytes → 16 floats).
# ============================================
.globl dequant_int4_avx2
.type  dequant_int4_avx2, @function
.align 32

dequant_int4_avx2:
    push    rbx
    push    r12

    mov     r12, rcx
    shr     r12, 4                  # n / 16 (16 elements per iteration)
    test    r12, r12
    jz      .Ldq4a_done

    # Constant: 0x0F mask for INT4
    vpbroadcastd ymm7, [rip + .Ldq4a_mask0f]
    # Constant: 7.5f for centering
    vbroadcastss ymm6, [rip + .Ldq4a_center]

    xor     rbx, rbx

.align 32
.Ldq4a_loop:
    # Load 8 packed bytes = 16 INT4 values
    vmovq   xmm0, [rdi]

    # Broadcast scale
    vbroadcastss ymm5, [rsi + rbx * 4]

    # Extract low nibbles (even indices): 8 values
    vpmovzxbd ymm1, xmm0           # Zero-extend 8 bytes to 8 dwords
    vpand   ymm2, ymm1, ymm7       # Low nibble = val & 0x0F

    vcvtdq2ps ymm2, ymm2
    vsubps  ymm2, ymm2, ymm6       # Center around 0
    vmulps  ymm2, ymm2, ymm5       # Apply scale
    vmovups [rdx], ymm2            # Store 8 floats

    # Extract high nibbles (odd indices): 8 values
    vpsrld  ymm3, ymm1, 4          # Shift right 4 bits
    vpand   ymm3, ymm3, ymm7       # High nibble

    vcvtdq2ps ymm3, ymm3
    vsubps  ymm3, ymm3, ymm6
    vmulps  ymm3, ymm3, ymm5
    vmovups [rdx + 32], ymm3       # Store next 8 floats

    # Advance
    add     rdi, 8                  # 8 bytes consumed
    add     rdx, 64                 # 16 floats * 4 bytes
    inc     rbx

    dec     r12
    jnz     .Ldq4a_loop

.Ldq4a_done:
    vzeroupper
    pop     r12
    pop     rbx
    ret

# ============================================
# Constants (shared, aligned for cache)
# ============================================
.section .rodata
.align 32

.Ldq2a_mask03:
    .long 0x00000003

.Ldq2a_center:
    .float 1.5

.Ldq4a_mask0f:
    .long 0x0000000F

.Ldq4a_center:
    .float 7.5
