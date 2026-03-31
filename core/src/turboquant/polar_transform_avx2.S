# TurboQuant: Polar Transform Assembly (AVX2)
# Fallback for CPUs without AVX-512 (e.g., AMD Ryzen 7 3700X)
#
# void polar_transform_avx2(
#     const float* x,          // rdi - input x values
#     const float* y,          // rsi - input y values
#     float* magnitude,        // rdx - output magnitudes
#     float* angle,            // rcx - output angles (atan2)
#     size_t n                 // r8  - number of elements
# )
#
# Computes: magnitude[i] = sqrt(x[i]^2 + y[i]^2)
#           angle[i]     = atan2(y[i], x[i])  (approximated)
#
# Processes 8 elements per iteration using YMM registers (256-bit).
# atan2 is approximated using a polynomial for speed.

.intel_syntax noprefix
.text
.globl polar_transform_avx2
.type  polar_transform_avx2, @function
.align 32

polar_transform_avx2:
    push    rbx
    push    r12
    push    r13

    mov     r12, r8                 # r12 = n
    shr     r12, 3                  # r12 = n / 8 (8 elements per iter)
    test    r12, r12
    jz      .Lpolar2_done

    # Save output pointers
    mov     r13, rcx                # r13 = angle output pointer

.align 32
.Lpolar2_loop:
    # Load 8 x values and 8 y values
    vmovups ymm0, [rdi]            # ymm0 = x[i..i+7]
    vmovups ymm1, [rsi]            # ymm1 = y[i..i+7]

    # ── Magnitude: sqrt(x^2 + y^2) ──
    vmulps  ymm2, ymm0, ymm0       # ymm2 = x^2
    vfmadd231ps ymm2, ymm1, ymm1   # ymm2 = x^2 + y^2 (FMA: ymm2 += y*y)
    vsqrtps ymm3, ymm2             # ymm3 = sqrt(x^2 + y^2) = magnitude
    vmovups [rdx], ymm3            # Store magnitudes

    # ── Angle: atan2(y, x) approximation ──
    # Using fast atan2 approximation:
    # atan2(y,x) ≈ atan(y/x) with quadrant correction
    # atan(t) ≈ t * (pi/4 + 0.273 * (1 - |t|))  for |t| <= 1
    #
    # For |y| > |x|, use: atan(x/y) and subtract from pi/2

    # Compute abs values for comparison
    vandps  ymm4, ymm0, [rip + .Lpolar2_absmask] # |x|
    vandps  ymm5, ymm1, [rip + .Lpolar2_absmask] # |y|

    # Determine which is larger: swap so we divide smaller by larger
    vcmpps  ymm6, ymm5, ymm4, 14   # ymm6 = mask where |y| > |x|
    vblendvps ymm7, ymm0, ymm1, ymm6 # ymm7 = max(|x|,|y|) contrib
    vblendvps ymm4, ymm1, ymm0, ymm6 # ymm4 = min contrib (numerator)

    # Avoid division by zero: add tiny epsilon to denominator
    vbroadcastss ymm5, [rip + .Lpolar2_eps]
    vaddps  ymm7, ymm7, ymm5       # denominator + eps

    vdivps  ymm4, ymm4, ymm7       # t = min/max (|t| <= 1)

    # Polynomial approximation: atan(t) ≈ t * (pi/4 + 0.273*(1-|t|))
    vandps  ymm5, ymm4, [rip + .Lpolar2_absmask] # |t|
    vbroadcastss ymm7, [rip + .Lpolar2_one]
    vsubps  ymm5, ymm7, ymm5       # 1 - |t|
    vbroadcastss ymm7, [rip + .Lpolar2_c273]
    vmulps  ymm5, ymm5, ymm7       # 0.273 * (1 - |t|)
    vbroadcastss ymm7, [rip + .Lpolar2_pi4]
    vaddps  ymm5, ymm5, ymm7       # pi/4 + 0.273*(1-|t|)
    vmulps  ymm5, ymm4, ymm5       # atan ≈ t * (...)

    # Quadrant correction for |y| > |x|: result = pi/2 - result
    vbroadcastss ymm7, [rip + .Lpolar2_pi2]
    vsubps  ymm4, ymm7, ymm5       # pi/2 - atan
    vblendvps ymm5, ymm5, ymm4, ymm6 # Select based on |y|>|x|

    # Sign correction: negate if x < 0
    vxorps  ymm4, ymm4, ymm4       # zero
    vcmpps  ymm7, ymm0, ymm4, 1    # mask where x < 0
    vbroadcastss ymm4, [rip + .Lpolar2_pi]
    vsubps  ymm6, ymm4, ymm5       # pi - atan
    vblendvps ymm5, ymm5, ymm6, ymm7 # fix for x < 0

    # Sign correction: negate result if y < 0
    vxorps  ymm4, ymm4, ymm4
    vcmpps  ymm7, ymm1, ymm4, 1    # mask where y < 0
    vmovups ymm3, [rip + .Lpolar2_pos] # ymm3 = +1.0 (default sign)
    vmovups ymm4, [rip + .Lpolar2_neg] # ymm4 = -1.0
    vblendvps ymm4, ymm3, ymm4, ymm7  # sign = y<0 ? -1 : +1
    vmulps  ymm5, ymm5, ymm4       # Apply sign

    vmovups [r13], ymm5             # Store angles

    # Advance pointers
    add     rdi, 32                 # 8 floats * 4 bytes
    add     rsi, 32
    add     rdx, 32
    add     r13, 32

    dec     r12
    jnz     .Lpolar2_loop

.Lpolar2_done:
    vzeroupper                      # Clean AVX state
    pop     r13
    pop     r12
    pop     rbx
    ret

# ============================================
# Constants
# ============================================
.section .rodata
.align 32

.Lpolar2_absmask:
    .long 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF
    .long 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF

.Lpolar2_eps:
    .float 1.0e-10

.Lpolar2_one:
    .float 1.0

.Lpolar2_c273:
    .float 0.273

.Lpolar2_pi4:
    .float 0.7853981633974483

.Lpolar2_pi2:
    .float 1.5707963267948966

.Lpolar2_pi:
    .float 3.141592653589793

.Lpolar2_neg:
    .float -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0

.Lpolar2_pos:
    .float 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
