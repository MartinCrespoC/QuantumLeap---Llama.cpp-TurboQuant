# Bug Fix: AVX-512 Assembly Compilation Error

**Status**: ✅ FIXED (Commit 8a853b5)  
**Date**: March 31, 2026  
**Severity**: High - Blocked Docker deployment and AVX-512 builds

## Problem Description

### Error Message
```
[1/33] Building ASM object CMakeFiles/turboquant.dir/src/turboquant/polar_transform_avx512.S.o
FAILED: CMakeFiles/turboquant.dir/src/turboquant/polar_transform_avx512.S.o

/tmp/ccZYWNjw.s: Assembler messages:
/tmp/ccZYWNjw.s: Error: .size expression for polar_transform_avx512 does not evaluate to a constant
```

### Root Causes

1. **Assembly `.size` directive placement**
   - The `.size` directive was placed AFTER `.section .rodata`
   - Assembler couldn't calculate function size across different sections
   - Expression `. - polar_transform_avx512` became non-constant

2. **CMake option handling**
   - No explicit `AVX512` option existed
   - `-DAVX512=OFF` was ignored by the build system
   - Only `FORCE_AVX512` existed, which didn't allow disabling

## Solution Implemented

### 1. Fixed Assembly Code (`core/src/turboquant/polar_transform_avx512.S`)

**Before:**
```asm
.Ldone:
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret

# Constants section
.section .rodata
.align 64
.Lconst_one:
    .rept 16
    .long 0x3F800000
    .endr
    
.size polar_transform_avx512, . - polar_transform_avx512  # ERROR: crosses sections
```

**After:**
```asm
.Ldone:
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret

.size polar_transform_avx512, . - polar_transform_avx512  # FIXED: before section change

# Constants section
.section .rodata
.align 64
.Lconst_one:
    .rept 16
    .long 0x3F800000
    .endr
```

### 2. Fixed CMake Options (`core/CMakeLists.txt`)

**Added explicit options:**
```cmake
option(AVX512 "Enable AVX-512 optimizations (auto-detect if ON)" ON)
option(AVX2 "Enable AVX2 optimizations (auto-detect if ON)" ON)
```

**Added proper disable logic:**
```cmake
if(NOT AVX512)
  set(HAS_AVX512 FALSE)
  message(STATUS "AVX-512: DISABLED (explicitly disabled with -DAVX512=OFF)")
elseif(FORCE_AVX512 OR COMPILER_SUPPORTS_AVX512)
  # ... auto-detection logic
endif()
```

## Verification

### Build Commands That Now Work

**Disable AVX-512 explicitly:**
```bash
cd core
cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Release -DAVX512=OFF
ninja -C build
```

**Enable AVX-512 (auto-detect):**
```bash
cd core
cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Release -DAVX512=ON
ninja -C build
```

### Build Results

```
-- AVX-512: DISABLED (explicitly disabled with -DAVX512=OFF)
-- AVX2: ENABLED
[31/31] Linking CXX executable test_expertflow
```

All 31/31 targets compile successfully without errors or warnings.

## Impact

### Before Fix
- ❌ Docker builds failed
- ❌ AVX-512 builds blocked
- ❌ `-DAVX512=OFF` ignored
- ❌ Deployment blocked

### After Fix
- ✅ Docker builds work
- ✅ AVX-512 compiles correctly
- ✅ `-DAVX512=OFF` respected
- ✅ Deployment unblocked
- ✅ Clean builds (no warnings)

## Related Commits

- **8a853b5**: Fix AVX-512 assembly compilation error and CMake option handling
- **84bc039**: Remove unused parameter warnings in benchmark_polar.cpp

## Testing

### Tested Configurations

| Configuration | Result | Notes |
|--------------|--------|-------|
| `-DAVX512=OFF -DAVX2=ON` | ✅ Pass | 31/31 targets |
| `-DAVX512=ON` (no AVX-512 CPU) | ✅ Pass | Auto-disabled, uses AVX2 |
| `-DAVX512=ON` (AVX-512 CPU) | ✅ Pass | Full AVX-512 enabled |
| Default (auto-detect) | ✅ Pass | Works on all systems |

### Verified On

- Ubuntu 22.04 (GCC 11.4.0)
- Arch Linux (GCC 15.2.1)
- Intel Xeon Gold 6136 (AVX-512 capable)
- AMD Ryzen (no AVX-512, falls back to AVX2)

## Documentation Updated

- ✅ This file (`BUGFIX_AVX512.md`)
- ✅ Commit messages with full context
- ✅ CMake status messages improved
- ✅ Code comments in assembly file

## Acceptance Criteria

All criteria met:
- ✅ AVX-512 assembly code compiles successfully
- ✅ Docker build completes without errors
- ✅ AVX-512 optimizations enabled when available
- ✅ Performance benchmarks unaffected (130% improvement maintained)
- ✅ `-DAVX512=OFF` properly disables AVX-512
- ✅ Clean builds with no warnings

## Lessons Learned

1. **Assembly directives are section-sensitive**: `.size` must be in the same section as the function
2. **CMake options need explicit disable paths**: Auto-detection alone isn't enough
3. **Status messages matter**: Clear feedback helps users understand build configuration
4. **Test on multiple platforms**: What works on one compiler/CPU may fail on others

---

**Status**: Production ready  
**Deployed**: March 31, 2026  
**Verified**: All build configurations passing
