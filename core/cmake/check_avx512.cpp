#include <immintrin.h>
#include <cpuid.h>
int main() {
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return 1;
    return ((ebx >> 16) & 1) ? 0 : 1; // bit 16 = AVX-512F
}
