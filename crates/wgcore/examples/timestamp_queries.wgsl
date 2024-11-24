@group(0) @binding(0)
var<storage, read_write> a: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let i = invocation_id.x;
    if i < arrayLength(&a) {
        const NUM_ITERS: u32 = 10000u;
        for (var k = 0u; k < NUM_ITERS; k++) {
            a[i] = collatz_iterations(a[i] * 7919);
        }
    }
}

// This is taken from the wgpu "hello_compute" example:
// https://github.com/gfx-rs/wgpu/blob/6f5014f0a3441bcbc3eb4223aee454b95904b087/examples/src/hello_compute/shader.wgsl
// (Apache 2 / MIT license)
//
// The Collatz Conjecture states that for any integer n:
// If n is even, n = n/2
// If n is odd, n = 3n+1
// And repeat this process for each new n, you will always eventually reach 1.
// Though the conjecture has not been proven, no counterexample has ever been found.
// This function returns how many times this recurrence needs to be applied to reach 1.
fn collatz_iterations(n_base: u32) -> u32{
    var n: u32 = n_base;
    var i: u32 = 0u;
    loop {
        if (n <= 1u) {
            break;
        }
        if (n % 2u == 0u) {
            n = n / 2u;
        }
        else {
            // Overflow? (i.e. 3*n + 1 > 0xffffffffu?)
            if (n >= 1431655765u) {   // 0x55555555u
                return 4294967295u;   // 0xffffffffu
            }

            n = 3u * n + 1u;
        }
        i = i + 1u;
    }
    return i;
}