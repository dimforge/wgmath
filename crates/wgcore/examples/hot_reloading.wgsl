@group(0) @binding(0)
var<storage, read_write> a: u32;

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    a = 1u; // Change this value and save the file while running the `hot_reloading` example.
}