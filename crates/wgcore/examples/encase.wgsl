@group(0) @binding(0)
var<storage, read_write> a: array<EncaseStruct>;
@group(0) @binding(1)
var<storage, read> b: array<BytemuckStruct>;

struct BytemuckStruct {
    value: f32,
}

struct EncaseStruct {
    value: f32,
    value2: vec4<f32>
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let i = invocation_id.x;
    if i < arrayLength(&a) {
        a[i].value += b[i].value;
        a[i].value2 += vec4(b[i].value);
    }
}