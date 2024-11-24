#import composable::module as Dependency

@group(0) @binding(0)
var<storage, read_write> a: array<Dependency::MyStruct>;
@group(0) @binding(1)
var<storage, read> b: array<Dependency::MyStruct>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let i = invocation_id.x;
    if i < arrayLength(&a) {
        a[i] = Dependency::shared_function(a[i], b[i]);
    }
}



