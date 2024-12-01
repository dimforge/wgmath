#define_import_path composable::module

struct MyStruct {
    value: f32,
}

fn shared_function(a: MyStruct, b: MyStruct) -> MyStruct {
    // Same as compose_dependency.wgsl but with a subtraction instead of an addition.
    return MyStruct(a.value - b.value);
}
