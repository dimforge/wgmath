#define_import_path composable::module

struct MyStruct {
    value: f32,
}

fn shared_function(a: MyStruct, b: MyStruct) -> MyStruct {
    return MyStruct(a.value + b.value);
}