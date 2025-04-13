#define_import_path wgebra::min_max

fn max2(v: vec2<f32>) -> f32 {
    return max(v.x, v.y);
}

fn amax2x2(m: mat2x2<f32>) -> f32 {
    let vm = max(abs(m.x), abs(m.y));
    return max(vm.x, vm.y);
}

fn max2x2(m: mat2x2<f32>) -> f32 {
    let vm = max(m.x, m.y);
    return max(vm.x, vm.y);
}

fn max3(v: vec3<f32>) -> f32 {
    return max(v.x, max(v.y, v.z));
}

fn amax3x3(m: mat3x3<f32>) -> f32 {
    let vm = max(abs(m.x), max(abs(m.y), abs(m.z)));
    return max(vm.x, max(vm.y, vm.z));
}

fn max3x3(m: mat3x3<f32>) -> f32 {
    let vm = max(m.x, max(m.y, m.z));
    return max(vm.x, max(vm.y, vm.z));
}

fn max4(v: vec4<f32>) -> f32 {
    return max(v.x, max(v.y, max(v.z, v.w)));
}

fn amax4x4(m: mat4x4<f32>) -> f32 {
    let vm = max(abs(m.x), max(abs(m.y), max(abs(m.z), abs(m.w))));
    return max(vm.x, max(vm.y, max(vm.z, vm.w)));
}

fn max4x4(m: mat4x4<f32>) -> f32 {
    let vm = max(m.x, max(m.y, max(m.z, m.w)));
    return max(vm.x, max(vm.y, max(vm.z, vm.w)));
}