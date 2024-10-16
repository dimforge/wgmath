#define_import_path wgebra::rot2


/// Compact representation of a 2D rotation.
struct Rot2 {
    cosSin: vec2<f32>
}

/// Initializes a 2D rotation from an angle (radians).
fn fromAngle(angle: f32) -> Rot2 {
    return Rot2(vec2(cos(angle), sin(angle)));
}

/// The quaternion representing an identity rotation.
fn identity() -> Rot2 {
    return Rot2(vec2(1.0, 0.0));
}

/// The inverse of a 2d rotation.
fn inv(r: Rot2) -> Rot2 {
    return Rot2(vec2(r.cosSin.x, -r.cosSin.y));
}

/// Multiplication of two 2D rotations.
fn mul(lhs: Rot2, rhs: Rot2) -> Rot2 {
    let new_cos = lhs.cosSin.x * rhs.cosSin.x - lhs.cosSin.y * rhs.cosSin.y;
    let new_sin = lhs.cosSin.y * rhs.cosSin.x + lhs.cosSin.x * rhs.cosSin.y;
    return Rot2(vec2(new_cos, new_sin));
}

/// Multiplies a 2D rotation by a vector (rotates the vector).
fn mulVec(r: Rot2, v: vec2<f32>) -> vec2<f32> {
    return vec2(r.cosSin.x * v.x - r.cosSin.y * v.y, r.cosSin.y * v.x + r.cosSin.x * v.y);
}

/// Multiplies the inverse of a 2D rotation by a vector (applies inverse rotation to the vector).
fn invMulVec(r: Rot2, v: vec2<f32>) -> vec2<f32> {
    return vec2(r.cosSin.x * v.x + r.cosSin.y * v.y, -r.cosSin.y * v.x + r.cosSin.x * v.y);
}