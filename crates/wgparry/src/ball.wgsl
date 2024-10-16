#import wgebra::sim3 as Pose
#import wgparry::ray as Ray

#define_import_path wgparry::ball


/// A ball, defined by its radius.
struct Ball {
    /// The ball’s radius.
    radius: f32,
}

/// Casts a ray on a ball.
///
/// Returns a negative value if there is no hit.
/// If there is a hit, the result is a scalar `t >= 0` such that the hit point is equal to `ray.origin + t * ray.dir`.
fn castLocalRay(ball: Ball, ray: Ray::Ray, maxTimeOfImpact: f32) -> f32 {
    // Ray origin relative to the ball’s center. It’s the origin itself since it’s in the ball’s local frame.
    let dcenter = ray.origin;
    let a = dot(ray.dir, ray.dir);
    let b = dot(dcenter, ray.dir);
    let c = dot(dcenter, dcenter) - ball.radius * ball.radius;
    let delta = b * b - a * c;
    let t = -b - sqrt(delta);

    if (c > 0.0 && (b > 0.0 || a == 0.0)) || delta < 0.0 || t > maxTimeOfImpact * a {
        // No hit.
        return -1.0;
    } else if a == 0.0 {
        // Dir is zero but the ray started inside the ball.
        return 0.0;
    } else {
        // Hit. If t <= 0, the origin is inside the ball.
        return max(t / a, 0.0);
    }
}

/// Casts a ray on a transformed ball.
///
/// Returns a negative value if there is no hit.
/// If there is a hit, the result is a scalar `t >= 0` such that the hit point is equal to `ray.origin + t * ray.dir`.
fn castRay(ball: Ball, pose: Pose::Sim3, ray: Ray::Ray, maxTimeOfImpact: f32) -> f32 {
    let localRay = Ray::Ray(Pose::invMulPt(pose, ray.origin), Pose::invMulVec(pose, ray.dir));
    return castLocalRay(ball, localRay, maxTimeOfImpact);
}

/// Projects a point on a ball.
///
/// If the point is inside the ball, the point itself is returned.
fn projectLocalPoint(ball: Ball, pt: vec3<f32>) -> vec3<f32> {
    let dist = length(pt);

    if dist >= ball.radius {
        // The point is outside the ball.
        return pt * (ball.radius / dist);
    } else {
        // The point is inside the ball.
        return pt;
    }
}

/// Projects a point on a transformed ball.
///
/// If the point is inside the ball, the point itself is returned.
fn projectPoint(ball: Ball, pose: Pose::Sim3, pt: vec3<f32>) -> vec3<f32> {
    let localPt = Pose::invMulPt(pose, pt);
    return Pose::mulPt(pose, projectLocalPoint(ball, localPt));
}
