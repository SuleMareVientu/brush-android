use ball_tree::BallTree;
use brush_render::{
    bounding_box::BoundingBox,
    camera::Camera,
    gaussian_splats::{SplatRenderMode, Splats, inverse_sigmoid},
};
use brush_serde::SplatData;
use burn::{config::Config, tensor::Device};
use glam::Vec3;
use rand::{Rng, RngExt};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use tracing::trace_span;

#[derive(Config, Debug)]
pub struct RandomSplatsConfig {
    #[config(default = 10)]
    pub init_count: usize,
}

/// Estimate scene scale from camera positions.
///
/// Uses the average nearest-neighbor distance between cameras,
/// with a minimum of 1.0 (1 meter baseline).
#[allow(dead_code)]
fn estimate_scene_scale(cameras: &[Camera]) -> f32 {
    if cameras.len() < 2 {
        return 1.0;
    }

    let mut total_nn_dist = 0.0f32;
    for (i, cam) in cameras.iter().enumerate() {
        let mut min_dist = f32::INFINITY;
        for (j, other) in cameras.iter().enumerate() {
            if i != j {
                let d = cam.position.distance(other.position);
                if d < min_dist {
                    min_dist = d;
                }
            }
        }
        total_nn_dist += min_dist;
    }

    let avg_nn = total_nn_dist / cameras.len() as f32;
    // Scene depth is roughly a few multiples of the camera spacing.
    // Use 3x the average spacing, with 1m floor.
    (avg_nn * 3.0).max(1.0)
}

/// Create initial splats by sampling random points inside camera frustums.
///
/// For each splat, a random camera is chosen, then a random ray direction
/// within its field of view is sampled, and a random depth along that ray
/// is picked.
pub fn create_random_splats(
    config: &RandomSplatsConfig,
    cameras: &[Camera],
    _scene_scale_override: Option<f32>,
    rng: &mut impl Rng,
    mode: SplatRenderMode,
    device: &Device,
) -> Splats {
    let num_points = config.init_count;

    // SLV Initialization: bounds from camera positions
    let mut min_pos = Vec3::splat(f32::INFINITY);
    let mut max_pos = Vec3::splat(f32::NEG_INFINITY);
    for cam in cameras {
        min_pos = min_pos.min(cam.position);
        max_pos = max_pos.max(cam.position);
    }
    
    // If there are no cameras or only one, default to a 1x1x1 box
    if min_pos.x == f32::INFINITY {
        min_pos = Vec3::splat(-1.0);
        max_pos = Vec3::splat(1.0);
    } else if min_pos == max_pos {
        min_pos -= Vec3::splat(1.0);
        max_pos += Vec3::splat(1.0);
    }

    let camera_dist = max_pos - min_pos;
    let bounds_extent = camera_dist.max_element();
    // Large variance for low-frequency structure
    let large_scale = (bounds_extent * 1.5).ln();

    let center = (min_pos + max_pos) / 2.0;
    let half_size = camera_dist * 1.5;
    let sample_min = center - half_size;
    let sample_max = center + half_size;

    // Sample points in 3x the camera bounding box
    let positions: Vec<f32> = (0..num_points)
        .flat_map(|_| {
            [
                rng.random_range(sample_min.x..sample_max.x),
                rng.random_range(sample_min.y..sample_max.y),
                rng.random_range(sample_min.z..sample_max.z),
            ]
        })
        .collect();

    // Random colors
    let sh_coeffs: Vec<f32> = (0..num_points)
        .flat_map(|_| {
            [
                rng.random_range(0.0..1.0),
                rng.random_range(0.0..1.0),
                rng.random_range(0.0..1.0),
            ]
        })
        .collect();

    // Random rotations (normalized quaternions)
    let rotations: Vec<f32> = (0..num_points)
        .flat_map(|_| {
            let x: f32 = rng.random_range(-1.0..1.0);
            let y: f32 = rng.random_range(-1.0..1.0);
            let z: f32 = rng.random_range(-1.0..1.0);
            let w: f32 = rng.random_range(-1.0..1.0);
            let len = (x * x + y * y + z * z + w * w).sqrt().max(1e-6);
            [x / len, y / len, z / len, w / len]
        })
        .collect();

    // Random opacities
    let opacities: Vec<f32> = (0..num_points)
        .map(|_| rng.random_range(inverse_sigmoid(0.1)..inverse_sigmoid(0.25)))
        .collect();

    // SLV: Scale is initialized to a large variance
    let log_scales: Vec<f32> = vec![large_scale; num_points * 3];

    Splats::from_raw(
        positions, rotations, log_scales, sh_coeffs, opacities, mode, device,
    )
}

pub fn bounds_from_pos(percentile: f32, means: &[f32]) -> BoundingBox {
    let (mut x_vals, mut y_vals, mut z_vals): (Vec<f32>, Vec<f32>, Vec<f32>) = means
        .chunks_exact(3)
        .map(|chunk| (chunk[0], chunk[1], chunk[2]))
        .collect();
    x_vals.retain(|x| x.is_finite());
    y_vals.retain(|y| y.is_finite());
    z_vals.retain(|z| z.is_finite());

    // If any axis is entirely non-finite, fall back to a unit box rather
    // than panicking on the percentile index.
    if x_vals.is_empty() || y_vals.is_empty() || z_vals.is_empty() {
        return BoundingBox::from_min_max(Vec3::splat(-1.0), Vec3::splat(1.0));
    }

    x_vals.sort_by(|a, b| a.total_cmp(b));
    y_vals.sort_by(|a, b| a.total_cmp(b));
    z_vals.sort_by(|a, b| a.total_cmp(b));

    let pick = |vals: &[f32]| -> (f32, f32) {
        let n = vals.len();
        let lo = ((1.0 - percentile) / 2.0 * n as f32) as usize;
        let hi = (n - 1).min(((1.0 + percentile) / 2.0 * n as f32) as usize);
        (vals[lo], vals[hi])
    };

    let (xmin, xmax) = pick(&x_vals);
    let (ymin, ymax) = pick(&y_vals);
    let (zmin, zmax) = pick(&z_vals);
    BoundingBox::from_min_max(Vec3::new(xmin, ymin, zmin), Vec3::new(xmax, ymax, zmax))
}

#[derive(PartialEq, Clone, Copy, Debug)]
struct BallPoint(glam::Vec3A);

impl ball_tree::Point for BallPoint {
    fn distance(&self, other: &Self) -> f64 {
        self.0.distance(other.0) as f64
    }

    fn move_towards(&self, other: &Self, d: f64) -> Self {
        Self(self.0.lerp(other.0, d as f32 / self.0.distance(other.0)))
    }

    fn midpoint(a: &Self, b: &Self) -> Self {
        Self((a.0 + b.0) / 2.0)
    }
}

/// Compute scales using KNN based on point density.
fn compute_knn_scales(pos_data: &[f32]) -> Vec<f32> {
    let _ = trace_span!("compute_knn_scales").entered();

    let n_splats = pos_data.len() / 3;

    if n_splats < 3 {
        return vec![0.0; n_splats * 3];
    }

    let bounding_box = trace_span!("Bounds from pose").in_scope(|| bounds_from_pos(0.75, pos_data));
    let median_size = bounding_box.median_size().max(0.01);

    trace_span!("Splats KNN scale init").in_scope(|| {
        let tree_points: Vec<BallPoint> = pos_data
            .as_chunks::<3>()
            .0
            .iter()
            .map(|v| BallPoint(glam::Vec3A::new(v[0], v[1], v[2])))
            .collect();

        let empty = vec![(); tree_points.len()];
        let tree = BallTree::new(tree_points.clone(), empty);

        tree_points
            .par_iter()
            .map_with(tree.query(), |query, p| {
                // Get half of the average of 2 nearest distances.
                let mut q = query.nn(p).skip(1);
                let a1 = q.next().unwrap().1 as f32;
                let a2 = q.next().unwrap().1 as f32;
                let dist = (a1 + a2) / 4.0;
                dist.clamp(1e-3, median_size * 0.1).ln()
            })
            .flat_map(|p| [p, p, p])
            .collect()
    })
}

pub fn to_init_splats(data: SplatData, mode: SplatRenderMode, device: &Device) -> Splats {
    let n_splats = data.num_splats();

    // Use KNN for scales if not provided
    let log_scales = data
        .log_scales
        .unwrap_or_else(|| compute_knn_scales(&data.means));

    // Default rotation = identity quaternion [1, 0, 0, 0]
    let rotations = data
        .rotations
        .unwrap_or_else(|| [1.0, 0.0, 0.0, 0.0].repeat(n_splats));

    // Default opacity = inverse_sigmoid(0.5)
    let opacities = data
        .raw_opacities
        .unwrap_or_else(|| vec![inverse_sigmoid(0.5); n_splats]);

    // Default SH coeffs = gray (0.5)
    let sh_coeffs = data.sh_coeffs.unwrap_or_else(|| vec![0.5; n_splats * 3]);

    Splats::from_raw(
        data.means, rotations, log_scales, sh_coeffs, opacities, mode, device,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_from_pos_all_nan_does_not_panic() {
        let means = vec![f32::NAN; 30];
        let bb = bounds_from_pos(0.8, &means);
        // We expect a finite fallback — no NaN leak, no panic.
        assert!(bb.center.is_finite(), "center: {:?}", bb.center);
        assert!(bb.extent.is_finite(), "extent: {:?}", bb.extent);
    }

    #[test]
    fn bounds_from_pos_empty_does_not_panic() {
        let bb = bounds_from_pos(0.8, &[]);
        assert!(bb.center.is_finite());
        assert!(bb.extent.is_finite());
    }

    #[test]
    fn bounds_from_pos_mixed_nan_and_finite() {
        // Half NaN, half finite. The finite half should determine the bounds.
        let mut means = Vec::new();
        for i in 0..100 {
            if i % 2 == 0 {
                means.extend_from_slice(&[f32::NAN, f32::NAN, f32::NAN]);
            } else {
                means.extend_from_slice(&[i as f32, i as f32, i as f32]);
            }
        }
        let bb = bounds_from_pos(0.8, &means);
        assert!(bb.center.is_finite());
        assert!(bb.extent.is_finite());
        // Extent should be reasonable (the finite values span 1..99).
        assert!(bb.extent.x > 0.0 && bb.extent.x < 100.0);
    }

    #[test]
    fn bounds_from_pos_one_axis_all_nan() {
        // x and z are OK, y is all NaN — we must not panic indexing into y.
        let mut means = Vec::new();
        for i in 0..50 {
            means.extend_from_slice(&[i as f32, f32::NAN, i as f32]);
        }
        let bb = bounds_from_pos(0.8, &means);
        // y axis collapses to the fallback, other axes should still be
        // reasonable.
        assert!(bb.center.is_finite());
        assert!(bb.extent.is_finite());
    }
}
