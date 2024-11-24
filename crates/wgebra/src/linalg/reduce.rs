use crate::linalg::shape::Shape;
use bytemuck::Pod;
use naga_oil::compose::{ComposerError, NagaModuleDescriptor};
use naga_oil::redirect::Redirector;
use nalgebra::DVector;
use wgcore::kernel::{KernelInvocationBuilder, KernelInvocationQueue};
use wgcore::tensor::{GpuScalar, GpuVectorView};
use wgcore::utils;
use wgcore::Shader;
use wgpu::{ComputePipeline, Device};

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[non_exhaustive]
/// The desired operation for the [`Reduce`] kernel.
pub enum ReduceOp {
    /// Minimum: `result = min(input[0], min(input[1], ...))`
    Min,
    /// Maximum: `result = max(input[0], max(input[1], ...))`
    Max,
    /// Sum: `result = input[0] + input[1] ...`
    Sum,
    /// Product: `result = input[0] * input[1] ...`
    Prod,
    /// Squared norm: `result = input[0] * input[0] + input[1] * input[1] ...`
    SqNorm,
}

impl ReduceOp {
    const fn init_fn(self) -> &'static str {
        match self {
            Self::Min => "init_max_f32",
            Self::Max => "init_min_f32",
            Self::Sum => "init_zero",
            Self::Prod => "init_one",
            Self::SqNorm => "init_zero",
        }
    }

    const fn workspace_fn(self) -> &'static str {
        match self {
            Self::Min => "reduce_min_f32",
            Self::Max => "reduce_max_f32",
            Self::Sum => "reduce_sum_f32",
            Self::Prod => "reduce_prod_f32",
            Self::SqNorm => "reduce_sqnorm_f32",
        }
    }

    const fn reduce_fn(self) -> &'static str {
        match self {
            Self::Min => "reduce_min_f32",
            Self::Max => "reduce_max_f32",
            Self::Sum => "reduce_sum_f32",
            Self::Prod => "reduce_prod_f32",
            Self::SqNorm => "reduce_sum_f32", // reduce_sqnorm only happens in workspace
        }
    }
}

/// A GPU kernel for performing the operation described by [`ReduceOp`].
pub struct Reduce(pub ComputePipeline, pub ReduceOp);

impl Reduce {
    /// WGSL source file for `Reduce`.
    pub const SRC: &'static str = include_str!("reduce.wgsl");
    /// The WGSL file path.
    pub const FILE_PATH: &'static str = "wgebra/src/reduce.wgsl";

    /// Creates the compute pipeline for the operation described by the given [`ReduceOp`].
    pub fn new(device: &Device, op: ReduceOp) -> Result<Self, ComposerError> {
        let module = Shape::composer()?.make_naga_module(NagaModuleDescriptor {
            source: Self::SRC,
            file_path: Self::FILE_PATH,
            ..Default::default()
        })?;
        let mut redirector = Redirector::new(module);
        redirector
            .redirect_function(
                "workspace_placeholder",
                op.workspace_fn(),
                &Default::default(),
            )
            .unwrap();
        redirector
            .redirect_function("init_placeholder", op.init_fn(), &Default::default())
            .unwrap();
        redirector
            .redirect_function("reduce_placeholder", op.reduce_fn(), &Default::default())
            .unwrap();

        Ok(Self(
            utils::load_module(device, "main", redirector.into_module().unwrap()),
            op,
        ))
    }

    /// Queues the operation for computing `result = reduce(value)` where `reduce` depends on the
    /// [`ReduceOp`] selected when creating `self`.
    pub fn queue<'a, 'b, T: Pod>(
        &'a self,
        queue: &mut KernelInvocationQueue<'a>,
        value: impl Into<GpuVectorView<'b, T>>,
        result: &GpuScalar<T>,
    ) {
        let value = value.into();
        let shape_buf = queue.shape_buffer(value.shape());
        KernelInvocationBuilder::new(queue, &self.0)
            .bind0([&shape_buf, value.buffer(), result.buffer()])
            .queue(1);
    }

    #[doc(hidden)]
    pub fn eval_cpu(&self, val: &DVector<f32>) -> f32 {
        match self.1 {
            ReduceOp::Min => val.min(),
            ReduceOp::Max => val.max(),
            ReduceOp::Prod => val.product(),
            ReduceOp::Sum => val.sum(),
            ReduceOp::SqNorm => val.norm_squared(),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::ops::reduce::ReduceOp;
    use nalgebra::DVector;
    use wgcore::gpu::GpuInstance;
    use wgcore::kernel::KernelInvocationQueue;
    use wgcore::tensor::TensorBuilder;
    use wgpu::BufferUsages;

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_reduce() {
        let gpu = GpuInstance::new().await.unwrap();

        let ops = [
            ReduceOp::Min,
            ReduceOp::Max,
            ReduceOp::Sum,
            ReduceOp::SqNorm,
            ReduceOp::Prod,
        ];

        for op in ops {
            println!("Testing: {:?}", op);
            let reduce = super::Reduce::new(gpu.device(), op);
            let mut queue = KernelInvocationQueue::new(gpu.device());
            let mut encoder = gpu.device().create_command_encoder(&Default::default());

            const LEN: usize = 345;
            let numbers: DVector<f32> = DVector::new_random(LEN);

            let vector = TensorBuilder::vector(numbers.len() as u32, BufferUsages::STORAGE)
                .build_init(gpu.device(), numbers.as_slice());
            let result = TensorBuilder::scalar(BufferUsages::STORAGE | BufferUsages::COPY_SRC)
                .build(gpu.device());
            let staging = TensorBuilder::scalar(BufferUsages::MAP_READ | BufferUsages::COPY_DST)
                .build(gpu.device());

            reduce.queue(&mut queue, &vector, &result);

            queue.encode(&mut encoder, None);
            staging.copy_from(&mut encoder, &result);
            gpu.queue().submit(Some(encoder.finish()));

            approx::assert_relative_eq!(
                staging.read(gpu.device()).await.unwrap()[0],
                reduce.eval_cpu(&numbers),
                epsilon = 1.0e-3
            );
        }
    }
}
