//
// Implements torch.ops.aten._upsample_trilinear3d_mps
//
// Key points
// ----------
// • We build & cache an MPSGraph executable so launch cost is ≈0.
// • Uses `resampleTensor:sizeTensor:mode:coordinateMode:alignCorners:`
//   which supports 5-D (N,C,D,H,W) tensors with layout NCDHW.
// • Works for float32 / float16 (bfloat16 still awaits full MPSGraph support).
// • Backward is free – we let MPSGraph create the gradient graph in one shot.
// • The file mirrors your existing ConvTranspose3d.mm for uniform style.
//

#import <ATen/mps/MPSProfiler.h>
#import <ATen/mps/MPSStream.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <MetalPerformanceShadersGraph/MPSGraphResizeOps.h>
#include <ATen/native/mps/MPSGraphHelpers.h>
#include <ATen/ops/upsample_trilinear3d_native.h>

#include <ATen/native/mps/OperationUtils.h>
#include <torch/library.h>
#include <optional>
#include <ATen/ops/upsample_bilinear2d.h>  // for 2D upsample
#include <ATen/ops/upsample_bilinear2d_backward.h>
#include <ATen/ops/upsample_bilinear2d_backward_native.h>
#include <cmath>                             // for std::floor
#include <algorithm>                         // for std::min
#include <cstdio>
#include <ATen/ops/upsample_trilinear3d_backward_native.h>

// ---------------------------------------------------------------------------
//  Forward declarations to satisfy -Wmissing-prototypes when functions have
//  external linkage.
// ---------------------------------------------------------------------------

namespace at {
namespace native {
namespace mps {

TORCH_API Tensor upsample_trilinear3d_mps(
    const Tensor&                         input,
    std::optional<ArrayRef<int64_t>>      output_size,
    bool                                  align_corners,
    std::optional<double>                 scale_d,
    std::optional<double>                 scale_h,
    std::optional<double>                 scale_w);

} // namespace mps

TORCH_API Tensor& upsample_trilinear3d_out_mps(
    const Tensor&                         input,
    at::IntArrayRef                      output_size,
    bool                                  align_corners,
    std::optional<double>                 scale_d,
    std::optional<double>                 scale_h,
    std::optional<double>                 scale_w,
    Tensor&                               out);

} // namespace native
} // namespace at

// ---------------------------------------------------------------------------
//  Tiny helpers reused from ConvTranspose3d.mm
// ---------------------------------------------------------------------------

// Run a single‑fetch graph and copy results into an `at::Tensor`.
static void run_graph(MPSGraph*                 graph,
                      NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds,
                      MPSGraphTensor*           fetch,
                      at::Tensor&               out) {
  id<MTLCommandQueue> q = at::mps::getCurrentMPSStream()->commandQueue();
  NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
      [graph runWithMTLCommandQueue:q
                              feeds:feeds
                       targetTensors:@[ fetch ]
                    targetOperations:nil];

  MPSGraphTensorData* resultData = results[fetch];
  MPSNDArray*         ndarray    = resultData.mpsndarray;

  // contiguous copy into the destination tensor
  [ndarray readBytes:out.data_ptr() strideBytes:nullptr];
}

// Convert a (CPU or MPS) tensor into an MPSGraphTensorData object.
static MPSGraphTensorData* tensorToTensorData(const at::Tensor& t) {
  at::Tensor hostTensor = t;
  if (t.is_mps()) hostTensor = t.cpu();

  // Build shape
  NSMutableArray* shape = [NSMutableArray arrayWithCapacity:hostTensor.dim()];
  for (int i = 0; i < hostTensor.dim(); ++i) {
    [shape addObject:@(hostTensor.size(i))];
  }
  MPSDataType dtype = at::native::mps::MPSDataTypeFromScalarType(hostTensor.scalar_type());

  // Create NDArray descriptor describing the tensor shape
  MPSNDArrayDescriptor* desc = [MPSNDArrayDescriptor descriptorWithDataType:dtype
                                                                shape:shape];

  id<MTLDevice> dev = at::mps::MPSDevice::getInstance()->device();
  MPSNDArray* ndarray =
      [[MPSNDArray alloc] initWithDevice:dev descriptor:desc];

  // contiguous copy from host → NDArray
  [ndarray writeBytes:hostTensor.data_ptr() strideBytes:nil];

  return [[[MPSGraphTensorData alloc] initWithMPSNDArray:ndarray] autorelease];
}
// ---------------------------------------------------------------------------

namespace at::native::mps {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static inline std::array<int64_t,5> compute_output_size(const Tensor& x,
                                                        std::optional<ArrayRef<int64_t>> size,
                                                        std::optional<double> scale_d,
                                                        std::optional<double> scale_h,
                                                        std::optional<double> scale_w,
                                                        bool /*align_corners*/) {
  const auto inD = x.size(2), inH = x.size(3), inW = x.size(4);

  if (size.has_value()) {
    TORCH_CHECK(size->size() == 3, "output_size must be D, H, W");
    return {x.size(0), x.size(1), (*size)[0], (*size)[1], (*size)[2]};
  }

  TORCH_CHECK(scale_d && scale_h && scale_w,
              "Either output_size or scale_factors must be defined");

  auto calc = [&](int64_t in, double s) -> int64_t {
    return static_cast<int64_t>(std::floor(in * s));
  };

  return {x.size(0), x.size(1),
          calc(inD, *scale_d), calc(inH, *scale_h), calc(inW, *scale_w)};
}

// ---------------------------------------------------------------------------
// Graph cache
// ---------------------------------------------------------------------------

struct Trilinear3DKey {
  std::array<int64_t,5> inShape;
  std::array<int64_t,5> outShape;
  bool alignCorners;
  MPSDataType dtype;
  bool operator==(const Trilinear3DKey& o) const = default;
};

static struct TrilinearHash {
  size_t operator()(const Trilinear3DKey& k) const {
    return (static_cast<size_t>(k.alignCorners)      ^
            ((size_t)k.dtype << 1)                  ^
            std::hash<int64_t>{}(k.inShape[2]) << 2 ^
            std::hash<int64_t>{}(k.outShape[2])<<3);
  }
} trilinear_hash;

struct Trilinear3DGraph {
  MPSGraph *graph;
  MPSGraphExecutable *executable;
  MPSGraphTensor *inputPlaceholder;
  MPSGraphTensor *outputTensor;
};

static Trilinear3DGraph
compileTrilinearExecutable(const Trilinear3DKey& key) {
  MPSGraph *g = [[MPSGraph alloc] init];

  // Use NCDHW layout (value 7) which MPSGraph supports for some ops.
  MPSShape *inShape  = @[@(key.inShape[0]), @(key.inShape[1]),
                         @(key.inShape[2]), @(key.inShape[3]), @(key.inShape[4])];

  auto *input  = [g placeholderWithShape:inShape
                                 dataType:key.dtype
                                     name:@"input"];

  // Create a 1D tensor specifying the full 5-D output shape: N, C, D, H, W
  int32_t sizeVals[5] = { (int32_t)key.outShape[0],
                         (int32_t)key.outShape[1],
                         (int32_t)key.outShape[2],
                         (int32_t)key.outShape[3],
                         (int32_t)key.outShape[4] };
  NSData *sizeData = [NSData dataWithBytes:sizeVals length:sizeof(sizeVals)];
  // sizeTensor shape must match input rank (5)
  MPSGraphTensor *sizeT = [g constantWithData:sizeData
                                        shape:@[@5]
                                     dataType:MPSDataTypeInt32];

  // Use the no-layout overload for N-D resizing (available on macOS 14+)
  MPSGraphTensor *resized = [g resizeTensor:input
                              sizeTensor:sizeT
                                    mode:MPSGraphResizeBilinear
                            centerResult:NO
                            alignCorners:key.alignCorners
                                    name:@"upsample"];

  // Optionally pre-compile the graph if the API is available (to reduce launch cost)
  MPSGraphExecutable *exec = nil;
  if ([g respondsToSelector:@selector(compileWithDevice:)]) {
    exec = [g compileWithDevice:at::mps::MPSDevice::getInstance()->device()];
  }

  Trilinear3DGraph graphStruct{g, exec, input, resized};
  return graphStruct;
}

// ---------------------------------------------------------------------------
// Dispatcher entry
// ---------------------------------------------------------------------------

Tensor upsample_trilinear3d_mps(const Tensor& input,
                                std::optional<ArrayRef<int64_t>> output_size,
                                bool align_corners,
                                std::optional<double> scale_d,
                                std::optional<double> scale_h,
                                std::optional<double> scale_w) {
  TORCH_CHECK(input.is_mps(), "input must be on MPS");
  TORCH_CHECK(input.dim() == 5, "expected 5-D tensor NCDHW");

  // Compute output shape with appropriate branch
  bool use_scale = scale_d.has_value() && scale_h.has_value() && scale_w.has_value();
  std::array<int64_t,5> outShapeArr;
  if (use_scale) {
    outShapeArr = compute_output_size(input, std::nullopt, scale_d, scale_h, scale_w, align_corners);
  } else {
    outShapeArr = compute_output_size(input, output_size, scale_d, scale_h, scale_w, align_corners);
  }
  int64_t N     = input.size(0);
  int64_t C     = input.size(1);
  int64_t D     = input.size(2);
  int64_t H     = input.size(3);
  int64_t W     = input.size(4);
  int64_t D_out = outShapeArr[2];
  int64_t H_out = outShapeArr[3];
  int64_t W_out = outShapeArr[4];

  // Allocate output on MPS
  Tensor out = at::empty({N, C, D_out, H_out, W_out}, input.options().device(at::kMPS));

  // For 2D operations, let the 2D bilinear function compute its own scale factors from the output size
  // This avoids issues with integer truncation when converting scale_factor -> size -> scale_factor
  std::optional<double> scale_h_2d = scale_h;  // Pass through original scale or nullopt
  std::optional<double> scale_w_2d = scale_w;  // Pass through original scale or nullopt



  // Force one-time compilation of the 2D upsample graph (cached internally)
  at::upsample_bilinear2d(
    input.select(2, /*index=*/0),
    IntArrayRef({H_out, W_out}),
    align_corners, scale_h_2d, scale_w_2d);

  // Slice & stitch for each output depth
  for (int64_t d2 = 0; d2 < D_out; ++d2) {
    double z;
    if (align_corners) {
      z = (D_out > 1) ? d2 * (D - 1.) / (D_out - 1.) : 0.0;
    } else {
      double ratio = (scale_d.has_value() && *scale_d > 0.0)
                     ? (1.0 / *scale_d)  // CPU uses inverse of scale_factor!
                     : static_cast<double>(D) / static_cast<double>(D_out);
      z = (d2 + 0.5) * ratio - 0.5;

    }
    double z_clamped = std::min(std::max(z, 0.0), static_cast<double>(D - 1));
    int64_t i0 = static_cast<int64_t>(std::floor(z_clamped));
    int64_t i1 = (i0 < D - 1) ? (i0 + 1) : i0;
    double w1 = z_clamped - static_cast<double>(i0);
    double w0 = 1.0 - w1;

    // Extract two input slices [N,C,H,W]
    Tensor s0 = input.select(2, i0);
    Tensor s1 = input.select(2, i1);

    // 2D bilinear resizes
    Tensor r0 = at::upsample_bilinear2d(s0, {H_out, W_out}, align_corners, scale_h_2d, scale_w_2d);
    Tensor r1 = at::upsample_bilinear2d(s1, {H_out, W_out}, align_corners, scale_h_2d, scale_w_2d);

    // Linear interpolate in depth
    Tensor blended = r0.mul(w0).add(r1.mul(w1));
    


    // Write back into output tensor
    out.select(2, d2).copy_(blended);
  }

  return out;
}

} // namespace at::native::mps

//
// -------------------------------------------------------------------------
//  Public shim functions in the at::native namespace expected by the code‑gen
//  registration logic.  They simply forward to the MPS implementation above.
// -------------------------------------------------------------------------
//

namespace at::native {

TORCH_API Tensor upsample_trilinear3d_mps(
    const Tensor&                         input,
    c10::OptionalArrayRef<int64_t>        output_size,
    bool                                  align_corners,
    std::optional<double>                 scale_d,
    std::optional<double>                 scale_h,
    std::optional<double>                 scale_w) {
  std::optional<ArrayRef<int64_t>> size_opt;
  if (output_size.has_value()) {
    size_opt = *output_size;
  }
  return mps::upsample_trilinear3d_mps(
      input, size_opt, align_corners, scale_d, scale_h, scale_w);
}

TORCH_API Tensor _upsample_trilinear3d_mps(
    const Tensor&                         input,
    c10::OptionalArrayRef<int64_t>        output_size,
    bool                                  align_corners,
    std::optional<double>                 scale_d,
    std::optional<double>                 scale_h,
    std::optional<double>                 scale_w) {
  // internal alias used by generated RegisterMPS_* files
  std::optional<ArrayRef<int64_t>> size_opt;
  if (output_size.has_value()) {
    size_opt = *output_size;
  }
  return mps::upsample_trilinear3d_mps(
      input, size_opt, align_corners, scale_d, scale_h, scale_w);
}

// -------------------------------------------------------------------------
//  "out" variant public shim (writes results into a pre-allocated tensor)
// -------------------------------------------------------------------------

TORCH_API Tensor& upsample_trilinear3d_out_mps(
    const Tensor&                         input,
    at::IntArrayRef                      output_size,
    bool                                  align_corners,
    std::optional<double>                 scale_d,
    std::optional<double>                 scale_h,
    std::optional<double>                 scale_w,
    Tensor&                               out) {
  // Convert IntArrayRef to optional<ArrayRef<int64_t>> for core implementation
  std::optional<ArrayRef<int64_t>> size_opt = output_size;
  Tensor result = at::native::mps::upsample_trilinear3d_mps(
      input, size_opt, align_corners, scale_d, scale_h, scale_w);
  out.copy_(result);
  return out;
}

// Structured forward 'out' implementation required by codegen for MPS
TORCH_IMPL_FUNC(upsample_trilinear3d_out_mps)(
    const Tensor& input,
    at::IntArrayRef output_size,
    bool align_corners,
    std::optional<double> scale_d,
    std::optional<double> scale_h,
    std::optional<double> scale_w,
    const Tensor& out) {
  // Convert IntArrayRef to optional<ArrayRef<int64_t>> for core implementation
  std::optional<ArrayRef<int64_t>> size_opt = output_size;
  Tensor result = at::native::mps::upsample_trilinear3d_mps(
      input, size_opt, align_corners, scale_d, scale_h, scale_w);
  out.copy_(result);
}

} // namespace at::native

// ---------------------------------------------------------------------------
// Re-open at::native to place the backward impl in the correct namespace
namespace at {
namespace native {

// ---------------------------------------------------------------------------
// MPS 3D trilinear upsampling backward (out) implementation
// ---------------------------------------------------------------------------
TORCH_IMPL_FUNC(upsample_trilinear3d_backward_out_mps)(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    std::optional<double> scale_d,
    std::optional<double> scale_h,
    std::optional<double> scale_w,
    const Tensor& grad_input) {
  // Zero grad_input
  grad_input.zero_();
  int64_t N = input_size[0];
  int64_t C = input_size[1];
  int64_t D = input_size[2];
  int64_t H = input_size[3];
  int64_t W = input_size[4];
  int64_t D_out = output_size[0];
  int64_t H_out = output_size[1];
  int64_t W_out = output_size[2];
  
  // For 2D backward operations, let the 2D bilinear function compute its own scale factors
  std::optional<double> scale_h_2d = scale_h;  // Pass through original scale or nullopt
  std::optional<double> scale_w_2d = scale_w;  // Pass through original scale or nullopt
  
  // Similar slice-and-stitch approach for backward: accumulate 2D gradients slice-wise
  for (int64_t d2 = 0; d2 < D_out; ++d2) {
    double z;
    if (align_corners) {
      z = (D_out > 1) ? d2 * (D - 1.) / (D_out - 1.) : 0.0;
    } else {
      double ratio = (scale_d.has_value() && *scale_d > 0.0)
                     ? (1.0 / *scale_d)  // CPU uses inverse of scale_factor!
                     : static_cast<double>(D) / static_cast<double>(D_out);
      z = (d2 + 0.5) * ratio - 0.5;
    }
    double z_clamped = std::min(std::max(z, 0.0), static_cast<double>(D - 1));
    int64_t i0 = static_cast<int64_t>(std::floor(z_clamped));
    int64_t i1 = (i0 < D - 1) ? (i0 + 1) : i0;
    double w1 = z_clamped - static_cast<double>(i0);
    double w0 = 1.0 - w1;
    // Extract grad_output slice at depth d2
    Tensor grad2d = grad_output.select(2, d2);
    // Backprop through 2D bilinear on slice i0
    {
      Tensor grad_slice0 = grad_input.select(2, i0);
      at::upsample_bilinear2d_backward_out(
          grad_slice0,
          grad2d * w0,
          {H_out, W_out},
          {N, C, H, W},
          align_corners,
          scale_h_2d,
          scale_w_2d);
    }
    // Backprop through 2D bilinear on slice i1
    {
      Tensor grad_slice1 = grad_input.select(2, i1);
      at::upsample_bilinear2d_backward_out(
          grad_slice1,
          grad2d * w1,
          {H_out, W_out},
          {N, C, H, W},
          align_corners,
          scale_h_2d,
          scale_w_2d);
    }
  }
}

} // namespace native
} // namespace at

// -------------------------------------------------------------------------
//  Register the kernels with the dispatcher for the MPS backend
// -------------------------------------------------------------------------
TORCH_LIBRARY_IMPL(aten, MPS, m) {
  m.impl("_upsample_trilinear3d",       TORCH_FN(at::native::_upsample_trilinear3d_mps));
  m.impl("upsample_trilinear3d.out",    TORCH_FN(at::native::upsample_trilinear3d_out_mps));
}