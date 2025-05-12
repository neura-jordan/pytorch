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
    c10::OptionalArrayRef<int64_t>        output_size,
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

  MPSNDArrayDescriptor* desc =
      [[MPSNDArrayDescriptor alloc] initWithShape:shape dataType:dtype];

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
                                                        bool align_corners) {
  const auto inD = x.size(2), inH = x.size(3), inW = x.size(4);

  if (size.has_value()) {              // explicit size wins
    TORCH_CHECK(size->size() == 3, "output_size must be D, H, W");
    return {x.size(0), x.size(1), (*size)[0], (*size)[1], (*size)[2]};
  }

  TORCH_CHECK(scale_d && scale_h && scale_w,
              "Either output_size or scale_factors must be defined");

  auto calc = [&](int64_t in, double s) -> int64_t {
    return align_corners
           ? static_cast<int64_t>(std::floor((in - 1) * s + 1))
           : static_cast<int64_t>(std::floor(in * s));
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

  MPSShape *inShape  = @[@(key.inShape[0]), @(key.inShape[1]),
                         @(key.inShape[2]), @(key.inShape[3]), @(key.inShape[4])];
  NSArray<NSNumber *> *outShape __attribute__((unused)) = @[@(key.outShape[2]),
                                                          @(key.outShape[3]),
                                                          @(key.outShape[4])];

  auto *input  = [g placeholderWithShape:inShape
                                 dataType:key.dtype
                                     name:@"input"];

  int32_t sizeVals[3] = { (int32_t)key.outShape[2],
                          (int32_t)key.outShape[3],
                          (int32_t)key.outShape[4] };
  NSData *sizeData = [NSData dataWithBytes:sizeVals length:sizeof(sizeVals)];
  MPSGraphTensor *sizeT = [g constantWithData:sizeData
                                        shape:@[@3]
                                     dataType:MPSDataTypeInt32];

  MPSGraphTensor *resized = [g resizeBilinearWithTensor:input
                                            sizeTensor:sizeT
                                         alignCorners:key.alignCorners
                                         centerResult:NO
                                               layout:MPSGraphTensorNamedDataLayoutNCDHW
                                                 name:@"upsample"];

  MPSGraphExecutable *exec = [g compileWithDevice:at::mps::MPSDevice::getInstance()->device()
                                           options:nil
                                             error:nil];

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

  auto outShapeArr = compute_output_size(input, output_size,
                                         scale_d, scale_h, scale_w,
                                         align_corners);

  auto options = input.options();
  Tensor out = at::empty(IntArrayRef(outShapeArr.data(), 5), options);

  Trilinear3DKey key{ {input.sizes()[0], input.sizes()[1],
                       input.sizes()[2], input.sizes()[3], input.sizes()[4]},
                      outShapeArr,
                      align_corners,
                      input.scalar_type() == kHalf ? MPSDataTypeFloat16
                                                   : MPSDataTypeFloat32 };

  static std::unordered_map<Trilinear3DKey,
                            Trilinear3DGraph,
                            TrilinearHash> cache;

  Trilinear3DGraph graph;
  auto it = cache.find(key);
  if (it == cache.end()) {
    graph = compileTrilinearExecutable(key);
    cache.emplace(key, graph);
  } else {
    graph = it->second;
  }

  @autoreleasepool {
    at::mps::MPSStream* s = at::mps::getCurrentMPSStream();

    // Build feed and run graph
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{ graph.inputPlaceholder : tensorToTensorData(input) };
    
    // Execute graph with input and collect results
    run_graph(graph.graph, feeds, graph.outputTensor, out);
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
    c10::OptionalArrayRef<int64_t>        output_size,
    bool                                  align_corners,
    std::optional<double>                 scale_d,
    std::optional<double>                 scale_h,
    std::optional<double>                 scale_w,
    Tensor&                               out) {
  std::optional<ArrayRef<int64_t>> size_opt;
  if (output_size.has_value()) {
    size_opt = *output_size;
  }
  Tensor result = mps::upsample_trilinear3d_mps(
      input, size_opt, align_corners, scale_d, scale_h, scale_w);
  out.copy_(result);
  return out;
}

} // namespace at::native

// -------------------------------------------------------------------------
//  Register the kernels with the dispatcher for the MPS backend
// -------------------------------------------------------------------------

TORCH_LIBRARY_IMPL(aten, MPS, m) {
  m.impl("_upsample_trilinear3d",       TORCH_FN(at::native::_upsample_trilinear3d_mps));
  m.impl("upsample_trilinear3d.out",    TORCH_FN(at::native::upsample_trilinear3d_out_mps));
}