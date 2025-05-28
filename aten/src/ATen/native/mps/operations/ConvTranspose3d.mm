// Copyright (c) 2025 PyTorch contributors
// SPDX-License-Identifier: BSD-3-Clause

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <MetalPerformanceShadersGraph/MPSGraphConvolutionOps.h>
#include <ATen/ATen.h>
#include <ATen/native/TensorProperties.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <torch/utils.h>   // for torch::NoGradGuard

#include <ATen/native/mps/MPSGraphHelpers.h>
#include <torch/library.h>   // for TORCH_LIBRARY_IMPL and dispatcher helpers
#include <c10/util/Optional.h>
#include <torch/types.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/autograd.h>

using namespace at::native::mps;
using namespace at::mps;

// ------------- helper to run a tiny graph -----------------
static void run_graph(MPSGraph*                 graph,
                      NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds,
                      MPSGraphTensor*           fetch,
                      at::Tensor&               out) {
  // Execute the graph in one shot using the queue from the current MPS stream.
  id<MTLCommandQueue> q = at::mps::getCurrentMPSStream()->commandQueue();

  // Run and obtain the result map.  We use the simpler convenience
  // `runWithMTLCommandQueue:feeds:targetTensors:targetOperations:` API so we
  // don't have to craft an MPSGraphExecutionDescriptor or a command buffer
  // manually.
  NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
      [graph runWithMTLCommandQueue:q
                              feeds:feeds
                       targetTensors:@[ fetch ]
                    targetOperations:nil];

  // Copy the result into the output tensor.
  MPSGraphTensorData* resultData = results[fetch];
  MPSNDArray*         ndarray    = resultData.mpsndarray;

  // Read the NDArray's contents directly into the tensor's buffer.
  // (strideBytes == nullptr ⇒ contiguous copy)
  [ndarray readBytes:out.data_ptr() strideBytes:nullptr];
}
// ----------------------------------------------------------

// -------------- helper for tensor -> MPSGraphTensorData -------------
static MPSGraphTensorData* tensorToTensorData(const at::Tensor& t) {
  // Ensure we pass a *host‑accessible* pointer to writeBytes:.  If the incoming
  // tensor lives on MPS memory, copy it to CPU first.
  at::Tensor hostTensor = t;

  // Build shape array (NSArray<NSNumber *> *)
  NSMutableArray* shape = [NSMutableArray arrayWithCapacity:t.dim()];
  for (int i = 0; i < t.dim(); ++i) {
    [shape addObject:@(t.size(i))];
  }

  MPSDataType dtype = MPSDataTypeFromScalarType(t.scalar_type());

  // If the tensor is already on MPS, embed its underlying MTLBuffer directly.
  if (t.is_mps()) {
    // Ensure a packed layout so that stride computations are correct.
    at::Tensor gpuTensor = t.contiguous();

    // Extract the Metal buffer from PyTorch's MPS storage.
    id<MTLBuffer> mtlBuffer = getMTLBufferStorage(gpuTensor);

    // Compute per-dimension *byte* strides expected by MPSGraph.
    const int dim = gpuTensor.dim();
    // Allocate stride array on the heap; per-Apple docs the pointer must
    // remain valid for the lifetime of the MPSGraphTensorData object.  We
    // intentionally leak these few words to satisfy that requirement.
    NSUInteger* byteStrides = (NSUInteger*)malloc(sizeof(NSUInteger) * dim);
    for (int i = 0; i < dim; ++i) {
      byteStrides[i] = gpuTensor.stride(i) * gpuTensor.element_size();
    }

    // Attempt to use the modern strideBytes initializer introduced in macOS 15.
    SEL strideInitSel = @selector(initWithMTLBuffer:shape:dataType:strideBytes:);
    if ([MPSGraphTensorData instancesRespondToSelector:strideInitSel]) {
      MPSGraphTensorData* graphData = [[MPSGraphTensorData alloc]
          initWithMTLBuffer:mtlBuffer
                     shape:shape
                  dataType:dtype
                strideBytes:byteStrides];
      return [graphData autorelease];
    }

    // Fallback for macOS 14.x: use the rowBytes initializer if rank ≤ 2.
    if (dim <= 2) {
      NSUInteger rowBytes = gpuTensor.stride(0) * gpuTensor.element_size();
      MPSGraphTensorData* graphData = [[MPSGraphTensorData alloc]
          initWithMTLBuffer:mtlBuffer
                     shape:shape
                  dataType:dtype
                   rowBytes:rowBytes];
      return [graphData autorelease];
    }
    // Otherwise fall back to the host copy path below.
  }

  if (t.is_mps()) {
    hostTensor = t.cpu();
  }

  // Prefer the modern constructor when the runtime provides it; otherwise fall
  // back to the legacy one.  We check *at runtime* because the headers we
  // compile against may be newer than the user's OS.
  MPSNDArrayDescriptor* desc = nil;
  SEL newCtorSel = @selector(descriptorWithShape:dataType:);
  SEL midCtorSel = @selector(descriptorWithDataType:shape:);
  SEL oldCtorSel = @selector(descriptorWithDataType:shape:count:);

  if ([MPSNDArrayDescriptor respondsToSelector:newCtorSel]) {
    desc = [MPSNDArrayDescriptor descriptorWithShape:shape dataType:dtype];
  } else if ([MPSNDArrayDescriptor respondsToSelector:midCtorSel]) {
    desc = [MPSNDArrayDescriptor descriptorWithDataType:dtype shape:shape];
  } else if ([MPSNDArrayDescriptor respondsToSelector:oldCtorSel]) {
    desc = [MPSNDArrayDescriptor descriptorWithDataType:dtype shape:shape count:hostTensor.dim()];
  } else {
    TORCH_CHECK(false,
      "MPSNDArrayDescriptor: neither modern, intermediate, nor legacy constructors are available on this macOS version.");
  }

  // Instantiate the NDArray that views the tensor's storage
  id<MTLDevice> mtlDevice = at::mps::MPSDevice::getInstance()->device();
  MPSNDArray* ndarray = [[MPSNDArray alloc] initWithDevice:mtlDevice
                                                descriptor:desc];

  // Copy the tensor's contents into the NDArray.
  @try {
    // Prefer the modern two‑arg variant (present in macOS 14.0+ on the core
    // framework and in macOS 15.0+ on the Graph framework).
    [ndarray writeBytes:hostTensor.data_ptr() strideBytes:nil];
  } @catch (NSException* _) {
    // If the selector doesn't exist on this particular MPSNDArray implementation,
    // fall back to the legacy single‑arg method (macOS 13.x).
    [ndarray writeBytes:hostTensor.data_ptr()];
  }

  // Wrap in an MPSGraphTensorData
  return [[[MPSGraphTensorData alloc] initWithMPSNDArray:ndarray] autorelease];
}
// -------------------------------------------------------------------

// -------- helper: infer kernel size (1-D) ---------------------------------
static inline int64_t infer_kernel_1d(int64_t in_sz,
                                      int64_t out_sz,
                                      int64_t stride,
                                      int64_t pad,
                                      int64_t out_pad,
                                      int64_t dil) {
  // out = stride*(in-1) + out_pad + (k-1)*dil + 1 − 2*pad
  int64_t k = ((out_sz + 2 * pad) - (stride * (in_sz - 1) + out_pad) - 1) / dil + 1;
  TORCH_INTERNAL_ASSERT(k > 0, "Unable to infer kernel size");
  return k;
}
// --------------------------------------------------------------------------

namespace at::native {

// Prototype for dispatcher linkage (extern)
TORCH_API Tensor conv_transpose3d_mps(
        const Tensor& input,
        const Tensor& weight,
        const std::optional<Tensor>& bias,
        IntArrayRef stride,
        IntArrayRef padding,
        IntArrayRef output_padding,
        int64_t groups,
        IntArrayRef dilation);

// Forward declaration for the backward function
std::tuple<at::Tensor, at::Tensor, at::Tensor> conv_transpose3d_backward_mps_impl(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    at::OptionalIntArrayRef bias_sizes,
    c10::IntArrayRef stride,
    c10::IntArrayRef padding,
    c10::IntArrayRef dilation,
    bool transposed,
    c10::IntArrayRef output_padding,
    int64_t groups,
    std::array<bool, 3> output_mask
);

// Calculated output size for transposed convolution
// Based on the formula from ConvUtils.h
static std::vector<int64_t> calc_output_size(
        const IntArrayRef& input_size,
        const IntArrayRef& weight_size,
        const IntArrayRef& padding,
        const IntArrayRef& output_padding,
        const IntArrayRef& stride,
        const IntArrayRef& dilation,
        int64_t groups) {
    
    auto dim = input_size.size() - 2;
    std::vector<int64_t> output_size(input_size.begin(), input_size.begin() + 2);
    output_size[1] = weight_size[1] * groups; // output channels
    
    for (const auto d : c10::irange(dim)) {
        // Correct PyTorch transposed convolution formula:
        // output_size = (input_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
        auto kernel_size = weight_size[d + 2];
        auto dilated_kernel_size = dilation[d] * (kernel_size - 1) + 1;
        auto output_dim = (input_size[d + 2] - 1) * stride[d] - 2 * padding[d] + dilated_kernel_size + output_padding[d];
        output_size.push_back(output_dim);
    }
    
    return output_size;
}

// Add helper to fill 3D conv descriptor (copied from Convolution.mm)
// ------------- helper to program 3D conv descriptor -----------------
static void fill_conv3d_desc(MPSGraphConvolution3DOpDescriptor* descriptor_,
                             NSUInteger strideInX,
                             NSUInteger strideInY,
                             NSUInteger strideInZ,
                             NSUInteger dilationRateInX,
                             NSUInteger dilationRateInY,
                             NSUInteger dilationRateInZ,
                             NSUInteger paddingHorizontal,
                             NSUInteger paddingVertical,
                             NSUInteger paddingDepth,
                             NSUInteger groups) {
  descriptor_.strideInX = strideInX;
  descriptor_.strideInY = strideInY;
  descriptor_.strideInZ = strideInZ;
  descriptor_.dilationRateInX = dilationRateInX;
  descriptor_.dilationRateInY = dilationRateInY;
  descriptor_.dilationRateInZ = dilationRateInZ;
  descriptor_.paddingStyle = MPSGraphPaddingStyleExplicit;
  descriptor_.paddingLeft = paddingHorizontal;
  descriptor_.paddingRight = paddingHorizontal;
  descriptor_.paddingTop = paddingVertical;
  descriptor_.paddingBottom = paddingVertical;
  descriptor_.paddingFront = paddingDepth;
  descriptor_.paddingBack = paddingDepth;
  descriptor_.dataLayout = MPSGraphTensorNamedDataLayoutNDHWC;  // NDHWC
  descriptor_.weightsLayout = MPSGraphTensorNamedDataLayoutOIDHW; // OIDHW
  descriptor_.groups = groups;
}
// ----------------------------------------------------------

Tensor conv_transpose3d_mps(
        const Tensor& input,
        const Tensor& weight,
        const std::optional<Tensor>& bias,
        IntArrayRef stride,
        IntArrayRef padding,
        IntArrayRef output_padding,
        int64_t groups,
        IntArrayRef dilation) {

  TORCH_CHECK(groups == 1, "MPS ConvTranspose3d currently only supports groups==1");
  TORCH_CHECK(input.device().is_mps() && weight.device().is_mps(),
              "inputs must be on MPS");

  // Compute output shape (N, Cout, D, H, W)
  auto out_sizes = calc_output_size(
      input.sizes(), weight.sizes(), padding, output_padding, stride, dilation, groups);

  // Compute output tensor
  Tensor output = at::empty(out_sizes, input.options().device(kMPS));
  @autoreleasepool {
    if (@available(macOS 14.0, *)) {
      // 1. Prepare tensors for MPSGraph (convert from NCDHW to NDHWC format)
      auto x = input.contiguous();
      auto x_ndhwc = x.permute({0, 2, 3, 4, 1});

      // PyTorch weights for ConvTranspose3d are (Cin, Cout/groups, kD, kH, kW).
      // For convolution3DDataGradientWithIncomingGradientTensor, we need to think about
      // what forward convolution would produce our input from our desired output.
      // The data gradient API computes: input = conv3d_data_grad(output, weight_fwd)
      // For transposed conv: output = conv_transpose3d(input, weight_transpose)
      // This is equivalent to: output = conv3d_data_grad(input, weight_fwd)
      // where weight_fwd should be weight_transpose with no permutation for OIDHW layout
      auto w_formps = weight.contiguous();

      // 2. Initialize the graph
      MPSGraph* graph = [[MPSGraph alloc] init];

      // 3. Create placeholder tensors
      MPSGraphTensor* gx = TorchTensorToMPSPlaceholder(graph, x_ndhwc, /*rank=*/5, "x");
      MPSGraphTensor* gw = TorchTensorToMPSPlaceholder(graph, w_formps, /*rank=*/5, "w");

      // 4. Configure convolution descriptor
      MPSGraphConvolution3DOpDescriptor* conv3dDescriptor_ = [[MPSGraphConvolution3DOpDescriptor new] autorelease];
      fill_conv3d_desc(conv3dDescriptor_,
                       /*strideInX*/ stride[2],
                       /*strideInY*/ stride[1],
                       /*strideInZ*/ stride[0],
                       /*dilationRateInX*/ dilation[2],
                       /*dilationRateInY*/ dilation[1],
                       /*dilationRateInZ*/ dilation[0],
                       /*paddingHorizontal*/ padding[2],
                       /*paddingVertical*/ padding[1],
                       /*paddingDepth*/ padding[0],
                       /*groups*/ groups);

      // 5. Create convolutionTranspose3D operation using data gradient API
      NSArray* outputShapeArr = @[
          @(out_sizes[0]),   // N
          @(out_sizes[2]),   // D
          @(out_sizes[3]),   // H
          @(out_sizes[4]),   // W
          @(out_sizes[1])    // C_out
      ];

      // Use the data gradient API (mathematically equivalent to transposed convolution)
      SEL selModernDataGrad = @selector(convolution3DDataGradientWithIncomingGradientTensor:weightsTensor:outputShape:forwardConvolutionDescriptor:name:);
      SEL selLegacyDataGrad = @selector(convolution3DDataGradientWithIncomingGradientTensor:weightsTensor:forwardConvolutionDescriptor:name:);
      MPSGraphTensor* gy_ndhwc = nil;
      if ([graph respondsToSelector:selModernDataGrad]) {
        gy_ndhwc = [graph convolution3DDataGradientWithIncomingGradientTensor:gx
                                                            weightsTensor:gw
                                                              outputShape:outputShapeArr
                                                 forwardConvolutionDescriptor:conv3dDescriptor_
                                                                        name:@"convT3D"];
      } else if ([graph respondsToSelector:selLegacyDataGrad]) {
        gy_ndhwc = [graph convolution3DDataGradientWithIncomingGradientTensor:gx
                                                            weightsTensor:gw
                                               forwardConvolutionDescriptor:conv3dDescriptor_
                                                                        name:@"convT3D"];
      } else {
        TORCH_CHECK(false, "3D transposed convolution on MPS requires macOS 14+ with data gradient support");
      }

      // 6. Prepare feeds and run the graph
      auto feeds = @{
        gx : tensorToTensorData(x_ndhwc),
        gw : tensorToTensorData(w_formps)
      };
      
      // Create a *CPU* tensor to receive the NDHWC result. readBytes: copies
      // into host-accessible memory; passing an MPS buffer crashes.
      auto y_ndhwc_cpu = at::empty({out_sizes[0], out_sizes[2], out_sizes[3], out_sizes[4], out_sizes[1]},
                                   input.options().device(at::kCPU));

      // Run the graph and read back into CPU tensor
      run_graph(graph, feeds, gy_ndhwc, y_ndhwc_cpu);

      // Move to MPS and convert from NDHWC back to NCDHW format
      auto y_ndhwc = y_ndhwc_cpu.to(kMPS);
      output = y_ndhwc.permute({0, 4, 1, 2, 3});

      [graph release];
    } else {
      TORCH_CHECK(false, "3D transposed convolution on MPS requires macOS 14+ / Xcode 15+");
    }
  }
  
  // Add bias if provided
  if (bias.has_value()) {
    output = output + bias.value().view({1, -1, 1, 1, 1});
  }
  
  return output;
}

// ==== Weight-gradient for ConvTranspose3d on MPS ==========================
static Tensor conv_transpose3d_weight_mps(const Tensor& grad_output,
                                          const Tensor& input,
                                          const Tensor& weight,
                                          IntArrayRef stride,
                                          IntArrayRef padding,
                                          IntArrayRef output_padding,
                                          int64_t groups,
                                          IntArrayRef dilation) {

  TORCH_CHECK(input.device().is_mps() && grad_output.device().is_mps(),
              "MPS conv_transpose3d.weight expects MPS tensors");
  TORCH_CHECK(groups == 1, "groups != 1 not yet supported");

  const int64_t Cin  = input.size(1);
  const int64_t Cout = grad_output.size(1);
  const int64_t kD = weight.size(2);
  const int64_t kH = weight.size(3);
  const int64_t kW = weight.size(4);

  // Allocate CPU buffer for weight-gradient (OIDHW) to avoid heap corruption
  Tensor tmp_oidhw_cpu = at::empty({Cout, Cin, kD, kH, kW},
                                    grad_output.options().device(at::kCPU));

  @autoreleasepool {
      auto gY = grad_output.contiguous().permute({0,2,3,4,1});
      auto  X = input.contiguous()       .permute({0,2,3,4,1});

      MPSGraph* graph = [[MPSGraph alloc] init];

      MPSGraphTensor* gyT = TorchTensorToMPSPlaceholder(graph, gY, 5, "dY");
      MPSGraphTensor*  xT = TorchTensorToMPSPlaceholder(graph,  X, 5, "X");

      MPSGraphConvolution3DOpDescriptor* desc = [[MPSGraphConvolution3DOpDescriptor new] autorelease];
      fill_conv3d_desc(desc,
                       stride[2], stride[1], stride[0],
                       dilation[2], dilation[1], dilation[0],
                       padding[2], padding[1], padding[0],
                       groups);

      NSArray* wShapeArr = @[ @(Cout), @(Cin), @(kD), @(kH), @(kW) ]; // OIDHW

      MPSGraphTensor* dW_oidhw =
          [graph convolution3DWeightsGradientWithIncomingGradientTensor:gyT
                                                           sourceTensor:xT
                                                            outputShape:wShapeArr
                                           forwardConvolutionDescriptor:desc
                                                                 name:@"dW"];

      auto feeds = @{ gyT : tensorToTensorData(gY),
                      xT  : tensorToTensorData(X) };

      // Read weight-gradient into CPU tensor
      run_graph(graph, feeds, dW_oidhw, tmp_oidhw_cpu);
      [graph release];
  }
  // Permute on CPU and move to MPS
  return tmp_oidhw_cpu.permute({1,0,2,3,4}).contiguous().to(kMPS);
}
// ==========================================================================

// ==== Input-gradient for ConvTranspose3d on MPS ==========================
static Tensor conv_transpose3d_input_mps(const Tensor& grad_output,
                                         const Tensor& weight,
                                         const Tensor& input,
                                         const std::optional<Tensor>& bias,
                                         IntArrayRef stride,
                                         IntArrayRef padding,
                                         IntArrayRef output_padding,
                                         int64_t groups,
                                         IntArrayRef dilation) {
    TORCH_CHECK(grad_output.device().is_mps() && weight.device().is_mps(),
                "MPS conv_transpose3d.input expects MPS tensors");
    TORCH_CHECK(groups == 1, "groups != 1 not yet supported");

    // grad_output shape: [N, C_out, D_out, H_out, W_out]
    // weight shape:      [C_in, C_out/groups, kD, kH, kW]
    // We compute grad_input by a forward convolution3D:
    //   grad_input = conv3d(grad_output, weight_for_fwd_conv, padding_adj, stride=dilation, dilation=stride)

    const int64_t kD = weight.size(2);
    const int64_t kH = weight.size(3);
    const int64_t kW = weight.size(4);
    // Compute adjusted padding for data gradient
    int64_t padD = (kD - 1) * dilation[0] - padding[0];
    int64_t padH = (kH - 1) * dilation[1] - padding[1];
    int64_t padW = (kW - 1) * dilation[2] - padding[2];

    // Permute to NDHWC
    auto gY_ndhwc = grad_output.contiguous().permute({0,2,3,4,1}); // grad_output (source for conv3d)

    // For conv3d(source, weights_fwd, ...), source.channels must match weights_fwd.InputChannels.
    // PyTorch weight is (Cin_tp, Cout_tp/G, k...).
    // If descriptor_.weightsLayout = OIDHW for conv3d, it expects weights_fwd as (O_fwd, I_fwd, k...).
    // Here, source is gY_ndhwc (Cout_tp channels). So I_fwd must be Cout_tp.
    // O_fwd will be the output channels of conv3d, which is Cin_tp (grad_input channels).
    // So, weights_fwd should be (Cin_tp, Cout_tp/G, k...).
    // This means the PyTorch weight (Cin_tp, Cout_tp/G, k...) needs to be permuted to (O_fwd=Cin_tp, I_fwd=Cout_tp/G, k...).
    // For MPS OIDHW format, the transposed conv weight (Cin, Cout, kD, kH, kW) is already
    // in the correct format for the forward conv weight used in the data gradient computation.
    auto W_fwd = weight.contiguous(); // No permutation needed

    auto  X_orig_ndhwc = input.contiguous().permute({0,2,3,4,1}); // Original input permuted

    // Prepare output shape (NDHWC) using the original input tensor's shape
    auto orig_input_sizes = X_orig_ndhwc.sizes();

    // Allocate CPU buffer for input-gradient using original input shape
    Tensor grad_input_cpu = at::empty_like(X_orig_ndhwc, grad_output.options().device(at::kCPU));

    @autoreleasepool {
        MPSGraph* graph = [[MPSGraph alloc] init];
        // placeholders
        MPSGraphTensor* gyT = TorchTensorToMPSPlaceholder(graph, gY_ndhwc, 5, "dY");
        MPSGraphTensor*  wT = TorchTensorToMPSPlaceholder(graph,  W_fwd, 5, "W");

        // descriptor
        MPSGraphConvolution3DOpDescriptor* desc = [[MPSGraphConvolution3DOpDescriptor new] autorelease];
        fill_conv3d_desc(desc,
                         /*strideX*/ dilation[2], dilation[1], dilation[0],
                         /*dilationX*/ stride[2],  stride[1],  stride[0],
                         /*padH*/    padW, padH, padD,
                         /*groups*/  groups);

        // Use the correct shape derived from the original input
        NSArray* outShapeArr = @[
          @(orig_input_sizes[0]), // N
          @(orig_input_sizes[1]), // D
          @(orig_input_sizes[2]), // H
          @(orig_input_sizes[3]), // W
          @(orig_input_sizes[4])  // C_in
        ];

        // Detect modern API with outputShape and convolutionDescriptor (macOS 16+)
        SEL selModern = @selector(convolution3DWithSourceTensor:weightsTensor:outputShape:convolutionDescriptor:name:);
        SEL selLegacy = @selector(convolution3DWithSourceTensor:weightsTensor:descriptor:name:);
        MPSGraphTensor* dx_ndhwc = nil;
        if ([graph respondsToSelector:selModern]) {
          // Modern API (macOS 16+): specify outputShape and convolutionDescriptor
          dx_ndhwc = [graph convolution3DWithSourceTensor:gyT
                                         weightsTensor:wT
                                           outputShape:outShapeArr
                               convolutionDescriptor:desc
                                               name:@"conv_input_grad"];
        } else if ([graph respondsToSelector:selLegacy]) {
          // Legacy API (macOS 14/15): descriptor only (no outputShape)
          dx_ndhwc = [graph convolution3DWithSourceTensor:gyT
                                         weightsTensor:wT
                                             descriptor:desc
                                                   name:@"conv_input_grad"];
        } else {
          TORCH_CHECK(false, "MPS conv_transpose3d.input requires macOS 14+ with 3D convolution support");
        }

        // run and read back
        NSDictionary* feeds = @{ gyT : tensorToTensorData(gY_ndhwc),
                                 wT  : tensorToTensorData(W_fwd) };
        // Read input-gradient into CPU tensor
        run_graph(graph, feeds, dx_ndhwc, grad_input_cpu);
        [graph release];
    }

    // Permute on CPU and move to MPS
    return grad_input_cpu.to(kMPS).permute({0,4,1,2,3}).contiguous();
}
// =========================================================================

// Wrapper function matching the native signature expected by the linker/autograd system
std::tuple<at::Tensor, at::Tensor, at::Tensor> conv_transpose3d_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    at::OptionalIntArrayRef bias_sizes,
    c10::IntArrayRef stride,
    c10::IntArrayRef padding,
    c10::IntArrayRef output_padding, // Correct order based on linker error
    int64_t groups,
    c10::IntArrayRef dilation,        // Correct order based on linker error
    std::array<bool, 3> output_mask
) {
    // Call the actual MPS implementation, passing true for the transposed flag
    // and ensuring argument order matches the _mps_impl function's definition.
    return conv_transpose3d_backward_mps_impl(
        grad_output,
        input,
        weight,
        bias_sizes,
        stride,
        padding,
        dilation,         // Order for _mps_impl
        /*transposed=*/true,
        output_padding,  // Order for _mps_impl
        groups,
        output_mask
    );
}

// Composite backward function for ConvTranspose3d on MPS
std::tuple<at::Tensor, at::Tensor, at::Tensor> conv_transpose3d_backward_mps_impl(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    at::OptionalIntArrayRef bias_sizes,
    c10::IntArrayRef stride,
    c10::IntArrayRef padding,
    c10::IntArrayRef dilation,
    bool transposed,
    c10::IntArrayRef output_padding,
    int64_t groups,
    std::array<bool, 3> output_mask
) {
    at::Tensor grad_input, grad_weight, grad_bias;

    // Bias gradient is not supported yet
    grad_bias = Tensor();

    if (output_mask[0]) {
        // Calculate grad_input
        grad_input = conv_transpose3d_input_mps(
            grad_output, weight, input, /*bias=*/std::nullopt,
            stride, padding, output_padding, groups, dilation);
    }

    if (output_mask[1]) {
        // Calculate grad_weight
        grad_weight = conv_transpose3d_weight_mps(
            grad_output, input, weight,
            stride, padding, output_padding, groups, dilation);
    }

    return std::make_tuple(grad_input, grad_weight, grad_bias);
}
} // namespace at::native

// Register the autograd implementation for AutogradMPS key
TORCH_LIBRARY_IMPL(aten, AutogradMPS, m) {
  // Register the backward kernel for conv_transpose3d
  m.impl("conv_transpose3d_backward",
         TORCH_FN(at::native::conv_transpose3d_backward));
}


// Register with dispatcher for MPS
TORCH_LIBRARY_IMPL(aten, MPS, m) {
  // Register the forward kernel for conv_transpose3d
  m.impl("conv_transpose3d", // <-- CORRECT NAME
         TORCH_FN(at::native::conv_transpose3d_mps));
}