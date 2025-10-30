//===-- SConv.cpp - Transform dialect Extension SConv ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// @brief Implements the SConv Transform dialect extension (init + ops).
//
// Debugging
//   Use: `sconv-opt <args> -debug-only=SConv`
//
// Dialect loading
//   - Dependent dialects (used by transform ops): Linalg.
//   - Generated dialects (may appear in payload results): Affine, Arith, Index,
//     SCF, Tensor.
//
// Paper reference
//   - Using MLIR Transform to Design Sliced Convolution Algorithm (see Readme)
//
// Paper anchors
//   - CSA and schedule selection (IS/WS): Section 2.2 / 5.2
//   - Edge-case splitting: Section 5.3
//   - Two-level tiling: Section 5.4
//   - Packing + Multipacking (affine maps): Section 4 / 5.5
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "SConv.h"

#include "CSA.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgMatchOps.h"
#include "mlir/Dialect/Linalg/TransformOps/Syntax.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include <cstdint>

#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/CallInterfaces.h"

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::linalg;
using namespace mlir::transform;

// To debug info, use: transform-opt your_parameters -debug-only=SConv
#define DEBUG_TYPE "sconv-transform"
#define DBGS() (llvm::dbgs() << "\n[" DEBUG_TYPE "] ")

#define GET_OP_CLASSES
#include "SConv.cpp.inc"

/// Define the SConv transform dialect. This uses the CRTP idiom to identify extensions.
class SConv
    : public transform::TransformDialectExtension<SConv> {
public:
  // The TypeID of this extension.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SConv)

  // The extension must derive the base constructor.
  using Base::Base;

  // This function initializes the extension, similarly to `initialize` in
  // dialect definitions. List individual operations and dependent dialects
  // here.
  void init();
};

void SConv::init() {
  // As an transform extension dialect, we must declare all dependent dialects.
  // These dialects will be loaded along with the extension and, therefore,
  // along with the Transform dialect. The dependent dialects contain the
  // attributes or types used by transform operations.
  declareDependentDialect<linalg::LinalgDialect>();

  // When transformations are applied, they may produce new operations from
  // previously unloaded dialects. Typically, a pass would need to declare
  // itself dependent on the dialects containing such new operations. To avoid
  // confusion with the dialects the extension itself depends on, the Transform
  // dialects differentiates between:
  //   - dependent dialects, which are used by the transform operations, and
  //   - generated dialects, which contain the entities (attributes, operations,
  //     types) that may be produced by applying the transformation even when
  //     not present in the original payload IR.
  declareGeneratedDialect<affine::AffineDialect>();
  declareGeneratedDialect<arith::ArithDialect>();
  declareGeneratedDialect<index::IndexDialect>();
  declareGeneratedDialect<scf::SCFDialect>();
  declareGeneratedDialect<tensor::TensorDialect>();
  declareGeneratedDialect<LLVM::LLVMDialect>();

  // Finally, we register the additional transform operations with the dialect.
  // List all operations generated from ODS. This call will perform additional
  // checks that the operations implement the transform and memory effect
  // interfaces required by the dialect interpreter and assert if they do not.
  registerTransformOps<
#define GET_OP_LIST
#include "SConv.cpp.inc"
      >();
}

// ====--------------------------------------------------------------------------
// Utilities for arithmetic on index values, and others.
// ====--------------------------------------------------------------------------

static bool hasAllOneValues(DenseIntElementsAttr attr) {
  return llvm::all_of(
      attr, [](const APInt &element) { return element.getSExtValue() == 1; });
}

static Value createAdd(Location loc, Value x, Value y, OpBuilder &builder) {
  if (isa<IntegerType>(x.getType()))
    return builder.create<arith::AddIOp>(loc, x, y);
  return builder.create<arith::AddFOp>(loc, x, y);
}

static Value createMul(Location loc, Value x, Value y, Type accType,
                       OpBuilder &builder) {
  // Linalg named ops specify signed extend for named ops.
  Value xConvert =
      convertScalarToDtype(builder, loc, x, accType, /*isUnsignedCast=*/false);
  Value yConvert =
      convertScalarToDtype(builder, loc, y, accType, /*isUnsignedCast=*/false);
  if (isa<IntegerType>(accType))
    return builder.create<arith::MulIOp>(loc, xConvert, yConvert);
  return builder.create<arith::MulFOp>(loc, xConvert, yConvert);
}

static tensor::CollapseShapeOp findCollapseUsing(Value src, Block *body) {
  tensor::CollapseShapeOp result = nullptr;
  body->walk([&](tensor::CollapseShapeOp op) {
    if (op.getSrc() == src)
      result = op;
  });
  return result;
}

/// Forward declaration — see definition below for full description.
static LogicalResult handleTilingOrSplit(RewriterBase &rewriter, Operation *transformOp, Operation* op,
    ConvInfo csaConv, CSA csa, CSAStrategy res, SmallVector<int64_t, 2> strides, SmallVector<int64_t, 2> dilations,
    SmallVector<SmallVector<Operation*, 6>> &resultLoops, SmallVector<Operation*> &resultConvs);


/// @brief Compute the effective split size (`ss`) for linearized H×W windows.
///
/// SConv sometimes needs a correction when the flattened output spatial
/// dimension (Oh*Ow) is not an integer multiple of Nwin. In that case, CSA
/// may produce a split of the epilogue: the “steady-state” runs with fully
/// aligned tiles and an epilogue handles the tail. This function validates
/// whether the `split_size_windows` attached to ConvInfo matches the CSA
/// schedule (IS or WS) and returns it; otherwise returns 0.
///
/// @param csa       Convolution Slicing Analysis parameters (Nc, K2, K3, Nwin).
/// @param csaConv   Concrete convolution instance (shapes, precomputed sizes).
/// @param res       CSA strategy result (schedule and per-level tiling).
/// @returns         A non-zero `ss` if an edge-split must be reflected in the
///                  input affine formulas (Section 4, Eq. 15); 0 otherwise.
///
/// @note Paper reference: Section 4 (Input packing), especially Edge Packing
///       and Eq. (15) where an extra offset (`Eoff`) participates in Ts.
///
/// @see computeLinearInputIndices, computeMultiPackInputIndices
int64_t getValidSplitSize(const CSA &csa, const ConvInfo &csaConv, const CSAStrategy &res) {

  int64_t expectedSplitSize = 0;
  int64_t splitSize = (csaConv.output_rows * csaConv.output_cols) % csa.mK_.nwindows;

  if (res.schd == WS) {
    expectedSplitSize = csaConv.output_rows * csaConv.output_cols - res.k2 * csa.mK_.nwindows - splitSize;
    if (csaConv.split_size_windows == expectedSplitSize)
      return csaConv.split_size_windows;
  } else if (res.schd == IS) {
    expectedSplitSize = csaConv.output_rows * csaConv.output_cols - res.k3 * csa.mK_.nwindows - splitSize;
    if (csaConv.split_size_windows == expectedSplitSize)
      return csaConv.split_size_windows;
  }
  return 0;
}

/// Delinearizes the given composite `index` by the basis specified in `factors`.
static SmallVector<Value> unrollIndex(OpBuilder &b, Location loc, Value index,
                                      ArrayRef<int64_t> factors) {
  assert(!factors.empty() && "empty factor list");
  SmallVector<Value> basis;
  for (int64_t f : factors)
    basis.push_back(b.create<arith::ConstantOp>(loc, b.getIndexAttr(f)));
  FailureOr<SmallVector<Value>> multiIndex =
      affine::delinearizeIndex(b, loc, index, basis);
  assert(!failed(multiIndex) && "Failed to linearize sconv index");
  return *multiIndex;
}

/// @brief Compute the linearized input index for a packed window element.
///
/// This materializes the affine expression that maps (Fh, Fw, Nwin) within a
/// tile to the source tensor’s linearized H×W dimension, honoring stride and
/// dilation. It corresponds to Section 4 equations for the “general case”
/// where row breaks can occur (Eqs. 11–12 combined with Eq. 8).
///
/// Symbols/Dims (notation parity with the paper):
///   - Ow: output width (windows per row).
///   - Iw: input width (columns of the input tensor).
///   - Ss: epilogue split-size in windows (0 if none).
///   - IOin/IOout: inner/outer loop iters selecting the current H×W tile.
///   - fhIndex, fwIndex, nwinIndex: local indices in {Fh, Fw, Nwin}.
///
/// Affine structure:
///   ILstart = IOin + IOout + Ss
///   iHw = (((ILstart + nwin) / Ow - ILstart / Ow) * strideH + fh*dilationH) * Iw
///         + ((ILstart + nwin) % Ow - ILstart % Ow) * strideW + fw*dilationW
///
/// @note Paper reference: Section 4, Eqs. (11), (12) and (8).
///
/// @see computeMultiPackInputIndices
static Value computeLinearInputIndices(
    OpBuilder &b, Location loc, Value fhIndex, Value fwIndex, Value nwinIndex, Value IOin,
    Value IOout, int64_t Ss, int64_t Ow, int64_t Iw, 
    SmallVector<int64_t, 2> strides, SmallVector<int64_t, 2> dilations) {

  MLIRContext *context = b.getContext();

  int64_t strideH = strides[0];
  int64_t strideW = strides[1];
  int64_t dilationH = dilations[0];
  int64_t dilationW = dilations[1];

  // Construct ILstart = IOin + IOout + Ss, using IOout as symbol
  AffineExpr d1, s0;
  bindDims(context, d1);
  bindSymbols(context, s0);
  AffineMap ILstartMap = AffineMap::get(1, 1, {d1 + s0 + Ss}, context);
  Value ILstart = affine::makeComposedAffineApply(b, loc, ILstartMap, {IOin, IOout});

  // Construct Hwindex (iHw) using ILstart as symbol
  AffineExpr dFh, dFw, nwin, s_ilstart;
  bindDims(context, dFh, dFw, nwin);
  bindSymbols(context, s_ilstart);
  AffineExpr iHwExpr = 
      ((((s_ilstart + nwin).floorDiv(Ow) - s_ilstart.floorDiv(Ow)) * strideH) +
       dFh * dilationH) * Iw +
      (((s_ilstart + nwin) % Ow - s_ilstart % Ow) * strideW) +
      dFw * dilationW;

  AffineMap iHwMap = AffineMap::get(3, 1, {iHwExpr}, context);
  Value HwIndex = b.create<affine::AffineApplyOp>(
      loc, 
      iHwMap,
      ValueRange{fhIndex, fwIndex, nwinIndex, ILstart});

  return HwIndex;
}

/// @brief Like computeLinearInputIndices, but for multi-packing
///
/// Multipacking adds a tile index `t` and skips full tiles at a higher level:
///   ILstart = IOout + Ss              (tiled at outer level)
///   iHw uses: (ILstart + t*Nwin + nwin)  // Eq. (14)
///
/// This is Section 4.2.1 (Multipacking), using Eq. (13) for Ts and Eq. (14)
/// for O_hw. Stride/dilation are applied identically to the single-pack case.
///
/// @note Paper reference: Section 4.2.1, Eqs. (13–14).
///
/// Preconditions:
///   - `res.schd == WS` (weight stationary) in the calling context.
///   - `Nwin` equals the microkernel window width.
static Value computeMultiPackInputIndices(
    OpBuilder &b, Location loc, Value tIndex, Value fhIndex, Value fwIndex, Value nwinIndex,
    Value IOut, int64_t Ss, int64_t Ow, int64_t Iw, int64_t Nwin,
    SmallVector<int64_t, 2> strides, SmallVector<int64_t, 2> dilations) {

  MLIRContext *context = b.getContext();

  int64_t strideH = strides[0];
  int64_t strideW = strides[1];
  int64_t dilationH = dilations[0];
  int64_t dilationW = dilations[1];

  Value ILstart;
  if (Ss == 0) 
    ILstart = IOut;
  else {
    // Construct ILstart = IOout + Ss, using IOout as symbol
    AffineExpr s0;
    bindSymbols(context, s0);
    AffineMap ILstartMap = AffineMap::get(0, 1, {s0 + Ss}, context);
    ILstart = affine::makeComposedAffineApply(b, loc, ILstartMap, {IOut});
  }

  // Construct Hwindex (iHw) using ILstart as symbol
  AffineExpr dt, dFh, dFw, dw, s_ilstart;
  bindDims(context, dt, dFh, dFw, dw);
  bindSymbols(context, s_ilstart);
  AffineExpr iHwExpr = 
      ((((s_ilstart + dt * Nwin + dw).floorDiv(Ow) - s_ilstart.floorDiv(Ow)) * strideH) + dFh * dilationH) * Iw +
      (((s_ilstart + dt * Nwin + dw) % Ow - s_ilstart % Ow) * strideW) + dFw * dilationW;

  AffineMap iHwMap = AffineMap::get(4, 1, {iHwExpr}, context);
  Value HwIndex = b.create<affine::AffineApplyOp>(
      loc, 
      iHwMap,
      ValueRange{tIndex, fhIndex, fwIndex, nwinIndex, ILstart});

  return HwIndex;
}

//===----------------------------------------------------------------------===//
// Helpers for promoteOpsOfTile
//===----------------------------------------------------------------------===//

/// @brief Build an affine.apply that computes the flattened H×W index for a window.
static Value createLinearizedAffineApply(OpBuilder &rewriter, Location loc,
                                         MLIRContext *context,
                                         Value ILrange, Value ILstart,
                                         int64_t strideH, int64_t strideW, int64_t ow, int64_t iw) {

  // Construct: (((ILrange/ow - ILstart/ow) * strideH) * iw + ((ILrange%ow - ILstart%ow) * strideW))
  AffineExpr s2, s1;
  bindSymbols(context, s2, s1);
  AffineExpr expr = (((s2.floorDiv(ow) - s1.floorDiv(ow)) * strideH) * iw +
                     (s2 % ow - s1 % ow) * strideW);
  AffineMap map = AffineMap::get(0, 2, {expr}, context);
  return rewriter.create<AffineApplyOp>(loc, map, ValueRange{ILrange, ILstart});
}

/// @brief Compute the starting linear index (ILstart) and window range offsets for the current tile.
static std::pair<Value, Value> computeILstartAndRange(OpBuilder &rewriter, Location loc,
                                                      MLIRContext *context,
                                                      Value inputValue0, Value inputValue1,
                                                      int64_t ss) {
  // Construct ILstart = inputValue0 + ss
  AffineExpr d0;
  bindDims(context, d0);
  AffineMap ILstartMap = AffineMap::get(1, 0, {d0 + ss}, context);
  Value ILstart = affine::makeComposedAffineApply(rewriter, loc, ILstartMap, {inputValue0});

  // Construct ILrange = inputValue1 + inputValue0 + ss
  AffineExpr d1, s0;
  bindDims(context, d1);
  bindSymbols(context, s0);
  AffineMap ILrangeMap = AffineMap::get(1, 1, {d1 + s0 + ss}, context);
  Value ILrange = affine::makeComposedAffineApply(rewriter, loc, ILrangeMap, {inputValue1, inputValue0});

  return {ILstart, ILrange};
}

/// @brief Hoist a pure affine.apply operation from the inner loop to the outer loop, preserving SSA uses.
static Value promoteSimpleAffineApply(OpBuilder &rewriter, Location loc,
                                      MLIRContext *context,
                                      Value d, Value s,
                                      int64_t strideH, int64_t strideW, int64_t ow, int64_t iw) {
  // Construct: (((d1 + s0)/ow - s0/ow) * strideH) * iw + ((d1 + s0)%ow - s0%ow) * strideW
  AffineExpr d1, s0;
  bindDims(context, d1);
  bindSymbols(context, s0);
  AffineExpr expr = (((d1 + s0).floorDiv(ow) - s0.floorDiv(ow)) * strideH) * iw +
                    ((d1 + s0) % ow - s0 % ow) * strideW;
  AffineMap map = AffineMap::get(1, 1, {expr}, context);
  return rewriter.create<AffineApplyOp>(loc, map, ValueRange{d, s});
}

/// @brief Hoist key affine.apply and extract_slice ops across loop levels.
///
/// After second-level tiling, we want the affine index computation for the
/// linearized input, and the first extract_slice (IS) or second (WS), to live
/// at the outer loop so that (a) the maps become invariant within the inner
/// microkernel loop and (b) we can rewrite them with the closed-form
/// formulas derived in Section 4.
///
/// Behavior:
///   - For IS: hoist affine.apply to the *outer* loop and promote the first
///     extract_slice. For certain rectangular convs with Fh==1, also patch
///     the slice to index via the new affine.apply.
///   - For WS: compute affine.apply at *inner* loop (needs inner iv), promote
///     the second extract_slice to the outer loop.
///
/// Preconditions:
///   - `loopOps[idx-1], loopOps[idx], loopOps[idx+1]` must be valid scf.for.
///   - The inner loop body must contain at least two tensor.extract_slice
///     (input, filter) and optionally the affine.apply we want to hoist.
///   - Tiling already produced the (root2, outer, inner) loops in order.
///
/// Notes:
///   - When `pointwise==true` we guard the affine path and only patch slices.
///   - The “split size” `ss` is folded in the new affine maps (see getValidSplitSize).
///
/// @returns success() on successful promotion; emitError otherwise.
///
/// @see Section 5.4 (tiling structure) and Section 4 (affine index rewrites).
static LogicalResult
promoteOpsOfTile(RewriterBase &rewriter, Operation *transformOp, CSA csa, 
    ConvInfo csaConv, CSAStrategy res, SmallVector<int64_t, 2> strides,
    SmallVector<Operation *> loopOps) {

  int idx = (res.k2 == 0 || res.k3 == 0 || res.tile_c == 0) ? 3 : 4;
  auto root2Loop = dyn_cast<scf::ForOp>(loopOps[idx - 1]);
  auto outerLoop = dyn_cast<scf::ForOp>(loopOps[idx]);
  auto innerLoop = dyn_cast<scf::ForOp>(loopOps[idx + 1]);
  if (!root2Loop || !outerLoop || !innerLoop)
    return transformOp->emitError("Loops must be scf.for");

  Block *root2Body = root2Loop.getBody();
  Block *outerBody = outerLoop.getBody();
  Block *innerBody = innerLoop.getBody();

  Operation *affineApply0 = nullptr;
  Operation *affineApply1 = nullptr;
  Operation *tensorExtractSlice1 = nullptr;
  Operation *tensorExtractSlice2 = nullptr;

  // Get the (only) affine operation at the root2 loop body
  for (Operation &op : root2Body->getOperations())
    if (!affineApply0 && isa<AffineApplyOp>(&op))
      affineApply0 = &op;

  // Get the target operations (affineApply & extractedSlice) in the inner loop body
  for (Operation &op : innerBody->getOperations()) {
    if (!affineApply1 && isa<AffineApplyOp>(&op))
      affineApply1 = &op;
    else if (!tensorExtractSlice1 && isa<tensor::ExtractSliceOp>(&op))
      tensorExtractSlice1 = &op;
    else if (!tensorExtractSlice2 && isa<tensor::ExtractSliceOp>(&op))
      tensorExtractSlice2 = &op;

    // Stop once all target operations are found
    if (affineApply1 && tensorExtractSlice1 && tensorExtractSlice2)
      break;
  }

  // In the case of pointwise, afineApply0 & affineApply1 equals null
  bool pointwise = (!affineApply0 && !affineApply1);
  if (!tensorExtractSlice1 || !tensorExtractSlice2)
    return transformOp->emitError("Failed to locate necessary operations for promotion");

  // Get context
  MLIRContext *context = rewriter.getContext();
  // Set insertion point at the start of root2 loop
  rewriter.setInsertionPointToStart(root2Body);

  Value inputValue0;
  Value inputValue1;
  Value newAffineApply1;

  Operation *promotedTensorExtractSlice1 = nullptr;
  Operation *promotedTensorExtractSlice2 = nullptr;
  Operation *newAffineApply0 = nullptr;

  int64_t iw = csaConv.input_cols;
  int64_t ow = csaConv.output_cols;
  int64_t fh = csaConv.kernel_rows;
  int64_t fw = csaConv.kernel_cols;
  int64_t strideH = strides[0];
  int64_t strideW = strides[1];

  int64_t ss = getValidSplitSize(csa, csaConv, res);

  if (!pointwise) inputValue0 = affineApply0->getOperand(0);

  if (!pointwise) {
    // Modify the first Affine op with new (fixed) map
    AffineExpr d0;
    bindDims(context, d0);
    AffineMap m0Map = AffineMap::get(1, 0, {((d0.floorDiv(ow)) * strideH) * iw + ((d0 % ow) * strideW)}, context);
    newAffineApply0 = rewriter.create<AffineApplyOp>(affineApply0->getLoc(), m0Map, inputValue0);
  }

  if (res.schd == IS) {
    rewriter.setInsertionPointToStart(outerBody);
    if (!pointwise) {
      Location loc = affineApply1->getLoc();
      inputValue1 = outerLoop.getInductionVar();
      if (ss == 0) {
        newAffineApply1 = promoteSimpleAffineApply(rewriter, loc, context, inputValue1, inputValue0, strideH, strideW, ow, iw);
      } else {
        auto [ILstart, ILrange] = computeILstartAndRange(rewriter, loc, context, inputValue0, inputValue1, ss);
        newAffineApply1 = createLinearizedAffineApply(rewriter, loc, context, ILrange, ILstart, strideH, strideW, ow, iw);
      }
    }

    // check if conv is rectangular & kernel_rows equals 1
    // In this case, insert the newAffineApply1 & modify tensorExtractSlice1
    if (pointwise && fh != fw && fh == 1) {
      Location loc = tensorExtractSlice1->getLoc();
      inputValue1 = outerLoop.getInductionVar();
      AffineExpr d0;
      bindDims(context, d0);
      AffineMap m1Map = AffineMap::get(1, 0, {((d0.floorDiv(ow)) * strideH) * iw + ((d0 % ow) * strideW)}, context);
      newAffineApply1 = rewriter.create<AffineApplyOp>(loc, m1Map, inputValue1);

      auto sliceOp = cast<tensor::ExtractSliceOp>(tensorExtractSlice1);
      SmallVector<OpFoldResult> offsets(sliceOp.getMixedOffsets());
      SmallVector<OpFoldResult> sizes(sliceOp.getMixedSizes());
      SmallVector<OpFoldResult> stridesVec(sliceOp.getMixedStrides());

      // Replace the induction var to affine.apply
      for (size_t i = 0; i < offsets.size(); ++i) {
        if (auto val = offsets[i].dyn_cast<Value>()) {
          if (val == inputValue1) {
            offsets[i] = newAffineApply1;
            break;
          }
        }
      }
      auto newSlice = rewriter.create<tensor::ExtractSliceOp>(
          loc, sliceOp.getSource(), offsets, sizes, stridesVec);
      sliceOp.getResult().replaceAllUsesWith(newSlice.getResult());
      rewriter.eraseOp(sliceOp);
      promotedTensorExtractSlice1 = newSlice.getOperation();

    } else {
      promotedTensorExtractSlice1 = rewriter.clone(*tensorExtractSlice1);
    }

  } else {
    if (!pointwise) {
      rewriter.setInsertionPointToStart(innerBody);
      Location loc = affineApply1->getLoc();
      inputValue1 = innerLoop.getInductionVar();
      if (ss == 0) {
        newAffineApply1 = promoteSimpleAffineApply(rewriter, loc, context, inputValue1, inputValue0, strideH, strideW, ow, iw);
      } else {
        auto [ILstart, ILrange] = computeILstartAndRange(rewriter, loc, context, inputValue0, inputValue1, ss);
        newAffineApply1 = createLinearizedAffineApply(rewriter, loc, context, ILrange, ILstart, strideH, strideW, ow, iw);
      }
    }
    rewriter.setInsertionPointToStart(outerBody);
    promotedTensorExtractSlice2 = rewriter.clone(*tensorExtractSlice2);
  }

  if (!pointwise) {
    root2Body->walk([&](Operation *op) {
      for (auto &operand : op->getOpOperands()) {
        if (operand.get() == affineApply0->getResult(0))
          operand.set(newAffineApply0->getResult(0));
        else if (operand.get() == affineApply1->getResult(0))
          operand.set(newAffineApply1);
      }
    });
  }

  root2Body->walk([&](Operation *op) {
    for (auto &operand : op->getOpOperands()) {
      if (res.schd == IS && operand.get() == tensorExtractSlice1->getResult(0))
        operand.set(promotedTensorExtractSlice1->getResult(0));
      else if (res.schd == WS && operand.get() == tensorExtractSlice2->getResult(0))
        operand.set(promotedTensorExtractSlice2->getResult(0));
    }
  });

  if (!pointwise) {
    if (affineApply0->use_empty()) rewriter.eraseOp(affineApply0);
    if (affineApply1->use_empty()) rewriter.eraseOp(affineApply1);
  }
  if (tensorExtractSlice1->use_empty()) rewriter.eraseOp(tensorExtractSlice1);
  if (tensorExtractSlice2->use_empty()) rewriter.eraseOp(tensorExtractSlice2);

  return success();
}

/// @brief Rewrite the ukernel linalg.generic to consume packed tensors.
///
/// Once packing is in place, the inner linalg.generic (ukernel) must read the
/// packed input/filter and expose iterator types + indexing maps consistent
/// with an outer-product microkernel:
///   iters: [parallel, parallel, parallel, reduction]
///   LHS map:   (d0,d1,d2,d3) -> (d0, d3, d2)    // input  tile
///   RHS map:   (d0,d1,d2,d3) -> (d3, d1)        // filter tile
///   OUT map:   (d0,d1,d2,d3) -> (d0, d1, d2)    // output tile
///
/// The body becomes: C += A * B, with types preserved (int/float).
///
/// Preconditions:
///   - `tiledOps.front()` is the ukernel linalg.generic of the current tile.
///   - `loopOps[idx], loopOps[idx+1]` are valid scf.for for outer/inner levels.
///   - Exactly one tensor::CollapseShapeOp result per loop body identifies the
///     packed value (outer = IS input / WS filter; inner = complement).
///
/// Paper reference: Section 5.5 (packing) and microkernel structure in Fig. 2.
///
/// Postconditions:
///   - The original ukernel op is replaced by a new linalg.generic that reads
///     from packed tensors and yields the same result type.
///   - Any users within the loop bodies are updated; dead ops are erased.
static LogicalResult
adjustLinalgOps(RewriterBase &rewriter, Operation *transformOp, CSAStrategy res,
                SmallVector<Operation *> &tiledOps, SmallVector<Operation *> loopOps) {

  // Get context
  MLIRContext *context = rewriter.getContext();

  // Validate input loops
  int idx = (res.k2 == 0 || res.k3 == 0 || res.tile_c == 0) ? 3 : 4;
  auto outerLoop = dyn_cast<scf::ForOp>(loopOps[idx]);
  auto innerLoop = dyn_cast<scf::ForOp>(loopOps[idx+1]);
  if (!outerLoop || !innerLoop)
    return transformOp->emitError("Loops must be scf.for");

  // Get the body of the loops
  Block *outerBody = outerLoop.getBody();
  Block *innerBody = innerLoop.getBody();

  // Get the uKernel (convOp)
  auto convOp = tiledOps.front();
  Location loc = convOp->getLoc();

  // set the insertion point
  rewriter.setInsertionPoint(convOp);

  // Cast uKernel to linalg::GenericOp
  auto linalgOp = dyn_cast<linalg::GenericOp>(convOp);
  if (!linalgOp) return transformOp->emitError("failed to get the uKernel (convOp)");

  // get input & filter based on the schedule
  Value outerPackOp;
  for (Operation &op : outerBody->getOperations()) {
    if (auto packOp = dyn_cast<tensor::CollapseShapeOp>(&op)) {
      outerPackOp = packOp.getResult();
      break;
    }
  }
  Value innerPackOp;
  for (Operation &op : innerBody->getOperations()) {
    if (auto packOp = dyn_cast<tensor::CollapseShapeOp>(&op)) {
      innerPackOp = packOp.getResult();
      break;
    }
  }

  // get the input & filter to new uKernel
  Value input = res.schd == IS ? outerPackOp : innerPackOp;
  Value filter = res.schd == WS ? outerPackOp : innerPackOp; 

  // Create the iterator types for new uKernel
  auto parallel = utils::IteratorType::parallel;
  auto reduction = utils::IteratorType::reduction;
  SmallVector<utils::IteratorType> newOpIterators = {parallel, parallel, parallel, reduction};

  // Create the new affine maps
  AffineExpr d0, d1, d2, d3;
  bindDims(context, d0, d1, d2, d3);
  auto lhsMap = AffineMap::get(4, 0, {d0, d3, d2}, context);
  auto rhsMap = AffineMap::get(4, 0, {d3, d1}, context);
  auto resultMap = AffineMap::get(4, 0, {d0, d1, d2}, context);

  // create the new uKernel
  auto genericOp = rewriter.create<linalg::GenericOp>(
    loc,
    linalgOp.getOutputs()[0].getType(),
    ValueRange{input, filter},
    linalgOp.getOutputs(),
    ArrayRef<AffineMap>{lhsMap, rhsMap, resultMap},
    newOpIterators, 
    [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
      Value mul = createMul(loc, args[0], args[1], args[2].getType(), nestedBuilder);
      Value add = createAdd(loc, mul, args[2], nestedBuilder);
      nestedBuilder.create<linalg::YieldOp>(nestedLoc, add);
    });

  // Replace all uses of the uKernel linalgOp to the new uKernel genericOp
  for (auto res : linalgOp.getResults()) {
    res.replaceAllUsesWith(genericOp.getResult(0));
  }

  // replace linalgOp with the new uKernel gennericOp
  rewriter.replaceOp(linalgOp, genericOp);

  // Replace all extractOp or affineOp that use the linalgOp.
  outerBody->walk([&](Operation *op) {
    if (auto extractOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
      if (extractOp.getSource().getDefiningOp() == linalgOp) {
        extractOp.replaceAllUsesWith(genericOp.getResult(0));
      }
    } else if (auto affineOp = dyn_cast<AffineApplyOp>(op)) {
      for (auto &operand : affineOp->getOpOperands()) {
        if (operand.get().getDefiningOp() == linalgOp) {
          operand.set(genericOp.getResult(0));
        }
      }
    }
  });

  // Collect all pending operations and remove them
  SmallVector<Operation *, 4> dependentOps;
  outerBody->walk<WalkOrder::PostOrder>([&](Operation *op) {
    if (llvm::any_of(op->getOperands(), [&](Value operand) {
          return operand.getDefiningOp() == linalgOp;
        })) {
      dependentOps.push_back(op);
    }
  });
  for (Operation *op : llvm::reverse(dependentOps)) {
    rewriter.eraseOp(op);
  }

  // Iterate through the tiledOps to find convOp and replace it to genericOp
  for (auto &op : tiledOps) {
    if (op == convOp) {
      op = genericOp;
      break;
    }
  }

  return success();
}

/// @brief Create the filter packing tensor and collapse it to 2D.
///
/// Packed layout (no replication):  [Ic, Fh, Fw, Nf]  → collapse → [Ic*Fh*Fw, Nf]
/// Iterates with a rank-4 linalg.generic that extracts from the original filter
/// (Nf, Ic, Fh, Fw) and yields to the packed tensor in-order.
///
/// Schedule placement:
///   - IS: insert at linalg.generic (so input remains stationary outside).
///   - WS: insert at the outer loop to align with weight-stationary policy.
///
/// Paper reference: Section 4.1 (Filter Packing), Eqs. (1–2).
///
/// results:
///   The packing is placed at the entry of the innermost loops (first or second)
///   created by the second-level tiling, depending on the IS or WS schedule.
static LogicalResult
applyFilterPacking(RewriterBase &rewriter, Operation *transformOp, CSAStrategy res,
                  SmallVector<Operation *> &tiledOps, SmallVector<Operation *> loopOps) {

  MLIRContext *context = rewriter.getContext();

  // select the loop based on IS or WS
  int idx = (res.k2 == 0 || res.k3 == 0 || res.tile_c == 0) ? 3 : 4;
  int loopIndex = res.schd == IS ? idx+1 : idx;

  // Cast to scf::ForOp the  selected loop
  auto loopOp = dyn_cast<scf::ForOp>(loopOps[loopIndex]);
  if (!loopOp) return transformOp->emitError("failed to get the inner scf::for op");
  Block *loopBody = loopOp.getBody();

  // Get the uKernel
  auto convOp = tiledOps.front();

  // Cast to linalg::GenericOp to get the input & filter, types & shapes
  auto linalgOp = dyn_cast<linalg::GenericOp>(convOp);
  if (!linalgOp) return transformOp->emitError("failed to get the inner convOp");

  // Locate the insertion point (scf::for Op if WS or linalg::generic Op if IS)
  Location loc = linalgOp.getLoc(); // location should never be null.
  if (res.schd == WS) {
    scf::ForOp innerLoop;
    for (Operation &op : loopBody->getOperations()) {
      if (auto loop = dyn_cast<scf::ForOp>(&op)) {
        innerLoop = loop;
        break;
      }
    }
    if (!innerLoop) return transformOp->emitError("failed to get the inner forOp");
    rewriter.setInsertionPoint(innerLoop);
    loc = innerLoop->getLoc();
  }
  else {
    rewriter.setInsertionPoint(linalgOp);
  }

  SmallVector<Value> inputs = linalgOp.getInputs();
  Value input = inputs[0];
  Value filter = inputs[1];

  auto inputType = cast<ShapedType>(input.getType());
  auto filterType = cast<ShapedType>(filter.getType());
  auto inputShape = inputType.getShape();
  auto filterShape = filterType.getShape();

  // Compute the Packed Filter Shape: {ic × fh × fw, nf}
  int64_t ic = inputShape[1];
  int64_t nf = filterShape[0];
  int64_t fh = filterShape[2];
  int64_t fw = filterShape[3];
  SmallVector<int64_t, 2> filterPackingShape = {ic, fh, fw, nf};
  Value filterPacking = rewriter.create<tensor::EmptyOp>(loc, filterPackingShape, filterType.getElementType());

  auto nloops = filterPackingShape.size();
  auto parallel = utils::IteratorType::parallel;
  SmallVector<utils::IteratorType, 2> packingIterators(nloops, parallel);
  SmallVector<AffineMap, 3> packingIndexingMaps = {AffineMap::getMultiDimIdentityMap(nloops, context)};

  auto packingTensor = rewriter.create<linalg::GenericOp>(
    loc, filterPacking.getType(),
    /*inputs=*/ValueRange{},
    /*outputs=*/filterPacking,
    /*indeexingMaps=*/ packingIndexingMaps,
    /*iteratorTypes=*/ packingIterators,
    [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
      // Get the iterators
      Value icIndex = nestedBuilder.create<linalg::IndexOp>(loc, 0);
      Value fhIndex = nestedBuilder.create<linalg::IndexOp>(loc, 1);
      Value fwIndex = nestedBuilder.create<linalg::IndexOp>(loc, 2);
      Value nfIndex = nestedBuilder.create<linalg::IndexOp>(loc, 3);

      // Create the extraction indices for the original filter tensor
      SmallVector<Value> extractionIndices{nfIndex, icIndex, fhIndex, fwIndex};

      // Extract the value from the original filter tensor
      Value filterVal = nestedBuilder.create<tensor::ExtractOp>(loc, filter, extractionIndices);

      // Yield the extracted value into the packed tensor
      nestedBuilder.create<linalg::YieldOp>(nestedLoc, filterVal);
    });

  packingTensor->setAttrs({{"packing", rewriter.getStringAttr("filter")},
                           {"multipacking", rewriter.getBoolAttr(false)}});

  // Collapsing [Ic, Fh, Fw, Nf] → [Ic*Fh*Fw, Nf] matches the “register-friendly”
  // 2D panel expected by the outer-product microkernel (Fig. 2 of the paper).
  SmallVector<ReassociationIndices> collapseDims = {{0, 1, 2}, {3}};
  Value collapsedTensor = rewriter.create<tensor::CollapseShapeOp>(
      loc,
      RankedTensorType::get({filterPackingShape[0] * filterPackingShape[1] * filterPackingShape[2],
        filterPackingShape[3]}, filterType.getElementType()),
      packingTensor.getResult(0),
      collapseDims);

  return success();
}

/// @brief Create the input packing tensor and collapse it to 3D.
///
/// Packed layout (with replication across windows):
///   [N, Ic, Fh, Fw, Nwin] → collapse → [N, Ic*Fh*Fw, Nwin]
/// Indexing follows Section 4 (Eqs. 11–12 and 8) and handles edge epilogue via
/// `ss = getValidSplitSize(...)`. The implementation computes iHw using
/// computeLinearInputIndices(), which folds stride/dilation and row wrapping.
///
/// Schedule placement:
///   - IS: insert packing close to the inner loop (input-stationary).
///   - WS: insert at ukernel op (weight-stationary).
///
/// Preconditions:
///   - `loopOps[loopIndex]` is the inner loop; `loopOps[outerIndex]` the outer.
///   - `Ow, Iw` are the *original* widths.
///
/// Paper reference: Section 4.2 (Input Packing), Eqs. (4–5, 8, 11–12, 15).
///
/// Results:
///   Apply the input packing at the start of the innermost loops (first or second) 
///   created by the second-level tiling, depending on the IS or WS schedule.
static LogicalResult
applyInputPacking(RewriterBase &rewriter, Operation *transformOp, ConvInfo csaConv,
    CSA csa, CSAStrategy res, SmallVector<int64_t, 2> strides, SmallVector<int64_t, 2> dilations,
    SmallVector<Operation *> &tiledOps, SmallVector<Operation *> loopOps) {

  MLIRContext *context = rewriter.getContext();

  int idx = (res.k2 == 0 || res.k3 == 0 || res.tile_c == 0) ? 3 : 4;
  // select the loops based on IS or WS
  int loopIndex = res.schd == IS ? idx : idx+1;
  int outerIndex = res.schd == IS ? 2 : 3;

  // Cast to scf::ForOp the inner loop
  auto loopOp = dyn_cast<scf::ForOp>(loopOps[loopIndex]);
  if (!loopOp) return transformOp->emitError("failed to get the inner scf::for op");
  Block *loopBody = loopOp.getBody();

  // Get the ukernel
  auto convOp = tiledOps.front();

  // Cast to linalg::GenericOp to get the input & filter, types & shapes
  auto linalgOp = dyn_cast<linalg::GenericOp>(convOp);
  if (!linalgOp) return transformOp->emitError("failed to get the uKernel");

  // Locate the insertion point (scf::for Op if IS or linalg::generic Op if WS)
  Location loc = linalgOp.getLoc(); // location should never be null.
  if (res.schd == IS) {
    scf::ForOp innerLoop;
    for (Operation &op : loopBody->getOperations()) {
      if (auto loop = dyn_cast<scf::ForOp>(&op)) {
        innerLoop = loop;
        break;
      }
    }
    if (!innerLoop) return transformOp->emitError("failed to get the inner loop");
    rewriter.setInsertionPoint(innerLoop);
    loc = innerLoop->getLoc();
  }
  else {
    rewriter.setInsertionPoint(linalgOp);
  }

  SmallVector<Value> inputs = linalgOp.getInputs();
  SmallVector<Value> outputs = linalgOp.getOutputs();
  Value input = inputs[0];
  Value filter = inputs[1];
  Value output = outputs[0];

  auto inputType = cast<ShapedType>(input.getType());
  auto inputShape = inputType.getShape();

  auto filterType = cast<ShapedType>(filter.getType());
  auto filterShape = filterType.getShape();

  auto outputType = cast<ShapedType>(output.getType());
  auto outputShape = outputType.getShape();

  int64_t n = inputShape[0];
  int64_t ic = inputShape[1];
  int64_t iw = csaConv.input_cols; // The iw of the original input
  int64_t fh = filterShape[2];
  int64_t fw = filterShape[3];
  int64_t oh = outputShape[1];
  int64_t ow = csaConv.output_cols; // The ow of the orignal output
  int64_t nw = csa.mK_.nwindows;

  // Compute the Packed Input Shape: {n, ic, fh, fw, nw}
  SmallVector<int64_t, 5> inputPackingShape = {n, ic, fh, fw, nw};
  Value inputPacking = rewriter.create<tensor::EmptyOp>(loc, inputPackingShape, inputType.getElementType());

  auto nloops = inputPackingShape.size();
  auto parallel = utils::IteratorType::parallel;
  SmallVector<utils::IteratorType, 5> packingIterators(nloops, parallel);
  SmallVector<AffineMap, 5> packingIndexingMaps = {AffineMap::getMultiDimIdentityMap(nloops, context)};

  // create the input packing
  Value IO_in  = dyn_cast<scf::ForOp>(loopOps[loopIndex]).getInductionVar();
  Value IO_out = dyn_cast<scf::ForOp>(loopOps[outerIndex]).getInductionVar();
  int64_t ss = getValidSplitSize(csa, csaConv, res);

  auto packingTensor = rewriter.create<linalg::GenericOp>(
    loc, inputPacking.getType(),
    /*inputs=*/ValueRange{},
    /*outputs=*/inputPacking,
    packingIndexingMaps,
    packingIterators,

    [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
      Value iN = nestedBuilder.create<linalg::IndexOp>(loc, 0);
      Value iC = nestedBuilder.create<linalg::IndexOp>(loc, 1);
      Value iFh = nestedBuilder.create<linalg::IndexOp>(loc, 2);
      Value iFw = nestedBuilder.create<linalg::IndexOp>(loc, 3);
      Value iNwin = nestedBuilder.create<linalg::IndexOp>(loc, 4);

      Value iHw = computeLinearInputIndices(
          nestedBuilder, nestedLoc, iFh, iFw, iNwin, IO_in, IO_out, ss, ow, iw, strides, dilations);

      // Create the extraction indices for the original input tensor
      SmallVector<Value> extractionIndices{iN, iC, iHw};

      // Extract the value from the original input tensor
      Value inputVal = nestedBuilder.create<tensor::ExtractOp>(loc, input, extractionIndices);

      // Yield the extracted value into the packed tensor
      nestedBuilder.create<linalg::YieldOp>(nestedLoc, inputVal);
    });

  packingTensor->setAttrs({{"packing", rewriter.getStringAttr("input")},
                           {"multipacking", rewriter.getBoolAttr(false)}});

  // Collapse [N, Ic, Fh, Fw, Nwin] → [N, Ic*Fh*Fw, Nwin] to form a 3D packed
  // tensor matching the microkernel’s expected input panel layout.
  // This layout linearizes the receptive field (Ic×Fh×Fw) for each window,
  // enabling contiguous memory access during the innermost computation.
  // See Section 4.2 (Input Packing), Eqs. (11–12) in the paper.
  SmallVector<ReassociationIndices> collapseDims = {{0}, {1, 2, 3}, {4}};
  Value collapsedTensor = rewriter.create<tensor::CollapseShapeOp>(
      loc,
      RankedTensorType::get({inputPackingShape[0],
        inputPackingShape[1]* inputPackingShape[2] * inputPackingShape[3],
        inputPackingShape[4]}, inputType.getElementType()),
      packingTensor.getResult(0),
      collapseDims);

  return success();
}

/// @brief Workaround for missing inner loop interchange effects in SCF tiler.
///
/// In IS schedule, the intended inner loop order may not be obtained using
/// `tileUsingSCF` with `innerInterchange`. This routine swaps the induction
/// variables and loop bounds manually to achieve the desired order.
///
/// Preconditions:
///   - Only applicable to IS. Early-return otherwise.
///   - `loopOps[idx]` and `loopOps[idx+1]` are valid scf.for.
///
/// Caveats:
///   - This is a structural rewrite; verify future SCF tiler changes before
///     removing it. Keep tests that pin the intended loop order.
///
/// Paper reference: aligns with Fig. 2 scheduling layers for IS.
static LogicalResult
swapInductionVars(RewriterBase &rewriter, Operation *transformOp, CSAStrategy res,
                  SmallVector<Operation *> &tiledOps, SmallVector<Operation *> &loopOps) {

  if (res.schd != IS) return success();

  int idx = (res.k2 == 0 || res.k3 == 0 || res.tile_c == 0) ? 3 : 4;
  auto outerLoop = dyn_cast<scf::ForOp>(loopOps[idx]);
  auto innerLoop = dyn_cast<scf::ForOp>(loopOps[idx+1]);
  if (!outerLoop || !innerLoop)
    return transformOp->emitError("Loops must be scf.for");

  // Collect induction variables and loop bodies
  Value outerIndVar = outerLoop.getInductionVar();
  Value innerIndVar = innerLoop.getInductionVar();
  Block &outerBody = *outerLoop.getBody();
  Block &innerBody = *innerLoop.getBody();

  // Introduce temporary variable to avoid conflicting replacements
  rewriter.setInsertionPointToStart(&outerBody);
  auto tempOp = rewriter.create<arith::ConstantIndexOp>(outerLoop.getLoc(), 0);
  Value tempIndVar = tempOp.getResult();

  // Replace outerIndVar with a temporary to prevent conflicts
  outerBody.walk([&](Operation *op) {
    op->replaceUsesOfWith(outerIndVar, tempIndVar);
  });

  // Replace innerIndVar with outerIndVar
  innerBody.walk([&](Operation *op) {
    op->replaceUsesOfWith(innerIndVar, outerIndVar);
  });

  // Replace temporary variable with innerIndVar in the outer loop body
  outerBody.walk([&](Operation *op) {
    op->replaceUsesOfWith(tempIndVar, innerIndVar);
  });

  // Erase the temporary variable immediately after the swap
  rewriter.eraseOp(tempOp);

  // Swap loop bounds and steps
  Value outerLowerBound = outerLoop.getLowerBound();
  Value outerUpperBound = outerLoop.getUpperBound();
  Value outerStep = outerLoop.getStep();

  Value innerLowerBound = innerLoop.getLowerBound();
  Value innerUpperBound = innerLoop.getUpperBound();
  Value innerStep = innerLoop.getStep();

  outerLoop.setLowerBound(innerLowerBound);
  outerLoop.setUpperBound(innerUpperBound);
  outerLoop.setStep(innerStep);

  innerLoop.setLowerBound(outerLowerBound);
  innerLoop.setUpperBound(outerUpperBound);
  innerLoop.setStep(outerStep);

  return success();
}

/// @brief WS-only optimization: pack multiple input tiles (K2) at a higher level.
///
/// Adds a dimension Ti (num tiles) so the packed input becomes:
///   [Ni, Ti, Nc, Fh, Fw, Nwin]
/// and we index the source using (t*Nwin + nwin) (Eq. 14).
///
/// Placement:
///   - Hoisted to the nest/outer loop so data reuse matches the L2 policy.
///   - An affine.apply is inserted at the beginning of the inner loop to index
///     the per-iteration slice (floorDiv by Nwin).
///
/// Paper reference: Section 4.2.1 (Multipacking).
///
/// Preconditions:
///   - `res.schd == WS`.
///   - `nestLoop`, `outerLoop`, `innerLoop` exist and are scf.for.
static LogicalResult
inputMultipackingOpt(RewriterBase &rewriter, Operation *transformOp, ConvInfo csaConv, CSA csa,
    CSAStrategy res, SmallVector<int64_t, 2> strides, SmallVector<int64_t, 2> dilations,
    SmallVector<Operation *> &tiledOps, SmallVector<Operation *> loopOps) {

  if (res.schd != WS)
    return success();

  MLIRContext *context = rewriter.getContext();

  int idx = (res.k2 == 0 || res.k3 == 0 || res.tile_c == 0) ? 2 : 3;
  auto nestLoop = dyn_cast<scf::ForOp>(loopOps[idx]);
  auto outerLoop = dyn_cast<scf::ForOp>(loopOps[idx+1]);
  auto innerLoop = dyn_cast<scf::ForOp>(loopOps[idx+2]);
  if (!nestLoop || !outerLoop || !innerLoop)
    return transformOp->emitError("Loops must be scf.for");

  Location loc = outerLoop.getLoc();
  rewriter.setInsertionPoint(outerLoop);

  // First, find the input & filter extracted_slice in the nestLoop
  Operation *inputSlice = nullptr;
  Operation *filterSlice = nullptr;
  int count = 0;
  nestLoop.getBody()->walk([&](tensor::ExtractSliceOp op) {
    if (!inputSlice) {  // inputSlice is the first occurrence of ExtractSliceOp
      inputSlice = op;
    }
    if (count == 1) {  // filterSlice is the second occurence
      filterSlice = op;
    }
    count++;
  });

  // Then, get the input extracted_slice used by the old inputPacking
  Operation *extractedSlice = nullptr;
  innerLoop.getBody()->walk([&](tensor::ExtractSliceOp op) {
    if (!extractedSlice) { // It's the first one in the inner loop
      extractedSlice = op;
    }
  });

  // Last, create the inputMultiPacking with shape {Ni, Ti, K, Nwin} 
  // where Ti = res.k2, K = Nc * Fh * Fw, Nwin = csa.mK_.nwindows
  auto oper0Type = cast<ShapedType>(inputSlice->getOperand(0).getType());
  auto oper0Shape = oper0Type.getShape(); // Used to do a workaround. See comment below

  auto input = inputSlice->getResult(0);
  auto inputType = cast<ShapedType>(input.getType());
  auto inputShape = inputType.getShape();

  auto filter = filterSlice->getResult(0);
  auto filterType = cast<ShapedType>(filter.getType());
  auto filterShape = filterType.getShape();

  int64_t Ti = res.k2;          // num of tiles to the input 
  int64_t Ni = inputShape[0];
  int64_t Nc = inputShape[1];
  int64_t iw = csaConv.input_cols;
  int64_t Fh = filterShape[2];
  int64_t Fw = filterShape[3];
  int64_t ow = csaConv.output_cols;
  int64_t Nwin = csa.mK_.nwindows;

  SmallVector<int64_t, 6> inputPackingShape = {Ni, Ti, Nc, Fh, Fw, Nwin};
  Value inputPacking = rewriter.create<tensor::EmptyOp>(loc, inputPackingShape, inputType.getElementType());

  auto nloops = inputPackingShape.size();
  auto parallel = utils::IteratorType::parallel;
  SmallVector<utils::IteratorType, 6> packingIterators(nloops, parallel);
  SmallVector<AffineMap, 6> packingIndexingMaps = {AffineMap::getMultiDimIdentityMap(nloops, context)};

  Value IO_out = nestLoop.getInductionVar();
  int64_t ss = getValidSplitSize(csa, csaConv, res);

  auto newPackingTensor = rewriter.create<linalg::GenericOp>(
    loc, inputPacking.getType(),
    ValueRange{},
    inputPacking,
    packingIndexingMaps,
    packingIterators,
    [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
    Value iN = nestedBuilder.create<linalg::IndexOp>(loc, 0);
    Value iT = nestedBuilder.create<linalg::IndexOp>(loc, 1);
    Value iC = nestedBuilder.create<linalg::IndexOp>(loc, 2);
    Value iFh = nestedBuilder.create<linalg::IndexOp>(loc, 3);
    Value iFw = nestedBuilder.create<linalg::IndexOp>(loc, 4);
    Value iNwin = nestedBuilder.create<linalg::IndexOp>(loc, 5);

    Value iHw = computeMultiPackInputIndices(nestedBuilder, nestedLoc,
        iT, iFh, iFw, iNwin, IO_out, ss, ow, iw, Nwin, strides, dilations);

    SmallVector<Value> extractionIndices{iN, iC, iHw};
    Value inputVal = nestedBuilder.create<tensor::ExtractOp>(loc, inputSlice->getResult(0), extractionIndices);
    nestedBuilder.create<linalg::YieldOp>(nestedLoc, inputVal);
  });

  newPackingTensor->setAttrs({{"packing", rewriter.getStringAttr("input")},
                              {"multipacking", rewriter.getBoolAttr(true)}});

  // Create the affine.apply at beginning of innerLoop body
  // to indexing the new inputSlice
  rewriter.setInsertionPointToStart(innerLoop.getBody());
  Value affineIndex = rewriter.create<AffineApplyOp>(
      innerLoop.getLoc(), AffineMap::get(1, 0, getAffineDimExpr(0, context).floorDiv(Nwin), context), innerLoop.getInductionVar());
 
  // Define new offsets, sizes, and strides for new extractedSlice
  SmallVector<OpFoldResult, 6> newSliceOffsets = {
      rewriter.getIndexAttr(0), affineIndex, rewriter.getIndexAttr(0),
      rewriter.getIndexAttr(0), rewriter.getIndexAttr(0), rewriter.getIndexAttr(0) };

  SmallVector<OpFoldResult, 6> newSliceSizes = {
      rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
      rewriter.getIndexAttr(inputPackingShape[2]), rewriter.getIndexAttr(inputPackingShape[3]),
      rewriter.getIndexAttr(inputPackingShape[4]), rewriter.getIndexAttr(inputPackingShape[5])};

  SmallVector<OpFoldResult, 6> newSliceStrides = {
      rewriter.getIndexAttr(1), rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
      rewriter.getIndexAttr(1), rewriter.getIndexAttr(1), rewriter.getIndexAttr(1) };

  Value updatedExtractedSlice = rewriter.create<tensor::ExtractSliceOp>(
    innerLoop.getLoc(), newPackingTensor.getResult(0), newSliceOffsets, newSliceSizes, newSliceStrides);

  // Apply tensor.collapse_shape on the dimensions of new extractedSlice
  SmallVector<ReassociationIndices> collapseDims = {{0, 1}, {2, 3, 4}, {5}};
  Value collapsedSlice = rewriter.create<tensor::CollapseShapeOp>(
      innerLoop.getLoc(),
      RankedTensorType::get({inputPackingShape[0],
        inputPackingShape[2]* inputPackingShape[3] * inputPackingShape[4],
        inputPackingShape[5]}, inputType.getElementType()),
      updatedExtractedSlice,
      collapseDims);

  // Get the old uKernel & old inputPacking
  linalg::GenericOp linalgOp;
  linalg::GenericOp oInputPacking;
  innerLoop.getBody()->walk([&](linalg::GenericOp op) {
    if (!oInputPacking) {
      oInputPacking = op; // Must be the first one
    }
    linalgOp = op; // It's the last one in the loop
  });

  // Get the old CollapseShapeOp
  auto oldCollapseOp = findCollapseUsing(oInputPacking.getResult(0), innerLoop.getBody());

  // Set the insertion point to old uKernel
  rewriter.setInsertionPoint(linalgOp);

  auto linalgInputs = linalgOp.getInputs(); 
  SmallVector<Value, 2> newInputs;
  newInputs.push_back(collapsedSlice);  // Replace old inputPacking
  newInputs.push_back(linalgInputs[1]); // Same filterPacking

  // Get the indexingMaps of linalgOp
  SmallVector<AffineMap, 3> indexingMaps;
  if (auto indexingMapsAttr = linalgOp.getIndexingMaps()) {
    for (auto attr : indexingMapsAttr) {
      if (auto affineMapAttr = dyn_cast<AffineMapAttr>(attr)) {
        indexingMaps.push_back(affineMapAttr.getValue());
      } else {
        return transformOp->emitError("Expected AffineMapAttr in indexingMaps.");
      }
    }
  } else {
    return transformOp->emitError("IndexingMaps not found on linalgOp.");
  }

  // Create the iterator types for new uKernel
  auto reduction = utils::IteratorType::reduction; // parallel already defined
  SmallVector<utils::IteratorType> iteratorTypes = {parallel, parallel, parallel, reduction};

  // create the new uKernel
  auto newLinalgOp = rewriter.create<linalg::GenericOp>(
    linalgOp.getLoc(), linalgOp->getResultTypes(), 
    newInputs, linalgOp.getOutputs(),
    indexingMaps, iteratorTypes);

  // clone the body of linalgOp into the new uKernel
  rewriter.inlineRegionBefore(linalgOp.getRegion(), newLinalgOp.getRegion(),
                              newLinalgOp.getRegion().begin());

  // Replace all uses of the uKernel linalgOp to new uKernel
  for (auto res : linalgOp.getResults()) {
    res.replaceAllUsesWith(newLinalgOp.getResult(0));
  }

  // replace linalgOp with the new uKernel
  rewriter.replaceOp(linalgOp, newLinalgOp->getResults());

  Operation *emptyOp = nullptr;
  innerLoop.getBody()->walk([&](tensor::EmptyOp op) {
    emptyOp = op;
  });

  // Locate the old affineApply operations within the inner loop body
  Operation *affineApply1 = nullptr;
  Operation *affineApply2 = nullptr;
  Operation *affineApply3 = nullptr;
  for (Operation &op : innerLoop.getBody()->getOperations()) {
    if (!affineApply1 && isa<AffineApplyOp>(&op))
      affineApply1 = &op; // It's the new AffineApply inserted 
    else if (!affineApply2 && isa<AffineApplyOp>(&op))
      affineApply2 = &op;
    else if (!affineApply3 && isa<AffineApplyOp>(&op))
      affineApply3 = &op;
    // Stop once all target operations are found
    if (affineApply1 && affineApply2 && affineApply3)
      break;
  }

  // Remove obsolet ops
  SmallVector<Operation *, 4> opsToRemove;
  opsToRemove.push_back(emptyOp);
  opsToRemove.push_back(affineApply2);
  opsToRemove.push_back(affineApply3);
  opsToRemove.push_back(extractedSlice); 
  opsToRemove.push_back(oInputPacking);
  opsToRemove.push_back(oldCollapseOp);

  for (Operation *op : llvm::reverse(opsToRemove)) {
    if (op && op->use_empty()) {
      rewriter.eraseOp(op);
    }
  }

  auto convOp = tiledOps.front();
  // Iterate through the tiledOps to find convOp and replace it to newLinalgOp
  for (auto &op : tiledOps) {
    if (op == convOp) {
      op = newLinalgOp;
      break;
    }
  }

  return success();
}

/// @brief IS-only optimization: pack multiple filter tiles at a higher level.
///
/// Adds a tile dimension Tf (number of filter tiles) so the packed filter becomes:
///   [Tf, Ic, Fh, Fw, Nf] → collapse → [Tf, Ic*Fh*Fw, Nf]
/// The source indexing uses (t*Nf + nf) within the inner loop, so that every
/// Nf iterations we advance to the next filter tile in the multipacked buffer.
///
/// Placement:
///   - Emitted at the outer loop of the second-level tiling to match the
///     input-stationary policy (filters vary across tiles).
///
/// Paper reference:
///   - Section 4.1 (Filter Packing)
///   - Section 4.2.1 (Multipacking, filter case)
///
/// Preconditions:
///   - `res.schd == IS`.
///   - `nestLoop`, `outerLoop`, and `innerLoop` are valid scf.for.
///   - Filter operand is normalized to (Nf, Ic, Fh, Fw).
///
/// Postconditions:
///   - Produces a multipacked panel per Tf; the inner loop consumes slices
///     selected via an affine index driven by the inner induction variable.
///
/// Notes:
///   - Keep the packed 2D view [Ic*Fh*Fw, Nf] to match the outer-product
///     microkernel’s RHS panel expectation (consistent with adjustLinalgOps()).
static LogicalResult
filterMultipackingOpt(RewriterBase &rewriter, Operation *transformOp,
                      CSAStrategy res, SmallVector<Operation *> &tiledOps,
                      SmallVector<Operation *> loopOps) {
  if (res.schd != IS)
    return success();

  MLIRContext *context = rewriter.getContext();

  int idx = (res.k2 == 0 || res.k3 == 0 || res.tile_c == 0) ? 2 : 3;
  auto nestLoop = dyn_cast<scf::ForOp>(loopOps[idx]);
  auto outerLoop = dyn_cast<scf::ForOp>(loopOps[idx+1]);
  auto innerLoop = dyn_cast<scf::ForOp>(loopOps[idx+2]);
  if (!nestLoop || !outerLoop || !innerLoop)
    return transformOp->emitError("Loops must be scf.for");

  Location loc = outerLoop.getLoc();
  rewriter.setInsertionPoint(outerLoop);

  // First, find the filter extracted_slice in the nestLoop
  Value filterSlice;
  int count = 0;
  nestLoop.getBody()->walk([&](tensor::ExtractSliceOp op) {
    if (count == 1) {  // It's the second occurrence of ExtractSliceOp
      filterSlice = op;
    }
    count++;
  });

  // Then, get the filter extracted_slice used by the old filterPacking
  Operation *extractedSlice = nullptr;
  innerLoop.getBody()->walk([&](tensor::ExtractSliceOp op) {
    if (!extractedSlice) { // It's the first one in the inner loop
      extractedSlice = op;
    }
  });

  // Last, create the filterMultiPacking with shape {Tf, Nc × Fh × Fw, Nf}
  auto filter = extractedSlice->getResult(0);
  auto filterType = cast<ShapedType>(filter.getType());
  auto filterShape = filterType.getShape();
  int64_t Tf = res.k2;          // num of tiles to the filter 
  int64_t Nf = filterShape[0];
  int64_t Nc = filterShape[1];
  int64_t Fh = filterShape[2];
  int64_t Fw = filterShape[3];
  SmallVector<int64_t, 5> filterPackingShape = {Tf, Nc, Fh, Fw, Nf};
  Value filterPacking = rewriter.create<tensor::EmptyOp>(loc, filterPackingShape, filterType.getElementType());

  auto nloops = filterPackingShape.size();
  auto parallel = utils::IteratorType::parallel;
  SmallVector<utils::IteratorType, 5> packingIterators(nloops, parallel);
  SmallVector<AffineMap, 5> packingIndexingMaps = {AffineMap::getMultiDimIdentityMap(nloops, context)};

  auto newPackingTensor = rewriter.create<linalg::GenericOp>(
    loc, filterPacking.getType(),
    ValueRange{},
    filterPacking,
    packingIndexingMaps,
    packingIterators,
    [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
    Value index0 = nestedBuilder.create<linalg::IndexOp>(loc, 0); // iTf
    Value index1 = nestedBuilder.create<linalg::IndexOp>(loc, 1); // iNc
    Value index2 = nestedBuilder.create<linalg::IndexOp>(loc, 2); // iFh
    Value index3 = nestedBuilder.create<linalg::IndexOp>(loc, 3); // iFw
    Value index4 = nestedBuilder.create<linalg::IndexOp>(loc, 4); // iNf

    AffineExpr d0, d1;
    bindDims(context, d0, d1);
    
    // Index Tf via (iv_inner / Nf) so that each group of Nf inner iterations
    // switches to the next filter tile (analogous to the input multipacking scheme).
    AffineMap tfMap = AffineMap::get(2, 0, {d0 * Nf + d1}, context);
    Value TfIndex = rewriter.create<affine::AffineApplyOp>(loc, tfMap, ValueRange{index0, index4});

    // Create the extraction indices for the original filter tensor
    SmallVector<Value> extractionIndices{TfIndex, index1, index2, index3};

    Value filterVal = nestedBuilder.create<tensor::ExtractOp>(loc, filterSlice, extractionIndices);
    nestedBuilder.create<linalg::YieldOp>(nestedLoc, filterVal);
  });

  newPackingTensor->setAttrs({{"packing", rewriter.getStringAttr("filter")},
                              {"multipacking", rewriter.getBoolAttr(true)}});

  // Create the affine.apply at beginning of innerLoop body to indexing the new filterSlice
  rewriter.setInsertionPointToStart(innerLoop.getBody());
  Value affineIndex = rewriter.create<AffineApplyOp>(
      innerLoop.getLoc(), AffineMap::get(1, 0, getAffineDimExpr(0, context).floorDiv(Nf), context), innerLoop.getInductionVar());

  // Define new offsets, sizes, and strides for new extractedSlice
  SmallVector<OpFoldResult, 5> newSliceOffsets = {
      affineIndex, rewriter.getIndexAttr(0), rewriter.getIndexAttr(0), rewriter.getIndexAttr(0), rewriter.getIndexAttr(0) };

  SmallVector<OpFoldResult, 5> newSliceSizes = {
      rewriter.getIndexAttr(1),
      rewriter.getIndexAttr(filterPackingShape[1]), rewriter.getIndexAttr(filterPackingShape[2]),
      rewriter.getIndexAttr(filterPackingShape[3]), rewriter.getIndexAttr(filterPackingShape[4])};

  SmallVector<OpFoldResult, 5> newSliceStrides = {
      rewriter.getIndexAttr(1), rewriter.getIndexAttr(1), rewriter.getIndexAttr(1), rewriter.getIndexAttr(1), rewriter.getIndexAttr(1) };

  Value updatedExtractedSlice = rewriter.create<tensor::ExtractSliceOp>(
      innerLoop.getLoc(), newPackingTensor.getResult(0), newSliceOffsets, newSliceSizes, newSliceStrides);

  // Apply tensor.collapse_shape on the dimensions of new extractedSlice
  SmallVector<ReassociationIndices> collapseDims = {{0, 1, 2, 3}, {4}};
  Value collapsedSlice = rewriter.create<tensor::CollapseShapeOp>(
      innerLoop.getLoc(),
      RankedTensorType::get({1 * filterPackingShape[1] * filterPackingShape[2] * filterPackingShape[3], filterPackingShape[4]},
                            filterType.getElementType()),
      updatedExtractedSlice,
      collapseDims);

  // Get the old uKernel & old filterPacking
  linalg::GenericOp linalgOp;
  linalg::GenericOp oFilterPacking;
  innerLoop.getBody()->walk([&](linalg::GenericOp op) {
    if (!oFilterPacking) {
      oFilterPacking = op; // Must be the first one
    }
    linalgOp = op; // It's the last one in the loop
  });

  // Get the old CollapseShapeOp
  auto oldCollapseOp = findCollapseUsing(oFilterPacking.getResult(0), innerLoop.getBody());

  // Set the insertion point to old uKernel
  rewriter.setInsertionPoint(linalgOp);

  auto linalgInputs = linalgOp.getInputs(); 
  SmallVector<Value, 2> newInputs;
  newInputs.push_back(linalgInputs[0]); // The same inputPacking
  newInputs.push_back(collapsedSlice);  // Replace the old filterPacking

  // Get the indexingMaps of linalgOp
  SmallVector<AffineMap, 3> indexingMaps;
  if (auto indexingMapsAttr = linalgOp.getIndexingMaps()) {
    for (auto attr : indexingMapsAttr) {
      if (auto affineMapAttr = dyn_cast<AffineMapAttr>(attr)) {
        indexingMaps.push_back(affineMapAttr.getValue());
      } else {
        return transformOp->emitError("Expected AffineMapAttr in indexingMaps.");
      }
    }
  } else {
    return transformOp->emitError("IndexingMaps not found on linalgOp.");
  }

  // Create the iterator types for new uKernel
  auto reduction = utils::IteratorType::reduction; // parallel already defined
  SmallVector<utils::IteratorType> iteratorTypes = {parallel, parallel, parallel, reduction};

  // create the new uKernel
  auto newLinalgOp = rewriter.create<linalg::GenericOp>(
    linalgOp.getLoc(), linalgOp->getResultTypes(), 
    newInputs, linalgOp.getOutputs(),
    indexingMaps, iteratorTypes);

  // clone the body of linalgOp into the new uKernel
  rewriter.inlineRegionBefore(linalgOp.getRegion(), newLinalgOp.getRegion(),
                              newLinalgOp.getRegion().begin());

  // Replace all uses of the uKernel linalgOp to new uKernel
  for (auto res : linalgOp.getResults()) {
    res.replaceAllUsesWith(newLinalgOp.getResult(0));
  }

  // replace linalgOp with the new uKernel
  rewriter.replaceOp(linalgOp, newLinalgOp->getResults());

  Operation *emptyOp = nullptr;
  innerLoop.getBody()->walk([&](tensor::EmptyOp op) {
    emptyOp = op;
  });

  // Remove obsolet ops
  SmallVector<Operation *, 4> opsToRemove;
  opsToRemove.push_back(emptyOp);
  opsToRemove.push_back(extractedSlice); 
  opsToRemove.push_back(oFilterPacking);
  opsToRemove.push_back(oldCollapseOp);

  for (Operation *op : llvm::reverse(opsToRemove)) {
    if (op && op->use_empty()) {
      rewriter.eraseOp(op);
    }
  }

  auto convOp = tiledOps.front();
  // Iterate through the tiledOps to find convOp and replace it to newLinalgOp
  for (auto &op : tiledOps) {
    if (op == convOp) {
      op = newLinalgOp;
      break;
    }
  }

  return success();
}

/// @brief Apply two-level tiling to the target convolution operation.
///
/// Performs hierarchical tiling of the convolution according to the CSA
/// (Convolution Slicing Analysis) parameters, producing a loop nest with
/// outer and inner tiles sized by (Nc, K2, K3, Nwin).
///
/// Behavior:
///   - The first-level tiling partitions the convolution into macro-tiles
///     that fit L2 cache (driven by Nc and K2).
///   - The second-level tiling refines each macro-tile into micro-tiles
///     aligned with the register-level microkernel (Nwin × Nf).
///   - The resulting loops form the structure over which packing and
///     multipacking are later applied.
///
/// Paper reference:
///   - Section 5.4 (Two-level Tiling)
///   - Figure 2 (Loop hierarchy and scheduling order)
///
/// Preconditions:
///   - The input op is a normalized linalg::GenericOp produced from a
///     linalg::Conv2DNchwFchwOp by SConv normalization.
///   - CSA analysis (`res`) provides valid tile sizes and schedule (IS/WS).
///
/// Postconditions:
///   - Emits a pair of nested scf.for loops per tiling level.
///   - Returns the tiled op and its loop handles for further packing steps.
///
/// Notes:
///   - The loop interchange order may differ between IS and WS; in the
///     Input-Stationary case, swapInductionVars() is invoked to ensure
///     the expected inner loop order when innerInterchange is ineffective.
static LogicalResult
applyTileTo(RewriterBase &rewriter, Operation *transformOp, Operation *target, ConvInfo csaConv,
    CSA csa, CSAStrategy res, SmallVector<int64_t, 2> strides, SmallVector<int64_t, 2> dilations,
    SmallVector<Operation*, 7> &outResults) {

  SmallVector<Operation *> tiledOps;
  SmallVector<Operation *> loopOps;

  auto tilingInterfaceOp = dyn_cast<TilingInterface>(target);
  if (!tilingInterfaceOp)
    return transformOp->emitError("only TilingInterface ops are supported");

  // Assign tile sizes
  // Input Stationary: N, Nf * K2, Nwin * K3, Nc, Fh, Fw  
  // Weight Stationary: N, Nf * K3, Nwin * K2, Nc, Fh, Fw  
  int64_t nFTiles = csa.mK_.num_filters * (res.schd == IS ? res.k2 : res.k3);
  int64_t nWinTiles = csa.mK_.nwindows * (res.schd == IS ? res.k3 : res.k2);
  SmallVector<int64_t, 6> tileSize = {1, nFTiles, nWinTiles, res.tile_c, 0, 0};
  SmallVector<OpFoldResult> tileSizesOfr = getAsIndexOpFoldResult(rewriter.getContext(), tileSize);

  // Order:
  // Input Stationary: N, Nc, Nwin, Nf
  // Weight Stationary: N, Nc, Nf, Nwin
  int64_t outer = res.schd == IS ? 2 : 1;
  int64_t inner = res.schd == IS ? 1 : 2;
  SmallVector<int64_t, 4> tileInterchange = {0, 3, outer, inner};

  scf::SCFTilingOptions tilingOptions;
  tilingOptions.setTileSizes(tileSizesOfr).setInterchange(tileInterchange);
  tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);

  rewriter.setInsertionPoint(target);
  FailureOr<scf::SCFTilingResult> tiledResults =
      scf::tileUsingSCF(rewriter, tilingInterfaceOp, tilingOptions);
  if (failed(tiledResults))
    return transformOp->emitError("First level tiling operation was failed.");

  // Perform the replacement of tiled and fused values.
  rewriter.replaceOp(tilingInterfaceOp, tiledResults->loops.empty()
                             ? tiledResults->tiledOps.front()->getResults()
                             : tiledResults->loops.front()->getResults());

  // Perform the tiling in the inner convolution
  auto innerOp = tiledResults->tiledOps.front();

  SmallVector<int64_t, 6> innerTileSize = {0, csa.mK_.num_filters, csa.mK_.nwindows, 0, 0, 0};
  SmallVector<OpFoldResult> innerTileSizesOfr = getAsIndexOpFoldResult(rewriter.getContext(), innerTileSize);

  int64_t innerFOrder = res.schd == IS ? 1 : 0;
  int64_t innerSOrder = res.schd == IS ? 0 : 1;
  SmallVector<int64_t, 2> innerInterchange = {innerFOrder, innerSOrder};

  auto innerTilingInterfaceOp = dyn_cast<TilingInterface>(innerOp);
  if (!innerTilingInterfaceOp)
    return transformOp->emitError("only TilingInterface ops are supported.");
  scf::SCFTilingOptions innerTilingOptions;
  innerTilingOptions.setTileSizes(innerTileSizesOfr).setInterchange(innerInterchange);
  innerTilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);

  rewriter.setInsertionPoint(innerOp);
  FailureOr<scf::SCFTilingResult> innerTiledResults =
      scf::tileUsingSCF(rewriter, innerTilingInterfaceOp, innerTilingOptions);
  if (failed(innerTiledResults))
    return transformOp->emitError("Second level tiling operation was failed.");

  // Perform the replacement of tiled and fused values.
  rewriter.replaceOp(innerTilingInterfaceOp, innerTiledResults->loops.empty()
                             ? innerTiledResults->tiledOps.front()->getResults()
                             : innerTiledResults->loops.front()->getResults());

  // Report back the relevant handles to the transform op.
  tiledOps.push_back(innerTiledResults->tiledOps.front());
  for (Operation *loop : tiledResults->loops)
    loopOps.push_back(loop);
  for (Operation *loop : innerTiledResults->loops)
    loopOps.push_back(loop);

  // ======== variable used for debug purposes:
  auto rootLoop = dyn_cast<scf::ForOp>(loopOps[0]);
  // ========

  // Swap the innner loops in the case of Input Stationary. (BUG in the interchange on scf::tileUsingSCF)
  LogicalResult result0 = swapInductionVars(rewriter, transformOp, res, tiledOps, loopOps);
  if (failed(result0)) return transformOp->emitError("failed to swap indvar Ops");
  // LLVM_DEBUG({
  //   DBGS() << "=== Loops after tiling (& swapInductionVars) === \n" << rootLoop << "\n";
  // });

  // Promote some inner loop Ops depending on the schedule (WS or IS)
  LogicalResult result1 = promoteOpsOfTile(rewriter, transformOp, csa, csaConv, res, strides, loopOps);
  if (failed(result1)) return transformOp->emitError("failed to hosting Ops");
  // LLVM_DEBUG({
  //   DBGS() << "=== Loops after promote === \n" << rootLoop << "\n";
  // });

  // Generate the filter packing
  LogicalResult result2 = applyFilterPacking(rewriter, transformOp, res, tiledOps, loopOps);
  if (failed(result2)) return transformOp->emitError("failed to apply the filter packing");
  // LLVM_DEBUG({
  //   DBGS() << "=== Loops after applyFilterPacking === \n" << rootLoop << "\n";
  // });

  // Generate the input packing
  LogicalResult result3 = applyInputPacking(rewriter, transformOp, csaConv, csa, res, strides, dilations, tiledOps, loopOps);
  if (failed(result3)) return transformOp->emitError("failed to apply the input packing");
  // LLVM_DEBUG({
  //   DBGS() << "=== Loops after applyInputPacking === \n" << rootLoop << "\n";
  // });

  // Fix the uKernel after packing input & filter
  LogicalResult result4 = adjustLinalgOps(rewriter, transformOp, res, tiledOps, loopOps);
  if (failed(result4)) return transformOp->emitError("failed to replace the uKernel after packing");
  LLVM_DEBUG({
    DBGS() << "=== Loops after adjustLinalgOps === \n" << rootLoop << "\n";
  });

  // Generate the filter Multi-Packing
  LogicalResult result5 = filterMultipackingOpt(rewriter, transformOp, res, tiledOps, loopOps);
  if (failed(result5)) return transformOp->emitError("failed to apply filter Multi-Packing optimization");
  LLVM_DEBUG({
    DBGS() << "=== Loops after filterMultipackingOpt === \n" << rootLoop << "\n";
  });

  // Generate the input Multi-Packing
  LogicalResult result6 = inputMultipackingOpt(rewriter, transformOp, csaConv, csa, res, strides, dilations, tiledOps, loopOps);
  if (failed(result6)) return transformOp->emitError("failed to apply input Multi-Packing optimization");
  LLVM_DEBUG({
    DBGS() << "=== Loops after all packings  === \n" << rootLoop << "\n";
  });

  // Add attributes to the microkernel
  auto ukernel = tiledOps.front();
  auto schedule = (res.schd == Scheduling::IS) ? "IS" : "WS";
  ukernel->setAttrs({{"microkernel", rewriter.getUnitAttr()},
                     {"schedule", rewriter.getStringAttr(schedule)}});

  // Store the results (Operation*) in the output variable (as Value)
  outResults.push_back(ukernel);
  for (auto &loop : loopOps)
    outResults.push_back(loop);

  return success();
}

//===----------------------------------------------------------------------===//
// Helpers for splitAndTileConvolution
//===----------------------------------------------------------------------===//

/// @brief Validate convolution dimensions and CSA parameters before performing any split.
/// Ensures that the dimension is valid and that the splitPoint, offsets, and sizes
/// are well-formed (i.e., static and within bounds).
LogicalResult validateSplitInputs(RewriterBase &rewriter, Operation *transformOp,
                                  TilingInterface op, unsigned dim,
                                  OpFoldResult splitPoint) {

  Location loc = op->getLoc();
  auto iterationSpace = op.getIterationDomain(rewriter);

  if (dim >= iterationSpace.size()) {
    return transformOp->emitError()
           << "Split dimension " << dim << " exceeds iteration domain size "
           << iterationSpace.size() << ".";
  }

  auto offset = iterationSpace[dim].offset;
  auto size = iterationSpace[dim].size;

  LLVM_DEBUG({ 
    DBGS() << "=== Iteration Space ===";
    for (unsigned i = 0; i < iterationSpace.size(); ++i) {
      DBGS() << "Dim " << i << ": ";
      OpFoldResult size = iterationSpace[i].size;
      if (auto attr = size.dyn_cast<Attribute>()) {
        attr.print(llvm::dbgs());
      } else if (auto val = size.dyn_cast<Value>()) {
        val.print(llvm::dbgs());
      } else {
        DBGS() << "Unknown\n";
      }
    }
  });

  // Ensure splitPoint is an index attribute (static)
  auto splitAttr = llvm::dyn_cast_if_present<Attribute>(splitPoint);
  if (!splitAttr)
    return transformOp->emitError("Expected static split point.");

  auto offsetAttr = llvm::dyn_cast_if_present<Attribute>(offset);
  auto sizeAttr = llvm::dyn_cast_if_present<Attribute>(size);
  if (!offsetAttr || !sizeAttr)
    return transformOp->emitError("Expected static offset and size in iteration space.");

  int64_t splitVal = cast<IntegerAttr>(splitAttr).getInt();
  int64_t offsetVal = cast<IntegerAttr>(offsetAttr).getInt();
  int64_t sizeVal = cast<IntegerAttr>(sizeAttr).getInt();

  if (splitVal >= sizeVal) {
    return transformOp->emitError()
           << "Invalid split point " << splitVal
           << " for iteration space size " << sizeVal << ".";
  }

  LLVM_DEBUG({ 
    DBGS() << "=== Split Point: Dim " << dim << " Size " << splitVal;
  });

  return success();
}

/// @brief Execute the actual tensor split along the chosen dimension,
/// producing main and edge parts.
/// This assumes the dimension is splittable (CSA already validated bounds).
std::pair<TilingInterface, TilingInterface>
performSplit(RewriterBase &rewriter, TilingInterface op,
             unsigned dimension, OpFoldResult splitPoint) {

  Location loc = op->getLoc();
  MLIRContext *context = rewriter.getContext();

  rewriter.setInsertionPoint(op); 

  // Defensive check: ensure dimension is valid
  auto iterationSpace = op.getIterationDomain(rewriter);
  if (dimension >= iterationSpace.size()) {
    llvm::errs() << "split dimension " << dimension << " out of bounds\n";
    return {TilingInterface(), TilingInterface()};
  }

  // Defensive check: static splitPoint must not exceed the size
  if (auto sizeAttr = iterationSpace[dimension].size.dyn_cast<Attribute>()) {
    auto splitAttr = llvm::dyn_cast_if_present<Attribute>(splitPoint);
    if (splitAttr) {
      int64_t splitVal = cast<IntegerAttr>(splitAttr).getInt();
      int64_t dimSize = cast<IntegerAttr>(sizeAttr).getInt();
      if (splitVal <= 0 || splitVal >= dimSize) {
        llvm::errs() << "splitPoint " << splitVal << " is invalid for dimension size " << dimSize << "\n";
        return {TilingInterface(), TilingInterface()};
      }
    }
  }

  // It actually performs the split
  return linalg::splitOp(rewriter, op, dimension, splitPoint);
}

/// @brief Split the convolution op into multiple regions (main + edge) based on
/// input and filter boundaries.
static LogicalResult splitConvolution(
    RewriterBase &rewriter, Operation *transformOp, Operation *target, int64_t splitDim,
    int64_t splitSize, Operation *&firstOp, Operation *&secondOp) {

  auto tilingInterfaceOp = dyn_cast<TilingInterface>(target);
  if (!tilingInterfaceOp)
    return transformOp->emitError("only TilingInterface ops are supported to split");

  OpFoldResult splitPoint = rewriter.getIndexAttr(splitSize);
  if (failed(validateSplitInputs(rewriter, transformOp, tilingInterfaceOp, splitDim, splitPoint))) {
    return transformOp->emitError("Invalid dimension or splitPoint to call linalg::splitOp");
  }
  // In case of splitVal == 0, the payload has only the edge case
  auto splitAttr = llvm::dyn_cast_if_present<Attribute>(splitPoint);
  int64_t splitVal = cast<IntegerAttr>(splitAttr).getInt();
  if (splitVal == 0) {
    secondOp = tilingInterfaceOp;
  } else {
    std::tie(firstOp, secondOp) = performSplit(rewriter, tilingInterfaceOp, splitDim, splitPoint);
  }

  LLVM_DEBUG({
    DBGS() << "=== Splitted kernels with Dim = " << splitDim << ", Split Size = " << splitSize << " ===\n";
    if (firstOp != nullptr) {
      DBGS() << "First :\n";
      firstOp->print(llvm::dbgs());
    }
    DBGS() << "Last :\n";
    secondOp->print(llvm::dbgs());
  });

  return success();
}

/// @brief Helper that creates and returns new linalg::GenericOp instances for
/// each split convolution region.
static LogicalResult
splitConvHelper(RewriterBase &rewriter, Operation *transformOp,
    Operation *target, int64_t splitDim, int64_t splitSize,
    Operation *&firstOp, Operation *&secondOp) {
  return splitConvolution(rewriter, transformOp, target, splitDim, splitSize, firstOp, secondOp);
}

/// @brief Apply a computed split to the current convolution and advance
/// to process the next fragment.
static LogicalResult
applySplitAndAdvance(RewriterBase &rewriter, Operation *transformOp,
    Operation *&currentOp, int64_t dim, int64_t size, CSAStrategy res,
    ConvInfo csaConv, CSA &csa, SmallVector<int64_t, 2> strides,
    SmallVector<int64_t, 2> dilations, const int64_t curEdgeCase,
    SmallVector<SmallVector<Operation *, 6>> &resultLoops,
    SmallVector<Operation *> &resultConvs) {

  Operation *first = nullptr, *second = nullptr;

  // Split the current operation
  if (failed(splitConvHelper(rewriter, transformOp, currentOp, dim, size, first, second)))
    return transformOp->emitError("Split failed at dimension ") << dim;

  // Apply the adjustment to a local copy of the strategy (zero out the current edge case)
  CSAStrategy localRes = res;
  if (curEdgeCase == 0) localRes.extra_k2 = 0;
  else if (curEdgeCase == 1) localRes.extra_k3 = 0;
  else localRes.extra_tile_c = 0;

  // Tile the 'first' part with the adjusted strategy
  if (first) {
    if (failed(handleTilingOrSplit(rewriter, transformOp, first, csaConv, csa, localRes, strides, dilations, resultLoops, resultConvs)))
      return transformOp->emitError("Tiling failed for the first part of the split at dimension ") << dim;
  }

  // Update the current operation for the next iteration
  currentOp = second;
  return success();
};

/// @brief Split edge-case regions (input/filter/CSA remainders) and apply two-level tiling.
///
/// Performs the edge handling and tiling pipeline over a convolution payload:
///   1) Detects and splits edge tiles due to input/filter shapes or CSA
///      remainder analysis (e.g., incomplete H×W windows or partial Nf).
///   2) For each resulting convolution (main + edge pieces), either:
///        - fast-path: if a split yields tiles smaller than the ukernel
///          thresholds (Nwin for windows, Nf for filters), keep it as-is
///          (only adjust indexing maps); or
///        - full-path: run two-level tiling (outer/inner) and proceed with
///          packing/multipacking according to the schedule (IS/WS).
///
/// Paper reference:
///   - Section 5.3 (Edge-case splitting)
///   - Section 5.4 (Two-level tiling) and Section 5.5 (Packing)
///
/// Preconditions:
///   - The input op is a normalized linalg::GenericOp produced from a named
///     2D conv (NCHW/FCHW).
///   - CSA result provides schedule (IS/WS) and tile parameters (Nc, K2, K3, Nwin).
///
/// Postconditions:
///   - Emits zero or more additional conv fragments for edges; each fragment is
///     either left in adjusted form (small remainder) or tiled for the full path.
///   - Returns handles to the tiled ops/loops for subsequent packing steps.
///
/// Notes:
///   - Small remainders (< Nwin or < Nf) are currently handled without padding
///     (maps adjusted only). A future pass may pad these fragments so they can
///     follow the same tiling+packing pipeline.
static LogicalResult
splitAndTileConvolution(RewriterBase &rewriter, Operation *transformOp, Operation* target, ConvInfo csaConv,
    CSA csa, CSAStrategy res, SmallVector<int64_t, 2> strides, SmallVector<int64_t, 2> dilations,
    SmallVector<SmallVector<Operation*, 6>> &resultLoops, SmallVector<Operation*> &resultConvs) {

  Operation *currentOp = target;
  int64_t splitSize = 0;

  // === Split edge case on k2 first
  if (res.extra_k2 > 0) {
    int64_t dim = (res.schd == WS ? 2 : 1);
    splitSize = (res.schd == WS)
                  ? csaConv.output_rows * csaConv.output_cols - res.extra_k2 * csa.mK_.nwindows - csaConv.split_size_windows
                  : csaConv.num_filters - res.extra_k2 * csa.mK_.num_filters - csaConv.split_size_filters;

    if (res.schd == WS)
      csaConv.split_size_windows = splitSize;
    else
      csaConv.split_size_filters = splitSize;

    if (failed(applySplitAndAdvance(rewriter, transformOp, currentOp, dim, splitSize,
            res, csaConv, csa, strides, dilations, 0, resultLoops, resultConvs)))
      return failure();
    res.k2 = res.extra_k2;
    res.extra_k2 = 0;
  }

  // === Split edge case on k3 next
  if (res.extra_k3 > 0) {
    int64_t dim = (res.schd == IS ? 2 : 1);
    splitSize = (res.schd == IS)
                  ? csaConv.output_rows * csaConv.output_cols - res.extra_k3 * csa.mK_.nwindows - csaConv.split_size_windows
                  : csaConv.num_filters - res.extra_k3 * csa.mK_.num_filters - csaConv.split_size_filters;

    if (res.schd == IS)
      csaConv.split_size_windows = splitSize;
    else
      csaConv.split_size_filters = splitSize;

    if (failed(applySplitAndAdvance(rewriter, transformOp, currentOp, dim, splitSize,
            res, csaConv, csa, strides, dilations, 1, resultLoops, resultConvs)))
      return failure();
    res.k3 = res.extra_k3;
    res.extra_k3 = 0;
  }

  // === Split edge case on tile_c last
  if (res.extra_tile_c > 0) {
    int64_t dim = 3; // always the channel dimension
    splitSize = csaConv.input_channels - res.extra_tile_c;

    if (failed(applySplitAndAdvance(rewriter, transformOp, currentOp, dim, splitSize,
            res, csaConv, csa, strides, dilations, 2, resultLoops, resultConvs)))
      return failure();
    res.tile_c = res.extra_tile_c;
    res.extra_tile_c = 0;
  }

  // Apply tiling to the final (last) portion of the convolution
  if (failed(handleTilingOrSplit(rewriter, transformOp, currentOp, csaConv, csa, res, strides, dilations, resultLoops, resultConvs)))
    return transformOp->emitError("Tiling failed on final convolution after all splits.");

  return success();
}

/// @brief Adjust the indexing map of the second linalg.generic for input edge tiles.
///
/// In edge cases where the microkernel operates on a partial row (i.e., the last
/// split of the input is smaller than Nwin), the default affine equations assume
/// that computation starts at the beginning of a full row. This causes a row
/// wrap in the final split. The adjustment below adds the split offset
/// (splitSize) to the affine map, ensuring that indexing correctly continues
/// from the proper macro offset within the input tensor.
static FailureOr<linalg::GenericOp> 
adjustSecondOpIndexingMap(RewriterBase &rewriter, Operation *secondOp, ConvInfo csaConv,
    SmallVector<int64_t, 2> strides, SmallVector<int64_t, 2> dilations, int64_t splitSize) {
  
  auto genericOp = dyn_cast<linalg::GenericOp>(secondOp);
  if (!genericOp)
    return failure();

  MLIRContext *context = rewriter.getContext();
  Location loc = genericOp.getLoc();

  // Prepare constant AffineExpr for splitSize
  AffineExpr offset = getAffineConstantExpr(splitSize, context);

  // Get dimension AffineExprs: d0, d1, d2, d3, d4, d5
  SmallVector<AffineExpr, 6> dims;
  for (unsigned i = 0; i < 6; ++i) {
    dims.push_back(getAffineDimExpr(i, context));
  }

  // Construct new AffineExpr for output index
  int64_t icols = csaConv.input_cols;
  int64_t ocols = csaConv.output_cols;
  int64_t strideH = strides[0];
  int64_t strideW = strides[1];
  int64_t dilationH = dilations[0];
  int64_t dilationW = dilations[1];

  AffineExpr d2_plus_offset = dims[2] + offset;
  AffineExpr term1 = (((d2_plus_offset.floorDiv(ocols)) - (offset.floorDiv(ocols))) * strideH + dims[4] * dilationH) * icols;
  AffineExpr term2 = ((d2_plus_offset % ocols - (offset % ocols)) * strideW) + dims[5] * dilationW;
  AffineExpr newd2Expr = term1 + term2;

  // Build the new indexing map: (d0, d3, <newd2Expr>)
  SmallVector<AffineExpr, 3> resultExprs = {dims[0], dims[3], newd2Expr};
  AffineMap newMap = AffineMap::get(6, 0, resultExprs, context);

  // Replace only the first indexing map
  SmallVector<AffineMap, 4> newIndexingMaps = genericOp.getIndexingMapsArray();
  newIndexingMaps[0] = newMap;

  // Update the 'indexing_maps' attribute
  auto attrMaps = llvm::to_vector<4>(
    llvm::map_range(newIndexingMaps, [&](AffineMap m) -> Attribute {
      return AffineMapAttr::get(m);
    }));
  genericOp->setAttr("indexing_maps", ArrayAttr::get(context, attrMaps));

  return genericOp;
}

/// @brief Decide whether to perform two-level tiling or edge-case splitting for the given convolution.
///
/// Based on CSA results and current convolution shape, this function determines
/// if the operation should go through the standard two-level tiling pipeline or
/// first be split due to edge conditions (input/filter/CSA remainders).
///
/// Used internally by multiple stages of the SConv pipeline to unify the decision
/// logic between treatEdgeTileConvolution() and splitAndTileConvolution().
static LogicalResult handleTilingOrSplit(RewriterBase &rewriter, Operation *transformOp, Operation* op,
    ConvInfo csaConv, CSA csa, CSAStrategy res, SmallVector<int64_t, 2> strides, SmallVector<int64_t, 2> dilations,
    SmallVector<SmallVector<Operation*, 6>> &resultLoops, SmallVector<Operation*> &resultConvs) {

  bool requiresSplit = (res.extra_k2 != 0) || (res.extra_k3 != 0) || (res.extra_tile_c != 0);
  if (requiresSplit) {
    if (failed(splitAndTileConvolution(rewriter, transformOp, op, csaConv, csa, res, strides, dilations, resultLoops, resultConvs)))
      return transformOp->emitError("Failed to apply split & tiling to the convolution operation");
  } else {
    SmallVector<Operation*, 7> firstResults;
    rewriter.setInsertionPoint(op);

    if (failed(applyTileTo(rewriter, transformOp, op, csaConv, csa, res, strides, dilations, firstResults)))
      return transformOp->emitError("Failed to apply tiling.");

    resultConvs.push_back(firstResults[0]);
    SmallVector<Operation*, 6> loopSet;
    for (int i = 1; i <= 6; ++i) {
      loopSet.push_back(firstResults[i]);
    }
    resultLoops.push_back(loopSet);
  }
  return success();
}

/// @brief Detect and handle edge-case convolutions before applying CSA-based tiling.
///
/// Checks for edge tiles caused by input or filter dimensions that are not
/// multiples of the microkernel sizes (Nwin, Nf). When such cases are found,
/// it splits the convolution into regular and remainder parts to ensure that
/// all valid regions can still follow the standard tiling and packing pipeline.
///
/// Paper reference: Section 5.3 (Edge-case splitting)
///
/// Postconditions:
///   - Produces one or more smaller convolutions covering edge regions.
///   - Returns the updated set of ops to be processed by splitAndTileConvolution().
static LogicalResult
treatEdgeTileConvolution(RewriterBase &rewriter, Operation *transformOp, Operation* target, ConvInfo csaConv,
    CSA csa, CSAStrategy res, SmallVector<int64_t, 2> strides, SmallVector<int64_t, 2> dilations,
    SmallVector<SmallVector<Operation*, 6>> &resultLoops, SmallVector<Operation*> &resultConvs) {

  MLIRContext *context = rewriter.getContext();
  Location loc = target->getLoc();
  Operation *currentOp = target;

  // Check for edge cases
  bool edgeInput = ((csaConv.output_rows * csaConv.output_cols) % csa.mK_.nwindows) != 0;
  bool edgeFilter = (csaConv.num_filters % csa.mK_.num_filters) != 0;

  if (edgeInput) {

    LLVM_DEBUG({
      DBGS() << "=== Edge of input ===\n";
    });

    // Handle edge case of input: split on dim=2
    csaConv.split_size_windows = (csaConv.output_rows * csaConv.output_cols) % csa.mK_.nwindows;
    int64_t splitSizeInput = (csaConv.output_rows * csaConv.output_cols) - csaConv.split_size_windows;
    int64_t splitDimInput = 2;

    Operation *firstOpInput = nullptr, *secondOpInput = nullptr;
    if (failed(splitConvolution(rewriter, transformOp, currentOp, splitDimInput, splitSizeInput, firstOpInput, secondOpInput)))
      return transformOp->emitError("Failed to split on input dimension");

    if (firstOpInput != nullptr && edgeFilter) {
      // Handle edge case of filter: split on dim=1, applied on firstOpInput
      csaConv.split_size_filters = csaConv.num_filters % csa.mK_.num_filters;
      int64_t splitSizeFilter = csaConv.num_filters - csaConv.split_size_filters;
      int64_t splitDimFilter = 1;

      LLVM_DEBUG({
        DBGS() << "=== Edge of filter ===\n";
      });

      Operation *firstOpFilter = nullptr, *secondOpFilter = nullptr;
      if (failed(splitConvolution(rewriter, transformOp, firstOpInput, splitDimFilter, splitSizeFilter, firstOpFilter, secondOpFilter)))
        return transformOp->emitError("Failed to split on filter dimension");

      if (firstOpFilter != nullptr) {
        // Apply tiling for firstOpFilter
        if (failed(handleTilingOrSplit(rewriter, transformOp, firstOpFilter, csaConv, csa, res, strides, dilations, resultLoops, resultConvs)))
          return transformOp->emitError("Failed on handleTilingOrSplit for filter edge.");
      }

      auto maybeGenericFilter = adjustSecondOpIndexingMap(rewriter, secondOpFilter, csaConv, strides, dilations, 0);
      if (failed(maybeGenericFilter))
        return transformOp->emitError("Failed to adjust indexing_map for small filter edge.");

    } else if (firstOpInput != nullptr) {
      // Apply tiling for firstOpInput
      if (failed(handleTilingOrSplit(rewriter, transformOp, firstOpInput, csaConv, csa, res, strides, dilations, resultLoops, resultConvs)))
        return transformOp->emitError("Failed on handleTilingOrSplit for input edge.");
    }

    // Adjust indexing map in the small kernel by adding offset 
    auto maybeGenericInput = adjustSecondOpIndexingMap(rewriter, secondOpInput, csaConv, strides, dilations, splitSizeInput);
    if (failed(maybeGenericInput))
      return transformOp->emitError("Failed to adjust indexing_map for small input edge.");

  } else if (edgeFilter) {

    LLVM_DEBUG({
      DBGS() << "=== Edge of filter ===\n";
    });

    csaConv.split_size_filters = csaConv.num_filters % csa.mK_.num_filters;
    int64_t splitSizeFilter = csaConv.num_filters - csaConv.split_size_filters;
    int64_t splitDimFilter = 1;

    Operation *firstOpFilter = nullptr, *secondOpFilter = nullptr;
    if (failed(splitConvolution(rewriter, transformOp, currentOp, splitDimFilter, splitSizeFilter, firstOpFilter, secondOpFilter)))
      return transformOp->emitError("Failed to split on filter dimension");

    if (firstOpFilter != nullptr) {
      if (failed(handleTilingOrSplit(rewriter, transformOp, firstOpFilter, csaConv, csa, res, strides, dilations, resultLoops, resultConvs)))
        return transformOp->emitError("Failed on handleTilingOrSplit for filter edge.");
    }

    // Adjust indexing map in the small kernel by adding offset 
    auto maybeGenericFilter = adjustSecondOpIndexingMap(rewriter, secondOpFilter, csaConv, strides, dilations, 0);
    if (failed(maybeGenericFilter))
      return transformOp->emitError("Failed to adjust indexing_map for small filter edge.");
  }

  return success();
}

/// @brief Entry point of the SConv transformation pipeline.
///
/// Implements the full SConv workflow over a linalg convolution payload:
///   1) Normalize named convolution ops (e.g., Conv2DNchwFchwOp) to
///      linalg::GenericOp with collapsed H×W dimensions.
///   2) Run the Convolution Slicing Analysis (CSA) to determine tiling
///      parameters (Nc, K2, K3, Nwin) and schedule policy (IS or WS).
///   3) Detect and process edge cases through treatEdgeTileConvolution(),
///      which may trigger splitting via splitAndTileConvolution().
///   4) Apply two-level tiling to the main and split convolutions, followed by
///      input/filter packing and optional multipacking depending on CSA.
///   5) Adjust affine maps and indexing of remainder tiles when necessary.
///
/// Paper reference:
///   - Sections 2–5 (overall algorithm and pipeline)
///   - Fig. 1 and Fig. 2 (SConv stages and loop hierarchy)
///
/// Preconditions:
///   - The payload operation must be a valid linalg 2D convolution
///     (NCHW/FCHW) or its normalized linalg::Generic equivalent.
///   - Dependent dialects (Linalg, Affine, SCF, Tensor) must be registered.
///
/// Postconditions:
///   - Produces a fully transformed convolution hierarchy suitable for
///     lowering to microkernel-level code or backend-specific schedules.
///   - Returns success() if all passes complete, or failure() otherwise.
///
/// Notes:
///   - This function orchestrates all major SConv phases and is the main
///     entry point invoked by the Transform dialect interpreter.
DiagnosedSilenceableFailure
transform::SConvOp::apply(transform::TransformRewriter &rewriter,
                          transform::TransformResults &results,
                          transform::TransformState &state) {

  // Initialize the default values of mKInfo & ArchInfo to the CSA Analysis.
  // It's dependent of the target machine. We used these values for Sorgan
  mKInfo mK = {16, 8, 128};
  ArchInfo arch = {
      (uint32_t)(32768 * 0.9),
      (uint32_t)(1048576 * 0.9),
      (uint32_t)(4194304 * 0.9),
      2, 10, 30, 300, 128
  };

  // Get the optional arguments
  auto mKInfoAttr = getMKInfo();
  auto archInfoAttr = getArchInfo();
  auto latencyAttr = getLatency();

  // If `mKInfoAttr` was provided, use the given values
  if (mKInfoAttr) {
    SmallVector<int64_t, 4> mKValues;
    for (auto attr : mKInfoAttr->getValue()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
        mKValues.push_back(intAttr.getInt());
      } else {
        return emitSilenceableError() << "Error: mKInfoAttr contains non-integer values!\n";
      }
    }
    if (mKValues.size() >= 2) {
      mK.nwindows = mKValues[0];
      mK.num_filters = mKValues[1];
      mK.noutput = mK.nwindows * mK.num_filters;
    } else {
      return emitSilenceableError() << "Error: mKInfoAttr does not contain enough values!\n";
    }
  }

  // If `archInfoAttr` was provided, use the given values
  if (archInfoAttr) {
    SmallVector<int64_t, 4> archValues;
    for (auto attr : archInfoAttr->getValue()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
        archValues.push_back(intAttr.getInt());
      } else {
        return emitSilenceableError() << "Error: archInfoAttr contains non-integer values!\n";
      }
    }
    if (archValues.size() >=4) {
      arch.l1_size = (uint32_t)(archValues[0] * 0.9);
      arch.l2_size = (uint32_t)(archValues[1] * 0.9);
      arch.l3_size = (uint32_t)(archValues[2] * 0.9);
      arch.cache_line = archValues[3];
    } else {
      return emitSilenceableError() << "Error: archInfoAttr does not contain enough values!\n";
    }
  }

  // If `latencyAttr` was provided, use the given values
  if (latencyAttr) {
    SmallVector<int64_t, 4> latencyValues;
    for (auto attr : latencyAttr->getValue()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
        latencyValues.push_back(intAttr.getInt());
      } else {
        return emitSilenceableError() << "Error: latencyAttr contains non-integer values!\n";
      }
    }
    if (latencyValues.size() >=4) {
      arch.l1_latency = latencyValues[0];
      arch.l2_latency = latencyValues[1];
      arch.l3_latency = latencyValues[2];
      arch.mem_latency = latencyValues[3];
    } else {
      return emitSilenceableError() << "Error: latencyAttr does not contain enough values!\n";
    }
  }

  // temporary variables to store all conv transformation
  SmallVector<Operation*> tempResultConvs;
  SmallVector<SmallVector<Operation*, 6>> tempResultLoops;

  // Get context and convOps
  MLIRContext *context = rewriter.getContext();
  auto targetOps = state.getPayloadOps(getTarget());

  for (Operation *targetOp : targetOps) {
    auto convOp = dyn_cast_or_null<linalg::Conv2DNchwFchwOp>(targetOp);
    if (!convOp)
      continue;  // expected only Conv2DNchwFchw convolutions for transformations

    // Starting generalize the named convolution
    rewriter.setInsertionPoint(convOp);
    Location loc = convOp.getLoc();

    SmallVector<Value> inputs = convOp.getDpsInputs();
    ValueRange outputs = convOp.getDpsInits();
    Value input = inputs[0];
    Value filter = inputs[1];
    Value output = outputs[0];

    auto inputType = cast<ShapedType>(input.getType());
    auto filterType = cast<ShapedType>(filter.getType());
    auto outputType = cast<ShapedType>(output.getType());

    if (!filterType.hasStaticShape())
      return emitSilenceableError() << "expected a static shape for the filter";

    if (!inputType.hasStaticShape())
      return emitSilenceableError() << "expected a static shape for the input";

    auto inputShape = inputType.getShape();
    auto filterShape = filterType.getShape();
    auto outputShape = outputType.getShape();
    int64_t n = outputShape[0];
    int64_t ic = inputShape[1];
    int64_t ih = inputShape[2];
    int64_t iw = inputShape[3];
    int64_t fn = filterShape[0];
    int64_t fh = filterShape[2];
    int64_t fw = filterShape[3];
    int64_t oc = outputShape[1];
    int64_t oh = outputShape[2];
    int64_t ow = outputShape[3];

    // Create the Collapsed shape of input tensor
    SmallVector<ReassociationIndices> inputReassocIndices = {{0}, {1}, {2, 3}};
    auto reshapedInputType = RankedTensorType::get({n, ic, ih * iw}, inputType.getElementType());
    Value reshapedInput = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedInputType, input, inputReassocIndices);

    // Create the Collapsed shape of output tensor
    SmallVector<ReassociationIndices> outputReassocIndices = {{0}, {1}, {2, 3}};
    auto reshapedOutputType = RankedTensorType::get({n, oc, oh * ow}, outputType.getElementType());
    Value reshapedOutput = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedOutputType, output, outputReassocIndices);

    // Create the affine maps, iterator types and output tensor shape
    auto parallel = utils::IteratorType::parallel;
    auto reduction = utils::IteratorType::reduction;
    SmallVector<utils::IteratorType> newOpIterators = {parallel, parallel, parallel, reduction, reduction, reduction};

    // Get the strides
    auto hstride = convOp.getStrides().getValues<int64_t>()[0];
    auto wstride = convOp.getStrides().getValues<int64_t>()[1];
    SmallVector<int64_t, 2> strides = {hstride, wstride};

    // Get the dilations
    auto hdilation = convOp.getDilations().getValues<int64_t>()[0];
    auto wdilation = convOp.getDilations().getValues<int64_t>()[1];
    SmallVector<int64_t, 2> dilations = {hdilation, wdilation};

    // Affine Expr definitions:
    // d0 = batch; d1 = filter; d2 = output height; d3 = output width; d4 = channels; d5 = filter height; d6 = filter width
    AffineExpr d0, d1, d3, d4, d5, d6;
    bindDims(context, d0, d1, d3, d4, d5, d6);
    auto lhsMap = AffineMap::get(6, 0, {d0, d4, (d3.floorDiv(oh) * hstride + d5 * hdilation) * iw + d3 % oh * wstride + d6 * wdilation}, context);
    auto rhsMap = AffineMap::get(6, 0, {d1, d4, d5, d6}, context);
    auto resultMap = AffineMap::get(6, 0, {d0, d1, d3}, context);

    // Create the new genericOp that replaces the named convolution
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc,
        reshapedOutputType,
        ValueRange{reshapedInput, inputs[1]},
        ValueRange{reshapedOutput},
        ArrayRef<AffineMap>{lhsMap, rhsMap, resultMap},
        newOpIterators,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          Value mul = createMul(loc, args[0], args[1], args[2].getType(), nestedBuilder);
          Value add = createAdd(loc, mul, args[2], nestedBuilder);
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, add);
        });

    // Create the Expanded Shape
    auto reshapedResult = rewriter.create<tensor::ExpandShapeOp>(loc, outputType, genericOp.getResults().front(), outputReassocIndices);

    // replace convOp with (reshapedOutput + genericOp + reshapedResult)
    rewriter.replaceOp(convOp, ArrayRef<Value>{reshapedResult});

    // Call the CSA Analysis
    ConvInfo csaConv = {ic, iw, oh, ow, fh, fw, fn, 0, 0, 4};
    CSA csa = createCSAPass(arch, csaConv, mK);
    CSAStrategy res = csa();
    
    LLVM_DEBUG({
      DBGS() << "schd " << res.schd << " K2 " << res.k2 << " R2 " << res.extra_k2 << " K3 " << res.k3
             << " R3 " << res.extra_k3 << " TC " << res.tile_c << " RT " << res.extra_tile_c;
      DBGS() << "=== GenericOp ===\n";
      genericOp->print(llvm::dbgs());
    });

    // check if has edge case of input or filter in the kernel.
    bool hasEdgeCase = ((oh * ow) % mK.nwindows != 0) || (fn % mK.num_filters != 0);
    if (hasEdgeCase) {
      if (failed(treatEdgeTileConvolution(rewriter, getOperation(), genericOp, csaConv, csa, res, strides, dilations, tempResultLoops, tempResultConvs)))
        return emitSilenceableError() << "Failed to treat ukernel edge cases to on of the convolution operation";
    }
    else {
      // Handle the edge case of CSA Analysis
      bool requiresSplit = (res.extra_k2 != 0) || (res.extra_k3 != 0) || (res.extra_tile_c != 0);
      if (requiresSplit) {
        if (failed(splitAndTileConvolution(rewriter, getOperation(), genericOp, csaConv, csa, res, strides, dilations, tempResultLoops, tempResultConvs)))
          return emitSilenceableError() << "Failed to apply split & tiling to the convolution operation";
      }
      else {
        SmallVector<Operation*, 7> localResults;
        if (failed(applyTileTo(rewriter, getOperation(), genericOp, csaConv, csa, res, strides, dilations, localResults)))
          return emitSilenceableError() << "Failed to apply tiling to one of the convolution operations";
        // The first result is the transformed linalg.generic (uKernel)
        tempResultConvs.push_back(localResults[0]);
        // Following, the generated loops
        SmallVector<Operation*, 6> loopSet;
        for (int i = 1; i <= 6; ++i)
          loopSet.push_back(localResults[i]);
        tempResultLoops.push_back(loopSet);
      }
    }
  }

  // Flatten tempResultLoops
  SmallVector<Operation*> flatResultLoops;
  for (const auto &loopSet : tempResultLoops) {
    flatResultLoops.append(loopSet.begin(), loopSet.end());
  }

  // Store results properly in TransformResults
  results.set(getOperation()->getOpResult(0), tempResultConvs);
  results.set(getOperation()->getOpResult(1), flatResultLoops);

  return DiagnosedSilenceableFailure::success();
}

void transform::SConvOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

LogicalResult
transform::SConvOp::verify() {
  // All necessary checks are done in the Apply
  return success();
}

void registerSConv(mlir::DialectRegistry &registry) {
  registry.addExtensions<SConv>();
}
