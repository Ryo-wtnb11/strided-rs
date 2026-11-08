//! Block size computation ported from Strided.jl
//!
//! This module computes optimal block sizes for cache-efficient iteration.
//! It uses Julia's `totalmemoryregion` for accurate memory region estimation
//! and `_computeblocks` for iterative block size reduction.
//!
//! # Arithmetic contract
//!
//! Byte strides, memory regions and halving scores are evaluated in `u128`
//! from exact byte magnitudes `|s| · elem_size`. For extents `d ≥ 1` whose
//! product fits in `usize` (callers check it with `kernel::total_len`), every
//! intermediate is exact. The result is then the block vector of the
//! mathematical-integer model, for any element size and any stride, provided
//! the core's stride-order table (operands × rank entries) has a representable
//! length and its storage is obtained.
//!
//! - **Checked-length fallback.** If that length overflows `usize`, the full
//!   dimensions are returned. They are a valid block vector (`1 ≤ b_i ≤ d_i`)
//!   but not, in general, the model's result.
//! - **Outside the shape-product precondition.** The region arithmetic
//!   saturates: both budget comparisons keep their unbounded outcome and
//!   nothing panics.

use crate::fuse::compute_costs;
use crate::inline::{InlineVec, OPERAND_CAP};
use crate::{BLOCK_MEMORY_SIZE, CACHE_LINE_SIZE};
use strided_view::auxiliary::index_order;

/// Operand-major scratch of the core's stride orders: inline up to
/// [`OPERAND_CAP`] operands of `INLINE_RANK` axes, spilling above.
type Scratch<T> = tinyvec::TinyVec<[T; OPERAND_CAP * strided_view::INLINE_RANK]>;

/// The core's per-operand metadata. Byte strides are formed on the fly from
/// the element strides (`|s| * elem_size`, exact in `u128`) rather than
/// stored; the stride orders sit in one flat table where row `k` holds
/// operand `k` at a pitch of the original rank on every recursion level, so
/// the suffix at `offset` is `[k * pitch + offset .. (k + 1) * pitch]`.
struct Rows<'a> {
    pitch: usize,
    strides: &'a [&'a [isize]],
    elem_size: u128,
    orders: &'a [usize],
}

impl<'a> Rows<'a> {
    fn operands(&self) -> usize {
        self.strides.len()
    }

    fn byte_strides(&self, operand: usize, offset: usize) -> impl Iterator<Item = u128> + 'a {
        let elem_size = self.elem_size;
        self.strides[operand][offset..]
            .iter()
            .map(move |&s| s.unsigned_abs() as u128 * elem_size)
    }

    fn orders(&self, operand: usize, offset: usize) -> &[usize] {
        &self.orders[operand * self.pitch + offset..(operand + 1) * self.pitch]
    }
}

/// Compute block sizes for tiled iteration.
///
/// This implementation follows Julia's `_computeblocks` algorithm on axes
/// that are already in iteration order:
/// 1. If the memory region fits in the block budget, use full dimensions
/// 2. Otherwise compute stride orders and costs, and reduce blocks by
///    cost-weighted halving and decrementing
///
/// Two exits return full dimensions before the orders are computed, both
/// with the same result as the full algorithm: a zero `elem_size` (every
/// byte stride is 0, so every stride order is 1 and every axis is kept), and
/// operands whose strides are all nonzero and strictly increasing in
/// magnitude (each operand's stride order is `1..=r`, so every axis is kept).
/// The region test and both exits read the strides directly; the core's
/// metadata is built only when they do not apply.
///
/// # Arguments
/// * `dims` - The dimensions, in iteration order, each at least 1
/// * `strides_list` - Slice of element stride arrays, one per array, in the
///   same order as `dims`
/// * `elem_size` - Size of each element in bytes
///
/// # Returns
/// Block sizes in iteration order
pub(crate) fn compute_block_sizes(
    dims: &[usize],
    strides_list: &[&[isize]],
    elem_size: usize,
) -> InlineVec<usize> {
    let mut blocks: InlineVec<usize> = dims.iter().copied().collect();
    if dims.is_empty() || elem_size == 0 {
        return blocks;
    }

    let elem_size = elem_size as u128;
    let fits = memory_region(
        dims,
        strides_list.iter().map(|strides| {
            strides
                .iter()
                .map(|&s| s.unsigned_abs() as u128 * elem_size)
        }),
    ) <= BLOCK_MEMORY_SIZE as u128;
    if fits || strides_list.iter().all(|s| strictly_increasing(s)) {
        return blocks;
    }

    let pitch = dims.len();
    // A table length beyond `usize` cannot be stored; full dimensions are a
    // valid block vector, though not the model's result in general.
    let Some(len) = strides_list.len().checked_mul(pitch) else {
        return blocks;
    };
    let mut orders: Scratch<usize> = Scratch::with_capacity(len);
    for strides in strides_list {
        // With `elem_size > 0` the byte strides are an exact positive
        // multiple of the element strides, so both have the same stride order.
        orders.extend_from_slice(&index_order(strides));
    }
    let costs = compute_costs(strides_list);
    let rows = Rows {
        pitch,
        strides: strides_list,
        elem_size,
        orders: &orders,
    };
    compute_blocks_at(&mut blocks, 0, &costs, &rows, BLOCK_MEMORY_SIZE);
    blocks
}

/// Whether every stride is nonzero and strictly larger in magnitude than the
/// previous one.
fn strictly_increasing(strides: &[isize]) -> bool {
    strides.first().is_some_and(|&s| s != 0)
        && strides
            .windows(2)
            .all(|w| w[0].unsigned_abs() < w[1].unsigned_abs())
}

/// [`compute_blocks_at`] over per-operand slices, for the unit tests.
#[cfg(test)]
fn compute_blocks(
    dims: &[usize],
    costs: &[isize],
    byte_strides: &[&[u128]],
    stride_orders: &[&[usize]],
    block_size: usize,
) -> InlineVec<usize> {
    let mut blocks: InlineVec<usize> = dims.iter().copied().collect();
    if dims.is_empty() {
        return blocks;
    }
    // The tests give byte strides directly: as element strides of a 1-byte
    // element they produce exactly those byte magnitudes.
    let strides: Vec<Vec<isize>> = byte_strides
        .iter()
        .map(|s| {
            s.iter()
                .map(|&b| isize::try_from(b).expect("test byte strides fit isize"))
                .collect()
        })
        .collect();
    let strides: Vec<&[isize]> = strides.iter().map(|s| s.as_slice()).collect();
    let flat_orders: Vec<usize> = stride_orders
        .iter()
        .flat_map(|s| s.iter().copied())
        .collect();
    let rows = Rows {
        pitch: dims.len(),
        strides: &strides,
        elem_size: 1,
        orders: &flat_orders,
    };
    compute_blocks_at(&mut blocks, 0, costs, &rows, block_size);
    blocks
}

/// Recursive core of Julia's `_computeblocks` on the axes `offset..` of
/// `blocks`, which hold the full dimensions there on entry and the blocks on
/// return. The leading `offset` axes are already fixed.
///
/// # Julia equivalent
/// ```julia
/// function _computeblocks(dims, costs, bytestrides, strideorders, blocksize)
///     if totalmemoryregion(dims, bytestrides) <= blocksize
///         return dims
///     end
///     # ... reduction logic
/// end
/// ```
fn compute_blocks_at(
    blocks: &mut [usize],
    offset: usize,
    costs: &[isize],
    rows: &Rows<'_>,
    block_size: usize,
) {
    if offset == blocks.len() {
        return;
    }
    let block_size_bytes = block_size as u128;
    let operands = rows.operands();
    let region = |blocks: &[usize]| {
        memory_region(blocks, (0..operands).map(|k| rows.byte_strides(k, offset)))
    };

    // If everything fits in cache, use full dims
    if region(&blocks[offset..]) <= block_size_bytes {
        return;
    }

    // Check if first dimension is smallest stride for all arrays
    let min_order = (0..operands)
        .filter_map(|k| rows.orders(k, offset).iter().min().copied())
        .min()
        .unwrap_or(1);
    if (0..operands).all(|k| rows.orders(k, offset).first() == Some(&min_order)) {
        // First dimension has smallest stride in all arrays
        // Keep first dimension, recurse on rest
        compute_blocks_at(blocks, offset + 1, costs, rows, block_size);
        return;
    }

    // Check if minimum stride is larger than block size
    let min_stride = (0..operands)
        .filter_map(|k| rows.byte_strides(k, offset).min())
        .min()
        .unwrap_or(0);
    let tail = &mut blocks[offset..];
    if min_stride > block_size_bytes {
        tail.fill(1);
        return;
    }

    // Iteratively reduce blocks
    let costs = &costs[offset..];

    // Phase 1: Halve until within 2x of target
    while region(tail) >= 2 * block_size_bytes {
        let Some(i) = last_argmax_weighted(tail, costs) else {
            break;
        };
        tail[i] = tail[i].div_ceil(2);
    }

    // Phase 2: Decrement until within target
    while region(tail) > block_size_bytes {
        let Some(i) = last_argmax_weighted(tail, costs) else {
            break;
        };
        tail[i] -= 1;
    }
}

/// Compute total memory region for cache considerations.
///
/// This estimates the memory footprint considering cache line effects.
/// Strides smaller than cache line contribute to contiguous region,
/// larger strides multiply the number of cache line blocks.
///
/// # Julia equivalent
/// ```julia
/// function totalmemoryregion(dims, bytestrides)
///     memoryregion = 0
///     for i in 1:length(bytestrides)
///         strides = bytestrides[i]
///         numcontigeouscachelines = 0
///         numcachelineblocks = 1
///         for (d, s) in zip(dims, strides)
///             if s < _cachelinelength
///                 numcontigeouscachelines += (d - 1) * s
///             else
///                 numcachelineblocks *= d
///             end
///         end
///         numcontigeouscachelines = div(numcontigeouscachelines, _cachelinelength) + 1
///         memoryregion += _cachelinelength * numcontigeouscachelines * numcachelineblocks
///     end
///     return memoryregion
/// end
/// ```
///
/// `byte_strides` yields each operand's byte stride magnitudes. The sums and
/// products saturate, which is exact whenever the extents' product fits in
/// `usize`.
fn memory_region<O, S>(dims: &[usize], byte_strides: O) -> u128
where
    O: Iterator<Item = S>,
    S: Iterator<Item = u128>,
{
    let cache_line = CACHE_LINE_SIZE as u128;
    let mut memory_region = 0u128;

    for strides in byte_strides {
        let mut num_contiguous_bytes = 0u128;
        let mut num_cache_line_blocks = 1u128;

        for (&d, s) in dims.iter().zip(strides) {
            if s < cache_line {
                // Small stride: contributes to contiguous region. The product
                // is below 2^70, so only the sum can saturate.
                num_contiguous_bytes =
                    num_contiguous_bytes.saturating_add(d.saturating_sub(1) as u128 * s);
            } else {
                // Large stride: multiplies cache line blocks
                num_cache_line_blocks = num_cache_line_blocks.saturating_mul(d as u128);
            }
        }

        // Convert to cache lines
        let contiguous_lines = num_contiguous_bytes / cache_line + 1;

        memory_region = memory_region.saturating_add(
            cache_line
                .saturating_mul(contiguous_lines)
                .saturating_mul(num_cache_line_blocks),
        );
    }

    memory_region
}

/// [`memory_region`] over per-operand slices, for the unit tests.
#[cfg(test)]
fn total_memory_region(dims: &[usize], byte_strides: &[&[u128]]) -> u128 {
    memory_region(dims, byte_strides.iter().map(|s| s.iter().copied()))
}

/// Find the last index with maximum (value - 1) * cost.
///
/// Julia: `_lastargmax((blocks .- 1) .* costs)`, over blocks above 1 only.
/// Costs are nonnegative (`compute_costs` gives at least 1); the score is
/// exact in `u128`.
fn last_argmax_weighted(blocks: &[usize], costs: &[isize]) -> Option<usize> {
    if blocks.is_empty() {
        return None;
    }

    let mut max_score = 0u128;
    let mut max_idx = None;

    for (i, (&b, &c)) in blocks.iter().zip(costs.iter()).enumerate() {
        if b <= 1 {
            continue;
        }
        let score = (b - 1) as u128 * c as u128;
        if score >= max_score {
            max_score = score;
            max_idx = Some(i);
        }
    }

    max_idx
}

#[cfg(test)]
#[path = "block/tests/tests.rs"]
mod tests;
