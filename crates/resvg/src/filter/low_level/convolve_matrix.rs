// Copyright 2020 the Resvg Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::{ImageRefMut, f32_bound};
use rgb::RGBA8;

/// An edges processing mode.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum EdgeMode {
    /// Skip pixels outside the image.
    None,
    /// Duplicate edge pixels.
    Duplicate,
    /// Wrap around the image.
    Wrap,
}

/// A convolve matrix.
#[derive(Clone, Copy, Debug)]
pub struct ConvolveMatrix<'a> {
    target_x: u32,
    target_y: u32,
    columns: u32,
    rows: u32,
    data: &'a [f32],
    divisor: f32,
    bias: f32,
    edge_mode: EdgeMode,
    preserve_alpha: bool,
}

impl<'a> ConvolveMatrix<'a> {
    /// Creates a new `ConvolveMatrix`.
    ///
    /// Returns `None` when:
    ///
    /// - `columns * rows != data.len()`
    /// - `target_x >= columns`
    /// - `target_y >= rows`
    /// - `divisor` is zero or not finite
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        target_x: u32,
        target_y: u32,
        columns: u32,
        rows: u32,
        data: &'a [f32],
        divisor: f32,
        bias: f32,
        edge_mode: EdgeMode,
        preserve_alpha: bool,
    ) -> Option<Self> {
        if (columns as usize).checked_mul(rows as usize)? != data.len()
            || target_x >= columns
            || target_y >= rows
            || !divisor.is_finite()
            || divisor == 0.0
        {
            return None;
        }

        Some(ConvolveMatrix {
            target_x,
            target_y,
            columns,
            rows,
            data,
            divisor,
            bias,
            edge_mode,
            preserve_alpha,
        })
    }

    #[inline]
    fn get(&self, x: u32, y: u32) -> f32 {
        self.data[(y * self.columns + x) as usize]
    }
}

/// Applies a convolve matrix.
///
/// Input image pixels should have a **premultiplied alpha** when `preserve_alpha=false`.
///
/// # Allocations
///
/// This method will allocate a copy of the `src` image as a back buffer.
pub fn apply(matrix: &ConvolveMatrix, src: ImageRefMut) {
    fn bound(min: i32, val: i32, max: i32) -> i32 {
        core::cmp::max(min, core::cmp::min(max, val))
    }

    let width_max = src.width as i32 - 1;
    let height_max = src.height as i32 - 1;

    let mut buf = vec![RGBA8::default(); src.data.len()];
    let mut buf = ImageRefMut::from_pixels(&mut buf, src.width, src.height);
    let mut x = 0;
    let mut y = 0;
    for in_p in src.data.iter() {
        let mut new_r = 0.0;
        let mut new_g = 0.0;
        let mut new_b = 0.0;
        let mut new_a = 0.0;
        for oy in 0..matrix.rows {
            for ox in 0..matrix.columns {
                let mut tx = x as i32 - matrix.target_x as i32 + ox as i32;
                let mut ty = y as i32 - matrix.target_y as i32 + oy as i32;

                match matrix.edge_mode {
                    EdgeMode::None => {
                        if tx < 0 || tx > width_max || ty < 0 || ty > height_max {
                            continue;
                        }
                    }
                    EdgeMode::Duplicate => {
                        tx = bound(0, tx, width_max);
                        ty = bound(0, ty, height_max);
                    }
                    EdgeMode::Wrap => {
                        while tx < 0 {
                            tx += src.width as i32;
                        }
                        tx %= src.width as i32;

                        while ty < 0 {
                            ty += src.height as i32;
                        }
                        ty %= src.height as i32;
                    }
                }

                let k = matrix.get(matrix.columns - ox - 1, matrix.rows - oy - 1);

                let p = src.pixel_at(tx as u32, ty as u32);
                new_r += (p.r as f32) / 255.0 * k;
                new_g += (p.g as f32) / 255.0 * k;
                new_b += (p.b as f32) / 255.0 * k;

                if !matrix.preserve_alpha {
                    new_a += (p.a as f32) / 255.0 * k;
                }
            }
        }

        if matrix.preserve_alpha {
            new_a = in_p.a as f32 / 255.0;
        } else {
            new_a = new_a / matrix.divisor + matrix.bias;
        }

        let bounded_new_a = f32_bound(0.0, new_a, 1.0);

        let calc = |x| {
            let x = x / matrix.divisor + matrix.bias * new_a;

            let x = if matrix.preserve_alpha {
                f32_bound(0.0, x, 1.0) * bounded_new_a
            } else {
                f32_bound(0.0, x, bounded_new_a)
            };

            (x * 255.0 + 0.5) as u8
        };

        let out_p = buf.pixel_at_mut(x, y);
        out_p.r = calc(new_r);
        out_p.g = calc(new_g);
        out_p.b = calc(new_b);
        out_p.a = (bounded_new_a * 255.0 + 0.5) as u8;

        x += 1;
        if x == src.width {
            x = 0;
            y += 1;
        }
    }

    // Do not use `mem::swap` because `data` referenced via FFI.
    src.data.copy_from_slice(buf.data);
}
