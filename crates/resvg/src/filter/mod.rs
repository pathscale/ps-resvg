// Copyright 2018 the Resvg Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::rc::Rc;

use rgb::FromSlice;
use tiny_skia::IntRect;
use usvg::{ApproxEqUlps, ApproxZeroUlps};

mod low_level;

use low_level::{
    ImageRef, ImageRefMut, box_blur, color_matrix, component_transfer, composite, convolve_matrix,
    displacement_map, iir_blur, lighting, morphology, turbulence,
};

// TODO: apply single primitive filters in-place

#[derive(Debug)]
pub(crate) enum Error {
    InvalidRegion,
    NoResults,
}

trait PixmapExt: Sized {
    fn try_create(width: u32, height: u32) -> Result<tiny_skia::Pixmap, Error>;
    fn copy_region(&self, region: IntRect) -> Result<tiny_skia::Pixmap, Error>;
    fn clear(&mut self);
    fn into_srgb(&mut self);
    fn into_linear_rgb(&mut self);
}

impl PixmapExt for tiny_skia::Pixmap {
    fn try_create(width: u32, height: u32) -> Result<tiny_skia::Pixmap, Error> {
        tiny_skia::Pixmap::new(width, height).ok_or(Error::InvalidRegion)
    }

    fn copy_region(&self, region: IntRect) -> Result<tiny_skia::Pixmap, Error> {
        let rect = IntRect::from_xywh(region.x(), region.y(), region.width(), region.height())
            .ok_or(Error::InvalidRegion)?;
        self.clone_rect(rect).ok_or(Error::InvalidRegion)
    }

    fn clear(&mut self) {
        self.fill(tiny_skia::Color::TRANSPARENT);
    }

    fn into_srgb(&mut self) {
        low_level::demultiply_alpha(self.data_mut());
        low_level::from_linear_rgb(self.data_mut());
        low_level::multiply_alpha(self.data_mut());
    }

    fn into_linear_rgb(&mut self) {
        low_level::demultiply_alpha(self.data_mut());
        low_level::into_linear_rgb(self.data_mut());
        low_level::multiply_alpha(self.data_mut());
    }
}

#[derive(Clone)]
struct Image {
    /// Filter primitive result.
    ///
    /// All images have the same size which is equal to the current filter region.
    image: Rc<tiny_skia::Pixmap>,

    /// Image's region that has actual data.
    ///
    /// Region is in global coordinates and not in `image` one.
    ///
    /// Image's content outside this region will be transparent/cleared.
    ///
    /// Currently used only for `feTile`.
    region: IntRect,

    /// The current color space.
    color_space: usvg::filter::ColorInterpolation,
}

impl Image {
    fn from_image(image: tiny_skia::Pixmap, color_space: usvg::filter::ColorInterpolation) -> Self {
        let (w, h) = (image.width(), image.height());
        Image {
            image: Rc::new(image),
            region: IntRect::from_xywh(0, 0, w, h).unwrap(),
            color_space,
        }
    }

    fn into_color_space(
        self,
        color_space: usvg::filter::ColorInterpolation,
    ) -> Result<Self, Error> {
        if color_space != self.color_space {
            let region = self.region;

            let mut image = self.take()?;

            match color_space {
                usvg::filter::ColorInterpolation::SRGB => image.into_srgb(),
                usvg::filter::ColorInterpolation::LinearRGB => image.into_linear_rgb(),
            }

            Ok(Image {
                image: Rc::new(image),
                region,
                color_space,
            })
        } else {
            Ok(self)
        }
    }

    fn take(self) -> Result<tiny_skia::Pixmap, Error> {
        match Rc::try_unwrap(self.image) {
            Ok(v) => Ok(v),
            Err(v) => Ok((*v).clone()),
        }
    }

    fn width(&self) -> u32 {
        self.image.width()
    }

    fn height(&self) -> u32 {
        self.image.height()
    }

    fn as_ref(&self) -> &tiny_skia::Pixmap {
        &self.image
    }
}

struct FilterResult {
    name: String,
    image: Image,
}

pub fn apply(
    filter: &usvg::filter::Filter,
    ts: tiny_skia::Transform,
    source: &mut tiny_skia::Pixmap,
) {
    let result = apply_inner(filter, ts, source);
    let result = result.and_then(|image| apply_to_canvas(image, source));

    // Clear on error.
    if result.is_err() {
        source.fill(tiny_skia::Color::TRANSPARENT);
    }

    match result {
        Ok(_) => {}
        Err(Error::InvalidRegion) => {
            log::warn!("Filter has an invalid region.");
        }
        Err(Error::NoResults) => {}
    }
}

fn apply_inner(
    filter: &usvg::filter::Filter,
    ts: usvg::Transform,
    source: &mut tiny_skia::Pixmap,
) -> Result<Image, Error> {
    let region = filter
        .rect()
        .transform(ts)
        .map(|r| r.to_int_rect())
        .ok_or(Error::InvalidRegion)?;

    // The source pixmap is clamped to `max_bbox` in `render_group`, so the filter
    // region (derived directly from the unclamped filter rect) can be larger than
    // the buffer we actually render into.
    let source_rect =
        IntRect::from_xywh(0, 0, source.width(), source.height()).ok_or(Error::InvalidRegion)?;
    let region = crate::geom::fit_to_rect(region, source_rect).ok_or(Error::InvalidRegion)?;

    let mut results: Vec<FilterResult> = Vec::new();

    for primitive in filter.primitives() {
        let mut subregion = primitive
            .rect()
            .transform(ts)
            .map(|r| r.to_int_rect())
            .ok_or(Error::InvalidRegion)?;

        // `feOffset` inherits its region from the input.
        if let usvg::filter::Kind::Offset(fe) = primitive.kind() {
            if let usvg::filter::Input::Reference(name) = fe.input() {
                if let Some(res) = results.iter().rev().find(|v| v.name == *name) {
                    subregion = res.image.region;
                }
            }
        }

        let cs = primitive.color_interpolation();

        let mut result = match primitive.kind() {
            usvg::filter::Kind::Blend(fe) => {
                let input1 = get_input(fe.input1(), region, source, &results)?;
                let input2 = get_input(fe.input2(), region, source, &results)?;
                apply_blend(fe, cs, region, input1, input2)
            }
            usvg::filter::Kind::DropShadow(fe) => {
                let input = get_input(fe.input(), region, source, &results)?;
                apply_drop_shadow(fe, cs, ts, input)
            }
            usvg::filter::Kind::Flood(fe) => apply_flood(fe, region),
            usvg::filter::Kind::GaussianBlur(fe) => {
                let input = get_input(fe.input(), region, source, &results)?;
                apply_blur(fe, cs, ts, input)
            }
            usvg::filter::Kind::Offset(fe) => {
                let input = get_input(fe.input(), region, source, &results)?;
                apply_offset(fe, ts, input)
            }
            usvg::filter::Kind::Composite(fe) => {
                let input1 = get_input(fe.input1(), region, source, &results)?;
                let input2 = get_input(fe.input2(), region, source, &results)?;
                apply_composite(fe, cs, region, input1, input2)
            }
            usvg::filter::Kind::Merge(fe) => apply_merge(fe, cs, region, source, &results),
            usvg::filter::Kind::Tile(fe) => {
                let input = get_input(fe.input(), region, source, &results)?;
                apply_tile(input, region)
            }
            usvg::filter::Kind::Image(fe) => apply_image(fe, region, subregion, ts),
            usvg::filter::Kind::ComponentTransfer(fe) => {
                let input = get_input(fe.input(), region, source, &results)?;
                apply_component_transfer(fe, cs, input)
            }
            usvg::filter::Kind::ColorMatrix(fe) => {
                let input = get_input(fe.input(), region, source, &results)?;
                apply_color_matrix(fe, cs, input)
            }
            usvg::filter::Kind::ConvolveMatrix(fe) => {
                let input = get_input(fe.input(), region, source, &results)?;
                apply_convolve_matrix(fe, cs, input)
            }
            usvg::filter::Kind::Morphology(fe) => {
                let input = get_input(fe.input(), region, source, &results)?;
                apply_morphology(fe, cs, ts, input)
            }
            usvg::filter::Kind::DisplacementMap(fe) => {
                let input1 = get_input(fe.input1(), region, source, &results)?;
                let input2 = get_input(fe.input2(), region, source, &results)?;
                apply_displacement_map(fe, region, cs, ts, input1, input2)
            }
            usvg::filter::Kind::Turbulence(fe) => apply_turbulence(fe, region, cs, ts),
            usvg::filter::Kind::DiffuseLighting(fe) => {
                let input = get_input(fe.input(), region, source, &results)?;
                apply_diffuse_lighting(fe, region, cs, ts, input)
            }
            usvg::filter::Kind::SpecularLighting(fe) => {
                let input = get_input(fe.input(), region, source, &results)?;
                apply_specular_lighting(fe, region, cs, ts, input)
            }
        }?;

        if region != subregion {
            // Clip result.

            // TODO: explain
            let subregion2 = if let usvg::filter::Kind::Offset(..) = primitive.kind() {
                // We do not support clipping on feOffset.
                region.translate_to(0, 0)
            } else {
                subregion.translate(-region.x(), -region.y())
            }
            .unwrap();

            let color_space = result.color_space;

            let pixmap = {
                // This is cropping by clearing the pixels outside the region.
                let mut paint = tiny_skia::Paint::default();
                paint.set_color(tiny_skia::Color::BLACK);
                paint.blend_mode = tiny_skia::BlendMode::Clear;

                let mut pixmap = result.take()?;
                let w = pixmap.width() as f32;
                let h = pixmap.height() as f32;

                if let Some(rect) = tiny_skia::Rect::from_xywh(0.0, 0.0, w, subregion2.y() as f32) {
                    pixmap.fill_rect(rect, &paint, tiny_skia::Transform::identity(), None);
                }

                if let Some(rect) = tiny_skia::Rect::from_xywh(0.0, 0.0, subregion2.x() as f32, h) {
                    pixmap.fill_rect(rect, &paint, tiny_skia::Transform::identity(), None);
                }

                if let Some(rect) = tiny_skia::Rect::from_xywh(subregion2.right() as f32, 0.0, w, h)
                {
                    pixmap.fill_rect(rect, &paint, tiny_skia::Transform::identity(), None);
                }

                if let Some(rect) =
                    tiny_skia::Rect::from_xywh(0.0, subregion2.bottom() as f32, w, h)
                {
                    pixmap.fill_rect(rect, &paint, tiny_skia::Transform::identity(), None);
                }

                pixmap
            };

            result = Image {
                image: Rc::new(pixmap),
                region: subregion,
                color_space,
            };
        }

        results.push(FilterResult {
            name: primitive.result().to_string(),
            image: result,
        });
    }

    if let Some(res) = results.pop() {
        Ok(res.image)
    } else {
        Err(Error::NoResults)
    }
}

fn get_input(
    input: &usvg::filter::Input,
    region: IntRect,
    source: &tiny_skia::Pixmap,
    results: &[FilterResult],
) -> Result<Image, Error> {
    match input {
        usvg::filter::Input::SourceGraphic => {
            let image = source.clone();

            Ok(Image {
                image: Rc::new(image),
                region,
                color_space: usvg::filter::ColorInterpolation::SRGB,
            })
        }
        usvg::filter::Input::SourceAlpha => {
            let mut image = source.clone();
            // Set RGB to black. Keep alpha as is.
            for p in image.data_mut().as_rgba_mut() {
                p.r = 0;
                p.g = 0;
                p.b = 0;
            }

            Ok(Image {
                image: Rc::new(image),
                region,
                color_space: usvg::filter::ColorInterpolation::SRGB,
            })
        }
        usvg::filter::Input::Reference(name) => {
            if let Some(v) = results.iter().rev().find(|v| v.name == *name) {
                Ok(v.image.clone())
            } else {
                // Technically unreachable.
                log::warn!("Unknown filter primitive reference '{}'.", name);
                get_input(&usvg::filter::Input::SourceGraphic, region, source, results)
            }
        }
    }
}

trait PixmapToImageRef<'a> {
    fn as_image_ref(&'a self) -> ImageRef<'a>;
    fn as_image_ref_mut(&'a mut self) -> ImageRefMut<'a>;
}

impl<'a> PixmapToImageRef<'a> for tiny_skia::Pixmap {
    fn as_image_ref(&'a self) -> ImageRef<'a> {
        ImageRef::new(self.data(), self.width(), self.height()).unwrap()
    }

    fn as_image_ref_mut(&'a mut self) -> ImageRefMut<'a> {
        let (width, height) = (self.width(), self.height());
        ImageRefMut::new(self.data_mut(), width, height).unwrap()
    }
}

fn apply_drop_shadow(
    fe: &usvg::filter::DropShadow,
    cs: usvg::filter::ColorInterpolation,
    ts: usvg::Transform,
    input: Image,
) -> Result<Image, Error> {
    let (dx, dy) = match scale_coordinates(fe.dx(), fe.dy(), ts) {
        Some(v) => v,
        None => return Ok(input),
    };

    let mut pixmap = tiny_skia::Pixmap::try_create(input.width(), input.height())?;
    let input_pixmap = input.into_color_space(cs)?.take()?;
    let mut shadow_pixmap = input_pixmap.clone();

    if let Some((std_dx, std_dy, use_box_blur)) =
        resolve_std_dev(fe.std_dev_x().get(), fe.std_dev_y().get(), ts)
    {
        if use_box_blur {
            box_blur::apply(std_dx, std_dy, shadow_pixmap.as_image_ref_mut());
        } else {
            iir_blur::apply(std_dx, std_dy, shadow_pixmap.as_image_ref_mut());
        }
    }

    // flood
    let color = tiny_skia::Color::from_rgba8(
        fe.color().red,
        fe.color().green,
        fe.color().blue,
        fe.opacity().to_u8(),
    );
    for p in shadow_pixmap.pixels_mut() {
        let mut color = color;
        color.apply_opacity(p.alpha() as f32 / 255.0);
        *p = color.premultiply().to_color_u8();
    }

    match cs {
        usvg::filter::ColorInterpolation::SRGB => shadow_pixmap.into_srgb(),
        usvg::filter::ColorInterpolation::LinearRGB => shadow_pixmap.into_linear_rgb(),
    }

    pixmap.draw_pixmap(
        dx as i32,
        dy as i32,
        shadow_pixmap.as_ref(),
        &tiny_skia::PixmapPaint::default(),
        tiny_skia::Transform::identity(),
        None,
    );

    pixmap.draw_pixmap(
        0,
        0,
        input_pixmap.as_ref(),
        &tiny_skia::PixmapPaint::default(),
        tiny_skia::Transform::identity(),
        None,
    );

    Ok(Image::from_image(pixmap, cs))
}

fn apply_blur(
    fe: &usvg::filter::GaussianBlur,
    cs: usvg::filter::ColorInterpolation,
    ts: usvg::Transform,
    input: Image,
) -> Result<Image, Error> {
    let (std_dx, std_dy, use_box_blur) =
        match resolve_std_dev(fe.std_dev_x().get(), fe.std_dev_y().get(), ts) {
            Some(v) => v,
            None => return Ok(input),
        };

    let mut pixmap = input.into_color_space(cs)?.take()?;

    if use_box_blur {
        box_blur::apply(std_dx, std_dy, pixmap.as_image_ref_mut());
    } else {
        iir_blur::apply(std_dx, std_dy, pixmap.as_image_ref_mut());
    }

    Ok(Image::from_image(pixmap, cs))
}

fn apply_offset(
    fe: &usvg::filter::Offset,
    ts: usvg::Transform,
    input: Image,
) -> Result<Image, Error> {
    let (dx, dy) = match scale_coordinates(fe.dx(), fe.dy(), ts) {
        Some(v) => v,
        None => return Ok(input),
    };

    if dx.approx_zero_ulps(4) && dy.approx_zero_ulps(4) {
        return Ok(input);
    }

    let mut pixmap = tiny_skia::Pixmap::try_create(input.width(), input.height())?;
    pixmap.draw_pixmap(
        dx as i32,
        dy as i32,
        input.as_ref().as_ref(),
        &tiny_skia::PixmapPaint::default(),
        tiny_skia::Transform::identity(),
        None,
    );

    Ok(Image::from_image(pixmap, input.color_space))
}

fn apply_blend(
    fe: &usvg::filter::Blend,
    cs: usvg::filter::ColorInterpolation,
    region: IntRect,
    input1: Image,
    input2: Image,
) -> Result<Image, Error> {
    let input1 = input1.into_color_space(cs)?;
    let input2 = input2.into_color_space(cs)?;

    let mut pixmap = tiny_skia::Pixmap::try_create(region.width(), region.height())?;

    pixmap.draw_pixmap(
        0,
        0,
        input2.as_ref().as_ref(),
        &tiny_skia::PixmapPaint::default(),
        tiny_skia::Transform::identity(),
        None,
    );

    pixmap.draw_pixmap(
        0,
        0,
        input1.as_ref().as_ref(),
        &tiny_skia::PixmapPaint {
            blend_mode: crate::render::convert_blend_mode(fe.mode()),
            ..tiny_skia::PixmapPaint::default()
        },
        tiny_skia::Transform::identity(),
        None,
    );

    Ok(Image::from_image(pixmap, cs))
}

fn apply_composite(
    fe: &usvg::filter::Composite,
    cs: usvg::filter::ColorInterpolation,
    region: IntRect,
    input1: Image,
    input2: Image,
) -> Result<Image, Error> {
    use usvg::filter::CompositeOperator as Operator;

    let input1 = input1.into_color_space(cs)?;
    let input2 = input2.into_color_space(cs)?;

    let mut pixmap = tiny_skia::Pixmap::try_create(region.width(), region.height())?;

    if let Operator::Arithmetic { k1, k2, k3, k4 } = fe.operator() {
        let pixmap1 = input1.take()?;
        let pixmap2 = input2.take()?;

        composite::arithmetic(
            k1,
            k2,
            k3,
            k4,
            pixmap1.as_image_ref(),
            pixmap2.as_image_ref(),
            pixmap.as_image_ref_mut(),
        );

        return Ok(Image::from_image(pixmap, cs));
    }

    pixmap.draw_pixmap(
        0,
        0,
        input2.as_ref().as_ref(),
        &tiny_skia::PixmapPaint::default(),
        tiny_skia::Transform::identity(),
        None,
    );

    let blend_mode = match fe.operator() {
        Operator::Over => tiny_skia::BlendMode::SourceOver,
        Operator::In => tiny_skia::BlendMode::SourceIn,
        Operator::Out => tiny_skia::BlendMode::SourceOut,
        Operator::Atop => tiny_skia::BlendMode::SourceAtop,
        Operator::Xor => tiny_skia::BlendMode::Xor,
        Operator::Arithmetic { .. } => tiny_skia::BlendMode::SourceOver,
    };

    pixmap.draw_pixmap(
        0,
        0,
        input1.as_ref().as_ref(),
        &tiny_skia::PixmapPaint {
            blend_mode,
            ..tiny_skia::PixmapPaint::default()
        },
        tiny_skia::Transform::identity(),
        None,
    );

    Ok(Image::from_image(pixmap, cs))
}

fn apply_merge(
    fe: &usvg::filter::Merge,
    cs: usvg::filter::ColorInterpolation,
    region: IntRect,
    source: &tiny_skia::Pixmap,
    results: &[FilterResult],
) -> Result<Image, Error> {
    let mut pixmap = tiny_skia::Pixmap::try_create(region.width(), region.height())?;

    for input in fe.inputs() {
        let input = get_input(input, region, source, results)?;
        let input = input.into_color_space(cs)?;
        pixmap.draw_pixmap(
            0,
            0,
            input.as_ref().as_ref(),
            &tiny_skia::PixmapPaint::default(),
            tiny_skia::Transform::identity(),
            None,
        );
    }

    Ok(Image::from_image(pixmap, cs))
}

fn apply_flood(fe: &usvg::filter::Flood, region: IntRect) -> Result<Image, Error> {
    let c = fe.color();

    let mut pixmap = tiny_skia::Pixmap::try_create(region.width(), region.height())?;
    pixmap.fill(tiny_skia::Color::from_rgba8(
        c.red,
        c.green,
        c.blue,
        fe.opacity().to_u8(),
    ));

    Ok(Image::from_image(
        pixmap,
        usvg::filter::ColorInterpolation::SRGB,
    ))
}

fn apply_tile(input: Image, region: IntRect) -> Result<Image, Error> {
    let subregion = input.region.translate(-region.x(), -region.y()).unwrap();

    let tile_pixmap = input.image.copy_region(subregion)?;
    let mut paint = tiny_skia::Paint::default();
    paint.shader = tiny_skia::Pattern::new(
        tile_pixmap.as_ref(),
        tiny_skia::SpreadMode::Repeat,
        tiny_skia::FilterQuality::Bicubic,
        1.0,
        tiny_skia::Transform::from_translate(subregion.x() as f32, subregion.y() as f32),
    );

    let mut pixmap = tiny_skia::Pixmap::try_create(region.width(), region.height())?;
    let rect = tiny_skia::Rect::from_xywh(0.0, 0.0, region.width() as f32, region.height() as f32)
        .unwrap();
    pixmap.fill_rect(rect, &paint, tiny_skia::Transform::identity(), None);

    Ok(Image::from_image(
        pixmap,
        usvg::filter::ColorInterpolation::SRGB,
    ))
}

fn apply_image(
    fe: &usvg::filter::Image,
    region: IntRect,
    subregion: IntRect,
    ts: usvg::Transform,
) -> Result<Image, Error> {
    let mut pixmap = tiny_skia::Pixmap::try_create(region.width(), region.height())?;

    let (sx, sy) = ts.get_scale();
    let transform = tiny_skia::Transform::from_row(
        sx,
        0.0,
        0.0,
        sy,
        subregion.x() as f32,
        subregion.y() as f32,
    );

    let ctx = crate::render::Context {
        max_bbox: tiny_skia::IntRect::from_xywh(0, 0, region.width(), region.height()).unwrap(),
    };

    crate::render::render_nodes(fe.root(), &ctx, transform, &mut pixmap.as_mut());

    Ok(Image::from_image(
        pixmap,
        usvg::filter::ColorInterpolation::SRGB,
    ))
}

fn apply_component_transfer(
    fe: &usvg::filter::ComponentTransfer,
    cs: usvg::filter::ColorInterpolation,
    input: Image,
) -> Result<Image, Error> {
    let mut pixmap = input.into_color_space(cs)?.take()?;

    low_level::demultiply_alpha(pixmap.data_mut());
    component_transfer::apply(
        to_transfer_function(fe.func_r()),
        to_transfer_function(fe.func_g()),
        to_transfer_function(fe.func_b()),
        to_transfer_function(fe.func_a()),
        pixmap.as_image_ref_mut(),
    );
    low_level::multiply_alpha(pixmap.data_mut());

    Ok(Image::from_image(pixmap, cs))
}

fn apply_color_matrix(
    fe: &usvg::filter::ColorMatrix,
    cs: usvg::filter::ColorInterpolation,
    input: Image,
) -> Result<Image, Error> {
    let mut pixmap = input.into_color_space(cs)?.take()?;

    low_level::demultiply_alpha(pixmap.data_mut());
    if let Some(matrix) = to_color_matrix(fe.kind()) {
        color_matrix::apply(&matrix, pixmap.as_image_ref_mut());
    }
    low_level::multiply_alpha(pixmap.data_mut());

    Ok(Image::from_image(pixmap, cs))
}

fn apply_convolve_matrix(
    fe: &usvg::filter::ConvolveMatrix,
    cs: usvg::filter::ColorInterpolation,
    input: Image,
) -> Result<Image, Error> {
    let mut pixmap = input.into_color_space(cs)?.take()?;

    if fe.preserve_alpha() {
        low_level::demultiply_alpha(pixmap.data_mut());
    }

    let edge_mode = match fe.edge_mode() {
        usvg::filter::EdgeMode::None => low_level::EdgeMode::None,
        usvg::filter::EdgeMode::Duplicate => low_level::EdgeMode::Duplicate,
        usvg::filter::EdgeMode::Wrap => low_level::EdgeMode::Wrap,
    };

    let matrix = low_level::ConvolveMatrix::new(
        fe.matrix().target_x(),
        fe.matrix().target_y(),
        fe.matrix().columns(),
        fe.matrix().rows(),
        fe.matrix().data(),
        fe.divisor().get(),
        fe.bias(),
        edge_mode,
        fe.preserve_alpha(),
    )
    .ok_or(Error::InvalidRegion)?;

    convolve_matrix::apply(&matrix, pixmap.as_image_ref_mut());

    Ok(Image::from_image(pixmap, cs))
}

fn apply_morphology(
    fe: &usvg::filter::Morphology,
    cs: usvg::filter::ColorInterpolation,
    ts: usvg::Transform,
    input: Image,
) -> Result<Image, Error> {
    let mut pixmap = input.into_color_space(cs)?.take()?;

    let (rx, ry) = match scale_coordinates(fe.radius_x().get(), fe.radius_y().get(), ts) {
        Some(v) => v,
        None => return Ok(Image::from_image(pixmap, cs)),
    };

    if !(rx > 0.0 && ry > 0.0) {
        pixmap.clear();
        return Ok(Image::from_image(pixmap, cs));
    }

    let operator = match fe.operator() {
        usvg::filter::MorphologyOperator::Erode => low_level::MorphologyOperator::Erode,
        usvg::filter::MorphologyOperator::Dilate => low_level::MorphologyOperator::Dilate,
    };

    morphology::apply(operator, rx, ry, pixmap.as_image_ref_mut());

    Ok(Image::from_image(pixmap, cs))
}

fn apply_displacement_map(
    fe: &usvg::filter::DisplacementMap,
    region: IntRect,
    cs: usvg::filter::ColorInterpolation,
    ts: usvg::Transform,
    input1: Image,
    input2: Image,
) -> Result<Image, Error> {
    let pixmap1 = input1.into_color_space(cs)?.take()?;
    let pixmap2 = input2.into_color_space(cs)?.take()?;

    let mut pixmap = tiny_skia::Pixmap::try_create(region.width(), region.height())?;

    let (sx, sy) = match scale_coordinates(fe.scale(), fe.scale(), ts) {
        Some(v) => v,
        None => return Ok(Image::from_image(pixmap1, cs)),
    };

    displacement_map::apply(
        to_color_channel(fe.x_channel_selector()),
        to_color_channel(fe.y_channel_selector()),
        fe.scale(),
        sx,
        sy,
        pixmap1.as_image_ref(),
        pixmap2.as_image_ref(),
        pixmap.as_image_ref_mut(),
    );

    Ok(Image::from_image(pixmap, cs))
}

fn apply_turbulence(
    fe: &usvg::filter::Turbulence,
    region: IntRect,
    cs: usvg::filter::ColorInterpolation,
    ts: usvg::Transform,
) -> Result<Image, Error> {
    let mut pixmap = tiny_skia::Pixmap::try_create(region.width(), region.height())?;

    let (sx, sy) = ts.get_scale();
    if sx.approx_zero_ulps(4) || sy.approx_zero_ulps(4) {
        return Ok(Image::from_image(pixmap, cs));
    }

    turbulence::apply(
        region.x() as f64 - ts.tx as f64,
        region.y() as f64 - ts.ty as f64,
        sx as f64,
        sy as f64,
        fe.base_frequency_x().get() as f64,
        fe.base_frequency_y().get() as f64,
        fe.num_octaves(),
        fe.seed(),
        fe.stitch_tiles(),
        fe.kind() == usvg::filter::TurbulenceKind::FractalNoise,
        pixmap.as_image_ref_mut(),
    );

    low_level::multiply_alpha(pixmap.data_mut());

    Ok(Image::from_image(pixmap, cs))
}

fn apply_diffuse_lighting(
    fe: &usvg::filter::DiffuseLighting,
    region: IntRect,
    cs: usvg::filter::ColorInterpolation,
    ts: usvg::Transform,
    input: Image,
) -> Result<Image, Error> {
    let mut pixmap = tiny_skia::Pixmap::try_create(region.width(), region.height())?;

    let light_source = transform_light_source(fe.light_source(), region, ts);

    let params = low_level::DiffuseLighting {
        surface_scale: fe.surface_scale(),
        diffuse_constant: fe.diffuse_constant(),
        lighting_color: to_rgb8(fe.lighting_color()),
    };

    lighting::diffuse_lighting(
        &params,
        light_source,
        input.as_ref().as_image_ref(),
        pixmap.as_image_ref_mut(),
    );

    Ok(Image::from_image(pixmap, cs))
}

fn apply_specular_lighting(
    fe: &usvg::filter::SpecularLighting,
    region: IntRect,
    cs: usvg::filter::ColorInterpolation,
    ts: usvg::Transform,
    input: Image,
) -> Result<Image, Error> {
    let mut pixmap = tiny_skia::Pixmap::try_create(region.width(), region.height())?;

    let light_source = transform_light_source(fe.light_source(), region, ts);

    let params = low_level::SpecularLighting {
        surface_scale: fe.surface_scale(),
        specular_constant: fe.specular_constant(),
        specular_exponent: fe.specular_exponent(),
        lighting_color: to_rgb8(fe.lighting_color()),
    };

    lighting::specular_lighting(
        &params,
        light_source,
        input.as_ref().as_image_ref(),
        pixmap.as_image_ref_mut(),
    );

    Ok(Image::from_image(pixmap, cs))
}

fn transform_light_source(
    source: usvg::filter::LightSource,
    region: IntRect,
    ts: usvg::Transform,
) -> low_level::LightSource {
    use std::f32::consts::SQRT_2;

    match source {
        usvg::filter::LightSource::DistantLight(light) => {
            low_level::LightSource::DistantLight(low_level::DistantLight {
                azimuth: light.azimuth,
                elevation: light.elevation,
            })
        }
        usvg::filter::LightSource::PointLight(light) => {
            let mut point = tiny_skia::Point::from_xy(light.x, light.y);
            ts.map_point(&mut point);
            low_level::LightSource::PointLight(low_level::PointLight {
                x: point.x - region.x() as f32,
                y: point.y - region.y() as f32,
                z: light.z * (ts.sx * ts.sx + ts.sy * ts.sy).sqrt() / SQRT_2,
            })
        }
        usvg::filter::LightSource::SpotLight(light) => {
            let sz = (ts.sx * ts.sx + ts.sy * ts.sy).sqrt() / SQRT_2;

            let mut point = tiny_skia::Point::from_xy(light.x, light.y);
            ts.map_point(&mut point);

            let mut points_at = tiny_skia::Point::from_xy(light.points_at_x, light.points_at_y);
            ts.map_point(&mut points_at);

            low_level::LightSource::SpotLight(low_level::SpotLight {
                x: point.x - region.x() as f32,
                y: point.y - region.y() as f32,
                z: light.z * sz,
                points_at_x: points_at.x - region.x() as f32,
                points_at_y: points_at.y - region.y() as f32,
                points_at_z: light.points_at_z * sz,
                specular_exponent: light.specular_exponent.get(),
                limiting_cone_angle: light.limiting_cone_angle,
            })
        }
    }
}

fn to_rgb8(color: usvg::Color) -> low_level::Rgb8 {
    low_level::Rgb8 {
        r: color.red,
        g: color.green,
        b: color.blue,
    }
}

fn to_color_channel(channel: usvg::filter::ColorChannel) -> low_level::ColorChannel {
    match channel {
        usvg::filter::ColorChannel::R => low_level::ColorChannel::R,
        usvg::filter::ColorChannel::G => low_level::ColorChannel::G,
        usvg::filter::ColorChannel::B => low_level::ColorChannel::B,
        usvg::filter::ColorChannel::A => low_level::ColorChannel::A,
    }
}

fn to_transfer_function(func: &usvg::filter::TransferFunction) -> low_level::TransferFunction<'_> {
    match func {
        usvg::filter::TransferFunction::Identity => low_level::TransferFunction::Identity,
        usvg::filter::TransferFunction::Table(values) => low_level::TransferFunction::Table(values),
        usvg::filter::TransferFunction::Discrete(values) => {
            low_level::TransferFunction::Discrete(values)
        }
        usvg::filter::TransferFunction::Linear { slope, intercept } => {
            low_level::TransferFunction::Linear {
                slope: *slope,
                intercept: *intercept,
            }
        }
        usvg::filter::TransferFunction::Gamma {
            amplitude,
            exponent,
            offset,
        } => low_level::TransferFunction::Gamma {
            amplitude: *amplitude,
            exponent: *exponent,
            offset: *offset,
        },
    }
}

fn to_color_matrix(kind: &usvg::filter::ColorMatrixKind) -> Option<low_level::ColorMatrix<'_>> {
    match kind {
        usvg::filter::ColorMatrixKind::Matrix(m) => m
            .as_slice()
            .try_into()
            .ok()
            .map(low_level::ColorMatrix::Matrix),
        usvg::filter::ColorMatrixKind::Saturate(v) => {
            Some(low_level::ColorMatrix::Saturate(v.get()))
        }
        usvg::filter::ColorMatrixKind::HueRotate(v) => Some(low_level::ColorMatrix::HueRotate(*v)),
        usvg::filter::ColorMatrixKind::LuminanceToAlpha => {
            Some(low_level::ColorMatrix::LuminanceToAlpha)
        }
    }
}

fn apply_to_canvas(input: Image, pixmap: &mut tiny_skia::Pixmap) -> Result<(), Error> {
    let input = input.into_color_space(usvg::filter::ColorInterpolation::SRGB)?;

    pixmap.fill(tiny_skia::Color::TRANSPARENT);
    pixmap.draw_pixmap(
        0,
        0,
        input.as_ref().as_ref(),
        &tiny_skia::PixmapPaint::default(),
        tiny_skia::Transform::identity(),
        None,
    );

    Ok(())
}

/// Calculates Gaussian blur sigmas for the current world transform.
///
/// If the last flag is set, then a box blur should be used. Or IIR otherwise.
fn resolve_std_dev(std_dx: f32, std_dy: f32, ts: usvg::Transform) -> Option<(f64, f64, bool)> {
    let (mut std_dx, mut std_dy) = scale_coordinates(std_dx, std_dy, ts)?;

    // 'A negative value or a value of zero disables the effect of the given filter primitive
    // (i.e., the result is the filter input image).'
    if std_dx.approx_eq_ulps(&0.0, 4) && std_dy.approx_eq_ulps(&0.0, 4) {
        return None;
    }

    // Ignore tiny sigmas. In case of IIR blur it can lead to a transparent image.
    if std_dx < 0.05 {
        std_dx = 0.0;
    }

    if std_dy < 0.05 {
        std_dy = 0.0;
    }

    const BLUR_SIGMA_THRESHOLD: f32 = 2.0;
    // Check that the current feGaussianBlur filter can be applied using a box blur.
    let box_blur = std_dx >= BLUR_SIGMA_THRESHOLD || std_dy >= BLUR_SIGMA_THRESHOLD;

    Some((std_dx as f64, std_dy as f64, box_blur))
}

fn scale_coordinates(x: f32, y: f32, ts: usvg::Transform) -> Option<(f32, f32)> {
    let (sx, sy) = ts.get_scale();
    Some((x * sx, y * sy))
}
