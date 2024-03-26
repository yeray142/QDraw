use pyo3::prelude::*;
use rayon::prelude::*;
use image::{GrayImage, Luma};
use imageproc::drawing::{draw_antialiased_line_segment_mut};
use imageproc::pixelops::interpolate;

fn scale_down_coordinate(coordinate: i32, old_size: i32, new_size: i32) -> i32 {
    let scale_factor = new_size as f32 / old_size as f32;
    (coordinate as f32 * scale_factor).floor() as i32
}

#[pyfunction]
fn sketches_to_images(sketches_batch: Vec<Vec<Vec<Vec<i32>>>>, old_size: i32, new_size: i32) -> PyResult<Vec<Vec<u8>>> {
    sketches_batch.par_iter().map(|sketches| {
        let mut img = GrayImage::new(new_size as u32, new_size as u32);
        for stroke in sketches {
            let x_coords = &stroke[0];
            let y_coords = &stroke[1];
            let scaled_x: Vec<i32> = x_coords.iter().map(|&x| scale_down_coordinate(x, old_size, new_size)).collect();
            let scaled_y: Vec<i32> = y_coords.iter().map(|&y| scale_down_coordinate(y, old_size, new_size)).collect();
            for i in 0..scaled_x.len() - 1 {
                draw_antialiased_line_segment_mut(&mut img, (scaled_x[i], scaled_y[i]), (scaled_x[i + 1], scaled_y[i + 1]), Luma([255u8]), interpolate);
            }
        }
        Ok(img.into_raw())
    }).collect()
}

#[pymodule]
fn rust_sketch(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sketches_to_images, m)?)?;
    Ok(())
}
