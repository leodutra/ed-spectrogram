use hound;
use rustfft::{FftPlanner, num_complex::Complex};
use image::{RgbImage, Rgb, imageops::resize};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let reader = hound::WavReader::open("example.wav")?;
    let spec = reader.spec();
    println!("WAV file spec: {:?}", spec);

    let samples: Vec<f32> = reader
        .into_samples::<i16>()
        .map(|s| s.expect("Failed to read sample"))
        .collect::<Vec<i16>>()
        .chunks(2)
        .map(|pair| (pair[0] as f32 + pair[1] as f32) / (2.0 * i16::MAX as f32))
        .collect();

    println!("Number of samples read: {}", samples.len());

    if samples.is_empty() {
        eprintln!("Error: No audio samples found in the file.");
        return Ok(());
    }

    // Increase FFT size and adjust hop size
    let fft_size = 2048;
    let hop_size = fft_size / 4;

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);

    let mut spectrogram = Vec::new();

    for chunk in samples.chunks(hop_size) {
        let mut input: Vec<Complex<f32>> = chunk
            .iter()
            .map(|&x| Complex { re: x, im: 0.0 })
            .collect();

        while input.len() < fft_size {
            input.push(Complex { re: 0.0, im: 0.0 });
        }

        fft.process(&mut input);

        let magnitudes: Vec<f32> = input
            .iter()
            .take(fft_size / 2)
            .map(|c| c.norm())
            .collect();
        spectrogram.push(magnitudes);
    }

    if spectrogram.is_empty() {
        eprintln!("Error: Spectrogram data is empty. Check the input file or processing logic.");
        return Ok(());
    }

    // Improve height and scaling of the image
    let height = spectrogram[0].len();
    let width = spectrogram.len();
    let mut img = RgbImage::new(width as u32, height as u32);

    for (x, spectrum) in spectrogram.iter().enumerate() {
        for (y, &value) in spectrum.iter().enumerate() {
            let intensity = ((value.log10() + 3.0).max(0.0) * 85.0) as u8; // Adjust log offset and multiplier
            img.put_pixel(x as u32, height as u32 - y as u32 - 1, Rgb([intensity, intensity, intensity]));
        }
    }

    // Scale the image for higher resolution
    let scaled_img = resize(
        &img,
        width as u32,
        (height * 2) as u32, // Increase height
        image::imageops::FilterType::Lanczos3,
    );

    scaled_img.save("spectrogram_high_res.png")?;
    println!("High-resolution spectrogram saved to spectrogram_high_res.png");

    Ok(())
}
