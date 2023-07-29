use image::io::Reader as ImageReader;
use kmeans::{KMeans, KMeansConfig};
use std::path::Path;

fn main() {
	let img_name = std::env::args().nth(1).expect("please provide image");
	let k = std::env::args().nth(2).expect("please provide number of clusters").parse().unwrap();

	let iters = std::env::var("KMEANS_ITERS")
		.ok()
		.map(|s| s.parse().ok())
		.flatten()
		.unwrap_or(100);

	let img = ImageReader::open(&img_name)
		.unwrap()
		.decode()
		.unwrap()
		.into_rgb32f();
	let samples = img.as_raw().clone().into_boxed_slice();
	let samples_len = samples.len();

	let kmeans = KMeans::new(samples.to_vec(), samples_len / 3, 3);
	let result = kmeans.kmeans_lloyd(
		k,
		iters,
		KMeans::init_kmeanplusplus,
		&KMeansConfig::default(),
	);

	eprintln!("Error: {}", result.distsum);

	let mapped = image::ImageBuffer::from_fn(img.dimensions().0, img.dimensions().1, |x, y| {
		let c = result.assignments[y as usize * img.dimensions().0 as usize + x as usize];
		image::Rgb([
			result.centroids[c * 3 + 0],
			result.centroids[c * 3 + 1],
			result.centroids[c * 3 + 2],
		])
	});

	let output_name = {
		let name = Path::new(&img_name).file_stem().unwrap().to_str().unwrap();
		format!("{name}-k{k}.png")
	};
	image::DynamicImage::ImageRgb32F(mapped)
		.into_rgb8()
		.save(output_name)
		.unwrap()
}
