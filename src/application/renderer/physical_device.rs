use vulkano::device::physical::PhysicalDevice;
use vulkano::format::{Format, FormatFeatures};
use vulkano::image::ImageTiling;
use vulkano::swapchain::Surface;

pub fn is_physical_device_suitable_for_surface(
    physical_device: &PhysicalDevice,
    surface: &Surface,
) -> bool {
    true
}

pub fn find_supported_format<I: IntoIterator<Item = Format>>(
    physical_device: &PhysicalDevice,
    tiling: ImageTiling,
    features: FormatFeatures,
    candidates: I,
) -> Option<Format> where <I as IntoIterator>::IntoIter: DoubleEndedIterator {
    candidates
        .into_iter()
        .rev()
        .map(|format| (format, physical_device.format_properties(format).unwrap()))
        .filter(|(_, format_props)| {
            tiling == ImageTiling::Linear && format_props.linear_tiling_features.contains(features)
                || tiling == ImageTiling::Optimal
                    && format_props.optimal_tiling_features.contains(features)
        })
        .map(|(format, _)| format)
        .last()
}

pub fn find_depth_format(physical_device: &PhysicalDevice) -> Format {
    find_supported_format(
        physical_device,
        ImageTiling::Optimal,
        FormatFeatures::DEPTH_STENCIL_ATTACHMENT,
        [
            Format::D32_SFLOAT_S8_UINT,
            Format::D24_UNORM_S8_UINT,
            Format::D32_SFLOAT,
        ],
    )
    .unwrap()
}
