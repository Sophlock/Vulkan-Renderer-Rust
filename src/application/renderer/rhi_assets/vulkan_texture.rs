use std::sync::Arc;
use vulkano::image::view::ImageView;
struct VKTexture {
    image: Arc<ImageView>
}