#[cfg(feature = "validation_layers")]
pub fn enabled_layers() -> Vec<String> {
    vec![String::from("VK_LAYER_KHRONOS_validation")]
}
#[cfg(not(feature = "validation_layers"))]
pub fn enabled_layers() -> Vec<String> {
    vec![]
}