use std::{ffi::CString, sync::Arc};

/*use ash::{
    vk,
    vk::{DeviceQueueCreateInfo, PhysicalDeviceFeatures, PhysicalDeviceFeatures2},
};*/
use vulkano::{
    VulkanObject,
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, physical::PhysicalDevice,
    },
    instance::Instance,
};

use crate::application::rhi::queue::{QueueCollection, QueueFamilyIndices};

pub fn create_logical_device(
    physical_device: &Arc<PhysicalDevice>,
    queue_indices: &QueueFamilyIndices,
) -> (Arc<Device>, QueueCollection) {
    let queue_create_infos = queue_indices.generate_create_infos();
    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        khr_fragment_shader_barycentric: true,
        nv_device_generated_commands: true,
        nv_device_generated_commands_compute: true,
        khr_buffer_device_address: true,
        khr_synchronization2: true,
        //nv_compute_shader_derivatives: true,
        ..DeviceExtensions::default()
    };
    let device_features = DeviceFeatures {
        sampler_anisotropy: true,
        //compute_derivative_group_quads: true,
        synchronization2: true,
        device_generated_commands: true,
        device_generated_compute: true,
        device_generated_compute_pipelines: true,
        geometry_shader: true,
        fragment_shader_barycentric: true,
        shader_int64: true,
        buffer_device_address: true,
        variable_pointers_storage_buffer: true,
        robust_buffer_access: true,
        robust_buffer_access2: true,
        ..DeviceFeatures::default()
    };
    let device_create_info = DeviceCreateInfo {
        queue_create_infos,
        enabled_extensions: device_extensions,
        enabled_features: device_features,
        physical_devices: vec![physical_device.clone()].into(),
        ..DeviceCreateInfo::default()
    };

    /*let extra_device_extensions = [
        //"VK_KHR_compute_shader_derivatives",
        //"VK_EXT_device_generated_commands",
    ];

    let raw_queue_create_infos = device_create_info
        .queue_create_infos
        .iter()
        .map(|queue_create_info| {
            DeviceQueueCreateInfo::default()
                .queue_family_index(queue_create_info.queue_family_index)
                .queue_priorities(queue_create_info.queues.as_slice())
        })
        .collect::<Vec<_>>();
    let raw_device_features = PhysicalDeviceFeatures {
        sampler_anisotropy: true.into(),
        ..PhysicalDeviceFeatures::default()
    };
    let mut raw_device_features_2 =
        PhysicalDeviceFeatures2::default().features(raw_device_features);

    let extension_names = device_create_info
        .enabled_extensions
        .into_iter()
        .filter_map(|(extension_name, enabled)| enabled.then_some(extension_name))
        .chain(extra_device_extensions)
        .map(|name| CString::new(name).unwrap())
        .collect::<Vec<_>>();
    let raw_device_extensions = extension_names
        .iter()
        .map(|name| name.as_ptr())
        .collect::<Vec<_>>();

    let raw_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(raw_queue_create_infos.as_slice())
        .enabled_extension_names(raw_device_extensions.as_slice())
        .push_next(&mut raw_device_features_2);

    let raw_instance_handle = physical_device.instance().handle();
    let raw_instance_fns = physical_device.instance().fns();

    let raw_instance = ash::Instance::from_parts_1_3(
        raw_instance_handle,
        raw_instance_fns.v1_0.clone(),
        raw_instance_fns.v1_1.clone(),
        raw_instance_fns.v1_3.clone(),
    );

    let raw_device =
        unsafe { raw_instance.create_device(physical_device.handle(), &raw_create_info, None) }
            .unwrap();*/

    let (device, queues) = Device::new(physical_device.clone(), device_create_info).unwrap();
    /*unsafe {
        Device::from_handle(
            physical_device.clone(),
            raw_device.handle(),
            device_create_info,
        )
    };*/
    (
        device,
        QueueCollection::new(queues.collect(), queue_indices),
    )
}

/*pub unsafe fn ash_instance(instance: &Arc<Instance>) -> ash::Instance {
    ash::Instance::from_parts_1_3(
        instance.handle(),
        instance.fns().v1_0.clone(),
        instance.fns().v1_1.clone(),
        instance.fns().v1_3.clone(),
    )
}

pub unsafe fn ash_device(device: &Arc<Device>) -> ash::Device {
    ash::Device::from_parts_1_3(
        device.handle(),
        device.fns().v1_0.clone(),
        device.fns().v1_1.clone(),
        device.fns().v1_2.clone(),
        device.fns().v1_3.clone(),
    )
}*/
