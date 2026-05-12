module visBufferGenerator;

struct ViewData {
    float4x4 viewProjection;
}

struct InstanceInput {
    float4x4 transform : TRANSFORM;
}

struct VertexInput {
    float3 position : POSITION;
}

struct VertexOutput {
    float4 position : SV_Position;
    uint instanceID : INSTANCE_ID;
}

struct FragmentResult {
    uint4 packedData : SV_Target;
}

uniform ViewData gViewData;

// Vertex shader to rasterize the visibility buffer
// vertexInput is taken from the vertex buffer, instanceInput from the instance buffer.
// We need to take instanceID in the vertex shader and pass it to the fragment shader (for some reason)
[shader("vertex")]
func vertexMain(VertexInput vertexInput, InstanceInput instanceInput, uint instanceID: SV_VulkanInstanceID)->VertexOutput {
    VertexOutput output = {};
    // Essentially, this is 'running the vertex material', but here it is fixed function
    output.position = mul(gViewData.viewProjection, mul(instanceInput.transform, float4(vertexInput.position, 1.)));
    // Forward the instance ID (interpolation does nothing evil here, as this is the same for all vertices in a triangle)
    output.instanceID = instanceID;
    return output;
}

// Fragment shader to rasterize the visibility buffer
// We take barycentrics and triangle ID (primitive ID) from system values, instance ID from the vertex shader output.
[shader("fragment")]
func fragmentMain(float3 barycentric: SV_BaryCentrics, uint triangleID: SV_PrimitiveID, uint instanceID: INSTANCE_ID) -> FragmentResult
{
    FragmentResult result = {};

    // Triangle ID is written directly
    result.packedData.x = triangleID;
    // Instance ID is increased by one to keep 0 as a marker for texels where nothing is visible
    result.packedData.y = instanceID + 1;
    // Barycentrics and their derivatives are packed
    result.packedData.z = packUnorm2x16(barycentric.xy);
    // We can get the derivatives here simply by using ddx
    result.packedData.w = packHalf2x16(ddx(barycentric.xy));
    // TODO: Add and store ddy as well, but this needs cleverer packing or another buffer. This is not a lot of work but we do not use the derivatives right now, so it would be wasted

    return result;
}