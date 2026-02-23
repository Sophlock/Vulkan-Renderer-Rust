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
}

struct FragmentResult {
    uint4 packedData : SV_Target;
}

uniform ViewData gViewData;

[shader("vertex")]
func vertexMain(VertexInput vertexInput, InstanceInput instanceInput)->VertexOutput {
    VertexOutput output = {};
    output.position = mul(gViewData.viewProjection, mul(instanceInput.transform, float4(vertexInput.position, 1.)));
    return output;
}

[shader("fragment")]
func fragmentMain(float3 barycentric : SV_BaryCentrics, uint triangleID : SV_PrimitiveID, uint instanceID : SV_InstanceID) -> FragmentResult
{
    FragmentResult result = {};

    result.packedData.x = triangleID;
    result.packedData.y = instanceID;
    result.packedData.z = packUnorm2x16(barycentric.xy);
    result.packedData.w = packHalf2x16(ddx(barycentric.xy));
    // TODO: Add ddy as well

    return result;
}