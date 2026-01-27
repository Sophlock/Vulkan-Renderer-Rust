module mainRaster;

import lights;
import material;
import geometry;
import globalData;

// Main entry point shader
// Holds global object and entry points

uniform ViewData gViewData;
uniform ModelData gModelData;

uniform ILightEnvironment gLightEnvironment;
uniform IMaterial gMaterial;

// Input to the vertex shader/Type of the vertex buffer
struct VertexInput
{
    float3 position : POSITION;
    float3 normal : NORMAL;
    float3 tangent : TANGENT;
    float2 textureCoordinate : UV;
}

// Vertex that was processed by the vertex shader
struct ProcessedVertex
{
    float3 worldPosition;
    float3 worldNormal;
    float3 worldTangent;
    float2 textureCoordinate;
}

// Output of the vertex shader
struct VertexOutput
{
    ProcessedVertex vertex : Vertex;
    float4 sv_position : SV_Position;
}


// Vertex shader
// Does nothing special
[shader("vertex")]
VertexOutput vertexMain(VertexInput input)
{
    VertexOutput output;
    output.vertex.worldPosition = mul(gModelData.modelTransform, float4(input.position, 1.)).xyz;
    output.vertex.worldNormal = mul(gModelData.inverseTransposeModelTransform, float4(input.normal, 0.)).xyz;
    output.vertex.worldTangent = mul(gModelData.inverseTransposeModelTransform, float4(input.tangent, 0.)).xyz;
    output.vertex.textureCoordinate = input.textureCoordinate;
    output.sv_position = mul(gViewData.viewProjection, float4(output.vertex.worldPosition, 1.));
    return output;
}

// Fragment shader
// Assembles surface geometry, evaluates the material and shades the resulting BRDF
[shader("fragment")]
float4 fragmentMain(ProcessedVertex vertex : Vertex) : SV_Target
{
    SurfaceGeometry geometry;
    geometry.worldPosition = vertex.worldPosition;
    geometry.worldNormal = normalize(vertex.worldNormal);
    geometry.textureCoordinate = vertex.textureCoordinate;
    geometry.modelData = gModelData;
    geometry.viewData = gViewData;
    float3 bitangent = normalize(cross(vertex.worldTangent, geometry.worldNormal));
    // We re-orthogonalize the tangent. This will be normalized because worldNormal and bitangent are orthogonal and normalized
    geometry.worldTangent = cross(geometry.worldNormal, bitangent);
    geometry.tangentToWorld = transpose(float3x3(geometry.worldTangent, bitangent, geometry.worldNormal));

    // viewDirection is used by BRDFs
    float3 viewDirection = normalize(gViewData.viewPosition - geometry.worldPosition);

    // Evaluates the material into a BRDF
    let materialResult = gMaterial.evaluate(geometry);
    // Shades the BRDF using the light environment
    float3 color = max(gLightEnvironment.illuminate(materialResult.geometry, materialResult.brdf, viewDirection) + materialResult.brdf.evaluateEmissive(viewDirection), 0.f);
    return float4(1.f - exp(-color * gViewData.exposureValue), 1.);
}