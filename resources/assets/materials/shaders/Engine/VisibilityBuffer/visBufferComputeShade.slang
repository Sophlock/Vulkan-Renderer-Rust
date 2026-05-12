module visBufferComputeShade;

import Core.geometry;
import Core.material;
import Core.globalData;
import visBufferData;
import Core.largeBlock;

struct BinnedShadeInput {
    StructuredBuffer<uint> texelCounts;
    StructuredBuffer<uint> offsets;

    StructuredBuffer<uint2> binnedTexels;
}

uniform BinnedShadeInput gBinInput;
uniform Texture2D<uint4> visBuffer;
uniform RWTexture2D<float4> outputRT;
uniform GlobalData gGlobalData;
uniform LargeBlock _;

// Shading without binning
[shader("compute")]
[numthreads(16, 16, 1)]
func shadeVisBufferNaive<MaterialType : IMaterial>(uint2 dispatch: SV_DispatchThreadID, uniform VisBufferShadeInput gInput)->void {
    // If we are outside the screen, do nothing
    float width, height;
    visBuffer.GetDimensions(width, height);
    if (dispatch.x >= width || dispatch.y >= height) {
        return;
    }

    // Sample and unpack the visibility buffer and get some data
    let packedVisBuffer = visBuffer.Load(int3(dispatch.xy, 0));

    let instanceID = packedVisBuffer.y - 1;

    let instance = gGlobalData.instances[instanceID];
    let materialInstance = gGlobalData.materialInstances[instance.materialInstanceIndex];
    let material = gGlobalData.materials[materialInstance.materialIndex];

    // If this dispatch is not the material of this texel, then do nothing to this texel
    [branch]
    if (gInput.thisMaterialIndex != materialInstance.materialIndex) {
        return;
    }

    // Perform shading and write to the output target
    outputRT[dispatch] = performVisBufferShade<MaterialType>(packedVisBuffer);
}

// Shading with binning
[shader("compute")]
[numthreads(16, 1, 1)]
func shadeVisBufferBinned<MaterialType : IMaterial>(uint dispatch: SV_DispatchThreadID, uniform VisBufferShadeInput gInput)->void {
    // If we are outside our region of the binned buffer, do nothing
    if (dispatch >= gBinInput.texelCounts[gInput.thisMaterialIndex]) {
        return;
    }

    // Compute the index of the current texel in the binned buffer
    let texelIndex = dispatch + gBinInput.offsets[gInput.thisMaterialIndex];
    // Load the texel's coordinates
    let texelPos = gBinInput.binnedTexels[texelIndex];

    // Sample the visibility buffer
    let packedVisBuffer = visBuffer.Load(int3(texelPos, 0));

    // Shade and write to the render target
    outputRT[texelPos] = performVisBufferShade<MaterialType>(packedVisBuffer);
}

func transformVertex(Vertex vertex, ModelData modelData)->Vertex {
    Vertex result = {};
    result.worldPosition = mul(modelData.modelTransform, float4(vertex.worldPosition.xyz, 1)).xyz;
    result.worldNormal = mul(modelData.inverseTransposeModelTransform, float4(vertex.worldNormal, 0)).xyz;
    result.worldTangent = mul(modelData.inverseTransposeModelTransform, float4(vertex.worldTangent, 0)).xyz;
    result.textureCoordinate = vertex.textureCoordinate;
    return result;
}

func performVisBufferShade<MaterialType : IMaterial>(uint4 packedVisBuffer)->float4 {
    // Load data based on the packed visibility buffer sample
    let instanceID = packedVisBuffer.y - 1;

    let instance = gGlobalData.instances[instanceID];
    let materialInstance = gGlobalData.materialInstances[instance.materialInstanceIndex];
    let material = gGlobalData.materials[materialInstance.materialIndex];

    let mesh = gGlobalData.meshes[instance.meshIndex];
    let triangle = gGlobalData.indexBuffer[mesh.firstPrimitive + packedVisBuffer.x];

    // The triangle stores the indices of each vertex
    let vertex1 = gGlobalData.vertexBuffer[mesh.firstVertex + triangle.triangleIndices.x];
    let vertex2 = gGlobalData.vertexBuffer[mesh.firstVertex + triangle.triangleIndices.y];
    let vertex3 = gGlobalData.vertexBuffer[mesh.firstVertex + triangle.triangleIndices.z];

    // Load all three barycentric coordinates
    let barycentricXY = unpackUnorm2x16ToFloat(packedVisBuffer.z);
    let barycentric = saturate(float3(barycentricXY, 1 - barycentricXY.x - barycentricXY.y));

    // Interpolate to the virtual vertex inside the triangle
    let vertexLocal = normalize(vertex1 * barycentric.x + vertex2 * barycentric.y + vertex3 * barycentric.z);

    // Prepare per-instance input data for the vertex material
    ModelData modelData = {};
    modelData.modelTransform = instance.modelTransform;
    modelData.inverseTransposeModelTransform = instance.inverseTransposeModelTransform;

    // Re-run the vertex material (right now this is a fixed-function operation)
    let vertex = transformVertex(vertexLocal, modelData);

    // Prepare the inputs for the shade material
    SurfaceGeometry geometry;
    geometry.worldPosition = vertex.worldPosition;
    geometry.worldNormal = normalize(vertex.worldNormal);
    geometry.textureCoordinate = vertex.textureCoordinate;
    geometry.modelData = modelData;
    geometry.viewData.viewProjection = gGlobalData.mutData.Load(0).viewMatrix;
    geometry.viewData.viewPosition = gGlobalData.mutData.Load(0).viewPosition;
    float3 bitangent = normalize(cross(vertex.worldTangent, geometry.worldNormal));
    // We re-orthogonalize the tangent. This will be normalized because worldNormal and bitangent are orthogonal and normalized
    geometry.worldTangent = cross(geometry.worldNormal, bitangent);
    geometry.tangentToWorld = transpose(float3x3(geometry.worldTangent, bitangent, geometry.worldNormal));

    // viewDirection is used by BRDFs
    float3 viewDirection = normalize(geometry.viewData.viewPosition - geometry.worldPosition);

    // Evaluates the material into a BRDF
    // TODO: Right now, we do not support passing material instance data in here. This is mostly limited by the CPU-side architecture
    // TODO: To support texture sampling, we need to reconstruct the derivatives here
    MaterialType materialInstanceData = {};
    let materialResult = materialInstanceData.evaluate(geometry);
    // Shades the BRDF 
    // TODO: This should use a light environment but this is currently not available on the CPU side. Also, ideally this should have some clustered lighting approach
    float3 color = max(/*gLightEnvironment.illuminate(materialResult.geometry, materialResult.brdf, viewDirection) +*/ materialResult.brdf.evaluateEmissive(viewDirection), 0.f);
    return float4(color, 1.);
}
