module visBufferComputeShade;

import Core.geometry;
import Core.material;
import Core.globalData;
import visBufferData;
import Core.largeBlock;

uniform Texture2D<uint4> visBuffer;
uniform RWTexture2D<float4> outputRT;
uniform GlobalData gGlobalData;
uniform LargeBlock _;

[shader("compute")]
[numthreads(16, 16, 1)]
func shadeVisBuffer<MaterialType : IMaterial>(uint2 dispatch: SV_DispatchThreadID, uniform VisBufferShadeInput gInput)->void {
    float width, height;
    visBuffer.GetDimensions(width, height);
    if (dispatch.x >= width || dispatch.y >= height) {
        return;
    }

    let packedVisBuffer = visBuffer.Load(int3(dispatch.xy, 0));
    if (packedVisBuffer.equals(uint4(0))) {
        // TODO: Use clear instead?
        outputRT[dispatch] = float4(0, 0, 0, 1);
        return;
    }
    let instanceID = packedVisBuffer.y - 1;

    let instance = gGlobalData.instances[instanceID];
    let materialInstance = gGlobalData.materialInstances[instance.materialInstanceIndex];
    let material = gGlobalData.materials[materialInstance.materialIndex];

    if (gInput.thisPipelineAddress != material.pipelineAddress) {
        return;
    }

    let mesh = gGlobalData.meshes[instance.meshIndex];
    let triangle = gGlobalData.indexBuffer[mesh.firstPrimitive + packedVisBuffer.x];

    let vertex1 = gGlobalData.vertexBuffer[mesh.firstVertex + triangle.triangleIndices.x];
    let vertex2 = gGlobalData.vertexBuffer[mesh.firstVertex + triangle.triangleIndices.y];
    let vertex3 = gGlobalData.vertexBuffer[mesh.firstVertex + triangle.triangleIndices.z];

    let barycentricXY = unpackUnorm2x16ToFloat(packedVisBuffer.z);
    let barycentric = saturate(float3(barycentricXY, 1 - barycentricXY.x - barycentricXY.y));

    let vertexLocal = normalize(vertex1 * barycentric.x + vertex2 * barycentric.y + vertex3 * barycentric.z);

    ModelData modelData = {};
    modelData.modelTransform = instance.modelTransform;
    modelData.inverseTransposeModelTransform = instance.inverseTransposeModelTransform;

    let vertex = transformVertex(vertexLocal, modelData);

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
    // TODO: Get data somehow
    MaterialType materialInstanceData = {};
    let materialResult = materialInstanceData.evaluate(geometry);
    // Shades the BRDF using the light environment*/
    float3 color = max(/*gLightEnvironment.illuminate(materialResult.geometry, materialResult.brdf, viewDirection) +*/ materialResult.brdf.evaluateEmissive(viewDirection), 0.f);
    color = mul(modelData.modelTransform, float4(0, 0, 0, 1)).xyz;
    outputRT[dispatch] = float4(color, 1.);
}

Vertex transformVertex(Vertex vertex, ModelData modelData) {
    Vertex result = {};
    result.worldPosition = mul(modelData.modelTransform, float4(vertex.worldPosition.xyz, 1)).xyz;
    result.worldNormal = mul(modelData.inverseTransposeModelTransform, float4(vertex.worldNormal, 0)).xyz;
    result.worldTangent = mul(modelData.inverseTransposeModelTransform, float4(vertex.worldTangent, 0)).xyz;
    result.textureCoordinate = vertex.textureCoordinate;
    return result;
}
