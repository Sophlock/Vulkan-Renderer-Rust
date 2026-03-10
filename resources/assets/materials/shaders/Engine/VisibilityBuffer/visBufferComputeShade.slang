module visBufferComputeShade;

import Core.geometry;
import Core.material;
import Core.globalData;
import visBufferData;
import Core.largeBlock;

Texture2D<uint4> visBuffer;
RWTexture2D<float4> outputRT;
LargeBlock _;

[shader("compute")]
[numthreads(16, 16, 1)]
func shadeVisBuffer<MaterialType : IMaterial>(uint2 dispatch: SV_DispatchThreadID, uniform VisBufferShadeInput gInput)->void {
    /*let packedVisBuffer = gInput.visBuffer.Load(int3(dispatch.xy, 0));
    let instanceID = packedVisBuffer.y;

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

    let vertex = normalize(vertex1 * barycentric.x + vertex2 * barycentric.y + vertex3 * barycentric.z);

    ModelData modelData = {};
    modelData.modelTransform = instance.modelTransform;
    modelData.inverseTransposeModelTransform = instance.inverseTransposeModelTransform;

    SurfaceGeometry geometry;
    geometry.worldPosition = vertex.worldPosition;
    geometry.worldNormal = normalize(vertex.worldNormal);
    geometry.textureCoordinate = vertex.textureCoordinate;
    geometry.modelData = modelData;
    geometry.viewData = gViewData;
    float3 bitangent = normalize(cross(vertex.worldTangent, geometry.worldNormal));
    // We re-orthogonalize the tangent. This will be normalized because worldNormal and bitangent are orthogonal and normalized
    geometry.worldTangent = cross(geometry.worldNormal, bitangent);
    geometry.tangentToWorld = transpose(float3x3(geometry.worldTangent, bitangent, geometry.worldNormal));

    // viewDirection is used by BRDFs
    float3 viewDirection = normalize(gViewData.viewPosition - geometry.worldPosition);

    // Evaluates the material into a BRDF
    // let materialResult = gMaterial.evaluate(geometry);
    // Shades the BRDF using the light environment*/
    float3 color = float3(1, 1, 1);//max(/*gLightEnvironment.illuminate(materialResult.geometry, materialResult.brdf, viewDirection) +*//* materialResult.brdf.evaluateEmissive(viewDirection), 0.f);*/
    outputRT[dispatch] = float4(color, 1.);
}
