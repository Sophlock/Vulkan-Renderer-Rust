module visBufferMaterialDataWrite;

import visBufferData;

struct MaterialWriteData {
    Texture2D<uint4> visBuffer;
    RWTexture2D<uint64_t> materialDataBuffer;
}

uniform MaterialWriteData gInput;

[shader("compute")]
[numthreads(16, 16, 1)]
func writeMaterialData(uint2 dispatch: SV_DispatchThreadID, uniform GlobalData* globalData)->void {
    let packedVisBuffer = gInput.visBuffer.Load(int3(dispatch.xy, 0));
    let instanceID = packedVisBuffer.y;

    let instance = globalData.instances.Load(instanceID);
    let materialInstance = globalData.materialInstances.Load(instance.materialInstanceIndex);
    let material = globalData.materials.Load(materialInstance.materialIndex);
    gInput.materialDataBuffer[dispatch.xy] = material.pipelineAddress;
}