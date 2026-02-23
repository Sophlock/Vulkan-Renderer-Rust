module visBufferMaterialDataWrite;

import visBufferData;

struct MaterialWriteData {
    Texture2D<uint4> visBuffer;
    RWTexture2D<uint64_t> materialDataBuffer;
}

uniform MaterialWriteData gInput;
uniform GlobalData gGlobalData;

[shader("compute")]
[numthreads(16, 16, 1)]
func writeMaterialData(uint2 dispatch: SV_DispatchThreadID)->void {
    let packedVisBuffer = gInput.visBuffer.Load(int3(dispatch.xy, 0));
    let instanceID = packedVisBuffer.y;

    let instance = gGlobalData.instances[instanceID];
    let material = gGlobalData.materials[instance.materialIndex];
    gInput.materialDataBuffer[dispatch.xy] = material.pipelineAddress;
}