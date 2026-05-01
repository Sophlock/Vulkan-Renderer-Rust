module visBufferTexelBin;

import visBufferData;
import Core.largeBlock;

struct BinInput {
    Texture2D<uint4> visBuffer;
    Texture2D<uint> relativePerMaterialOffsets;

    StructuredBuffer<uint> perMaterialOffsets;

    RWStructuredBuffer<uint2> outBinnedTexels;
}

uniform LargeBlock _;
uniform BinInput gInput;
uniform GlobalData gGlobalData;

[shader("compute")]
[numthreads(16, 16, 1)]
func binTexels(uint2 dispatch: SV_DispatchThreadID)->void {
    float width, height;
    gInput.visBuffer.GetDimensions(width, height);
    if (dispatch.x >= width || dispatch.y >= height) {
        return;
    }

    let packedVisBuffer = gInput.visBuffer.Load(int3(dispatch.xy, 0));
    if (packedVisBuffer.equals(uint4(0))) {
        return;
    }
    let instanceID = packedVisBuffer.y - 1;

    let instance = gGlobalData.instances[instanceID];
    let materialInstance = gGlobalData.materialInstances[instance.materialInstanceIndex];
    let materialIndex = materialInstance.materialIndex;

    let localOffset = gInput.relativePerMaterialOffsets.Load(uint3(dispatch, 0));
    let materialOffset = gInput.perMaterialOffsets.Load(materialIndex);

    gInput.outBinnedTexels[materialOffset + localOffset] = dispatch;
}
