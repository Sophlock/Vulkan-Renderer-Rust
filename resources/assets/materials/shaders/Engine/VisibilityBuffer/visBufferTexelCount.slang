module visBufferScan;

import visBufferData;
import Core.largeBlock;

struct VisBufferData {
    Texture2D<uint4> visBuffer;
    RWStructuredBuffer<Atomic<uint>> materialFragmentCounts;
    RWTexture2D<uint> relativePerMaterialOffset;
}

uniform LargeBlock _;
uniform VisBufferData gInput;
uniform GlobalData gGlobalData;

[shader("compute")]
[numthreads(16, 16, 1)]
func countTexels(uint2 dispatch: SV_DispatchThreadID)->void {
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

    let instance = gGlobalData.instances.Load(instanceID);
    let materialInstance = gGlobalData.materialInstances.Load(instance.materialInstanceIndex);
    gInput.relativePerMaterialOffset[dispatch] = gInput.materialFragmentCounts[materialInstance.materialIndex].add(1);
}