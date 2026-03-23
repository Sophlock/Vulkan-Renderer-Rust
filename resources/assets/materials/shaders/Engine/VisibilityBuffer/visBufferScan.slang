module visBufferScan;

import visBufferData;
import Core.largeBlock;

struct VisBufferData {
    Texture2D<uint4> visBuffer;
    RWStructuredBuffer<Atomic<uint>> materialFragmentCounts;
}

uniform LargeBlock _;
uniform VisBufferData gInput;
uniform GlobalData gGlobalData;

[shader("compute")]
[numthreads(16, 16, 1)]
func countTexels(uint2 dispatch: SV_DispatchThreadID)->void {
    let packedVisBuffer = gInput.visBuffer.Load(int3(dispatch.xy, 0));
    let instanceID = packedVisBuffer.y;

    let instance = gGlobalData.instances.Load(instanceID);
    let materialInstance = gGlobalData.materialInstances.Load(instance.materialInstanceIndex);
    gInput.materialFragmentCounts[materialInstance.materialIndex].add(1);
}