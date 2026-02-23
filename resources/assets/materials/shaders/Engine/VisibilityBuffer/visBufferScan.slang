module visBufferScan;

import visBufferData;

struct VisBufferData {
    Texture2D<uint4> visBuffer;
    RWStructuredBuffer<Atomic<uint>> materialFragmentCounts;
}

uniform VisBufferData gInput;
uniform GlobalData gGlobalData;

[shader("compute")]
[numthreads(16, 16, 1)]
func countTexels(uint2 dispatch: SV_DispatchThreadID)->void {
    let packedVisBuffer = gInput.visBuffer.Load(int3(dispatch.xy, 0));
    let instanceID = packedVisBuffer.y;

    let instance = gGlobalData.instances[instanceID];
    gInput.materialFragmentCounts[instance.materialIndex].add(1);
}