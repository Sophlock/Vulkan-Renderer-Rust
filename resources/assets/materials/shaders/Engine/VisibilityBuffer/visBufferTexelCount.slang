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

// Counts texels per material
[shader("compute")]
[numthreads(16, 16, 1)]
func countTexels(uint2 dispatch: SV_DispatchThreadID)->void {
    // If we are outside the visibility buffer, do nothing
    float width, height;
    gInput.visBuffer.GetDimensions(width, height);
    if (dispatch.x >= width || dispatch.y >= height) {
        return;
    }

    // Sample the visibility buffer
    let packedVisBuffer = gInput.visBuffer.Load(int3(dispatch.xy, 0));

    // If nothing is visible at the current texel, do nothing
    if (packedVisBuffer.equals(uint4(0))) {
        return;
    }
    
    let instanceID = packedVisBuffer.y - 1;

    // Find the material of the current texel
    let instance = gGlobalData.instances.Load(instanceID);
    let materialInstance = gGlobalData.materialInstances.Load(instance.materialInstanceIndex);
    
    // Increment the atomic counter of the material. Save the result to keep an index of the texel relative to the material (used later for binning)
    gInput.relativePerMaterialOffset[dispatch] = gInput.materialFragmentCounts[materialInstance.materialIndex].add(1);
}