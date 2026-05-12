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

// Groups texels by their materials
[shader("compute")]
[numthreads(16, 16, 1)]
func binTexels(uint2 dispatch: SV_DispatchThreadID)->void {
    // If we are outside the visibility buffer, do nothing
    float width, height;
    gInput.visBuffer.GetDimensions(width, height);
    if (dispatch.x >= width || dispatch.y >= height) {
        return;
    }

    // If nothing is visible at the texel, do nothing
    let packedVisBuffer = gInput.visBuffer.Load(int3(dispatch.xy, 0));
    if (packedVisBuffer.equals(uint4(0))) {
        return;
    }
    
    // Find the material index from the visibility buffer
    let instanceID = packedVisBuffer.y - 1;

    let instance = gGlobalData.instances[instanceID];
    let materialInstance = gGlobalData.materialInstances[instance.materialInstanceIndex];
    let materialIndex = materialInstance.materialIndex;

    // Load this texel's local offset and the material's offset
    let localOffset = gInput.relativePerMaterialOffsets.Load(uint3(dispatch, 0));
    let materialOffset = gInput.perMaterialOffsets.Load(materialIndex);

    // Write this texel's coordinates at the final offset
    gInput.outBinnedTexels[materialOffset + localOffset] = dispatch;
}
