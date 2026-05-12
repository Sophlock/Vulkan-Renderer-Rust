module visBufferMaterialPrefixSum;

import visBufferData;
import Core.largeBlock;

struct PrefixSumInput {
    StructuredBuffer<uint> texelCounts;

    StructuredBuffer<uint> materialIndices;
    StructuredBuffer<uint> materialCount;

    RWStructuredBuffer<Atomic<uint>> currentOffset;

    RWStructuredBuffer<uint> outPerMaterialOffset;
}

uniform LargeBlock _;
uniform PrefixSumInput gInput;
uniform GlobalData gGlobalData;

// Accumulate per-material offsets
// Runs an unstable prefix sum over texel counts.
// This will run twice, once for drawn and once for culled materials.
[shader("compute")]
[numthreads(16, 1, 1)]
func computeOffsets(uint dispatch: SV_DispatchThreadID)->void {
    // If we are outside the desired bounds, do nothing
    if (dispatch >= gInput.materialCount[0] || dispatch >= gInput.materialIndices.getCount()) {
        return;
    }

    // Load material index and texel count
    let materialIndex = gInput.materialIndices.Load(dispatch);
    let texelCount = gInput.texelCounts.Load(materialIndex);

    // Accumulate the texel count onto the offset
    if (texelCount > 0) {
        let myOffset = gInput.currentOffset[0].add(texelCount);
        gInput.outPerMaterialOffset[materialIndex] = myOffset;
    }
}