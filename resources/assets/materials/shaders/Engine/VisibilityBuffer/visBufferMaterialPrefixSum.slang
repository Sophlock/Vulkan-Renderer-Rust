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

[shader("compute")]
[numthreads(16, 1, 1)]
func computeOffsets(uint dispatch: SV_DispatchThreadID)->void {
    if (dispatch >= gInput.materialCount[0] || dispatch >= gInput.materialIndices.getCount()) {
        return;
    }

    let materialIndex = gInput.materialIndices.Load(dispatch);
    let texelCount = gInput.texelCounts.Load(materialIndex);

    if (texelCount > 0) {
        let myOffset = gInput.currentOffset[0].add(texelCount);
        gInput.outPerMaterialOffset[materialIndex] = myOffset;
    }
}