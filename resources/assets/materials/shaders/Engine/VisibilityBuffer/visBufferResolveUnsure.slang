module visBufferResolveUnsure;

import visBufferData;
import Core.largeBlock;

struct ResolveUnsureInput {
    StructuredBuffer<uint> unsureIndices;
    StructuredBuffer<uint> unsureCount;

    RWStructuredBuffer<uint> drawnIndices;
    RWStructuredBuffer<Atomic<uint>> currentDrawnIndex;

    MaterialCollection culled;
}

uniform LargeBlock _;
uniform ResolveUnsureInput gInput;

[shader("compute")]
[numthreads(16, 1, 1)]
func resolveUnsure(uint dispatch: SV_DispatchThreadID)->void {
    if (dispatch >= gInput.unsureCount[0]) {
        return;
    }

    let myIndex = gInput.currentDrawnIndex[0].add(1);
    let myMaterial = gInput.unsureIndices.Load(dispatch);

    if (myIndex < gInput.drawnIndices.getCount()) {
        gInput.drawnIndices[myIndex] = myMaterial;
    }
    else {
        // No more pipelines allowed
        gInput.culled.write(myMaterial);
    }
}