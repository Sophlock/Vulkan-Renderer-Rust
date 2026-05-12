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

// Resolves unsure materials from the first culling step
// This amounts to writing all materials from the unsure buffer into the culled or drawn buffer.
// Note that it is essentially random (and in particular unstable over time) which materials are culled here!
[shader("compute")]
[numthreads(16, 1, 1)]
func resolveUnsure(uint dispatch: SV_DispatchThreadID)->void {
    // If we are outside the unsure buffer, do nothing
    if (dispatch >= gInput.unsureCount[0]) {
        return;
    }

    // Increment the drawn counter
    let myIndex = gInput.currentDrawnIndex[0].add(1);
    // Load the material
    let myMaterial = gInput.unsureIndices.Load(dispatch);

    // If the drawn buffer still has capacity, draw the material, else cull it
    if (myIndex < gInput.drawnIndices.getCount()) {
        gInput.drawnIndices[myIndex] = myMaterial;
    }
    else {
        gInput.culled.write(myMaterial);
    }
}