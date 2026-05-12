module visBufferShaderCullNaive;

import visBufferData;
import Core.largeBlock;

struct PipelineBindParameter {
    uint64_t pipelineAddress; 
}

struct ComputeDispatchParameter {
    uint3 dispatch;
}

struct MaterialCountData {
    StructuredBuffer<uint> texelCounts;
    RWStructuredBuffer<Atomic<uint>> index;
    RWStructuredBuffer<PipelineBindParameter> pipelineBindParameters;
    RWStructuredBuffer<ComputeDispatchParameter> computeDispatchParameters;
    RWStructuredBuffer<VisBufferShadeInput> shadePushConstants;
}

uniform LargeBlock _;
uniform MaterialCountData gInput;
uniform GlobalData gGlobalData;

// Naive material culling
// Simply puts all visible materials into the commands streams until their capacity is full.
[shader("compute")]
[numthreads(16, 1, 1)]
func cullShaders(uint dispatch: SV_DispatchThreadID)->void {
    // If we are outside the material buffer, do nothing
    if (dispatch >= gGlobalData.materials.getCount()) {
        return;
    }

    // Load material index and texel count
    let materialIndex = dispatch;
    let texelCount = gInput.texelCounts[materialIndex];

    // Ignore invisible materials
    if (texelCount > 0) {
        // Increase the global index counter
        let myIndex = gInput.index[0].add(1);

        // If the sequence buffers are full, return
        uint maxSequence, _;
        gInput.pipelineBindParameters.GetDimensions(maxSequence, _);
        if (myIndex >= maxSequence) return;

        // Otherwise write the material into the commands streams
        let material = gGlobalData.materials.Load(materialIndex);
        let pipelineAddress = material.pipelineAddress;

        // Write pipeline device address
        gInput.pipelineBindParameters[myIndex] = {pipelineAddress};
        // Write a dispatch over the entire screen
        gInput.computeDispatchParameters[myIndex] = { uint3(gGlobalData.mutData.Load(0).screenSize / 16 + 1, 1) };
        // Pass the material ID/index as a push constant
        gInput.shadePushConstants[myIndex] = { materialIndex };
    }
}