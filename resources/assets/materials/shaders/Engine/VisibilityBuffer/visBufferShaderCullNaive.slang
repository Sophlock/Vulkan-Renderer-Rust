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

[shader("compute")]
[numthreads(16, 1, 1)]
func cullShaders(uint dispatch: SV_DispatchThreadID)->void {
    if (dispatch >= gGlobalData.materials.getCount()) {
        return;
    }

    let materialIndex = dispatch;
    let texelCount = gInput.texelCounts[materialIndex];

    if (texelCount > 0) {
        let myIndex = gInput.index[0].add(1);
        uint maxSequence, _;
        gInput.pipelineBindParameters.GetDimensions(maxSequence, _);
        if (myIndex >= maxSequence) return;
        let material = gGlobalData.materials.Load(materialIndex);
        let pipelineAddress = material.pipelineAddress;
        gInput.pipelineBindParameters[myIndex] = {pipelineAddress};
        gInput.computeDispatchParameters[myIndex] = { uint3(gGlobalData.mutData.Load(0).screenSize / 16 + 1, 1) };
        gInput.shadePushConstants[myIndex] = { materialIndex };
    }
}