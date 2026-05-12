module visBufferFillAll;

import visBufferData;
import Core.largeBlock;

struct PipelineBindParameter {
    uint64_t pipelineAddress;
}

struct ComputeDispatchParameter {
    uint3 dispatch;
}

struct FillData {
    RWStructuredBuffer<PipelineBindParameter> pipelineBindParameters;
    RWStructuredBuffer<ComputeDispatchParameter> computeDispatchParameters;
    RWStructuredBuffer<VisBufferShadeInput> shadePushConstants;
}

uniform LargeBlock _;
uniform FillData gInput;
uniform GlobalData gGlobalData;

// Fills all indirect commands streams with data for all materials
[shader("compute")]
[numthreads(16, 1, 1)]
func fillIndirectCommandsStreams(uint dispatch: SV_DispatchThreadID)->void {
    // If we are outside the buffer's bounds, do nothing
    if (dispatch >= gGlobalData.materials.getCount()) {
        return;
    }

    // Load the pipeline device address
    let pipelineAddress = gGlobalData.materials.Load(dispatch).pipelineAddress;

    // Write the address
    gInput.pipelineBindParameters[dispatch] = { pipelineAddress };
    // Write a dispatch for the entire screen
    gInput.computeDispatchParameters[dispatch] = { uint3(gGlobalData.mutData.Load(0).screenSize / 16 + 1, 1) };
    // Write the material index (in this case, dispatch)
    gInput.shadePushConstants[dispatch] = { dispatch };
}