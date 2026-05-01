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

[shader("compute")]
[numthreads(16, 1, 1)]
func fillIndirectCommandsStreams(uint dispatch: SV_DispatchThreadID)->void {
    if (dispatch >= gGlobalData.materials.getCount()) {
        return;
    }

    let pipelineAddress = gGlobalData.materials.Load(dispatch).pipelineAddress;

    gInput.pipelineBindParameters[dispatch] = { pipelineAddress };
    gInput.computeDispatchParameters[dispatch] = { uint3(gGlobalData.mutData.Load(0).screenSize / 16 + 1, 1) };
    gInput.shadePushConstants[dispatch] = { pipelineAddress };
}