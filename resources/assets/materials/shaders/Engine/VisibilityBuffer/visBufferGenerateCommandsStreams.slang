module visBufferGenerateCommandsStreams;

import visBufferData;
import Core.largeBlock;

struct PipelineBindParameter {
    uint64_t pipelineAddress;
}

struct ComputeDispatchParameter {
    uint3 dispatch;
}

struct GenerateCommandsStreamsInput {
    StructuredBuffer<uint> drawnMaterials;
    StructuredBuffer<uint> drawnMaterialCount;

    RWStructuredBuffer<uint> texelCounts;

    StructuredBuffer<uint> texelsWithoutFallback;
    StructuredBuffer<uint> totalTexelCount;
    RWStructuredBuffer<uint> finalMaterialCount;
    RWStructuredBuffer<uint> perMaterialOffsets;

    RWStructuredBuffer<PipelineBindParameter> pipelineBindParameters;
    RWStructuredBuffer<ComputeDispatchParameter> computeDispatchParameters;
    RWStructuredBuffer<VisBufferShadeInput> shadePushConstants;
}

uniform LargeBlock _;
uniform GlobalData gGlobalData;
uniform GenerateCommandsStreamsInput gInput;

[shader("compute")]
[numthreads(16, 1, 1)]
func generateCommandsStreams(uint dispatch: SV_DispatchThreadID)->void {
    let count = min(gInput.drawnMaterialCount[0], gInput.drawnMaterials.getCount());
    if (dispatch > count) {
        return;
    }

    uint materialIndex;
    uint texelCount;
    if (dispatch == count) {
        // Fallback material is always at index 0
        materialIndex = 0;
        texelCount = gInput.totalTexelCount[0] - gInput.texelsWithoutFallback[0];

        // We need to 'register' the fallback pipeline here
        gInput.finalMaterialCount[0] = count + 1;

        gInput.perMaterialOffsets[materialIndex] = gInput.texelsWithoutFallback[0];
        gInput.texelCounts[materialIndex] = texelCount;
    }
    else {
        materialIndex = gInput.drawnMaterials[dispatch];
        texelCount = gInput.texelCounts[materialIndex];
    }

    let pipelineAddress = gGlobalData.materials[materialIndex].pipelineAddress;

    gInput.pipelineBindParameters[dispatch] = { pipelineAddress };
    gInput.computeDispatchParameters[dispatch] = { uint3(texelCount / 16 + 1, 1, 1) };
    gInput.shadePushConstants[dispatch] = { materialIndex };
}

[shader("compute")]
[numthreads(16, 1, 1)]
func generateCommandsStreamsNoIndirect(uint dispatch: SV_DispatchThreadID)->void {
    if (dispatch >= gGlobalData.materials.getCount()) {
        return;
    }

    let materialIndex = dispatch;

    uint texelCount;
    if (dispatch == 0) {
        // Fallback material
        texelCount = gInput.totalTexelCount[0] - gInput.texelsWithoutFallback[0];

        gInput.perMaterialOffsets[materialIndex] = gInput.texelsWithoutFallback[0];
        gInput.texelCounts[materialIndex] = texelCount;
    }
    else {
        texelCount = gInput.texelCounts[materialIndex];
    }

    gInput.computeDispatchParameters[materialIndex] = { uint3(texelCount / 16 + 1, 1, 1) };
}