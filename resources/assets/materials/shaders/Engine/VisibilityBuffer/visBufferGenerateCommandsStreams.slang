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

// Generates commands streams from the processed visibility buffer data
// This version is for use with device generated commands and processes only drawn materials.
[shader("compute")]
[numthreads(16, 1, 1)]
func generateCommandsStreams(uint dispatch: SV_DispatchThreadID)->void {
    // The number of materials that are drawn is the minimum of the two
    let count = min(gInput.drawnMaterialCount[0], gInput.drawnMaterials.getCount());
    // If we are out of bounds, do nothing. dispatch == count represents the fallback material
    if (dispatch > count) {
        return;
    }

    uint materialIndex;
    uint texelCount;
    // If we are looking at the fallback material, do some extra stuff
    if (dispatch == count) {
        // Fallback material is always at index 0
        materialIndex = 0;
        // Texel count is the difference between the number of all texels and the number of texels shaded with their desired material
        texelCount = gInput.totalTexelCount[0] - gInput.texelsWithoutFallback[0];

        // We need to 'register' the fallback material here
        gInput.finalMaterialCount[0] = count + 1;

        // And write its size and offset
        gInput.perMaterialOffsets[materialIndex] = gInput.texelsWithoutFallback[0];
        gInput.texelCounts[materialIndex] = texelCount;
    }
    else {
        // Else just load material index and texel count
        materialIndex = gInput.drawnMaterials[dispatch];
        texelCount = gInput.texelCounts[materialIndex];
    }

    // Load the pipeline device address
    let pipelineAddress = gGlobalData.materials[materialIndex].pipelineAddress;

    // Write the pipeline address
    gInput.pipelineBindParameters[dispatch] = { pipelineAddress };
    // Write a 1D dispatch over this material's region in the binned buffer
    gInput.computeDispatchParameters[dispatch] = { uint3(texelCount / 16 + 1, 1, 1) };
    // Pass the material ID/index as push constant
    gInput.shadePushConstants[dispatch] = { materialIndex };
}

// Generates commands streams from the processed visibility buffer data
// This version is for use without device generated commands and processes all materials.
[shader("compute")]
[numthreads(16, 1, 1)]
func generateCommandsStreamsNoIndirect(uint dispatch: SV_DispatchThreadID)->void {
    // If we are outside the material buffer, do nothing
    if (dispatch >= gGlobalData.materials.getCount()) {
        return;
    }

    // Material index is dispatch
    let materialIndex = dispatch;

    uint texelCount;
    // If we are looking at the fallback material, do some extra stuff
    if (dispatch == 0) {
        // Texel count is the difference between the number of all texels and the number of texels shaded with their desired material
        texelCount = gInput.totalTexelCount[0] - gInput.texelsWithoutFallback[0];

        // Register texel count and offset for the fallback material
        gInput.perMaterialOffsets[materialIndex] = gInput.texelsWithoutFallback[0];
        gInput.texelCounts[materialIndex] = texelCount;
    }
    else {
        // Else just load the texel count
        texelCount = gInput.texelCounts[materialIndex];
    }

    // For this, we only need the dispatch params (since we can not do the other parts indirectly)
    // Dispatch params are the same as in the version above
    gInput.computeDispatchParameters[materialIndex] = { uint3(texelCount / 16 + 1, 1, 1) };
}