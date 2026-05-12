module visBufferShaderCull;

import visBufferData;
import Core.largeBlock;

struct MaterialCountData {
    StructuredBuffer<uint> texelCounts;
    MaterialCollection drawn;
    MaterialCollection culled;
    MaterialCollection unsure;
    float minDrawnPixelFootprint;
    float maxCulledPixelFootprint;
}

uniform LargeBlock _;
uniform MaterialCountData gInput;
uniform GlobalData gGlobalData;

// Sorts all visible materials into drawn, culled and unsure buffers
// The unsure materials are resolved in a separate step.
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

    // If the material is invisible, do nothing
    if (texelCount <= 0) {
        return;
    }

    // Compute pixel footprint
    let screenSize = gGlobalData.mutData.Load(0).screenSize;
    let pixelFootprint = texelCount * 1.0f / (screenSize.x * screenSize.y);

    // Classify the material based on pixel footprint into the three categories
    if (pixelFootprint > gInput.maxCulledPixelFootprint) {
        // Draw this material, since its footprint is large
        gInput.drawn.write(materialIndex);
    }
    else if (pixelFootprint < gInput.minDrawnPixelFootprint) {
        // Cull this material, since its footprint is too small
        gInput.culled.write(materialIndex);
    }
    else {
        // Can't be decided yet, defer decision
        gInput.unsure.write(materialIndex);
    }
}