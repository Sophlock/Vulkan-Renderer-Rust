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

[shader("compute")]
[numthreads(16, 1, 1)]
func cullShaders(uint dispatch: SV_DispatchThreadID)->void {
    if (dispatch >= gGlobalData.materials.getCount()) {
        return;
    }

    let materialIndex = dispatch;
    let texelCount = gInput.texelCounts[materialIndex];

    if (texelCount <= 0) {
        return;
    }

    let screenSize = gGlobalData.mutData.Load(0).screenSize;
    let pixelFootprint = texelCount * 1.0f / (screenSize.x * screenSize.y);

    if (pixelFootprint > gInput.maxCulledPixelFootprint) {
        // Draw this pipeline
        gInput.drawn.write(materialIndex);
    }
    else if (pixelFootprint < gInput.minDrawnPixelFootprint) {
        // Cull this pipeline
        gInput.culled.write(materialIndex);
    }
    else {
        gInput.unsure.write(materialIndex);
    }
}