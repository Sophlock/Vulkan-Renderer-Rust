module compute_test;

struct PostProcessSettings {
    float exposureValue;
}

struct PostProcessData {
    Sampler2D input;
    RWTexture2D result;

    PostProcessSettings settings;
}

uniform PostProcessData gPostProcessData;

[shader("compute")]
[numthreads(16, 16, 1)]
func postProcessMain(uint2 dispatch: SV_DispatchThreadID) -> void {
    float width, height;
    gPostProcessData.result.GetDimensions(width, height);
    if (dispatch.x >= width || dispatch.y >= height) {
        return;
    }

    let uv = dispatch / float2(width, height);
    let color = gPostProcessData.input.SampleLevel(uv, 0).rgb;
    gPostProcessData.result[dispatch] = float4(1.f - exp(-color * gPostProcessData.settings.exposureValue), 1.);
}