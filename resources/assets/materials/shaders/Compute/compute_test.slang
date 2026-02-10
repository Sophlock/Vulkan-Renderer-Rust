module compute_test;

struct ComputeData {
    Sampler2D input;
    float2 screenSize;
    float exposureValue;

    RWTexture2D result;
}

uniform ComputeData gComputeInput;

[shader("compute")]
[numthreads(16, 16, 1)]
void compute_test(uint2 dispatch: SV_DispatchThreadID) {
    let uv = dispatch / gComputeInput.screenSize;
    let color = gComputeInput.input.SampleLevel(uv, 0).rgb;
    gComputeInput.result[dispatch] = float4(color, 1);//float4(1.f - exp(-color * gComputeInput.exposureValue), 1.);
}