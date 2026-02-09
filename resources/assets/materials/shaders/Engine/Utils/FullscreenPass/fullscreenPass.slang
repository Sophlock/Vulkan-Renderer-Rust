module fullscreenPass;

import Core.largeBlock;

struct VertexInput {
    float3 position : POSITION;
    float2 textureCoordinate : UV;
}

struct VertexOutput {
    float4 position : SV_Position;
    float2 textureCoordinate : UV;
}

struct FragmentOutput {
    float4 finalColor : SV_Target;
}

struct FullscreenPassInput {
    Sampler2D colorInput;
    LargeBlock _filler;
}

uniform FullscreenPassInput gInput;

[shader("vertex")]
func vertexMain(input: VertexInput)->VertexOutput {
    VertexOutput output = {};
    output.position = float4(input.position, 0);
    output.textureCoordinate = input.textureCoordinate;
    return output;
}

[shader("fragment")]
func fragmentMain(uv: float2 : UV)->FragmentOutput
{
    FragmentOutput output = {};
    output.finalColor = gInput.colorInput.SampleLevel(uv, 0);
    return output;
}
