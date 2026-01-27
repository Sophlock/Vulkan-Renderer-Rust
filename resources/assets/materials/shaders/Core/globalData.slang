module globalData;

// Data describing the camera
public struct ViewData
{
    public float4x4 viewProjection;
    public float3 viewPosition;
    public float exposureValue;
}

// Data describing the model
public struct ModelData
{
    public float4x4 modelTransform;
    public float4x4 inverseTransposeModelTransform;
}
