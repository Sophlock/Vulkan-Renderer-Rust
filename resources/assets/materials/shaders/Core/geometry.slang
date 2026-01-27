module geometry;

import globalData;

// Data describing the surface at a single point
public struct SurfaceGeometry
{
    public float3 worldPosition;
    public float3 worldNormal;
    public float3 worldTangent;
    public float2 textureCoordinate;
    public ModelData modelData;
    public ViewData viewData;
    public float3x3 tangentToWorld;

    public float3 getLocalPosition()
    {
        return mul(transpose(modelData.inverseTransposeModelTransform), float4(worldPosition, 1.f)).xyz;
    }
}