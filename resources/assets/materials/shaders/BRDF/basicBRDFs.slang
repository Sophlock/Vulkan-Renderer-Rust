module basicBRDFs;

import Core.brdf;
import Core.material;
import Core.geometry;
import Core.indirectLighting;

import Core.largeBlock;

// Collection of simple BRDFs

// BRDF that only has an emissive component (i.e., that radiates light uniformly in all directions while perfectly absorbing all incoming light)
public struct UnlitBRDF : IBRDF
{
    public float3 emissive;

    public float3 evaluate(float3 viewDirection, float3 lightDirection, float3 lightColor)
    {
        return float3(0.f);
    }

    public float3 evaluateIndirect<Environment : IIndirectLightEnvironment>(float3 viewDirection, Environment environment)
    {
        return float3(0.f);
    }

    public float3 evaluateEmissive(float3 viewDirection)
    {
        return emissive;
    }
}

// Simple wrapper material for UnlitBRDF
struct ConstantUnlitMaterial : IMaterial
{
    typedef UnlitBRDF BRDF;

    float3 emissive;

    LargeBlock _;

    MaterialResult<UnlitBRDF> evaluate(SurfaceGeometry geometry)
    {
        UnlitBRDF brdf = {};
        brdf.emissive = emissive;
        return {brdf, geometry};
    }
}
