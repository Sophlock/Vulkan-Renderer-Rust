module basicMaterials;

import Core.material;
import Core.geometry;

import BRDF.basicBRDFS;

import Core.largeBlock;

// Implementations for basic materials

// Simple, unlit sky sphere material based on a cubemap and intensity
struct SkySphereMaterial : IMaterial
{
    typedef UnlitBRDF BRDF;

    SamplerCube cubemap;
    float emissiveIntensity;

    LargeBlock _;

    MaterialResult<UnlitBRDF> evaluate(SurfaceGeometry geometry)
    {
        UnlitBRDF brdf = {};
        float3 environmentColor = cubemap.Sample(-geometry.worldNormal).rgb;
        brdf.emissive = environmentColor.rgb * emissiveIntensity;
        return {brdf, geometry};
    }
}

struct SingleColorUnlitMaterial : IMaterial
{
    typedef UnlitBRDF BRDF;

    LargeBlock _;

    MaterialResult<UnlitBRDF> evaluate(SurfaceGeometry geometry)
    {
        UnlitBRDF brdf = {};
        brdf.emissive = geometry.worldNormal;

        for (int i = 0; i < 100000; ++i) {
            brdf.emissive = float3(brdf.emissive.y, brdf.emissive.z, brdf.emissive.x);
        }

        return {brdf, geometry};
    }
}

struct FallbackMaterial : IMaterial
{
    typedef UnlitBRDF BRDF;

    LargeBlock _;

    MaterialResult<UnlitBRDF> evaluate(SurfaceGeometry geometry)
    {
        UnlitBRDF brdf = {};
        brdf.emissive = float3(1, 0, 1);

        return { brdf, geometry };
    }
}
