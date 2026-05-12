module basicMaterials;

import Core.material;
import Core.geometry;

import BRDF.basicBRDFs;

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

// The test material that is used as the desired material for all objects
// This will be created many times. All of them use the same code but this is invisible to the GPU and driver
struct SingleColorUnlitMaterial : IMaterial
{
    typedef UnlitBRDF BRDF;

    LargeBlock _;

    MaterialResult<UnlitBRDF> evaluate(SurfaceGeometry geometry)
    {
        // The actual shading is very simple: We just output the world normal, to show that we can accurately reconstruct data
        UnlitBRDF brdf = {};
        brdf.emissive = geometry.worldNormal;

        // Then we waste a bit of time to have the material be more represenatative of real world applications
        for (int i = 0; i < 100000; ++i) {
            brdf.emissive = float3(brdf.emissive.y, brdf.emissive.z, brdf.emissive.x);
        }

        return {brdf, geometry};
    }
}

// Global fallback material. This is just pink, but in reality, this would be a Uber shader
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
