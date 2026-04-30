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
        brdf.emissive = geometry.worldNormal; //float3(0, .8, 1);

        //for (int i = 0; i < 1000000; ++i) {
            brdf.emissive = float3(brdf.emissive.y, brdf.emissive.z, brdf.emissive.x);
        //}

        return {brdf, geometry};
    }
}
