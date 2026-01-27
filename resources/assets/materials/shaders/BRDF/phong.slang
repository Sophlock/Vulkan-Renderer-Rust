module phong;

import Core.brdf;
import Core.material;
import Core.geometry;
import Core.indirectLighting;

// Simple Phong BRDF to test all the backends for the simplest possible case
struct PhongBRDF : IBRDF
{
    float3 kd;
    float3 ks;
    float specularity;
    float3 normal;
    float3 emissive;

    float3 evaluate(float3 viewDirection, float3 lightDirection, float3 lightColor)
    {
        float nDotL = saturate(dot(normal, lightDirection));
        float3 H = normalize(lightDirection + viewDirection);
        float nDotH = saturate(dot(normal, H));

        return lightColor * (kd * nDotL + ks * pow(nDotH, specularity));
    }

    float3 evaluateIndirect<Environment : IIndirectLightEnvironment>(float3 viewDirection, Environment environment)
    {
        return environment.sampleEnvironment(normal, 1.f) * kd;
    }

    float3 evaluateEmissive(float3 viewDirection)
    {
        return emissive;
    }
}

// Simple material that wraps the Phong BRDF for quick testing
struct ConstantPhongMaterial : IMaterial
{
    float3 diffuseColor;
    float3 specularColor;
    float specularity;
    float3 emissiveColor;

    typedef PhongBRDF BRDF;

    MaterialResult<PhongBRDF> evaluate(SurfaceGeometry geometry)
    {
        PhongBRDF brdf;
        brdf.kd = diffuseColor;
        brdf.ks = specularColor;
        brdf.specularity = specularity;
        brdf.normal = geometry.worldNormal;
        brdf.emissive = emissiveColor;
        return {brdf, geometry};
    }
}