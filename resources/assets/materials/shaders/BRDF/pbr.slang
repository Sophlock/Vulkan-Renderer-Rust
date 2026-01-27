module pbr;

import Core.brdf;
import Core.material;
import Core.geometry;
import Core.indirectLighting;

import Library.common;
import Library.fresnel;
import Library.diffuse;
import Library.specular;

// Simple PBR material based on the Microfacet Model
public struct PBRBRDF : IBRDF
{
    // Right now, the types for diffuse, fresnel and specular are hardcoded but they will be refactored into type parameters to make the BRDF more modular
    public DefaultDiffuseTerm diffuse;
    public F0F90Fresnel<SchlickFresnel> fresnel;
    public DefaultSpecularTerm specular;
    public float roughness;
    public float3 normal;
    public float3 emissive;
    public float ambientOcclusion;

    public float3 evaluate(float3 viewDirection, float3 lightDirection, float3 lightColor)
    {
        float3 f = fresnel.getFresnelTerm(viewDirection, lightDirection);
        float alpha = roughness * roughness;
        float3 d = diffuse.evaluateDiffuseTerm(normal, viewDirection, lightDirection, f, alpha);
        float3 s = specular.evaluateSpecularTerm(normal, viewDirection, lightDirection, f, alpha);
        return (s + d) * lightColor;
    }

    public float3 evaluateIndirect<Environment : IIndirectLightEnvironment>(float3 viewDirection, Environment environment)
    {
        float3 f = fresnel.getFresnelTerm(viewDirection, normal);
        float alpha = roughness * roughness;
        float3 diffuse = diffuse.evaluateUnattenuatedDiffuseTerm(normal, viewDirection, f, alpha) * environment.sampleEnvironment(normal, 1.f);
        float3 r = -reflect(viewDirection, normal);
        float3 s = specular.evaluateOcclusion(normal, viewDirection, r, alpha) * environment.sampleEnvironment(r, roughness) * f;
        return (s + diffuse) * ambientOcclusion;
    }

    public float3 evaluateEmissive(float3 viewDirection)
    {
        return max(emissive, 0.f);
    }
}

// Simple material that wraps the PBR BRDF for easy testing
struct ConstantPBRMaterial : IMaterial
{
    typedef PBRBRDF BRDF;

    float3 albedo;
    float3 f0;
    float3 f90;
    float3 emissiveColor;
    float roughness;

    MaterialResult<PBRBRDF> evaluate(SurfaceGeometry geometry)
    {
        PBRBRDF brdf = {};
        brdf.diffuse.albedo = albedo;
        brdf.normal = geometry.worldNormal;
        brdf.fresnel.f0 = f0;
        brdf.fresnel.f90 = f90;
        brdf.roughness = roughness;
        brdf.emissive = emissiveColor;
        brdf.ambientOcclusion = 1.f;
        return {brdf, geometry};
    }
}
