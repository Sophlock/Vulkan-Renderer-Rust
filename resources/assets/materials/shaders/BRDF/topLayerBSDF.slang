module topLayerBSDF;

import Core.indirectLighting;

import Library.fresnel;
import Library.specular;

// Interface for layers that are rendered on top of other BRDFs
public interface ITopLayerBSDF
{
    // Computes the amount of light that goes from the surface of the top layer to the bottom layer
    // reflected should hold the specular contribution of the layer
    float3 transmitForward(float3 light, float3 view, float3 lightColor, out float3 reflected, out float3 transmittedDirection, out float3 surfacePenetratingLight);
    // Computes the percentage of light that goes from the bottom layer out at the surface of the top layer
    float3 transmitBackwards(float3 view, out float3 refractedView, out float3 internalReflectionColor, out float3 absorptionFactor);
    // Computes the percentage of light that reaches the bottom layer from the top, but parametrized at the bottom layer (used for indirect lighting)
    // This takes roughness into account
    float3 transmitInternal(float3 direction, float roughness, out float3 refractedDirection, out float combinedRoughness);
    // Immediate indirect contribution of the layer to indirect lighting
    float3 evaluateIndirect<Environment : IIndirectLightEnvironment>(float3 viewDirection, Environment environment);
    // Emissive contribution of the layer
    float3 evaluateEmissive(float3 view);

    // TODO: This may introduce dynamic dispatch if used improperly. It is however the only solution since slang does not support higher kinded types.
    IIndirectLightEnvironment modifyIndirectLightEnvironment<BaseEnvironment : IIndirectLightEnvironment>(BaseEnvironment baseEnvironment);
}

// General purpose environment map that takes a base environment map and views it through a given layer
public struct IndirectLightEnvironmentThroughLayer<BaseEnvironment : IIndirectLightEnvironment, Layer : ITopLayerBSDF> : IIndirectLightEnvironment
{
    BaseEnvironment baseEnvironment;
    Layer layer;

    // Refracts the direction (direction is inside the layer, we need to sample baseEnvironment with the corresponding direction on the outside) and computes absorption
    // In case refracted becomes 0, we have total internal reflection
    // In this case, we return 0
    public float3 sampleEnvironment(float3 direction, float roughness)
    {
        float3 refracted;
        float combinedRoughness;
        float3 transmittedScale = layer.transmitInternal(direction, roughness, refracted, combinedRoughness);
        if (all(refracted == float3(0.f)))
        {
            return 0.f;
        }
        return baseEnvironment.sampleEnvironment(refracted, combinedRoughness) * transmittedScale;
    }
}

// Top layer implementation based on the Beer-Lambert law
// Takes a specular term and fresnel term as type parameters
public struct BeerLambertTopLayerBSDF<Fresnel : IFresnel, SpecularTerm : ISpecularTerm> : ITopLayerBSDF
{
    public float3 normal;
    // Coverage is the probability for material to "be there"
    // It is not really physically based but gives artistic control
    public float coverage;
    public float thickness;
    public float roughness;
    public float3 absorption;
    public float ior;
    public Fresnel fresnel;
    public SpecularTerm specular;
    public float3 emissive;
    public float3 ambientOcclusion;

    // The three transmit functions all share the same concepts:
    // We take the direction of light/view inside the layer and correct the thickness according to this direction
    // We evaluate the fresnel term at the top layer's surface, compute the amount/percentage of absorbed light (using the Beer-Lambert law) and scale according to coverage
    // The details of the three functions differ slightly and are commented in the code below

    public float3 transmitForward(float3 light, float3 view, float3 lightColor, out float3 reflected, out float3 transmittedDirection, out float3 surfacePenetratingLight)
    {
        transmittedDirection = -refract(-light, normal, rcp(ior));
        float cosl = saturate(dot(normal, transmittedDirection));
        float effectiveThickness = thickness / cosl;
        float3 fl = fresnel.getFresnelTermDirectly(normal, light);
        surfacePenetratingLight = lightColor * (1.f - fl);
        float3 fullCoverageTransmittedLight = surfacePenetratingLight * exp(-effectiveThickness * absorption);
        float3 transmittedLight = lerp(surfacePenetratingLight, fullCoverageTransmittedLight, coverage);
        // For direct lighting, we also need the specular component of the top layer
        // Therefore, we also need another fresnel term using the half vector
        float3 f = fresnel.getFresnelTerm(view, light);
        reflected = specular.evaluateSpecularTerm(normal, view, light, f, roughness * roughness) * lightColor * coverage * fl;
        return (1.f - f) * transmittedLight * saturate(dot(light, normal));
    }

    public float3 transmitBackwards(float3 view, out float3 refractedView, out float3 internalReflectionColor, out float3 absorptionFactor)
    {
        refractedView = -refract(-view, normal, rcp(ior));
        float cosl = saturate(dot(normal, refractedView));
        float effectiveThickness = thickness / cosl;
        float3 fullCoverageTransmittedLight = exp(-effectiveThickness * absorption);
        float3 fv = fresnel.getFresnelTermDirectly(normal, view);
        absorptionFactor = lerp(1.f, fullCoverageTransmittedLight, coverage);
        internalReflectionColor = absorptionFactor * fv;
        return absorptionFactor - internalReflectionColor;
    }

    public float3 transmitInternal(float3 direction, float roughness, out float3 refractedDirection, out float combinedRoughness)
    {
        // Here, direction is inside the layer already
        float cosl = saturate(dot(normal, direction));
        float effectiveThickness = thickness / cosl;
        float3 fullCoverageTransmittedLight = exp(-effectiveThickness * absorption);
        float3 transmittedLight = lerp(1.f, fullCoverageTransmittedLight, coverage);
        // In principle, we would need a fresnel term here as well but I can't see a difference between with and without
        // float3 f = fresnel.getFresnelTermDirectly(-direction, normal);
        // TODO: Revisit if this fresnel term makes a difference after image based lighting is implemented

        // We refract the ray from inside the layer to the outside, therefore we use the inverse IOR
        refractedDirection = -refract(-direction, normal, ior);
        combinedRoughness = 1.f - ((1.f - this.roughness) * (1.f - roughness));
        return transmittedLight;// * (1.f - f);
    }

    // Immediate indirect contribution
    // Similar to PBR BRDF
    public float3 evaluateIndirect<Environment : IIndirectLightEnvironment>(float3 viewDirection, Environment environment)
    {
        float3 f = fresnel.getFresnelTerm(viewDirection, normal);
        float alpha = roughness * roughness;
        float3 r = -reflect(viewDirection, normal);
        // TODO: Should the occlusion really be here?
        float3 s = specular.evaluateOcclusion(normal, viewDirection, r, alpha) * environment.sampleEnvironment(r, roughness) * f;
        return s * ambientOcclusion * coverage;
    }

    // Emissive component
    public float3 evaluateEmissive(float3 view)
    {
        return emissive * coverage;
    }

    // The modified light environment just returns an IndirectLightEnvironmentThroughLayer with the base environment and this layer
    public IIndirectLightEnvironment modifyIndirectLightEnvironment<BaseEnvironment : IIndirectLightEnvironment>(BaseEnvironment baseEnvironment)
    {
        IndirectLightEnvironmentThroughLayer<BaseEnvironment, This> modified;
        modified.baseEnvironment = baseEnvironment;
        modified.layer = this;
        return modified;
    }
}

public typedef BeerLambertTopLayerBSDF<DefaultFresnel, DefaultSpecularTerm> DefaultTopLayerBSDF;
