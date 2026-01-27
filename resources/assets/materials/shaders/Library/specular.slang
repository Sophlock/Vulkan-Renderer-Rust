module specular;

import Library.common;

// Implementations for the specular component

// Interface for a specular distribution term
public interface ISpecularDistributionTerm
{
    // Evaluates the distribution term
    float evaluateDistributionTerm(float3 normal, float3 view, float3 light, float alpha);
}

// GGX distribution term without anisotropy support
public struct GGXIsotropicDistributionTerm : ISpecularDistributionTerm
{
    public float evaluateDistributionTerm(float3 normal, float3 view, float3 light, float alpha)
    {
        float3 h = halfVector(view, light);
        float alphaSq = alpha * alpha;
        float dotNH = saturate(dot(normal, h));
        float denominator = dotNH * dotNH * (alphaSq - 1.f) + 1.f;
        return alphaSq / (denominator * denominator * float.getPi());
    }
}

public typedef GGXIsotropicDistributionTerm DefaultSpecularDistributionTerm;

// Interface for specular geometry terms
public interface ISpecularGeometryTerm
{
    // Evaluates the geometry term
    float evaluateGeometryTerm(float3 normal, float3 view, float3 light, float alpha);
}

// GGX geometry term
public struct GGXGeometryTerm : ISpecularGeometryTerm
{
    private float occlusion(float3 normal, float3 direction, float alpha)
    {
        float dotNV = saturate(dot(normal, direction));
        float alphaSq = alpha * alpha;
        float rootTerm = sqrt(alphaSq + (1.f - alphaSq) * dotNV * dotNV);
        return 2.f * dotNV / (dotNV + rootTerm);
    }

    public float evaluateGeometryTerm(float3 normal, float3 view, float3 light, float alpha)
    {
        return occlusion(normal, view, alpha) * occlusion(normal, light, alpha);
    }
}

public typedef GGXGeometryTerm DefaultSpecularGeometryTerm;

// Interface for specular terms
public interface ISpecularTerm
{
    // Evaluates the specular term. This should include angular and fresnel attenuation
    float3 evaluateSpecularTerm(float3 normal, float3 view, float3 light, float3 fresnel, float alpha);
    // Evaluates only the occlusion/geometry term. Used for indirect lighting
    float3 evaluateOcclusion(float3 normal, float3 view, float3 light, float alpha);
}

// Cook-Torrance specular term based on a distribution and geometry term
public struct CookTorranceSpecularTerm<Distribution : ISpecularDistributionTerm, Geometry : ISpecularGeometryTerm> : ISpecularTerm
{
    // These are parameters because some might hold data (like anisotropy)
    public Distribution distribution;
    public Geometry geometry;

    public float3 evaluateSpecularTerm(float3 normal, float3 view, float3 light, float3 fresnel, float alpha)
    {
        float g = saturate(geometry.evaluateGeometryTerm(normal, view, light, alpha));
        float d = distribution.evaluateDistributionTerm(normal, view, light, alpha);
        return max(fresnel * g * d / saturate(dot(normal, view)) * 4.f, 0); // / dot(normal, light) <- not needed because we multiply by it anyways later
    }

    public float3 evaluateOcclusion(float3 normal, float3 view, float3 light, float alpha)
    {
        return saturate(geometry.evaluateGeometryTerm(normal, view, light, alpha));
    }
}

public typedef CookTorranceSpecularTerm<DefaultSpecularDistributionTerm, DefaultSpecularGeometryTerm> DefaultSpecularTerm;
