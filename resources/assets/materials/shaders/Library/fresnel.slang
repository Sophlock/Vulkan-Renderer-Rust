module fresnel;

import Library.common;

// Implementations for fresnel term and fresnel approximations

// Interface for fresnel approximations
public interface IFresnelApproximation
{
    // Returns the (single-channel) fresnel factor
    static float approximateFresnel(float3 view, float3 half);
}

// Schlick-Fresnel approximation
public struct SchlickFresnel : IFresnelApproximation
{
    public static float approximateFresnel(float3 view, float3 half)
    {
        float base = 1.f - saturate(dot(view, half));
        float baseSq = base * base;
        return baseSq * baseSq * base;
    }
}

// Interface for (colored) fresnel terms
public interface IFresnel
{
    // Evaluates the fresnel term based on the half vector between view and light vector
    float3 getFresnelTerm(float3 view, float3 light);
    // Evaluates the fresnel term based on normal and direction without computing a half vector
    float3 getFresnelTermDirectly(float3 normal, float3 direction);
}

// Fresnel term that uses an approximation and interpolates between and F0 and F90 value
public struct F0F90Fresnel<FresnelApproximation : IFresnelApproximation> : IFresnel
{
    public float3 f0;
    public float3 f90;

    public float3 getFresnelTerm(float3 view, float3 light)
    {
        float3 half = halfVector(view, light);
        return getFresnelTermDirectly(half, light);
    }

    public float3 getFresnelTermDirectly(float3 normal, float3 direction)
    {
        float fresnel = FresnelApproximation.approximateFresnel(normal, direction);
        return lerp(f0, f90, fresnel);
    }
}

public typedef F0F90Fresnel<SchlickFresnel> DefaultFresnel;
