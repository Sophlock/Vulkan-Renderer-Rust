module lights;

import brdf;
import geometry;
import indirectLighting;

// Interface for the scene's lighting environment
public interface ILightEnvironment
{
    // Illuminate a given BRDF
    float3 illuminate<B:IBRDF>(SurfaceGeometry geometry, B brdf, float3 viewDirection);
}

// Array of lights of a single type
// Use this together with LightPair to construct the scene's light environment
struct LightArray<L : ILightEnvironment, let N : int> : ILightEnvironment
{
    int count;
    L lights[N];

    float3 illuminate<B:IBRDF>(SurfaceGeometry geometry, B brdf, float3 viewDirection)
    {
        float3 sum = 0.;
        for (int i = 0; i < count; ++i)
        {
            sum += lights[i].illuminate(geometry, brdf, viewDirection);
        }
        return sum;
    }
}

// Pair of lights with two different types
// Use this together with LightArray to construct the scene's light environment
struct LightPair<T : ILightEnvironment, U : ILightEnvironment> : ILightEnvironment
{
    T first;
    U second;

    float3 illuminate<B:IBRDF>(SurfaceGeometry geometry, B brdf, float3 viewDirection)
    {
        return first.illuminate(geometry, brdf, viewDirection) + second.illuminate(geometry, brdf, viewDirection);
    }
};

// Light environment corresponding to no lights
struct EmptyLight : ILightEnvironment
{
    float3 illuminate<B:IBRDF>(SurfaceGeometry geometry, B brdf, float3 viewDirection)
    {
        return 0.;
    }
}

// Simple directional light
struct DirectionalLight : ILightEnvironment
{
    float3 direction;
    float3 color;
    float intensity;

    float3 illuminate<B:IBRDF>(SurfaceGeometry geometry, B brdf, float3 viewDirection)
    {
        float3 realColor = intensity * color;
        return brdf.evaluate(viewDirection, direction, realColor);
    }
}

// Simple point light
struct PointLight : ILightEnvironment
{
    float3 position;
    float3 color;
    float intensity;

    float3 illuminate<B:IBRDF>(SurfaceGeometry geometry, B brdf, float3 viewDirection)
    {
        float3 realColor = intensity * color;
        float3 delta = geometry.worldPosition - position;
        float distance = length(delta);
        float3 direction = normalize(delta);
        float3 illumination = realColor / (distance * distance);
        return brdf.evaluate(viewDirection, direction, illumination);
    }
}

// Simple, single color ambient light
// Will use indirect lighting
struct AmbientLight : ILightEnvironment, IIndirectLightEnvironment
{
    float3 color;
    float intensity;

    float3 illuminate<B:IBRDF>(SurfaceGeometry geometry, B brdf, float3 viewDirection)
    {
        return brdf.evaluateIndirect(viewDirection, this);
    }

    // Just return the color of the light
    float3 sampleEnvironment(float3 direction, float roughness)
    {
        return color * intensity;
    }
}

// Ambient light based on a cubemap
// Will use indirect lighting
struct AmbientCubemapLight : ILightEnvironment, IIndirectLightEnvironment
{
    SamplerCube environmentMap;
    float intensity;

    float3 illuminate<B:IBRDF>(SurfaceGeometry geometry, B brdf, float3 viewDirection)
    {
        return brdf.evaluateIndirect(viewDirection, this);
    }

    // Will sample the environmentMap in the give direction
    // Will use higher mip levels to account for roughness
    float3 sampleEnvironment(float3 direction, float roughness)
    {
        float width, height, levels;
        environmentMap.GetDimensions(0, width, height, levels);
        float level = max(levels - 2, 0.f) * sqrt(roughness);
        float3 color = environmentMap.SampleLevel(direction, level).rgb;
        return color * intensity;
    }
}
