module indirectLighting;

// Interface for environment maps used for indirect lighting
public interface IIndirectLightEnvironment
{
    // Sample the environment map in a direction with a given roughness
    float3 sampleEnvironment(float3 direction, float roughness);
}