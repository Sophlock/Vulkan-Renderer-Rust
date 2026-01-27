module largeBlock;

// A struct to fill up types so that they can't be optimized into the space of an interface
// This is currently required because of a bug in the slang compiler and missing support in my Shader Cursors
public struct LargeBlock
{
    // TODO: This should be replaced with actual fixes
    // Trash to handle a shader cursor and slang compiler bug
    float3 albedo;
    float3 f0;
    float3 f90;
    float3 emissiveColor;
    float roughness;
}