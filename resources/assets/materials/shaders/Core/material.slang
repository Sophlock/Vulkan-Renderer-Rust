module material;

import brdf;
import geometry;

// Result of a material evaluation
// geometry is not used anywhere right now and will likely be removed in the future
public struct MaterialResult<BRDF : IBRDF>
{
    public BRDF brdf;
    public SurfaceGeometry geometry;
}

// Interface for materials
public interface IMaterial
{
    // The BRDF type of this material
    associatedtype BRDF : IBRDF;

    // Compute a result (BRDF) for this material depending on the SurfaceGeometry
    MaterialResult<BRDF> evaluate(SurfaceGeometry geometry);
}