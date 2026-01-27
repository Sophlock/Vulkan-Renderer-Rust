module combinedBRDFs;

import Core.brdf;
import Core.indirectLighting;
import BRDF.topLayerBSDF;

// Implementation for combined BRDFs
// These BRDFs can be nested for arbitrary combinations of BRDFs

// Blends two BRDFs based on a blend factor
// Note that this will evaluate both BRDFs to prevent artifacts
public struct HorizontalBlendBRDF<BRDF1 : IBRDF, BRDF2 : IBRDF> : IBRDF
{
    public BRDF1 brdf1;
    public BRDF2 brdf2;
    public float lerpFactor;

    // The functions here are rather trivial
    // They just evaluate both BRDFs and lerp between them

    public float3 evaluate(float3 viewDirection, float3 lightDirection, float3 lightColor)
    {
        return lerp(brdf1.evaluate(viewDirection, lightDirection, lightColor), brdf2.evaluate(viewDirection, lightDirection, lightColor), saturate(lerpFactor));
    }

    public float3 evaluateIndirect<Environment : IIndirectLightEnvironment>(float3 viewDirection, Environment environment)
    {
        return lerp(brdf1.evaluateIndirect(viewDirection, environment), brdf2.evaluateIndirect(viewDirection, environment), saturate(lerpFactor));
    }

    public float3 evaluateEmissive(float3 viewDirection)
    {
        return lerp(brdf1.evaluateEmissive(viewDirection), brdf2.evaluateEmissive(viewDirection), saturate(lerpFactor));
    }
}

// Simulates a bottom BRDF coated with a top layer
// By making BottomBRDF a VerticalBlendBRDF, one can stack an arbitrary amount of layers
public struct VerticalBlendBRDF<BottomBRDF : IBRDF, TopBSDF : ITopLayerBSDF> : IBRDF
{
    public BottomBRDF bottomBrdf;
    public TopBSDF topBsdf;

    public float3 evaluate(float3 viewDirection, float3 lightDirection, float3 lightColor)
    {
        // The amount of light reflected at the top layer into the view direction
        // This will usually be the specular contribution of the top layer
        float3 reflected;
        // The direction of light inside the top layer
        float3 transmittedDirection;
        // The amount of light that entered the the top layer
        float3 surfacePenetratingLight;
        // The amount of light that reached the bottom layer
        float3 transmittedLight = topBsdf.transmitForward(lightDirection, viewDirection, lightColor, reflected, transmittedDirection, surfacePenetratingLight);

        // The direction of view inside the top layer (this is the relevant view vector for the bottom layer to take into account the refraction when light exists the top layer at the end)
        float3 refractedView;
        // The percentage of light that is reflected internally at the top layer (absorption is already applied to this)
        float3 internalReflectionFactor;
        // The percentage of light that was not absorbed by the top layer
        float3 absorptionFactor;
        // The percentage of light (that started at the bottom layer) that exits the top layer in the direction of view
        float3 backTransmittanceFactor = topBsdf.transmitBackwards(viewDirection, refractedView, internalReflectionFactor, absorptionFactor);
        // The relevant light right after interacting with the bottom layer
        float3 bottomLight = bottomBrdf.evaluate(refractedView, transmittedDirection, transmittedLight);
        // The light that leaves the top layer into the view direction after the first bounce (= the bounce at the bottom layer where the bottom BRDF was evaluated)
        float3 firstBounceExitLight = bottomLight * backTransmittanceFactor;
        // The amount of light that was internally reflected back towards the bottom layer at the surface of the top layer after the first bounce
        float3 firstInternalBounceLight = bottomLight * internalReflectionFactor;

        // For one round trip of light (top->bottom->top), the light going back to the bottom has lost this factor of energy
        float3 relativeInternalReflection = saturate(firstInternalBounceLight / surfacePenetratingLight);
        // We approximate the the total energy going into the top layer for and infinite number of bounces with a geometric series
        float3 internalReflectionGeometricSeries = 1.f / (1.f - relativeInternalReflection);

        // One bounce produced firstBounceExitLight
        // Therefore, infinite bounces produce firstBounceExitLight * internalReflectionGeometricSeries
        return reflected + firstBounceExitLight * internalReflectionGeometricSeries;
    }

    public float3 evaluateIndirect<Environment : IIndirectLightEnvironment>(float3 viewDirection, Environment environment)
    {
        // Immediate indirect contribution of the top layer
        float3 indirectTop = topBsdf.evaluateIndirect(viewDirection, environment);

        // These are the same as for evaluate() above
        float3 refractedView;
        float3 internalReflectionFactor;
        float3 absorptionFactor;
        float3 backTransmittanceFactor = topBsdf.transmitBackwards(viewDirection, refractedView, internalReflectionFactor, absorptionFactor);
        // We don't have the actual light values here (unlike with evaluate())
        // We approximate the geometric series by assuming that, on the way "down", the same percentage of light is absorbed as on the way "up"
        float3 internalReflectionGeometricSeries = 1.f / (1.f - internalReflectionFactor * absorptionFactor);

        // The light environment used for indirect lighting of the bottom layer
        // This represents environment viewed through the top layer
        let modifiedEnvironment = topBsdf.modifyIndirectLightEnvironment(environment);
        // bottomBrdf.evaluateIndirect() returns the amount of light right at the bottom layer, so we scale it by backTransmittanceFactor to account for absorption on the way back up
        // and by internalReflectionGeometricSeries to account for internal reflection
        float3 indirectBottom = bottomBrdf.evaluateIndirect(refractedView, modifiedEnvironment) * backTransmittanceFactor * internalReflectionGeometricSeries;

        return indirectTop + indirectBottom;
    }

    // Evaluates the emissive components of top and bottom layer and combines them taking into account the absorption of the top layer
    public float3 evaluateEmissive(float3 viewDirection)
    {
        float3 refractedView;
        float3 internalReflectionFactor;
        float3 absorptionFactor;
        float3 backTransmittanceFactor = topBsdf.transmitBackwards(viewDirection, refractedView, internalReflectionFactor, absorptionFactor);

        // Internal reflection is ignored here. It would require evaluating the bottom BRDF
        return topBsdf.evaluateEmissive(viewDirection) + bottomBrdf.evaluateEmissive(refractedView) * backTransmittanceFactor;
    }
}
