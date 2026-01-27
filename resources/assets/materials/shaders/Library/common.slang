module common;

// Implementations for common functionality

// Computes the half vector between two vectors
public float3 halfVector(float3 v1, float3 v2)
{
    return normalize(v1 + v2);
}

// Performs cubic interpolation between 0 and 1
// This can be used to perform cubic interpolation between arbitrary values
// To do this, use lerp(x, y, cubicInterpolationFactor(alpha))
// Result will be clamped between 0 and 1.
public float cubicInterpolationFactor(float inAlpha)
{
    float x = saturate(inAlpha);
    float xsq = x * x;
    float xcb = xsq * x;
    return -2 * xcb + 3 * xsq;
}

// Convert a hue value (between 0 and 1) into the (fully saturated, brightness=1) RGB color
public float3 hueToRgb(float hue)
{
    float r = min(hue, 1.f - hue);
    float div = .333333333f;
    float g = abs(hue - div);
    float b = abs(hue - (div + div));
    float3 rgb = saturate(div - float3(r, g, b));
    return rgb / (rgb.r + rgb.g + rgb.b);
}

// Convert an HSV color to an RGB color. Supports HDR
public float3 hsvToRgb(float3 hsv)
{
    float3 hue = hueToRgb(hsv.x);
    return lerp(1.f, hue, hsv.y) * hsv.z;
}

// Derive the F0 with respect to vacuum from the Index of Refraction
// This can be used for arbitrary combinations of IORs by passing in the ratio between the IORs
// Note that f0FromIOR(ior) == f0FromIOR(1 / ior), so order does not matter for different multiple IORs
public float f0FromIOR(float ior)
{
    float f0sqrt = (ior - 1.f) / (ior + 1.f);
    return f0sqrt * f0sqrt;
}

// Linearly remaps inValue from the interval (inValueMin, inValueMax) to the interval (outValueMin, outValueMax)
// Optionally clamp the result (default)
public float remapValueRange(float inValue, float outValueMin, float outValueMax, float inValueMin, float inValueMax, bool clampResult = true)
{
    float alpha = (inValue - inValueMin) / (inValueMax - inValueMin);
    float result = lerp(outValueMin, outValueMax, alpha);
    if (clampResult)
    {
        return clamp(result, outValueMin, outValueMax);
    }
    return result;
}

// Remaps inValue from the interval (inValueMin, inValueMax) to the interval (outValueMin, outValueMax) with control over a center and a threshold
// An inValue below inValueThreshold will be mapped below outValueCenter and inValue above inValueThreshold will be mapped above outValueCenter
// Optionally clamp the result (default)
public float remapDistribution(float inValue, float outValueCenter, float inValueThreshold = .5, bool clampResult = true, float outValueMin = 0., float outValueMax = 1., float inValueMin = 0., float inValueMax = 1.)
{
    float lowRange = remapValueRange(inValue, outValueMin, outValueCenter, inValueMin, inValueThreshold, clampResult);
    float highRange = remapValueRange(inValue, outValueCenter, outValueMax, inValueThreshold, inValueMax, clampResult);
    return inValue < inValueThreshold ? lowRange : highRange;
}

// Computes a bump offset (aka Parallax Occlusion Mapping with one iteration)
public float2 bumpOffset(float height, float3 viewDirectionTS)
{
    return height * float2(-viewDirectionTS.x, viewDirectionTS.y) / viewDirectionTS.z;
}
