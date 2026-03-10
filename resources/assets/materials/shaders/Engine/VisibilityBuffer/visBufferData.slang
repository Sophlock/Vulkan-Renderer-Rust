module visBufferData;

public struct InstanceData {
    public uint meshIndex;
    public uint materialInstanceIndex;
    public float4x4 modelTransform;
    public float4x4 inverseTransposeModelTransform;
}

public struct MaterialData {
    public uint64_t pipelineAddress;
}

public struct MaterialInstanceData {
    public uint materialIndex;
}

public struct MeshData {
    public uint firstPrimitive;
    public uint primitiveCount;
    public uint firstVertex;
    public uint vertexCount;
}

public struct Triangle {
    public uint3 triangleIndices;
}

public struct Vertex {
    public float3 worldPosition;
    public float3 worldNormal;
    public float3 worldTangent;
    public float2 textureCoordinate;
}

public struct MutatingData {
    public uint2 screenSize;
    public float4x4 viewMatrix;
}

public struct GlobalData {
    public StructuredBuffer<InstanceData>* instances;
    public StructuredBuffer<MaterialData>* materials;
    public StructuredBuffer<MaterialInstanceData>* materialInstances;
    public StructuredBuffer<MeshData>* meshes;
    public StructuredBuffer<Triangle>* indexBuffer;
    public StructuredBuffer<Vertex>* vertexBuffer;
    public StructuredBuffer<MutatingData>* mutData;
}

public Vertex operator +(Vertex v1, Vertex v2) {
    Vertex result = {};
    result.worldPosition = v1.worldPosition + v2.worldPosition;
    result.worldNormal = v1.worldNormal + v2.worldNormal;
    result.worldTangent = v1.worldTangent + v2.worldTangent;
    result.textureCoordinate = v1.textureCoordinate + v2.textureCoordinate;
    return result;
}

public Vertex operator *(Vertex v, float s) {
    Vertex result = {};
    result.worldPosition = v.worldPosition * s;
    result.worldNormal = v.worldNormal * s;
    result.worldTangent = v.worldTangent * s;
    result.textureCoordinate = v.textureCoordinate * s;
    return result;
}

public Vertex normalize(Vertex v) {
    Vertex result = v;
    result.worldNormal = normalize(v.worldNormal);
    result.worldTangent = normalize(v.worldTangent);
    return result;
}

public struct VisBufferShadeInput {
    public GlobalData* globalData;
    public uint64_t thisPipelineAddress;
}

