#pragma once

// Self-contained D3D12 compute implementation of a causal, coarse-to-fine
// luminance optical flow. The worker compiles the required entry points once
// at startup; no ReShade runtime or external shader file is required.
constexpr const char* kCausalFlowShader = R"HLSL(
cbuffer FlowParameters : register(b0)
{
    uint2 OutputSize;
    uint2 FullSize;
    uint ResetFlow;
    uint PyramidLevel;
    uint Dilation;
    uint Quality;
    float ConfidenceThreshold;
    float DepthStrength;
    float2 Padding0;
    float4 Padding1;
};

Texture2D<float4> Input0 : register(t0);
Texture2D<float4> Input1 : register(t1);
Texture2D<float4> Input2 : register(t2);
Texture2D<float4> Input3 : register(t3);
Texture2D<float4> Input4 : register(t4);
Texture2D<float4> Input5 : register(t5);
RWTexture2D<float> ScalarOutput : register(u0);
RWTexture2D<float2> VectorOutput : register(u1);
SamplerState LinearClamp : register(s0);

static const float kEpsilon = 1.0e-6;

float2 PixelUv(uint2 pixel, uint2 size)
{
    return (float2(pixel) + 0.5) / float2(size);
}

float Luma(float3 color)
{
    float value = dot(max(color, 0.0), float3(0.2126, 0.7152, 0.0722));
    return value / (1.0 + value);
}

float PatchCost(float2 currentUv, float2 previousUv, float2 texel)
{
    static const int2 offsets[9] = {
        int2( 0,  3), int2( 0,  1),
        int2(-3,  0), int2(-1,  0), int2(0, 0), int2(1, 0), int2(3, 0),
        int2( 0, -1), int2( 0, -3)
    };
    float currentSamples[9];
    float previousSamples[9];
    float currentMean = 0.0;
    float previousMean = 0.0;
    [unroll] for (int sampleIndex = 0; sampleIndex < 9; ++sampleIndex)
    {
        float2 delta = float2(offsets[sampleIndex]) * texel;
        currentSamples[sampleIndex] = Input0.SampleLevel(LinearClamp, currentUv + delta, 0).x;
        previousSamples[sampleIndex] = Input1.SampleLevel(LinearClamp, previousUv + delta, 0).x;
        currentMean += currentSamples[sampleIndex];
        previousMean += previousSamples[sampleIndex];
    }
    currentMean /= 9.0;
    previousMean /= 9.0;
    float error = 0.0;
    [unroll] for (int errorIndex = 0; errorIndex < 9; ++errorIndex)
    {
        error += abs((currentSamples[errorIndex] - currentMean) -
                     (previousSamples[errorIndex] - previousMean));
    }
    return error / 9.0 + kEpsilon;
}

[numthreads(8, 8, 1)]
void PackLuma(uint3 dispatchId : SV_DispatchThreadID)
{
    if (any(dispatchId.xy >= OutputSize)) return;
    ScalarOutput[dispatchId.xy] = Luma(Input0.Load(int3(dispatchId.xy, 0)).rgb);
}

[numthreads(8, 8, 1)]
void DownsampleLuma(uint3 dispatchId : SV_DispatchThreadID)
{
    if (any(dispatchId.xy >= OutputSize)) return;
    uint sourceWidth, sourceHeight;
    Input0.GetDimensions(sourceWidth, sourceHeight);
    int2 base = int2(dispatchId.xy) * 2;
    int2 maximum = int2(sourceWidth, sourceHeight) - 1;
    float sum = 0.0;
    sum += Input0.Load(int3(clamp(base + int2(0, 0), 0, maximum), 0)).x;
    sum += Input0.Load(int3(clamp(base + int2(1, 0), 0, maximum), 0)).x;
    sum += Input0.Load(int3(clamp(base + int2(0, 1), 0, maximum), 0)).x;
    sum += Input0.Load(int3(clamp(base + int2(1, 1), 0, maximum), 0)).x;
    ScalarOutput[dispatchId.xy] = sum * 0.25;
}

[numthreads(8, 8, 1)]
void CoarseFlow(uint3 dispatchId : SV_DispatchThreadID)
{
    if (any(dispatchId.xy >= OutputSize)) return;
    if (ResetFlow != 0)
    {
        VectorOutput[dispatchId.xy] = 0.0;
        return;
    }
    float2 uv = PixelUv(dispatchId.xy, OutputSize);
    uint lumaWidth, lumaHeight;
    Input0.GetDimensions(lumaWidth, lumaHeight);
    float2 lumaTexel = rcp(float2(lumaWidth, lumaHeight));
    float2 previousSeed = Input2.SampleLevel(LinearClamp, uv, 0).xy;
    float zeroCost = PatchCost(uv, uv, lumaTexel);
    float seedCost = PatchCost(uv, uv + previousSeed, lumaTexel);
    float2 best = zeroCost <= seedCost ? float2(0.0, 0.0) : previousSeed;
    float bestCost = min(zeroCost, seedCost);
    int radius = 2 + int(min(Quality, 2u));
    [loop] for (int y = -radius; y <= radius; ++y)
    {
        [loop] for (int x = -radius; x <= radius; ++x)
        {
            if (x == 0 && y == 0) continue;
            float2 candidate = best + float2(x, y) * lumaTexel;
            float cost = PatchCost(uv, uv + candidate, lumaTexel);
            if (cost < bestCost)
            {
                bestCost = cost;
                best = candidate;
            }
        }
    }
    VectorOutput[dispatchId.xy] = best;
}

[numthreads(8, 8, 1)]
void RefineFlow(uint3 dispatchId : SV_DispatchThreadID)
{
    if (any(dispatchId.xy >= OutputSize)) return;
    if (ResetFlow != 0)
    {
        VectorOutput[dispatchId.xy] = 0.0;
        return;
    }
    float2 uv = PixelUv(dispatchId.xy, OutputSize);
    uint coarseWidth, coarseHeight;
    Input2.GetDimensions(coarseWidth, coarseHeight);
    float2 coarseTexel = rcp(float2(coarseWidth, coarseHeight));
    uint lumaWidth, lumaHeight;
    Input0.GetDimensions(lumaWidth, lumaHeight);
    float2 lumaTexel = rcp(float2(lumaWidth, lumaHeight));

    float2 candidates[10];
    candidates[0] = Input2.SampleLevel(LinearClamp, uv, 0).xy;
    candidates[1] = Input2.SampleLevel(LinearClamp, uv + float2(-coarseTexel.x, 0), 0).xy;
    candidates[2] = Input2.SampleLevel(LinearClamp, uv + float2( coarseTexel.x, 0), 0).xy;
    candidates[3] = Input2.SampleLevel(LinearClamp, uv + float2(0, -coarseTexel.y), 0).xy;
    candidates[4] = Input2.SampleLevel(LinearClamp, uv + float2(0,  coarseTexel.y), 0).xy;
    candidates[5] = Input2.SampleLevel(LinearClamp, uv + float2(-coarseTexel.x, -coarseTexel.y), 0).xy;
    candidates[6] = Input2.SampleLevel(LinearClamp, uv + float2( coarseTexel.x, -coarseTexel.y), 0).xy;
    candidates[7] = Input2.SampleLevel(LinearClamp, uv + float2(-coarseTexel.x,  coarseTexel.y), 0).xy;
    candidates[8] = Input2.SampleLevel(LinearClamp, uv + float2( coarseTexel.x,  coarseTexel.y), 0).xy;
    candidates[9] = Input3.SampleLevel(LinearClamp, uv, 0).xy;

    float2 best = candidates[0];
    float bestCost = 1.0e20;
    [unroll] for (int index = 0; index < 10; ++index)
    {
        float cost = PatchCost(uv, uv + candidates[index], lumaTexel);
        if (cost < bestCost)
        {
            bestCost = cost;
            best = candidates[index];
        }
    }

    float left = PatchCost(uv, uv + best - float2(lumaTexel.x, 0), lumaTexel);
    float right = PatchCost(uv, uv + best + float2(lumaTexel.x, 0), lumaTexel);
    float up = PatchCost(uv, uv + best - float2(0, lumaTexel.y), lumaTexel);
    float down = PatchCost(uv, uv + best + float2(0, lumaTexel.y), lumaTexel);
    float2 subpixel;
    subpixel.x = (left - right) / (4.0 * (left + right - 2.0 * bestCost) + kEpsilon);
    subpixel.y = (up - down) / (4.0 * (up + down - 2.0 * bestCost) + kEpsilon);
    best += clamp(subpixel, -0.5, 0.5) * lumaTexel;
    VectorOutput[dispatchId.xy] = best;
}

[numthreads(8, 8, 1)]
void MedianFlow(uint3 dispatchId : SV_DispatchThreadID)
{
    if (any(dispatchId.xy >= OutputSize)) return;
    uint inputWidth, inputHeight;
    Input0.GetDimensions(inputWidth, inputHeight);
    int2 maximum = int2(inputWidth, inputHeight) - 1;
    float2 values[9];
    int cursor = 0;
    [unroll] for (int y = -1; y <= 1; ++y)
    {
        [unroll] for (int x = -1; x <= 1; ++x)
        {
            int2 coordinate = clamp(int2(dispatchId.xy) + int2(x, y), 0, maximum);
            values[cursor++] = Input0.Load(int3(coordinate, 0)).xy;
        }
    }
    [unroll] for (int sortPass = 0; sortPass < 5; ++sortPass)
    {
        [unroll] for (int index = 0; index < 8 - sortPass; ++index)
        {
            float2 a = values[index];
            float2 b = values[index + 1];
            values[index] = min(a, b);
            values[index + 1] = max(a, b);
        }
    }
    VectorOutput[dispatchId.xy] = values[4];
}

[numthreads(8, 8, 1)]
void FlowConfidence(uint3 dispatchId : SV_DispatchThreadID)
{
    if (any(dispatchId.xy >= OutputSize)) return;
    if (ResetFlow != 0)
    {
        ScalarOutput[dispatchId.xy] = 0.0;
        return;
    }
    float2 uv = PixelUv(dispatchId.xy, OutputSize);
    float2 flow = Input0.Load(int3(dispatchId.xy, 0)).xy;
    float2 previousUv = uv + flow;
    if (any(previousUv < 0.0) || any(previousUv > 1.0))
    {
        ScalarOutput[dispatchId.xy] = 0.0;
        return;
    }

    uint lumaWidth, lumaHeight;
    Input1.GetDimensions(lumaWidth, lumaHeight);
    float2 lumaTexel = rcp(float2(lumaWidth, lumaHeight));
    static const int2 offsets[5] = {
        int2(0, 1), int2(-1, 0), int2(0, 0), int2(1, 0), int2(0, -1)
    };
    float sumCurrent = 0.0, sumCurrent2 = 0.0;
    float sumPrevious = 0.0, sumPrevious2 = 0.0;
    [unroll] for (int index = 0; index < 5; ++index)
    {
        float2 delta = float2(offsets[index]) * lumaTexel;
        float currentValue = Input1.SampleLevel(LinearClamp, uv + delta, 0).x;
        float previousValue = Input2.SampleLevel(LinearClamp, previousUv + delta, 0).x;
        sumCurrent += currentValue;
        sumCurrent2 += currentValue * currentValue;
        sumPrevious += previousValue;
        sumPrevious2 += previousValue * previousValue;
    }
    float currentVariance = max(sumCurrent2 / 5.0 - pow(sumCurrent / 5.0, 2.0), 0.0);
    float previousVariance = max(sumPrevious2 / 5.0 - pow(sumPrevious / 5.0, 2.0), 0.0);
    float patternConfidence = 1.0 - saturate(
        abs(sqrt(currentVariance) - sqrt(previousVariance)) /
        (sqrt(currentVariance) + 0.01));

    uint flowWidth, flowHeight;
    Input0.GetDimensions(flowWidth, flowHeight);
    int2 maximum = int2(flowWidth, flowHeight) - 1;
    int2 center = int2(dispatchId.xy);
    float2 north = Input0.Load(int3(clamp(center + int2(0, -1), 0, maximum), 0)).xy;
    float2 south = Input0.Load(int3(clamp(center + int2(0,  1), 0, maximum), 0)).xy;
    float2 east = Input0.Load(int3(clamp(center + int2(1,  0), 0, maximum), 0)).xy;
    float2 west = Input0.Load(int3(clamp(center + int2(-1, 0), 0, maximum), 0)).xy;
    float2 neighborMean = (north + south + east + west) * 0.25;
    float magnitude = length(flow);
    float spatialThreshold = magnitude * 0.5 + rcp(float(FullSize.x));
    float spatialConfidence = saturate(1.0 - distance(flow, neighborMean) /
        (spatialThreshold + kEpsilon));
    float onePixel = length(rcp(float2(FullSize)));
    float lengthConfidence = magnitude <= onePixel ? 1.0 :
        rcp((magnitude / onePixel) * 0.05 + 1.0);
    float currentConfidence = patternConfidence * spatialConfidence * lengthConfidence;
    float historyConfidence = Input3.SampleLevel(LinearClamp, previousUv, 0).x;
    if (historyConfidence < 0.001)
    {
        ScalarOutput[dispatchId.xy] = saturate(currentConfidence);
        return;
    }
    float alpha = currentConfidence < historyConfidence - 0.05 ? 0.5 : 0.1;
    ScalarOutput[dispatchId.xy] = saturate(lerp(historyConfidence, currentConfidence, alpha));
}

[numthreads(8, 8, 1)]
void FilterFlow(uint3 dispatchId : SV_DispatchThreadID)
{
    if (any(dispatchId.xy >= OutputSize)) return;
    if (ResetFlow != 0)
    {
        VectorOutput[dispatchId.xy] = 0.0;
        return;
    }
    float2 uv = PixelUv(dispatchId.xy, OutputSize);
    uint flowWidth, flowHeight;
    Input0.GetDimensions(flowWidth, flowHeight);
    float2 flowTexel = rcp(float2(flowWidth, flowHeight));
    float2 centerFlow = Input0.Load(int3(dispatchId.xy, 0)).xy;
    float centerConfidence = max(Input1.Load(int3(dispatchId.xy, 0)).x, 0.01);
    float centerLuma = Input2.SampleLevel(LinearClamp, uv, 0).x;
    float centerDepth = Input3.SampleLevel(LinearClamp, uv, 0).x;
    float2 sum = centerFlow * centerConfidence;
    float total = centerConfidence;
    static const int2 offsets[8] = {
        int2(-1,-1), int2(0,-1), int2(1,-1), int2(-1,0),
        int2( 1, 0), int2(-1,1), int2(0, 1), int2( 1,1)
    };
    [unroll] for (int index = 0; index < 8; ++index)
    {
        float2 sampleUv = uv + float2(offsets[index]) * flowTexel * float(Dilation);
        float2 sampleFlow = Input0.SampleLevel(LinearClamp, sampleUv, 0).xy;
        float sampleConfidence = Input1.SampleLevel(LinearClamp, sampleUv, 0).x;
        float sampleLuma = Input2.SampleLevel(LinearClamp, sampleUv, 0).x;
        float sampleDepth = Input3.SampleLevel(LinearClamp, sampleUv, 0).x;
        float confidenceWeight = pow(saturate(sampleConfidence), 3.0);
        float lumaWeight = saturate(1.0 - abs(centerLuma - sampleLuma) * 10.0);
        float depthWeight = exp2(-abs(centerDepth - sampleDepth) * DepthStrength);
        float2 deltaPixels = (sampleFlow - centerFlow) * float2(FullSize);
        float motionWeight = exp2(-dot(deltaPixels, deltaPixels) / 64.0);
        motionWeight = lerp(1.0, motionWeight, saturate(centerConfidence));
        float weight = confidenceWeight * lumaWeight * depthWeight * motionWeight;
        sum += sampleFlow * weight;
        total += weight;
    }
    VectorOutput[dispatchId.xy] = sum / (total + kEpsilon);
}

[numthreads(8, 8, 1)]
void ExpandFlow(uint3 dispatchId : SV_DispatchThreadID)
{
    if (any(dispatchId.xy >= OutputSize)) return;
    if (ResetFlow != 0)
    {
        VectorOutput[dispatchId.xy] = 0.0;
        ScalarOutput[dispatchId.xy] = 0.0;
        return;
    }
    float2 uv = PixelUv(dispatchId.xy, OutputSize);
    float2 flowUv = Input0.SampleLevel(LinearClamp, uv, 0).xy;
    float confidence = saturate(Input1.SampleLevel(LinearClamp, uv, 0).x);
    float confidenceGate = ConfidenceThreshold > 0.0
        ? smoothstep(ConfidenceThreshold * 0.5, ConfidenceThreshold, confidence)
        : 1.0;
    float2 flowPixels = flowUv * float2(FullSize);
    float magnitude = length(flowPixels);
    float subpixelGate = saturate((magnitude - 0.15) / 0.35);
    VectorOutput[dispatchId.xy] = flowPixels * confidenceGate * subpixelGate;
    ScalarOutput[dispatchId.xy] = confidence;
}
)HLSL";

