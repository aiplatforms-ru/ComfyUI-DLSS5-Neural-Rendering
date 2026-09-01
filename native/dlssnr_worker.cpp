#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include <Windows.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nvsdk_ngx.h"
#include "causal_flow_hlsl.h"

using Microsoft::WRL::ComPtr;

namespace {

constexpr uint32_t kNvidiaVendorId = 0x10DE;
constexpr unsigned long long kSnippetApplicationId = 0x0876232Cull;
constexpr NVSDK_NGX_Version kSnippetAbi = static_cast<NVSDK_NGX_Version>(0x15);
constexpr NVSDK_NGX_Feature kNeuralRenderingFeature = static_cast<NVSDK_NGX_Feature>(18);

std::string Hex(uint32_t value) {
    std::ostringstream output;
    output << "0x" << std::hex << std::uppercase << std::setw(8) << std::setfill('0') << value;
    return output.str();
}

void CheckHr(HRESULT result, const char* operation) {
    if (FAILED(result)) {
        throw std::runtime_error(std::string(operation) + " failed (HRESULT " +
            Hex(static_cast<uint32_t>(result)) + ")");
    }
}

void CheckNgx(NVSDK_NGX_Result result, const char* operation) {
    if (NVSDK_NGX_FAILED(result)) {
        throw std::runtime_error(std::string(operation) + " failed (NGX " +
            Hex(static_cast<uint32_t>(result)) + ")");
    }
}

std::wstring Widen(const std::string& value) {
    if (value.empty()) return {};
    const int length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
        static_cast<int>(value.size()), nullptr, 0);
    if (!length) throw std::runtime_error("Invalid UTF-8 command-line value.");
    std::wstring result(static_cast<size_t>(length), L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
        static_cast<int>(value.size()), result.data(), length);
    return result;
}

std::string Narrow(const std::wstring& value) {
    if (value.empty()) return {};
    const int length = WideCharToMultiByte(CP_UTF8, 0, value.data(), static_cast<int>(value.size()),
        nullptr, 0, nullptr, nullptr);
    std::string result(static_cast<size_t>(length), '\0');
    WideCharToMultiByte(CP_UTF8, 0, value.data(), static_cast<int>(value.size()),
        result.data(), length, nullptr, nullptr);
    return result;
}

class Arguments {
public:
    Arguments(int argc, char** argv) {
        for (int index = 1; index < argc; ++index) {
            const std::string key = argv[index];
            if (!key.starts_with("--") || index + 1 >= argc) {
                throw std::runtime_error("Expected --name value command-line arguments.");
            }
            values_[key] = argv[++index];
        }
    }

    const std::string& Required(const std::string& key) const {
        const auto found = values_.find(key);
        if (found == values_.end()) throw std::runtime_error("Missing argument " + key + ".");
        return found->second;
    }

    int Int(const std::string& key) const { return std::stoi(Required(key)); }
    float Float(const std::string& key) const { return std::stof(Required(key)); }

private:
    std::map<std::string, std::string> values_;
};

struct Settings {
    uint32_t width = 0;
    uint32_t height = 0;
    int adapterIndex = -1;
    int preset = 0;
    int style = 0;
    float intensity = 1.0f;
    float tone = 1.0f;
    float structure = 1.0f;
    float skin = -1.0f;
    bool autoMask = false;
    bool uiCorrection = false;
    bool depthInverted = true;
    int motionQuality = 0;
    float motionConfidence = 0.15f;
    bool diagnostics = false;
    std::wstring runtimeDll;
    std::wstring inputMap;
    std::wstring outputMap;
};

Settings ParseSettings(const Arguments& arguments) {
    Settings value;
    value.width = static_cast<uint32_t>(arguments.Int("--width"));
    value.height = static_cast<uint32_t>(arguments.Int("--height"));
    value.adapterIndex = arguments.Int("--adapter");
    value.preset = arguments.Int("--preset");
    value.style = arguments.Int("--style");
    value.intensity = arguments.Float("--intensity");
    value.tone = arguments.Float("--tone");
    value.structure = arguments.Float("--structure");
    value.skin = arguments.Float("--skin");
    value.autoMask = arguments.Int("--auto-mask") != 0;
    value.uiCorrection = arguments.Int("--ui-correction") != 0;
    value.depthInverted = arguments.Int("--depth-inverted") != 0;
    value.motionQuality = arguments.Int("--motion-quality");
    value.motionConfidence = arguments.Float("--motion-confidence");
    value.diagnostics = arguments.Int("--diagnostics") != 0;
    value.runtimeDll = Widen(arguments.Required("--runtime-dll"));
    value.inputMap = Widen(arguments.Required("--input-map"));
    value.outputMap = Widen(arguments.Required("--output-map"));

    if (!value.width || !value.height || value.width > 65535 || value.height > 65535)
        throw std::runtime_error("Width and height must be in the range 1..65535.");
    if (value.adapterIndex < -1)
        throw std::runtime_error("Adapter index must be -1 or greater.");
    if (value.preset < 0 || value.preset > 3 || value.style < 0 || value.style > 2)
        throw std::runtime_error("Preset or style is outside its supported range.");
    if (value.motionQuality < 0 || value.motionQuality > 2 ||
        value.motionConfidence < 0.0f || value.motionConfidence > 1.0f)
        throw std::runtime_error("Motion quality or confidence is outside its supported range.");
    if (!std::filesystem::is_regular_file(value.runtimeDll))
        throw std::runtime_error("nvngx_dlssnr.dll does not exist at the configured path.");
    return value;
}

class SharedMapping {
public:
    SharedMapping(const std::wstring& name, size_t expectedSize, DWORD access)
        : size_(expectedSize) {
        mapping_.reset(OpenFileMappingW(access, FALSE, name.c_str()));
        if (!mapping_) {
            throw std::runtime_error("OpenFileMappingW failed for shared frame memory (Win32 " +
                std::to_string(GetLastError()) + ").");
        }
        data_ = static_cast<uint8_t*>(MapViewOfFile(mapping_.get(), access, 0, 0, expectedSize));
        if (!data_) {
            throw std::runtime_error("MapViewOfFile failed (Win32 " +
                std::to_string(GetLastError()) + ").");
        }
    }

    ~SharedMapping() {
        if (data_) UnmapViewOfFile(data_);
    }

    SharedMapping(const SharedMapping&) = delete;
    SharedMapping& operator=(const SharedMapping&) = delete;

    uint8_t* data() const { return data_; }
    size_t size() const { return size_; }

private:
    struct HandleCloser {
        void operator()(void* handle) const { if (handle) CloseHandle(handle); }
    };
    std::unique_ptr<void, HandleCloser> mapping_;
    uint8_t* data_ = nullptr;
    size_t size_ = 0;
};

struct AdapterSelection {
    ComPtr<IDXGIAdapter4> adapter;
    std::wstring description;
    int nvidiaIndex = -1;
};

AdapterSelection SelectAdapter(int requestedIndex) {
    ComPtr<IDXGIFactory6> factory;
    CheckHr(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)), "CreateDXGIFactory2");

    int nvidiaIndex = 0;
    for (UINT ordinal = 0;; ++ordinal) {
        ComPtr<IDXGIAdapter1> candidate;
        const HRESULT enumeration = factory->EnumAdapterByGpuPreference(
            ordinal, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&candidate));
        if (enumeration == DXGI_ERROR_NOT_FOUND) break;
        CheckHr(enumeration, "EnumAdapterByGpuPreference");

        DXGI_ADAPTER_DESC1 description{};
        CheckHr(candidate->GetDesc1(&description), "IDXGIAdapter1::GetDesc1");
        if ((description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) || description.VendorId != kNvidiaVendorId)
            continue;

        if (requestedIndex >= 0 && requestedIndex != nvidiaIndex) {
            ++nvidiaIndex;
            continue;
        }

        if (FAILED(D3D12CreateDevice(candidate.Get(), D3D_FEATURE_LEVEL_12_0,
                __uuidof(ID3D12Device), nullptr))) {
            ++nvidiaIndex;
            continue;
        }

        AdapterSelection result;
        CheckHr(candidate.As(&result.adapter), "Query IDXGIAdapter4");
        result.description = description.Description;
        result.nvidiaIndex = nvidiaIndex;
        return result;
    }

    if (requestedIndex >= 0)
        throw std::runtime_error("NVIDIA adapter index " + std::to_string(requestedIndex) +
            " was not found or does not support D3D12 feature level 12_0.");
    throw std::runtime_error("No suitable NVIDIA D3D12 adapter was found.");
}

class CommandContext {
public:
    explicit CommandContext(ID3D12Device* device) : device_(device) {
        D3D12_COMMAND_QUEUE_DESC queueDescription{};
        queueDescription.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        CheckHr(device_->CreateCommandQueue(&queueDescription, IID_PPV_ARGS(&queue_)),
            "CreateCommandQueue");
        CheckHr(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS(&allocator_)), "CreateCommandAllocator");
        CheckHr(device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator_.Get(),
            nullptr, IID_PPV_ARGS(&list_)), "CreateCommandList");
        CheckHr(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)),
            "CreateFence");
        event_.reset(CreateEventW(nullptr, FALSE, FALSE, nullptr));
        if (!event_) throw std::runtime_error("CreateEventW failed.");
    }

    ID3D12GraphicsCommandList* list() const { return list_.Get(); }

    void ExecuteAndWait() {
        CheckHr(list_->Close(), "Close command list");
        ID3D12CommandList* lists[] = {list_.Get()};
        queue_->ExecuteCommandLists(1, lists);
        const uint64_t value = ++fenceValue_;
        CheckHr(queue_->Signal(fence_.Get(), value), "Signal fence");
        if (fence_->GetCompletedValue() < value) {
            CheckHr(fence_->SetEventOnCompletion(value, event_.get()), "SetEventOnCompletion");
            WaitForSingleObject(event_.get(), INFINITE);
        }
    }

    void Reset() {
        CheckHr(allocator_->Reset(), "Reset command allocator");
        CheckHr(list_->Reset(allocator_.Get(), nullptr), "Reset command list");
    }

private:
    struct HandleCloser {
        void operator()(void* handle) const { if (handle) CloseHandle(handle); }
    };
    ID3D12Device* device_ = nullptr;
    ComPtr<ID3D12CommandQueue> queue_;
    ComPtr<ID3D12CommandAllocator> allocator_;
    ComPtr<ID3D12GraphicsCommandList> list_;
    ComPtr<ID3D12Fence> fence_;
    std::unique_ptr<void, HandleCloser> event_;
    uint64_t fenceValue_ = 0;
};

D3D12_HEAP_PROPERTIES HeapProperties(D3D12_HEAP_TYPE type) {
    D3D12_HEAP_PROPERTIES properties{};
    properties.Type = type;
    properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    properties.CreationNodeMask = 1;
    properties.VisibleNodeMask = 1;
    return properties;
}

ComPtr<ID3D12Resource> CreateBuffer(ID3D12Device* device, uint64_t size,
    D3D12_HEAP_TYPE heapType, D3D12_RESOURCE_STATES initialState) {
    D3D12_RESOURCE_DESC description{};
    description.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    description.Width = size;
    description.Height = 1;
    description.DepthOrArraySize = 1;
    description.MipLevels = 1;
    description.SampleDesc.Count = 1;
    description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    const D3D12_HEAP_PROPERTIES heap = HeapProperties(heapType);
    ComPtr<ID3D12Resource> resource;
    CheckHr(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &description,
        initialState, nullptr, IID_PPV_ARGS(&resource)), "Create buffer");
    return resource;
}

ComPtr<ID3D12Resource> CreateTexture(ID3D12Device* device, uint32_t width, uint32_t height,
    DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES initialState) {
    D3D12_RESOURCE_DESC description{};
    description.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    description.Width = width;
    description.Height = height;
    description.DepthOrArraySize = 1;
    description.MipLevels = 1;
    description.Format = format;
    description.SampleDesc.Count = 1;
    description.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    description.Flags = flags;

    const D3D12_HEAP_PROPERTIES heap = HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
    ComPtr<ID3D12Resource> resource;
    CheckHr(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &description,
        initialState, nullptr, IID_PPV_ARGS(&resource)), "Create texture");
    return resource;
}

D3D12_RESOURCE_BARRIER Transition(ID3D12Resource* resource,
    D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    return barrier;
}

D3D12_RESOURCE_BARRIER UavBarrier(ID3D12Resource* resource) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = resource;
    return barrier;
}

uint32_t DivideRoundUp(uint32_t value, uint32_t divisor) {
    return (value + divisor - 1) / divisor;
}

struct FlowTexture {
    ComPtr<ID3D12Resource> resource;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    uint32_t width = 0;
    uint32_t height = 0;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
};

struct TextureView {
    ID3D12Resource* resource = nullptr;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
};

struct alignas(16) FlowConstants {
    uint32_t outputWidth = 0;
    uint32_t outputHeight = 0;
    uint32_t fullWidth = 0;
    uint32_t fullHeight = 0;
    uint32_t reset = 0;
    uint32_t pyramidLevel = 0;
    uint32_t dilation = 1;
    uint32_t quality = 0;
    float confidenceThreshold = 0.15f;
    float depthStrength = 64.0f;
    float padding0[2]{};
    float padding1[4]{};
};
static_assert(sizeof(FlowConstants) == 16 * sizeof(uint32_t));

class CausalOpticalFlow {
public:
    CausalOpticalFlow(ID3D12Device* device, uint32_t width, uint32_t height,
        ID3D12Resource* color, ID3D12Resource* depth, ID3D12Resource* motion,
        int quality, float confidenceThreshold)
        : device_(device), width_(width), height_(height), color_(color), depth_(depth),
          motion_(motion), quality_(quality), confidenceThreshold_(confidenceThreshold) {
        CreateRootSignature();
        CreatePipelines();
        CreateResources();
        CreateDescriptorTables();
    }

    ID3D12Resource* confidenceTexture() const { return fullConfidence_.resource.Get(); }

    void Run(ID3D12GraphicsCommandList* commandList, bool reset) {
        ID3D12DescriptorHeap* heaps[] = {descriptorHeap_.Get()};
        commandList->SetDescriptorHeaps(1, heaps);
        commandList->SetComputeRootSignature(rootSignature_.Get());

        Dispatch(commandList, packLuma_.Get(), packTable_, width_, height_, reset, 0, 1,
            {}, {&currentLuma_[0]});
        for (size_t level = 1; level < currentLuma_.size(); ++level) {
            Dispatch(commandList, downsampleLuma_.Get(), downsampleTables_[level - 1],
                currentLuma_[level].width, currentLuma_[level].height, reset,
                static_cast<uint32_t>(level), 1,
                {&currentLuma_[level - 1]}, {&currentLuma_[level]});
        }

        Dispatch(commandList, coarseFlow_.Get(), coarseTable_, flow128_.width, flow128_.height,
            reset, 5, 1, {&currentLuma_[5], &previousLuma_[5], &previousFlow_},
            {&flow128_});
        Dispatch(commandList, refineFlow_.Get(), refineTables_[0], flow64A_.width, flow64A_.height,
            reset, 4, 1, {&currentLuma_[4], &previousLuma_[4], &flow128_, &previousFlow_},
            {&flow64A_});
        Dispatch(commandList, medianFlow_.Get(), medianTables_[0], flow64B_.width, flow64B_.height,
            reset, 4, 1, {&flow64A_}, {&flow64B_});
        Dispatch(commandList, refineFlow_.Get(), refineTables_[1], flow32A_.width, flow32A_.height,
            reset, 3, 1, {&currentLuma_[3], &previousLuma_[3], &flow64B_, &previousFlow_},
            {&flow32A_});
        Dispatch(commandList, medianFlow_.Get(), medianTables_[1], flow32B_.width, flow32B_.height,
            reset, 3, 1, {&flow32A_}, {&flow32B_});
        Dispatch(commandList, refineFlow_.Get(), refineTables_[2], flow16A_.width, flow16A_.height,
            reset, 2, 1, {&currentLuma_[2], &previousLuma_[2], &flow32B_, &previousFlow_},
            {&flow16A_});
        Dispatch(commandList, medianFlow_.Get(), medianTables_[2], flow16B_.width, flow16B_.height,
            reset, 2, 1, {&flow16A_}, {&flow16B_});
        Dispatch(commandList, refineFlow_.Get(), refineTables_[3], flow8A_.width, flow8A_.height,
            reset, 1, 1, {&currentLuma_[1], &previousLuma_[1], &flow16B_, &previousFlow_},
            {&flow8A_});
        Dispatch(commandList, confidenceFlow_.Get(), confidenceTable_, confidence_.width,
            confidence_.height, reset, 2, 1,
            {&flow8A_, &currentLuma_[2], &previousLuma_[2], &previousConfidence_},
            {&confidence_});
        Dispatch(commandList, filterFlow_.Get(), filterTables_[0], flow8B_.width, flow8B_.height,
            reset, 3, 1, {&flow8A_, &confidence_, &currentLuma_[3], nullptr}, {&flow8B_});
        Dispatch(commandList, filterFlow_.Get(), filterTables_[1], flow8A_.width, flow8A_.height,
            reset, 3, 2, {&flow8B_, &confidence_, &currentLuma_[3], nullptr}, {&flow8A_});
        Dispatch(commandList, expandFlow_.Get(), expandTable_, width_, height_, reset, 0, 1,
            {&flow8A_, &confidence_}, {&fullConfidence_});

        StoreHistory(commandList);
    }

private:
    static TextureView View(const FlowTexture& texture) {
        return {texture.resource.Get(), texture.format};
    }

    FlowTexture MakeTexture(uint32_t width, uint32_t height, DXGI_FORMAT format,
        D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
        FlowTexture result;
        result.resource = CreateTexture(device_, width, height, format,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, state);
        result.format = format;
        result.width = width;
        result.height = height;
        result.state = state;
        return result;
    }

    void TransitionTexture(ID3D12GraphicsCommandList* commandList, FlowTexture& texture,
        D3D12_RESOURCE_STATES state) {
        if (texture.state == state) return;
        const D3D12_RESOURCE_BARRIER barrier = Transition(texture.resource.Get(), texture.state, state);
        commandList->ResourceBarrier(1, &barrier);
        texture.state = state;
    }

    ComPtr<ID3DBlob> Compile(const char* entryPoint) {
        ComPtr<ID3DBlob> bytecode;
        ComPtr<ID3DBlob> errors;
        const UINT flags = D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3;
        const HRESULT result = D3DCompile(kCausalFlowShader, std::strlen(kCausalFlowShader),
            "causal_flow.hlsl", nullptr, nullptr, entryPoint, "cs_5_1", flags, 0,
            &bytecode, &errors);
        if (FAILED(result)) {
            std::string message = "Compile optical-flow shader ";
            message += entryPoint;
            if (errors && errors->GetBufferPointer()) {
                message += ": ";
                message.append(static_cast<const char*>(errors->GetBufferPointer()),
                    errors->GetBufferSize());
            }
            throw std::runtime_error(message);
        }
        return bytecode;
    }

    ComPtr<ID3D12PipelineState> MakePipeline(const char* entryPoint) {
        const ComPtr<ID3DBlob> bytecode = Compile(entryPoint);
        D3D12_COMPUTE_PIPELINE_STATE_DESC description{};
        description.pRootSignature = rootSignature_.Get();
        description.CS.pShaderBytecode = bytecode->GetBufferPointer();
        description.CS.BytecodeLength = bytecode->GetBufferSize();
        ComPtr<ID3D12PipelineState> pipeline;
        CheckHr(device_->CreateComputePipelineState(&description, IID_PPV_ARGS(&pipeline)),
            "Create optical-flow pipeline");
        return pipeline;
    }

    void CreateRootSignature() {
        D3D12_DESCRIPTOR_RANGE ranges[2]{};
        ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        ranges[0].NumDescriptors = 6;
        ranges[0].BaseShaderRegister = 0;
        ranges[0].RegisterSpace = 0;
        ranges[0].OffsetInDescriptorsFromTableStart = 0;
        ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        ranges[1].NumDescriptors = 2;
        ranges[1].BaseShaderRegister = 0;
        ranges[1].RegisterSpace = 0;
        ranges[1].OffsetInDescriptorsFromTableStart = 6;

        D3D12_ROOT_PARAMETER parameters[2]{};
        parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        parameters[0].Constants.ShaderRegister = 0;
        parameters[0].Constants.Num32BitValues = sizeof(FlowConstants) / sizeof(uint32_t);
        parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        parameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        parameters[1].DescriptorTable.NumDescriptorRanges = 2;
        parameters[1].DescriptorTable.pDescriptorRanges = ranges;

        D3D12_STATIC_SAMPLER_DESC sampler{};
        sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler.MipLODBias = 0.0f;
        sampler.MaxAnisotropy = 1;
        sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
        sampler.BorderColor = D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK;
        sampler.MinLOD = 0.0f;
        sampler.MaxLOD = D3D12_FLOAT32_MAX;
        sampler.ShaderRegister = 0;
        sampler.RegisterSpace = 0;
        sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

        D3D12_ROOT_SIGNATURE_DESC description{};
        description.NumParameters = 2;
        description.pParameters = parameters;
        description.NumStaticSamplers = 1;
        description.pStaticSamplers = &sampler;
        ComPtr<ID3DBlob> serialized;
        ComPtr<ID3DBlob> errors;
        CheckHr(D3D12SerializeRootSignature(&description, D3D_ROOT_SIGNATURE_VERSION_1,
            &serialized, &errors), "Serialize optical-flow root signature");
        CheckHr(device_->CreateRootSignature(0, serialized->GetBufferPointer(),
            serialized->GetBufferSize(), IID_PPV_ARGS(&rootSignature_)),
            "Create optical-flow root signature");
    }

    void CreatePipelines() {
        packLuma_ = MakePipeline("PackLuma");
        downsampleLuma_ = MakePipeline("DownsampleLuma");
        coarseFlow_ = MakePipeline("CoarseFlow");
        refineFlow_ = MakePipeline("RefineFlow");
        medianFlow_ = MakePipeline("MedianFlow");
        confidenceFlow_ = MakePipeline("FlowConfidence");
        filterFlow_ = MakePipeline("FilterFlow");
        expandFlow_ = MakePipeline("ExpandFlow");
    }

    void CreateResources() {
        uint32_t lumaWidth = width_;
        uint32_t lumaHeight = height_;
        for (size_t level = 0; level < currentLuma_.size(); ++level) {
            currentLuma_[level] = MakeTexture(lumaWidth, lumaHeight, DXGI_FORMAT_R16_FLOAT);
            previousLuma_[level] = MakeTexture(lumaWidth, lumaHeight, DXGI_FORMAT_R16_FLOAT,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            lumaWidth = (std::max)(1u, DivideRoundUp(lumaWidth, 2));
            lumaHeight = (std::max)(1u, DivideRoundUp(lumaHeight, 2));
        }
        const auto makeFlow = [this](uint32_t divisor) {
            return MakeTexture((std::max)(1u, DivideRoundUp(width_, divisor)),
                (std::max)(1u, DivideRoundUp(height_, divisor)), DXGI_FORMAT_R16G16_FLOAT);
        };
        flow128_ = makeFlow(128);
        flow64A_ = makeFlow(64);
        flow64B_ = makeFlow(64);
        flow32A_ = makeFlow(32);
        flow32B_ = makeFlow(32);
        flow16A_ = makeFlow(16);
        flow16B_ = makeFlow(16);
        flow8A_ = makeFlow(8);
        flow8B_ = makeFlow(8);
        previousFlow_ = MakeTexture(flow8A_.width, flow8A_.height, DXGI_FORMAT_R16G16_FLOAT,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        confidence_ = MakeTexture(flow8A_.width, flow8A_.height, DXGI_FORMAT_R16_FLOAT);
        previousConfidence_ = MakeTexture(flow8A_.width, flow8A_.height, DXGI_FORMAT_R16_FLOAT,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        fullConfidence_ = MakeTexture(width_, height_, DXGI_FORMAT_R16_FLOAT);

        D3D12_DESCRIPTOR_HEAP_DESC heapDescription{};
        heapDescription.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heapDescription.NumDescriptors = 256;
        heapDescription.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        CheckHr(device_->CreateDescriptorHeap(&heapDescription, IID_PPV_ARGS(&descriptorHeap_)),
            "Create optical-flow descriptor heap");
        descriptorSize_ = device_->GetDescriptorHandleIncrementSize(
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }

    D3D12_GPU_DESCRIPTOR_HANDLE MakeTable(const std::array<TextureView, 6>& inputs,
        TextureView scalarOutput, TextureView vectorOutput) {
        if (descriptorCursor_ + 8 > 256) throw std::runtime_error("Optical-flow descriptor heap overflow.");
        D3D12_CPU_DESCRIPTOR_HANDLE cpu = descriptorHeap_->GetCPUDescriptorHandleForHeapStart();
        cpu.ptr += static_cast<SIZE_T>(descriptorCursor_) * descriptorSize_;
        D3D12_GPU_DESCRIPTOR_HANDLE gpu = descriptorHeap_->GetGPUDescriptorHandleForHeapStart();
        gpu.ptr += static_cast<UINT64>(descriptorCursor_) * descriptorSize_;
        for (const TextureView& input : inputs) {
            D3D12_SHADER_RESOURCE_VIEW_DESC description{};
            description.Format = input.format;
            description.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            description.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            description.Texture2D.MipLevels = 1;
            device_->CreateShaderResourceView(input.resource, &description, cpu);
            cpu.ptr += descriptorSize_;
        }
        for (const TextureView& output : {scalarOutput, vectorOutput}) {
            D3D12_UNORDERED_ACCESS_VIEW_DESC description{};
            description.Format = output.format;
            description.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
            device_->CreateUnorderedAccessView(output.resource, nullptr, &description, cpu);
            cpu.ptr += descriptorSize_;
        }
        descriptorCursor_ += 8;
        return gpu;
    }

    std::array<TextureView, 6> Inputs(std::initializer_list<TextureView> values) const {
        const TextureView fallback = View(currentLuma_[0]);
        std::array<TextureView, 6> result{fallback, fallback, fallback, fallback, fallback, fallback};
        size_t index = 0;
        for (const TextureView& value : values) {
            if (index == result.size()) break;
            result[index++] = value;
        }
        return result;
    }

    void CreateDescriptorTables() {
        const TextureView scalarDummy = View(fullConfidence_);
        const TextureView vectorDummy = View(flow8B_);
        packTable_ = MakeTable(Inputs({{color_, DXGI_FORMAT_R16G16B16A16_FLOAT}}),
            View(currentLuma_[0]), vectorDummy);
        for (size_t level = 1; level < currentLuma_.size(); ++level) {
            downsampleTables_[level - 1] = MakeTable(Inputs({View(currentLuma_[level - 1])}),
                View(currentLuma_[level]), vectorDummy);
        }
        coarseTable_ = MakeTable(Inputs({View(currentLuma_[5]), View(previousLuma_[5]),
            View(previousFlow_)}), scalarDummy, View(flow128_));

        const FlowTexture* currentLevels[4] = {
            &currentLuma_[4], &currentLuma_[3], &currentLuma_[2], &currentLuma_[1]};
        const FlowTexture* previousLevels[4] = {
            &previousLuma_[4], &previousLuma_[3], &previousLuma_[2], &previousLuma_[1]};
        const FlowTexture* coarseLevels[4] = {&flow128_, &flow64B_, &flow32B_, &flow16B_};
        FlowTexture* refinedLevels[4] = {&flow64A_, &flow32A_, &flow16A_, &flow8A_};
        for (size_t index = 0; index < 4; ++index) {
            refineTables_[index] = MakeTable(Inputs({View(*currentLevels[index]),
                View(*previousLevels[index]), View(*coarseLevels[index]), View(previousFlow_)}),
                scalarDummy, View(*refinedLevels[index]));
        }
        medianTables_[0] = MakeTable(Inputs({View(flow64A_)}), scalarDummy, View(flow64B_));
        medianTables_[1] = MakeTable(Inputs({View(flow32A_)}), scalarDummy, View(flow32B_));
        medianTables_[2] = MakeTable(Inputs({View(flow16A_)}), scalarDummy, View(flow16B_));
        confidenceTable_ = MakeTable(Inputs({View(flow8A_), View(currentLuma_[2]),
            View(previousLuma_[2]), View(previousConfidence_)}), View(confidence_), vectorDummy);
        filterTables_[0] = MakeTable(Inputs({View(flow8A_), View(confidence_),
            View(currentLuma_[3]), {depth_, DXGI_FORMAT_R32_FLOAT}}), scalarDummy, View(flow8B_));
        filterTables_[1] = MakeTable(Inputs({View(flow8B_), View(confidence_),
            View(currentLuma_[3]), {depth_, DXGI_FORMAT_R32_FLOAT}}), scalarDummy, View(flow8A_));
        expandTable_ = MakeTable(Inputs({View(flow8A_), View(confidence_)}),
            View(fullConfidence_), {motion_, DXGI_FORMAT_R16G16_FLOAT});
    }

    void Dispatch(ID3D12GraphicsCommandList* commandList, ID3D12PipelineState* pipeline,
        D3D12_GPU_DESCRIPTOR_HANDLE table, uint32_t outputWidth, uint32_t outputHeight,
        bool reset, uint32_t pyramidLevel, uint32_t dilation,
        std::initializer_list<FlowTexture*> reads,
        std::initializer_list<FlowTexture*> writes) {
        for (FlowTexture* texture : reads) {
            if (texture) TransitionTexture(commandList, *texture,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        for (FlowTexture* texture : writes) {
            if (texture) TransitionTexture(commandList, *texture,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        FlowConstants constants;
        constants.outputWidth = outputWidth;
        constants.outputHeight = outputHeight;
        constants.fullWidth = width_;
        constants.fullHeight = height_;
        constants.reset = reset ? 1u : 0u;
        constants.pyramidLevel = pyramidLevel;
        constants.dilation = dilation;
        constants.quality = static_cast<uint32_t>(quality_);
        constants.confidenceThreshold = confidenceThreshold_;
        constants.depthStrength = 48.0f + static_cast<float>(quality_) * 24.0f;
        commandList->SetPipelineState(pipeline);
        commandList->SetComputeRoot32BitConstants(0,
            sizeof(FlowConstants) / sizeof(uint32_t), &constants, 0);
        commandList->SetComputeRootDescriptorTable(1, table);
        commandList->Dispatch(DivideRoundUp(outputWidth, 8), DivideRoundUp(outputHeight, 8), 1);
        for (FlowTexture* texture : writes) {
            if (!texture) continue;
            const D3D12_RESOURCE_BARRIER barrier = UavBarrier(texture->resource.Get());
            commandList->ResourceBarrier(1, &barrier);
        }
        if (pipeline == expandFlow_.Get()) {
            const D3D12_RESOURCE_BARRIER barriers[2] = {
                UavBarrier(motion_), UavBarrier(fullConfidence_.resource.Get())};
            commandList->ResourceBarrier(2, barriers);
        }
    }

    void StoreHistory(ID3D12GraphicsCommandList* commandList) {
        for (size_t level = 0; level < currentLuma_.size(); ++level) {
            TransitionTexture(commandList, currentLuma_[level], D3D12_RESOURCE_STATE_COPY_SOURCE);
            TransitionTexture(commandList, previousLuma_[level], D3D12_RESOURCE_STATE_COPY_DEST);
            commandList->CopyResource(previousLuma_[level].resource.Get(),
                currentLuma_[level].resource.Get());
            TransitionTexture(commandList, currentLuma_[level],
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            TransitionTexture(commandList, previousLuma_[level],
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        TransitionTexture(commandList, flow8A_, D3D12_RESOURCE_STATE_COPY_SOURCE);
        TransitionTexture(commandList, previousFlow_, D3D12_RESOURCE_STATE_COPY_DEST);
        commandList->CopyResource(previousFlow_.resource.Get(), flow8A_.resource.Get());
        TransitionTexture(commandList, flow8A_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        TransitionTexture(commandList, previousFlow_,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        TransitionTexture(commandList, confidence_, D3D12_RESOURCE_STATE_COPY_SOURCE);
        TransitionTexture(commandList, previousConfidence_, D3D12_RESOURCE_STATE_COPY_DEST);
        commandList->CopyResource(previousConfidence_.resource.Get(), confidence_.resource.Get());
        TransitionTexture(commandList, confidence_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        TransitionTexture(commandList, previousConfidence_,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }

    ID3D12Device* device_ = nullptr;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    ID3D12Resource* color_ = nullptr;
    ID3D12Resource* depth_ = nullptr;
    ID3D12Resource* motion_ = nullptr;
    int quality_ = 0;
    float confidenceThreshold_ = 0.15f;
    ComPtr<ID3D12RootSignature> rootSignature_;
    ComPtr<ID3D12PipelineState> packLuma_;
    ComPtr<ID3D12PipelineState> downsampleLuma_;
    ComPtr<ID3D12PipelineState> coarseFlow_;
    ComPtr<ID3D12PipelineState> refineFlow_;
    ComPtr<ID3D12PipelineState> medianFlow_;
    ComPtr<ID3D12PipelineState> confidenceFlow_;
    ComPtr<ID3D12PipelineState> filterFlow_;
    ComPtr<ID3D12PipelineState> expandFlow_;
    ComPtr<ID3D12DescriptorHeap> descriptorHeap_;
    UINT descriptorSize_ = 0;
    UINT descriptorCursor_ = 0;
    std::array<FlowTexture, 6> currentLuma_;
    std::array<FlowTexture, 6> previousLuma_;
    FlowTexture flow128_;
    FlowTexture flow64A_, flow64B_;
    FlowTexture flow32A_, flow32B_;
    FlowTexture flow16A_, flow16B_;
    FlowTexture flow8A_, flow8B_;
    FlowTexture previousFlow_;
    FlowTexture confidence_;
    FlowTexture previousConfidence_;
    FlowTexture fullConfidence_;
    D3D12_GPU_DESCRIPTOR_HANDLE packTable_{};
    std::array<D3D12_GPU_DESCRIPTOR_HANDLE, 5> downsampleTables_{};
    D3D12_GPU_DESCRIPTOR_HANDLE coarseTable_{};
    std::array<D3D12_GPU_DESCRIPTOR_HANDLE, 4> refineTables_{};
    std::array<D3D12_GPU_DESCRIPTOR_HANDLE, 3> medianTables_{};
    D3D12_GPU_DESCRIPTOR_HANDLE confidenceTable_{};
    std::array<D3D12_GPU_DESCRIPTOR_HANDLE, 2> filterTables_{};
    D3D12_GPU_DESCRIPTOR_HANDLE expandTable_{};
};

int RunFlowSelfTest() {
    const AdapterSelection selection = SelectAdapter(-1);
    ComPtr<ID3D12Device> device;
    CheckHr(D3D12CreateDevice(selection.adapter.Get(), D3D_FEATURE_LEVEL_12_0,
        IID_PPV_ARGS(&device)), "D3D12CreateDevice(flow self-test)");
    constexpr uint32_t width = 320;
    constexpr uint32_t height = 192;
    constexpr D3D12_RESOURCE_STATES shaderRead = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    const ComPtr<ID3D12Resource> color = CreateTexture(device.Get(), width, height,
        DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_NONE, shaderRead);
    const ComPtr<ID3D12Resource> depth = CreateTexture(device.Get(), width, height,
        DXGI_FORMAT_R32_FLOAT, D3D12_RESOURCE_FLAG_NONE, shaderRead);
    const ComPtr<ID3D12Resource> motion = CreateTexture(device.Get(), width, height,
        DXGI_FORMAT_R16G16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CausalOpticalFlow flow(device.Get(), width, height, color.Get(), depth.Get(), motion.Get(), 1, 0.15f);
    CommandContext context(device.Get());
    flow.Run(context.list(), true);
    context.ExecuteAndWait();
    context.Reset();
    flow.Run(context.list(), false);
    context.ExecuteAndWait();
    std::cout << "FLOW_SELF_TEST_OK " << Narrow(selection.description) << std::endl;
    return 0;
}

struct TextureTransfer {
    ComPtr<ID3D12Resource> texture;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    uint32_t compactRowBytes = 0;
    uint32_t rowCount = 0;
};

// The signed 0x15 feature uses a newer parameter-map ABI than the public SDK
// headers currently expose. These calls deliberately use the runtime vtable.
void** ParameterVtable(NVSDK_NGX_Parameter* parameters) {
    return *reinterpret_cast<void***>(parameters);
}

void ParameterUll(NVSDK_NGX_Parameter* parameters, const char* name, unsigned long long value) {
    using Function = void(NVSDK_CONV*)(NVSDK_NGX_Parameter*, const char*, unsigned long long);
    reinterpret_cast<Function>(ParameterVtable(parameters)[0])(parameters, name, value);
}

void ParameterResource(NVSDK_NGX_Parameter* parameters, const char* name, ID3D12Resource* value) {
    using Function = void(NVSDK_CONV*)(NVSDK_NGX_Parameter*, const char*, ID3D12Resource*);
    reinterpret_cast<Function>(ParameterVtable(parameters)[1])(parameters, name, value);
}

void ParameterUi(NVSDK_NGX_Parameter* parameters, const char* name, unsigned int value) {
    using Function = void(NVSDK_CONV*)(NVSDK_NGX_Parameter*, const char*, unsigned int);
    reinterpret_cast<Function>(ParameterVtable(parameters)[3])(parameters, name, value);
}

void ParameterInt(NVSDK_NGX_Parameter* parameters, const char* name, int value) {
    using Function = void(NVSDK_CONV*)(NVSDK_NGX_Parameter*, const char*, int);
    reinterpret_cast<Function>(ParameterVtable(parameters)[4])(parameters, name, value);
}

void ParameterFloat(NVSDK_NGX_Parameter* parameters, const char* name, float value) {
    using Function = void(NVSDK_CONV*)(NVSDK_NGX_Parameter*, const char*, float);
    reinterpret_cast<Function>(ParameterVtable(parameters)[6])(parameters, name, value);
}

NVSDK_NGX_Result ParameterGetUi(NVSDK_NGX_Parameter* parameters, const char* name,
    unsigned int* value) {
    using Function = NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Parameter*, const char*, unsigned int*);
    return reinterpret_cast<Function>(ParameterVtable(parameters)[11])(parameters, name, value);
}

NVSDK_NGX_Result NVSDK_CONV ScalingRatioCallback(NVSDK_NGX_Parameter* parameters) {
    if (!parameters) return NVSDK_NGX_Result_FAIL_InvalidParameter;
    unsigned int upscaling = 0;
    ParameterGetUi(parameters, "DLSSNR.Upscaling", &upscaling);
    ParameterFloat(parameters, "DLSSNR.ScalingRatio", upscaling ? 0.5f : 1.0f);
    return NVSDK_NGX_Result_Success;
}

using GetModuleFileNameFunction = DWORD(WINAPI*)(HMODULE, LPWSTR, DWORD);
GetModuleFileNameFunction gOriginalGetModuleFileName = nullptr;
HMODULE gSnippetModule = nullptr;
void** gPatchedImportSlot = nullptr;

bool ReplacePointer(void** slot, void* replacement) {
    DWORD oldProtection = 0;
    if (!VirtualProtect(slot, sizeof(void*), PAGE_READWRITE, &oldProtection)) return false;
    InterlockedExchangePointer(slot, replacement);
    DWORD ignored = 0;
    VirtualProtect(slot, sizeof(void*), oldProtection, &ignored);
    FlushInstructionCache(GetCurrentProcess(), slot, sizeof(void*));
    return true;
}

DWORD WINAPI RoutedGetModuleFileName(HMODULE module, LPWSTR filename, DWORD capacity) {
    if (!gOriginalGetModuleFileName) return 0;
    if (module == gSnippetModule) return gOriginalGetModuleFileName(module, filename, capacity);

    HMODULE core = GetModuleHandleW(L"_nvngx.dll");
    if (!core) core = GetModuleHandleW(L"nvngx.dll");
    if (core) return gOriginalGetModuleFileName(core, filename, capacity);

    if (!filename || !capacity) return 0;
    const wchar_t fallback[] = L"nvngx.dll";
    constexpr DWORD fallbackLength = static_cast<DWORD>(std::size(fallback) - 1);
    if (capacity <= fallbackLength) {
        SetLastError(ERROR_INSUFFICIENT_BUFFER);
        return capacity;
    }
    std::memcpy(filename, fallback, sizeof(fallback));
    return fallbackLength;
}

void InstallSnippetImportRoute(HMODULE snippet) {
    auto* image = reinterpret_cast<uint8_t*>(snippet);
    const auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(image);
    if (dos->e_magic != IMAGE_DOS_SIGNATURE) throw std::runtime_error("Invalid DLSS-NR PE header.");
    const auto* nt = reinterpret_cast<IMAGE_NT_HEADERS*>(image + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE) throw std::runtime_error("Invalid DLSS-NR NT header.");

    const IMAGE_DATA_DIRECTORY& imports =
        nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (!imports.VirtualAddress) throw std::runtime_error("DLSS-NR runtime has no import table.");

    auto* descriptor = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(image + imports.VirtualAddress);
    for (; descriptor->Name; ++descriptor) {
        const char* library = reinterpret_cast<const char*>(image + descriptor->Name);
        if (_stricmp(library, "KERNEL32.dll") && _stricmp(library, "KERNELBASE.dll")) continue;
        auto* slots = reinterpret_cast<IMAGE_THUNK_DATA*>(image + descriptor->FirstThunk);
        auto* names = descriptor->OriginalFirstThunk
            ? reinterpret_cast<IMAGE_THUNK_DATA*>(image + descriptor->OriginalFirstThunk) : nullptr;
        for (size_t index = 0; slots[index].u1.Function; ++index) {
            if (!names || IMAGE_SNAP_BY_ORDINAL(names[index].u1.Ordinal)) continue;
            const auto* entry = reinterpret_cast<IMAGE_IMPORT_BY_NAME*>(
                image + names[index].u1.AddressOfData);
            if (std::strcmp(reinterpret_cast<const char*>(entry->Name), "GetModuleFileNameW")) continue;

            auto** slot = reinterpret_cast<void**>(&slots[index].u1.Function);
            gOriginalGetModuleFileName =
                reinterpret_cast<GetModuleFileNameFunction>(slots[index].u1.Function);
            gSnippetModule = snippet;
            if (!ReplacePointer(slot, reinterpret_cast<void*>(&RoutedGetModuleFileName))) {
                gOriginalGetModuleFileName = nullptr;
                gSnippetModule = nullptr;
                throw std::runtime_error("Could not patch the signed runtime caller-path import.");
            }
            gPatchedImportSlot = slot;
            return;
        }
    }
    throw std::runtime_error("GetModuleFileNameW import was not found in nvngx_dlssnr.dll.");
}

void RemoveSnippetImportRoute() {
    if (gPatchedImportSlot && gOriginalGetModuleFileName) {
        ReplacePointer(gPatchedImportSlot, reinterpret_cast<void*>(gOriginalGetModuleFileName));
    }
    gPatchedImportSlot = nullptr;
    gOriginalGetModuleFileName = nullptr;
    gSnippetModule = nullptr;
}

class NgxRuntime {
public:
    using Init = NVSDK_NGX_Result(NVSDK_CONV*)(unsigned long long, const wchar_t*, ID3D12Device*,
        NVSDK_NGX_Version, const NVSDK_NGX_Parameter*);
    using Shutdown = NVSDK_NGX_Result(NVSDK_CONV*)(ID3D12Device*);
    using Create = NVSDK_NGX_Result(NVSDK_CONV*)(ID3D12GraphicsCommandList*, NVSDK_NGX_Feature,
        const NVSDK_NGX_Parameter*, NVSDK_NGX_Handle**);
    using Evaluate = NVSDK_NGX_Result(NVSDK_CONV*)(ID3D12GraphicsCommandList*,
        const NVSDK_NGX_Handle*, const NVSDK_NGX_Parameter*, PFN_NVSDK_NGX_ProgressCallback);
    using Release = NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Handle*);

    NgxRuntime(ID3D12Device* device, const std::wstring& runtimeDll) : device_(device) {
        try {
            const std::filesystem::path runtimePath(runtimeDll);
            const std::wstring featureDirectory = runtimePath.parent_path().wstring();
            const wchar_t* featurePaths[] = {featureDirectory.c_str()};
            NVSDK_NGX_FeatureCommonInfo common{};
            common.PathListInfo.Path = featurePaths;
            common.PathListInfo.Length = 1;

            wchar_t temporary[MAX_PATH]{};
            const DWORD temporaryLength = GetTempPathW(MAX_PATH, temporary);
            std::filesystem::path appData = temporaryLength ? temporary : L".";
            appData /= L"ComfyUI-DLSSNR";
            std::filesystem::create_directories(appData);

            CheckNgx(NVSDK_NGX_D3D12_Init_with_ProjectID(
                "cde2cf68-2231-4f68-b173-7e65e7c7cddb", NVSDK_NGX_ENGINE_TYPE_CUSTOM,
                "ComfyUI-DLSSNR-0.1", appData.c_str(), device_, &common), "Initialize NGX core");
            coreInitialized_ = true;

            module_ = LoadLibraryExW(runtimeDll.c_str(), nullptr,
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
            if (!module_) {
                throw std::runtime_error("LoadLibraryExW(nvngx_dlssnr.dll) failed (Win32 " +
                    std::to_string(GetLastError()) + ").");
            }
            init_ = Resolve<Init>("NVSDK_NGX_D3D12_Init_Ext");
            shutdown_ = Resolve<Shutdown>("NVSDK_NGX_D3D12_Shutdown1");
            create_ = Resolve<Create>("NVSDK_NGX_D3D12_CreateFeature");
            evaluate_ = Resolve<Evaluate>("NVSDK_NGX_D3D12_EvaluateFeature");
            release_ = Resolve<Release>("NVSDK_NGX_D3D12_ReleaseFeature");
            InstallSnippetImportRoute(module_);
            CheckNgx(init_(kSnippetApplicationId, appData.c_str(), device_, kSnippetAbi, nullptr),
                "Initialize signed DLSS-NR runtime");
            snippetInitialized_ = true;
        } catch (...) {
            ShutdownAll();
            throw;
        }
    }

    ~NgxRuntime() { ShutdownAll(); }

    NVSDK_NGX_Handle* CreateFeature(ID3D12GraphicsCommandList* commandList,
        NVSDK_NGX_Parameter* parameters) {
        NVSDK_NGX_Handle* handle = nullptr;
        CheckNgx(create_(commandList, kNeuralRenderingFeature, parameters, &handle),
            "Create DLSS-NR feature 18");
        if (!handle) throw std::runtime_error("DLSS-NR returned a null feature handle.");
        return handle;
    }

    void EvaluateFeature(ID3D12GraphicsCommandList* commandList, const NVSDK_NGX_Handle* handle,
        NVSDK_NGX_Parameter* parameters) {
        CheckNgx(evaluate_(commandList, handle, parameters, nullptr), "Evaluate DLSS-NR feature 18");
    }

    void ReleaseFeature(NVSDK_NGX_Handle* handle) {
        if (handle && release_) CheckNgx(release_(handle), "Release DLSS-NR feature 18");
    }

private:
    template <typename Type>
    Type Resolve(const char* name) {
        const FARPROC address = GetProcAddress(module_, name);
        if (!address) throw std::runtime_error(std::string("Missing runtime export ") + name + ".");
        return reinterpret_cast<Type>(address);
    }

    void ShutdownAll() noexcept {
        if (snippetInitialized_ && shutdown_) shutdown_(device_);
        snippetInitialized_ = false;
        RemoveSnippetImportRoute();
        if (module_) FreeLibrary(module_);
        module_ = nullptr;
        if (coreInitialized_) NVSDK_NGX_D3D12_Shutdown1(device_);
        coreInitialized_ = false;
    }

    ID3D12Device* device_ = nullptr;
    HMODULE module_ = nullptr;
    Init init_ = nullptr;
    Shutdown shutdown_ = nullptr;
    Create create_ = nullptr;
    Evaluate evaluate_ = nullptr;
    Release release_ = nullptr;
    bool coreInitialized_ = false;
    bool snippetInitialized_ = false;
};

class NeuralRenderer {
public:
    explicit NeuralRenderer(const Settings& settings)
        : settings_(settings), selection_(SelectAdapter(settings.adapterIndex)) {
        CheckHr(D3D12CreateDevice(selection_.adapter.Get(), D3D_FEATURE_LEVEL_12_0,
            IID_PPV_ARGS(&device_)), "D3D12CreateDevice");
        context_ = std::make_unique<CommandContext>(device_.Get());
        runtime_ = std::make_unique<NgxRuntime>(device_.Get(), settings_.runtimeDll);
        CreateResources();
        opticalFlow_ = std::make_unique<CausalOpticalFlow>(device_.Get(), settings_.width,
            settings_.height, colorTransfer_.texture.Get(), depthTransfer_.texture.Get(),
            motionTexture_.Get(), settings_.motionQuality, settings_.motionConfidence);
        if (settings_.diagnostics) CreateDiagnosticReadbacks();
        CreateFeature();
    }

    ~NeuralRenderer() {
        try {
            if (feature_ && runtime_) runtime_->ReleaseFeature(feature_);
        } catch (const std::exception& error) {
            std::cerr << "DLSS-NR release warning: " << error.what() << std::endl;
        }
        feature_ = nullptr;
        if (parameters_) NVSDK_NGX_D3D12_DestroyParameters(parameters_);
        parameters_ = nullptr;
    }

    const AdapterSelection& selection() const { return selection_; }

    size_t InputBytes() const {
        return static_cast<size_t>(settings_.width) * settings_.height * (8 + 4);
    }

    size_t OutputBytes() const {
        const size_t bytesPerPixel = settings_.diagnostics ? (8 + 4 + 2) : 8;
        return static_cast<size_t>(settings_.width) * settings_.height * bytesPerPixel;
    }

    void Process(const uint8_t* input, uint8_t* output, bool resetHistory) {
        const size_t pixelCount = static_cast<size_t>(settings_.width) * settings_.height;
        const uint8_t* color = input;
        const uint8_t* depth = color + pixelCount * 8;
        CopyRowsToUpload(colorTransfer_, color);
        CopyRowsToUpload(depthTransfer_, depth);

        context_->Reset();
        ID3D12GraphicsCommandList* commandList = context_->list();
        CopyUploadToTexture(commandList, colorTransfer_);
        CopyUploadToTexture(commandList, depthTransfer_);

        constexpr D3D12_RESOURCE_STATES shaderRead =
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        std::array<D3D12_RESOURCE_BARRIER, 2> begin = {
            Transition(colorTransfer_.texture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, shaderRead),
            Transition(depthTransfer_.texture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, shaderRead),
        };
        commandList->ResourceBarrier(static_cast<UINT>(begin.size()), begin.data());

        opticalFlow_->Run(commandList, resetHistory);
        D3D12_RESOURCE_BARRIER motionForNgx = Transition(motionTexture_.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, shaderRead);
        commandList->ResourceBarrier(1, &motionForNgx);
        SetDispatchParameters(resetHistory);
        runtime_->EvaluateFeature(commandList, feature_, parameters_);
        D3D12_RESOURCE_BARRIER uav = UavBarrier(outputTexture_.Get());
        commandList->ResourceBarrier(1, &uav);
        D3D12_RESOURCE_BARRIER outputToCopy = Transition(outputTexture_.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        commandList->ResourceBarrier(1, &outputToCopy);

        CopyTextureToReadback(commandList, outputTexture_.Get(), outputReadback_);

        if (settings_.diagnostics) {
            const D3D12_RESOURCE_BARRIER diagnosticBegin[2] = {
                Transition(motionTexture_.Get(), shaderRead, D3D12_RESOURCE_STATE_COPY_SOURCE),
                Transition(opticalFlow_->confidenceTexture(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE),
            };
            commandList->ResourceBarrier(2, diagnosticBegin);
            CopyTextureToReadback(commandList, motionTexture_.Get(), motionReadback_);
            CopyTextureToReadback(commandList, opticalFlow_->confidenceTexture(), confidenceReadback_);
        }

        std::vector<D3D12_RESOURCE_BARRIER> finish;
        finish.push_back(Transition(colorTransfer_.texture.Get(), shaderRead,
            D3D12_RESOURCE_STATE_COPY_DEST));
        finish.push_back(Transition(depthTransfer_.texture.Get(), shaderRead,
            D3D12_RESOURCE_STATE_COPY_DEST));
        finish.push_back(Transition(outputTexture_.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        finish.push_back(Transition(motionTexture_.Get(), settings_.diagnostics
            ? D3D12_RESOURCE_STATE_COPY_SOURCE : shaderRead,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        if (settings_.diagnostics) {
            finish.push_back(Transition(opticalFlow_->confidenceTexture(),
                D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        }
        commandList->ResourceBarrier(static_cast<UINT>(finish.size()), finish.data());
        context_->ExecuteAndWait();

        CopyReadbackRows(outputReadback_, output);
        if (settings_.diagnostics) {
            uint8_t* motionOutput = output + pixelCount * 8;
            uint8_t* confidenceOutput = motionOutput + pixelCount * 4;
            CopyReadbackRows(motionReadback_, motionOutput);
            CopyReadbackRows(confidenceReadback_, confidenceOutput);
        }
    }

private:
    TextureTransfer MakeTransfer(DXGI_FORMAT format, uint32_t compactRowBytes, uint64_t baseOffset) {
        TextureTransfer transfer;
        transfer.texture = CreateTexture(device_.Get(), settings_.width, settings_.height, format,
            D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
        const D3D12_RESOURCE_DESC description = transfer.texture->GetDesc();
        uint64_t totalBytes = 0;
        device_->GetCopyableFootprints(&description, 0, 1, baseOffset, &transfer.footprint,
            &transfer.rowCount, nullptr, &totalBytes);
        transfer.compactRowBytes = compactRowBytes;
        uploadBytes_ = (std::max)(uploadBytes_, transfer.footprint.Offset +
            static_cast<uint64_t>(transfer.footprint.Footprint.RowPitch) * transfer.rowCount);
        return transfer;
    }

    void CreateResources() {
        colorTransfer_ = MakeTransfer(DXGI_FORMAT_R16G16B16A16_FLOAT, settings_.width * 8, 0);
        depthTransfer_ = MakeTransfer(DXGI_FORMAT_R32_FLOAT, settings_.width * 4,
            AlignUpload(uploadBytes_));
        upload_ = CreateBuffer(device_.Get(), uploadBytes_, D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ);
        D3D12_RANGE noCpuRead{0, 0};
        CheckHr(upload_->Map(0, &noCpuRead, reinterpret_cast<void**>(&uploadData_)), "Map upload buffer");

        motionTexture_ = CreateTexture(device_.Get(), settings_.width, settings_.height,
            DXGI_FORMAT_R16G16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        outputTexture_ = CreateTexture(device_.Get(), settings_.width, settings_.height,
            DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        outputReadback_ = MakeReadback(outputTexture_.Get(), settings_.width * 8);
    }

    struct ReadbackSurface {
        ComPtr<ID3D12Resource> buffer;
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        uint8_t* data = nullptr;
        uint32_t compactRowBytes = 0;
    };

    ReadbackSurface MakeReadback(ID3D12Resource* texture, uint32_t compactRowBytes) {
        ReadbackSurface result;
        const D3D12_RESOURCE_DESC description = texture->GetDesc();
        uint32_t rows = 0;
        uint64_t bytes = 0;
        device_->GetCopyableFootprints(&description, 0, 1, 0, &result.footprint, &rows,
            nullptr, &bytes);
        result.compactRowBytes = compactRowBytes;
        result.buffer = CreateBuffer(device_.Get(), bytes, D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_RANGE fullRead{0, static_cast<SIZE_T>(bytes)};
        CheckHr(result.buffer->Map(0, &fullRead, reinterpret_cast<void**>(&result.data)),
            "Map texture readback buffer");
        return result;
    }

    void CreateDiagnosticReadbacks() {
        motionReadback_ = MakeReadback(motionTexture_.Get(), settings_.width * 4);
        confidenceReadback_ = MakeReadback(opticalFlow_->confidenceTexture(), settings_.width * 2);
    }

    static void CopyTextureToReadback(ID3D12GraphicsCommandList* commandList,
        ID3D12Resource* texture, const ReadbackSurface& readback) {
        D3D12_TEXTURE_COPY_LOCATION source{};
        source.pResource = texture;
        source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        D3D12_TEXTURE_COPY_LOCATION destination{};
        destination.pResource = readback.buffer.Get();
        destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        destination.PlacedFootprint = readback.footprint;
        commandList->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
    }

    void CopyReadbackRows(const ReadbackSurface& readback, uint8_t* destination) const {
        for (uint32_t row = 0; row < settings_.height; ++row) {
            std::memcpy(destination + static_cast<size_t>(row) * readback.compactRowBytes,
                readback.data + readback.footprint.Offset + static_cast<size_t>(row) *
                    readback.footprint.Footprint.RowPitch,
                readback.compactRowBytes);
        }
    }

    static uint64_t AlignUpload(uint64_t value) {
        constexpr uint64_t alignment = D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT;
        return (value + alignment - 1) & ~(alignment - 1);
    }

    void CopyRowsToUpload(const TextureTransfer& transfer, const uint8_t* source) {
        for (uint32_t row = 0; row < transfer.rowCount; ++row) {
            std::memcpy(uploadData_ + transfer.footprint.Offset + static_cast<size_t>(row) *
                    transfer.footprint.Footprint.RowPitch,
                source + static_cast<size_t>(row) * transfer.compactRowBytes,
                transfer.compactRowBytes);
        }
    }

    void CopyUploadToTexture(ID3D12GraphicsCommandList* commandList,
        const TextureTransfer& transfer) {
        D3D12_TEXTURE_COPY_LOCATION source{};
        source.pResource = upload_.Get();
        source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        source.PlacedFootprint = transfer.footprint;
        D3D12_TEXTURE_COPY_LOCATION destination{};
        destination.pResource = transfer.texture.Get();
        destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        destination.SubresourceIndex = 0;
        commandList->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
    }

    void CreateFeature() {
        CheckNgx(NVSDK_NGX_D3D12_AllocateParameters(&parameters_), "Allocate NGX parameters");
        if (!parameters_) throw std::runtime_error("NGX returned a null parameter map.");

        const int width = static_cast<int>(settings_.width);
        const int height = static_cast<int>(settings_.height);
        ParameterInt(parameters_, "Width", width);
        ParameterInt(parameters_, "Height", height);
        ParameterInt(parameters_, "OutWidth", width);
        ParameterInt(parameters_, "OutHeight", height);
        ParameterInt(parameters_, "DLSSNR.Width", width);
        ParameterInt(parameters_, "DLSSNR.Height", height);
        ParameterInt(parameters_, "DLSSNR.InputWidth", width);
        ParameterInt(parameters_, "DLSSNR.InputHeight", height);
        ParameterInt(parameters_, "DLSSNR.OutputWidth", width);
        ParameterInt(parameters_, "DLSSNR.OutputHeight", height);
        ParameterInt(parameters_, "DLSSNR.Output.Width", width);
        ParameterInt(parameters_, "DLSSNR.Output.Height", height);
        ParameterInt(parameters_, "DLSSNR.Hint.Render.Preset", settings_.preset);
        ParameterInt(parameters_, "CreationNodeMask", 1);
        ParameterInt(parameters_, "VisibilityNodeMask", 1);
        ParameterUi(parameters_, "DLSS.Output.Subrect.Base.X", 0);
        ParameterUi(parameters_, "DLSS.Output.Subrect.Base.Y", 0);
        ParameterUi(parameters_, "DLSSNR.Upscaling", 0);
        ParameterFloat(parameters_, "DLSSNR.Scale", 1.0f);
        ParameterFloat(parameters_, "DLSSNR.ScalingRatio", 1.0f);
        ParameterUll(parameters_, "DLSSNRComputeScalingRatioCallback",
            reinterpret_cast<unsigned long long>(&ScalingRatioCallback));

        feature_ = runtime_->CreateFeature(context_->list(), parameters_);
        context_->ExecuteAndWait();
    }

    void SetDispatchParameters(bool resetHistory) {
        const int width = static_cast<int>(settings_.width);
        const int height = static_cast<int>(settings_.height);
        ParameterResource(parameters_, "DLSSNR.Color", colorTransfer_.texture.Get());
        ParameterResource(parameters_, "DLSSNR.Output", outputTexture_.Get());
        ParameterResource(parameters_, "DLSSNR.MVec", motionTexture_.Get());
        ParameterResource(parameters_, "DLSSNR.Depth", depthTransfer_.texture.Get());
        ParameterInt(parameters_, "Width", width);
        ParameterInt(parameters_, "Height", height);
        ParameterInt(parameters_, "OutWidth", width);
        ParameterInt(parameters_, "OutHeight", height);

        for (const char* input : {"Color", "Depth", "MVec"}) {
            const std::string prefix = std::string("DLSSNR.") + input + "Subrect";
            ParameterInt(parameters_, (prefix + "BaseX").c_str(), 0);
            ParameterInt(parameters_, (prefix + "BaseY").c_str(), 0);
            ParameterInt(parameters_, (prefix + "Width").c_str(), width);
            ParameterInt(parameters_, (prefix + "Height").c_str(), height);
        }
        ParameterInt(parameters_, "DLSSNR.OutputSubrectBaseX", 0);
        ParameterInt(parameters_, "DLSSNR.OutputSubrectBaseY", 0);
        ParameterInt(parameters_, "DLSSNR.OutputSubrectWidth", width);
        ParameterInt(parameters_, "DLSSNR.OutputSubrectHeight", height);
        ParameterFloat(parameters_, "DLSSNR.MVecScaleX", 1.0f);
        ParameterFloat(parameters_, "DLSSNR.MVecScaleY", 1.0f);
        ParameterUi(parameters_, "DLSSNR.DepthInverted", settings_.depthInverted ? 1u : 0u);
        ParameterUi(parameters_, "DLSSNR.Enabled", 1);
        ParameterUi(parameters_, "DLSSNR.Reset", resetHistory ? 1u : 0u);
        ParameterFloat(parameters_, "DLSSNR.Intensity", settings_.intensity);
        ParameterFloat(parameters_, "DLSSNR.LocalToneStrength", settings_.tone);
        ParameterFloat(parameters_, "DLSSNR.LocalStructureStrength", settings_.structure);
        ParameterFloat(parameters_, "DLSSNR.SkinStructureStrength", settings_.skin);
        ParameterUi(parameters_, "DLSSNR.UseAutoMask", settings_.autoMask ? 1u : 0u);
        ParameterInt(parameters_, "DLSSNR.Style", settings_.style);
        ParameterUi(parameters_, "DLSSNR.UICorrection", settings_.uiCorrection ? 1u : 0u);
    }

    Settings settings_;
    AdapterSelection selection_;
    ComPtr<ID3D12Device> device_;
    std::unique_ptr<CommandContext> context_;
    std::unique_ptr<NgxRuntime> runtime_;
    std::unique_ptr<CausalOpticalFlow> opticalFlow_;
    TextureTransfer colorTransfer_;
    TextureTransfer depthTransfer_;
    ComPtr<ID3D12Resource> motionTexture_;
    ComPtr<ID3D12Resource> outputTexture_;
    ComPtr<ID3D12Resource> upload_;
    ReadbackSurface outputReadback_;
    ReadbackSurface motionReadback_;
    ReadbackSurface confidenceReadback_;
    uint8_t* uploadData_ = nullptr;
    uint64_t uploadBytes_ = 0;
    NVSDK_NGX_Parameter* parameters_ = nullptr;
    NVSDK_NGX_Handle* feature_ = nullptr;
};

int Run(int argc, char** argv) {
    const Arguments arguments(argc, argv);
    const Settings settings = ParseSettings(arguments);
    NeuralRenderer renderer(settings);
    SharedMapping input(settings.inputMap, renderer.InputBytes(), FILE_MAP_READ);
    SharedMapping output(settings.outputMap, renderer.OutputBytes(), FILE_MAP_WRITE);

    std::cout << "READY " << renderer.selection().nvidiaIndex << " "
              << Narrow(renderer.selection().description) << std::endl;

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "QUIT") return 0;
        if (!line.starts_with("PROCESS ")) {
            std::cout << "ERROR Unknown worker command." << std::endl;
            continue;
        }
        try {
            const bool reset = std::stoi(line.substr(8)) != 0;
            renderer.Process(input.data(), output.data(), reset);
            std::cout << "OK" << std::endl;
        } catch (const std::exception& error) {
            std::cout << "ERROR " << error.what() << std::endl;
        }
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--flow-self-test") {
            return RunFlowSelfTest();
        }
        return Run(argc, argv);
    } catch (const std::exception& error) {
        std::cout << "ERROR " << error.what() << std::endl;
        return 2;
    }
}

