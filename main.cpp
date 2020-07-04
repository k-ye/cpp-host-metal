// clang++ -std=c++11 main.cpp -framework Metal -framework CoreGraphics
// -framework Foundation

// Frameworks:
// * Metal for obvious reason
// * CoreGraphics:
// https://developer.apple.com/documentation/metal/1433401-mtlcreatesystemdefaultdevice?language=objc
// * Foundation: objc runtime,
// https://gist.github.com/TooTallNate/1073294/675f28984cd120cdec14b66b41d86cf01140b1e6
//
// References:
//
// * Halide
// * taichi
// * https://github.com/naleksiev/mtlpp
//
// All <Metal/*> headers are for Obj-C, we cannot import them.
#include <objc/message.h>
#include <objc/objc.h>
#include <objc/runtime.h>
#include <sys/mman.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

extern "C" {
void NSLog(id /* NSString * */ format, ...);
id MTLCreateSystemDefaultDevice();
}

namespace {

constexpr int kNSUTF8StringEncoding = 4;
//  Need to have \n at the end, otherwise the compiled library is garbage...
// const char *kernel_src =
//     "#include <metal_stdlib>\n"
//     "using namespace metal;\n"
//     "\n"
//     "kernel void add1(device int *data [[buffer(0)]],\n"
//     "                 device int *eptr_arr [[buffer(1)]],\n"
//     "                 const uint tid [[thread_position_in_grid]]) {\n"
//     "  int64_t data_addr = (int64_t)(data + tid);\n"
//     "  data[tid] = ((tid & 1) == 0) ? (data_addr) : (data_addr &
//     0xffffffff);\n"
//     // "  int64_t eptr_val = eptr_arr[1];\n"
//     // "  eptr_val = (eptr_val << 32);\n"
//     // "  eptr_val += eptr_arr[0];\n"
//     // "  device int *eptr = reinterpret_cast<device int *>(eptr_val);\n"
//     // "  eptr[tid] = data[tid];\n"
//     "}\n";

using NSString = objc_object;
struct MTLDevice;
struct MTLLibrary;
struct MTLComputePipelineState;
struct MTLCommandQueue;
struct MTLCommandBuffer;
struct MTLComputeCommandEncoder;
struct MTLFunction;
struct MTLComputePipelineState;
struct MTLBuffer;

template <typename R, typename O, typename... Args>
R cast_call(O *i, const char *select, Args... args) {
  using func = R (*)(id, SEL, Args...);
  return ((func)(objc_msgSend))(reinterpret_cast<id>(i), sel_getUid(select),
                                args...);
}

template <typename O, typename... Args>
id call(O *i, const char *select, Args... args) {
  return cast_call<id>(i, select, args...);
}

template <typename C = id, typename... Args>
C clscall(const char *class_name, const char *select, Args... args) {
  using func = C (*)(id, SEL, Args...);
  return ((func)(objc_msgSend))((id)objc_getClass(class_name),
                                sel_getUid(select), args...);
}

template <typename O> void release_ns_object(O *obj) {
  call(reinterpret_cast<id>(obj), "release");
}

template <typename O> class NsObjDeleter {
public:
  void operator()(O *o) { call(o, "release"); }
};

template <typename O>
using nsobj_unique_ptr = std::unique_ptr<O, NsObjDeleter<O>>;

template <typename O> nsobj_unique_ptr<O> wrap_as_nsobj_unique_ptr(O *nsobj) {
  return nsobj_unique_ptr<O>(nsobj);
}

size_t roundup_to_pagesize(size_t s) {
  const size_t pagesize = getpagesize();
  return ((s + pagesize - 1) / pagesize) * pagesize;
}

MTLDevice *mtl_create_system_default_device() {
  id dev = MTLCreateSystemDefaultDevice();
  return reinterpret_cast<MTLDevice *>(dev);
}

nsobj_unique_ptr<MTLCommandQueue> new_command_queue(MTLDevice *dev) {
  auto *queue = cast_call<MTLCommandQueue *>(dev, "newCommandQueue");
  return wrap_as_nsobj_unique_ptr(queue);
}

nsobj_unique_ptr<MTLCommandBuffer> new_command_buffer(MTLCommandQueue *queue) {
  auto *buffer = cast_call<MTLCommandBuffer *>(queue, "commandBuffer");
  return wrap_as_nsobj_unique_ptr(buffer);
}

nsobj_unique_ptr<MTLComputeCommandEncoder>
new_compute_command_encoder(MTLCommandBuffer *buffer) {
  auto *encoder =
      cast_call<MTLComputeCommandEncoder *>(buffer, "computeCommandEncoder");
  return wrap_as_nsobj_unique_ptr(encoder);
}

NSString *wrap_string_as_ns_string(const char *str, size_t len) {
  id ns_string = clscall("NSString", "alloc");
  return reinterpret_cast<NSString *>(
      call(ns_string, "initWithBytesNoCopy:length:encoding:freeWhenDone:", str,
           len, kNSUTF8StringEncoding, false));
}

nsobj_unique_ptr<MTLLibrary> new_library_with_source(MTLDevice *device,
                                                     const char *source,
                                                     size_t source_len) {
  NSString *source_str = wrap_string_as_ns_string(source, source_len);

  id options = clscall("MTLCompileOptions", "alloc");
  options = call(options, "init");
  call(options, "setFastMathEnabled:", false);

  auto *lib = cast_call<MTLLibrary *>(
      device, "newLibraryWithSource:options:error:", source_str, options,
      nullptr);

  release_ns_object(options);
  release_ns_object(source_str);

  return wrap_as_nsobj_unique_ptr(lib);
}

nsobj_unique_ptr<MTLFunction>
new_function_with_name(MTLLibrary *library, const char *name, size_t name_len) {
  NSString *name_str = wrap_string_as_ns_string(name, name_len);
  auto *func =
      cast_call<MTLFunction *>(library, "newFunctionWithName:", name_str);
  release_ns_object(name_str);
  return wrap_as_nsobj_unique_ptr(func);
}

MTLComputePipelineState *
new_compute_pipeline_state_with_function(MTLDevice *device,
                                         MTLFunction *function) {
  id pipeline_state = call(
      device, "newComputePipelineStateWithFunction:error:", function, nullptr);
  return reinterpret_cast<MTLComputePipelineState *>(pipeline_state);
}

void set_compute_pipeline_state(MTLComputeCommandEncoder *encoder,
                                MTLComputePipelineState *pipeline_state) {
  call(encoder, "setComputePipelineState:", pipeline_state);
}

size_t
get_max_total_threads_per_threadgroup(MTLComputePipelineState *pipeline_state) {
  // The value of the pointer returned by call is the actual result
  return (size_t)call(pipeline_state, "maxTotalThreadsPerThreadgroup");
}

void end_encoding(MTLComputeCommandEncoder *encoder) {
  call(encoder, "endEncoding");
}

MTLBuffer *new_mtl_buffer(MTLDevice *device, size_t length) {
  constexpr int kMtlBufferResourceOptions = 0;
  id buffer = call(device, "newBufferWithLength:options:", length,
                   kMtlBufferResourceOptions);
  return reinterpret_cast<MTLBuffer *>(buffer);
}

MTLBuffer *new_mtl_buffer_no_copy(MTLDevice *device, void *ptr, size_t length) {
  // MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared
  constexpr int kMtlBufferResourceOptions = 0;

  id buffer =
      call(device, "newBufferWithBytesNoCopy:length:options:deallocator:", ptr,
           length, kMtlBufferResourceOptions, nullptr);
  return reinterpret_cast<MTLBuffer *>(buffer);
}

void set_mtl_buffer(MTLComputeCommandEncoder *encoder, MTLBuffer *buffer,
                    size_t offset, size_t index) {
  call(encoder, "setBuffer:offset:atIndex:", buffer, offset, index);
}

void set_mtl_bytes(MTLComputeCommandEncoder *encoder, void *bytes,
                   size_t length, size_t index) {
  call(encoder, "setBytes:length:atIndex:", bytes, length, index);
}

void dispatch_threadgroups(MTLComputeCommandEncoder *encoder, int32_t blocks_x,
                           int32_t blocks_y, int32_t blocks_z,
                           int32_t threads_x, int32_t threads_y,
                           int32_t threads_z) {
  struct MTLSize {
    uint64_t width;
    uint64_t height;
    uint64_t depth;
  };

  MTLSize threadgroups_per_grid;
  threadgroups_per_grid.width = blocks_x;
  threadgroups_per_grid.height = blocks_y;
  threadgroups_per_grid.depth = blocks_z;

  MTLSize threads_per_threadgroup;
  threads_per_threadgroup.width = threads_x;
  threads_per_threadgroup.height = threads_y;
  threads_per_threadgroup.depth = threads_z;

  call(encoder,
       "dispatchThreadgroups:threadsPerThreadgroup:", threadgroups_per_grid,
       threads_per_threadgroup);
}

// 1D
void dispatch_threadgroups(MTLComputeCommandEncoder *encoder, int32_t blocks_x,
                           int32_t threads_x) {
  dispatch_threadgroups(encoder, blocks_x, 1, 1, threads_x, 1, 1);
}

void commit_command_buffer(MTLCommandBuffer *cmd_buffer) {
  call(cmd_buffer, "commit");
}

void wait_until_completed(MTLCommandBuffer *cmd_buffer) {
  call(cmd_buffer, "waitUntilCompleted");
}

void *mtl_buffer_contents(MTLBuffer *buffer) {
  return call(buffer, "contents");
}

template <typename T> void set_label(T *mtl_obj, const std::string &label) {
  // Set labels on Metal command buffer and encoders, so that they can be
  // tracked in Instrument - Metal System Trace
  if constexpr (std::is_same_v<T, MTLComputeCommandEncoder> ||
                std::is_same_v<T, MTLCommandBuffer>) {
    auto *label_str = wrap_string_as_ns_string(label.data(), label.size());
    call(mtl_obj, "setLabel:", label_str);
    release_ns_object(label_str);
  }
}

class VMRaii {
public:
  explicit VMRaii(size_t size) : size_(roundup_to_pagesize(size)) {
    ptr_ = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, /*fd=*/-1,
                /*offset=*/0);
    std::memset(ptr_, 0, size_);
  }

  size_t size() const { return size_; }
  void *ptr() const { return ptr_; }

  ~VMRaii() { munmap(ptr_, size_); }

private:
  const size_t size_;
  void *ptr_;
};

std::string load_file(const std::string &filename) {
  std::ifstream infile(filename);
  std::string line;
  std::stringstream ss;
  while (std::getline(infile, line)) {
    ss << line << '\n';
  }
  return ss.str();
}

#define TI_ASSERT(x) assert(x)

struct KernelAttributes {
  enum class Buffers {
    Root,
    GlobalTmps,
    Context,
    Runtime,
    Print,
  };
  std::string name;
  int num_threads;

  std::vector<Buffers> buffers;
};

using BufferEnum = KernelAttributes::Buffers;
using InputBuffersMap = std::unordered_map<BufferEnum, MTLBuffer *>;

class CompiledMtlKernel {
public:
  struct Params {
    const KernelAttributes *kernel_attribs;
    MTLDevice *device;
    MTLFunction *mtl_func;
    std::optional<int> context_size;
  };

  explicit CompiledMtlKernel(Params &params)
      : kernel_attribs_(*params.kernel_attribs),
        pipeline_state_(new_compute_pipeline_state_with_function(
            params.device, params.mtl_func)) {
    if (params.context_size) {
      ctx_mem_ = std::make_unique<VMRaii>(params.context_size.value());
      ctx_buffer_ = new_mtl_buffer_no_copy(params.device, ctx_mem_->ptr(),
                                           ctx_mem_->size());
      TI_ASSERT(ctx_buffer_ != nullptr);
    }
  }

  inline KernelAttributes *kernel_attribs() { return &kernel_attribs_; }

  void launch(InputBuffersMap &input_buffers,
              MTLCommandBuffer *command_buffer) {
    BindBuffers buffers;
    for (const auto b : kernel_attribs_.buffers) {
      if (b == BufferEnum::Context) {
        TI_ASSERT(ctx_buffer_ != nullptr);
        buffers.push_back({ctx_buffer_, b});
      } else {
        buffers.push_back({input_buffers.find(b)->second, b});
      }
    }
    launch_if_not_empty(std::move(buffers), command_buffer);
  }

protected:
  using BindBuffers = std::vector<std::pair<MTLBuffer *, BufferEnum>>;

  void launch_if_not_empty(BindBuffers buffers,
                           MTLCommandBuffer *command_buffer) {
    const int num_threads = kernel_attribs_.num_threads;
    if (num_threads == 0) {
      return;
    }
    TI_ASSERT(buffers.size() == kernel_attribs_.buffers.size());
    auto encoder = new_compute_command_encoder(command_buffer);
    TI_ASSERT(encoder != nullptr);

    set_label(encoder.get(), kernel_attribs_.name);
    set_compute_pipeline_state(encoder.get(), pipeline_state_.get());

    for (int bi = 0; bi < buffers.size(); ++bi) {
      auto &b = buffers[bi];
      TI_ASSERT(b.second == kernel_attribs_.buffers[bi]);
      set_mtl_buffer(encoder.get(), b.first, /*offset=*/0, bi);
    }
    const int num_threads_per_group =
        get_max_total_threads_per_threadgroup(pipeline_state_.get());
    const int num_groups =
        ((num_threads + num_threads_per_group - 1) / num_threads_per_group);
    dispatch_threadgroups(encoder.get(), num_groups,
                          std::min(num_threads, num_threads_per_group));
    end_encoding(encoder.get());
  }

  KernelAttributes kernel_attribs_;
  nsobj_unique_ptr<MTLComputePipelineState> pipeline_state_;
  std::unique_ptr<VMRaii> ctx_mem_;
  MTLBuffer *ctx_buffer_;
};

class KernelManager {
public:
  struct Params {
    int root_size;
    int runtime_size = 1024 * 1024;
  };

  explicit KernelManager(const Params &params) {
    device_ = mtl_create_system_default_device();
    TI_ASSERT(device_ != nullptr);
    std::cout << "mtl_device=" << device_ << "\n";
    command_queue_ = new_command_queue(device_);
    TI_ASSERT(command_queue_ != nullptr);
    create_new_command_buffer();

    // compiled_structs_.root_size, mem_pool_);
    root_mem_ = std::make_unique<VMRaii>(params.root_size);
    root_buffer_ =
        new_mtl_buffer_no_copy(device_, root_mem_->ptr(), root_mem_->size());
    TI_ASSERT(root_buffer_ != nullptr);
    std::cout << "Metal root buffer size: " << root_mem_->size() << " bytes";

    global_tmps_mem_ = std::make_unique<VMRaii>(1024 * 1024);
    global_tmps_buffer_ = new_mtl_buffer_no_copy(
        device_, global_tmps_mem_->ptr(), global_tmps_mem_->size());
    TI_ASSERT(global_tmps_buffer_ != nullptr);

    runtime_mem_ = std::make_unique<VMRaii>(params.runtime_size);
    runtime_buffer_ = new_mtl_buffer_no_copy(device_, runtime_mem_->ptr(),
                                             runtime_mem_->size());
    TI_ASSERT(runtime_buffer_ != nullptr);

    print_mem_ = std::make_unique<VMRaii>(1024 * 1024);
    print_buffer_ =
        new_mtl_buffer_no_copy(device_, print_mem_->ptr(), print_mem_->size());
    TI_ASSERT(print_buffer_ != nullptr);
  }

  inline MTLDevice *device() { return device_; }

  int register_mtl_kernel(std::unique_ptr<CompiledMtlKernel> kernel) {
    const int id = compiled_mtl_kernels_.size();
    compiled_mtl_kernels_.push_back(std::move(kernel));
    return id;
  }

  void launch_mtl_kernel(int id) {
    InputBuffersMap input_buffers = {
        {BufferEnum::Root, root_buffer_},
        {BufferEnum::GlobalTmps, global_tmps_buffer_},
        {BufferEnum::Runtime, runtime_buffer_},
        {BufferEnum::Print, print_buffer_},
    };
    auto &k = compiled_mtl_kernels_[id];
    pending_kernels_.push_back(k->kernel_attribs()->name);
    compiled_mtl_kernels_[id]->launch(input_buffers, cur_command_buffer_.get());
  }

  void synchronize() {
    using namespace std::chrono;
    const auto start = high_resolution_clock::now();
    commit_command_buffer(cur_command_buffer_.get());
    wait_until_completed(cur_command_buffer_.get());
    const auto stop = high_resolution_clock::now();
    const auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "-------------------------------\n";
    std::cout << "Metal synchronize took " << duration.count() << " us\n";
    std::cout << "Lauched kernels:\n";
    for (const auto &k : pending_kernels_) {
      std::cout << "* " << k << "\n";
    }
    std::cout << "-------------------------------\n";
    pending_kernels_.clear();
    create_new_command_buffer();
  }

private:
  void create_new_command_buffer() {
    cur_command_buffer_ = new_command_buffer(command_queue_.get());
    TI_ASSERT(cur_command_buffer_ != nullptr);
    set_label(cur_command_buffer_.get(),
              "command_buffer_" + std::to_string(command_buffer_id_++));
  }

  MTLDevice *device_;
  nsobj_unique_ptr<MTLCommandQueue> command_queue_;
  nsobj_unique_ptr<MTLCommandBuffer> cur_command_buffer_;
  std::size_t command_buffer_id_;
  std::unique_ptr<VMRaii> root_mem_;
  MTLBuffer *root_buffer_;
  std::unique_ptr<VMRaii> global_tmps_mem_;
  MTLBuffer *global_tmps_buffer_;
  std::unique_ptr<VMRaii> runtime_mem_;
  MTLBuffer *runtime_buffer_;
  std::unique_ptr<VMRaii> print_mem_;
  MTLBuffer *print_buffer_;
  std::vector<std::unique_ptr<CompiledMtlKernel>> compiled_mtl_kernels_;
  std::vector<std::string> pending_kernels_;
};

} // namespace

#if 0
void run_kernel(const std::string &kernel_name, MTLDevice *device,
                MTLLibrary *kernel_lib, MTLBuffer *root_buffer) {
  MTLFunction *kernel_func = new_function_with_name(
      kernel_lib, kernel_name.data(), kernel_name.size());
  std::cout << "kernel_func=" << kernel_func << "\n";

  MTLComputePipelineState *pipeline_state =
      new_compute_pipeline_state_with_function(device, kernel_func);
  std::cout << "pipeline_state=" << pipeline_state << "\n";

  MTLCommandQueue *cmd_queue = new_command_queue(device);
  std::cout << "cmd_queue=" << cmd_queue << "\n";
  MTLCommandBuffer *cmd_buffer = new_command_buffer(cmd_queue);
  std::cout << "cmd_buffer=" << cmd_buffer << "\n";

  MTLComputeCommandEncoder *encoder = new_compute_command_encoder(cmd_buffer);
  std::cout << "compute encoder=" << encoder << "\n";

  set_compute_pipeline_state(encoder, pipeline_state);
  std::cout << "set_compute_pipeline_state done\n";

  set_mtl_buffer(encoder, root_buffer, /*offset=*/0, /*index=*/0);
  // set_mtl_buffer(encoder, args_buffer, /*offset=*/0,
  // /*index=*/1);
  std::cout << "set_mtl_buffer done\n";

  constexpr int kThreadsPerGroup = 1024;
  const int kThreadGroups =
      (kRootBufferLen + kThreadsPerGroup - 1) / kThreadsPerGroup;
  std::cout << "kThreadsPerGroup=" << kThreadsPerGroup
            << " kThreadGroups=" << kThreadGroups << "\n";

  dispatch_threadgroups(encoder, kThreadGroups, kThreadsPerGroup);
  std::cout << "dispatch_threadgroups done\n";

  end_encoding(encoder);
  std::cout << "end_encoding done\n";

  commit_command_buffer(cmd_buffer);
  std::cout << "commit_command_buffer done\n";

  wait_until_completed(cmd_buffer);
  std::cout << "wait_until_completed done\n";
}
#endif

std::unique_ptr<KernelManager> init_kernel_mgr() {
  KernelManager::Params km_params;
  km_params.root_size = 8392704;
  km_params.runtime_size = 1024 * 1024 * 10;
  auto km = std::make_unique<KernelManager>(km_params);
  return km;
}

constexpr char kKernelStep0[] = "mtl_k0006_linear_solver_step_c14_0_0";

std::unordered_map<std::string, int>
init_mtl_kernels(KernelManager *kernel_mgr) {
  const auto kernel_src_str = load_file("kernel.metal");
  auto *device = kernel_mgr->device();
  auto kernel_lib = new_library_with_source(device, kernel_src_str.data(),
                                            kernel_src_str.size());
  std::cout << "kernel_lib=" << kernel_lib.get() << "\n";

  std::unordered_map<std::string, int> kname_to_id;
  KernelAttributes ka;
  ka.name = kKernelStep0;
  ka.num_threads = 65536;
  // device byte* root_addr [[buffer(0)]],
  //   device byte* global_tmps_addr [[buffer(1)]],
  //   device byte* ctx_addr [[buffer(2)]],
  //   device byte* runtime_addr [[buffer(3)]],
  //   device byte* print_addr [[buffer(4)]],
  ka.buffers = {
      BufferEnum::Root,    BufferEnum::GlobalTmps, BufferEnum::Context,
      BufferEnum::Runtime, BufferEnum::Print,
  };
  CompiledMtlKernel::Params kp;
  kp.kernel_attribs = &ka;
  kp.device = device;
  kp.context_size = 1024 * 4;

  {
    auto mtl_func = new_function_with_name(kernel_lib.get(), ka.name.data(),
                                           ka.name.size());
    kp.mtl_func = mtl_func.get();
    const auto id = kernel_mgr->register_mtl_kernel(
        std::make_unique<CompiledMtlKernel>(kp));
    kname_to_id[ka.name] = id;
  }
  return kname_to_id;
}

int main() {
  auto kernel_mgr = init_kernel_mgr();
  const auto kname_to_id = init_mtl_kernels(kernel_mgr.get());

  for (int i = 0; i < 10; ++i) {
    const auto id = kname_to_id.at(kKernelStep0);
    kernel_mgr->launch_mtl_kernel(id);
    kernel_mgr->synchronize();
  }
  return 0;
}
