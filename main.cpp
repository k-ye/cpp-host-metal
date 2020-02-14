// clang++ -std=c++11 main.cpp -framework Metal -framework CoreGraphics -framework Foundation

// Frameworks:
// * Metal for obvious reason
// * CoreGraphics:
// https://developer.apple.com/documentation/metal/1433401-mtlcreatesystemdefaultdevice?language=objc
// * Foundation: objc runtime, https://gist.github.com/TooTallNate/1073294/675f28984cd120cdec14b66b41d86cf01140b1e6
//
// References:
//
// * Halide
// * taichi
// * https://github.com/naleksiev/mtlpp
//
// All <Metal/*> headers are for Obj-C, we cannot import them.
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

#include <objc/message.h>
#include <objc/objc.h>
#include <objc/runtime.h>

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

template <typename O, typename C = id, typename... Args>
C call(O *i, const char *select, Args... args) {
  using func = C (*)(id, SEL, Args...);
  return ((func)(objc_msgSend))(reinterpret_cast<id>(i), sel_getUid(select),
                                args...);
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

size_t roundup_to_pagesize(size_t s) {
  const size_t pagesize = getpagesize();
  return ((s + pagesize - 1) / pagesize) * pagesize;
}

MTLDevice *mtl_create_system_default_device() {
  id dev = MTLCreateSystemDefaultDevice();
  return reinterpret_cast<MTLDevice *>(dev);
}

MTLCommandQueue *new_command_queue(MTLDevice *dev) {
  id queue = call(dev, "newCommandQueue");
  return reinterpret_cast<MTLCommandQueue *>(queue);
}

MTLCommandBuffer* new_command_buffer(MTLCommandQueue* queue) {
  id buffer = call(queue, "commandBuffer");
  return reinterpret_cast<MTLCommandBuffer*>(buffer);
}

MTLComputeCommandEncoder* new_compute_command_encoder(MTLCommandBuffer* buffer) {
  id encoder = call(buffer, "computeCommandEncoder");
  return reinterpret_cast<MTLComputeCommandEncoder*>(encoder);
}

NSString *wrap_string_as_ns_string(const char *str, size_t len) {
  id ns_string = clscall("NSString", "alloc");
  return reinterpret_cast<NSString *>(
      call(ns_string, "initWithBytesNoCopy:length:encoding:freeWhenDone:", str,
           len, kNSUTF8StringEncoding, false));
}

MTLLibrary *new_library_with_source(MTLDevice *device, const char *source,
                                    size_t source_len) {
  NSString *source_str = wrap_string_as_ns_string(source, source_len);

  id options = clscall("MTLCompileOptions", "alloc");
  options = call(options, "init");
  call(options, "setFastMathEnabled:", false);

  id lib = call(device, "newLibraryWithSource:options:error:", source_str,
                options, nullptr);

  release_ns_object(options);
  release_ns_object(source_str);

  return reinterpret_cast<MTLLibrary *>(lib);
}

MTLFunction *new_function_with_name(MTLLibrary *library, const char *name,
                                    size_t name_len) {
  NSString *name_str = wrap_string_as_ns_string(name, name_len);
  id func = call(library, "newFunctionWithName:", name_str);
  release_ns_object(name_str);
  return reinterpret_cast<MTLFunction *>(func);
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

constexpr size_t kRootBufferLen = 16;
constexpr size_t kRootBufferBytes = sizeof(int32_t) * kRootBufferLen;

} // namespace

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
  MTLCommandBuffer* cmd_buffer = new_command_buffer(cmd_queue);
  std::cout << "cmd_buffer=" << cmd_buffer << "\n";

  MTLComputeCommandEncoder* encoder = new_compute_command_encoder(cmd_buffer);
  std::cout << "compute encoder=" << encoder << "\n";

  set_compute_pipeline_state(encoder, pipeline_state);
  std::cout << "set_compute_pipeline_state done\n";

  set_mtl_buffer(encoder, root_buffer, /*offset=*/0, /*index=*/0);
  // set_mtl_buffer(encoder, args_buffer, /*offset=*/0, /*index=*/1);
  std::cout << "set_mtl_buffer done\n";

  constexpr int kThreadsPerGroup = 1024;
  const int kThreadGroups =
      (kRootBufferLen + kThreadsPerGroup - 1) / kThreadsPerGroup;
  std::cout << "kThreadsPerGroup=" << kThreadsPerGroup
            << " kThreadGroups=" << kThreadGroups << "\n";

  dispatch_threadgroups(encoder, kThreadsPerGroup, kThreadGroups);
  std::cout << "dispatch_threadgroups done\n";

  end_encoding(encoder);
  std::cout << "end_encoding done\n";

  commit_command_buffer(cmd_buffer);
  std::cout << "commit_command_buffer done\n";

  wait_until_completed(cmd_buffer);
  std::cout << "wait_until_completed done\n";
}

void run_broken_test(MTLDevice *device, MTLLibrary *kernel_lib,
                     MTLBuffer *root_buffer) {

  run_kernel("broken_test", device, kernel_lib, root_buffer);
}

void run_human_readable_test(MTLDevice *device, MTLLibrary *kernel_lib,
                             MTLBuffer *root_buffer) {
  run_kernel("human_readable_test", device, kernel_lib, root_buffer);
}

void print_result(const int32_t *buffer_contents) {
  constexpr int kRowLen = 8;
  for (int i = 0; i < kRootBufferLen / kRowLen; ++i) {
    for (int j = 0; j < kRowLen; ++j) {
      const auto a = buffer_contents[i * kRowLen + j];
      const auto e = j;
      if (a != e) {
        // Red
        std::cout << "\033[1;31m";
      } else {
        // Green
        std::cout << "\033[1;32m";
      }
      std::cout << "root_buffer[" << i << ", " << j << "]=" << a
                << " expected=" << e;
      if (a != e) {
        std::cout << " MISMATCH!!";
      }
      std::cout << "\033[0m\n";
    }
  }
}

int main() {
  MTLDevice *device = mtl_create_system_default_device();
  std::cout << "mtl_device=" << device << "\n";

  const auto kernel_src_str = load_file("kernel.metal");
  MTLLibrary *kernel_lib = new_library_with_source(
      device, kernel_src_str.data(), kernel_src_str.size());
  std::cout << "kernel_lib=" << kernel_lib << "\n";

  VMRaii root_mem(kRootBufferBytes);
  std::cout << "root_mem=" << root_mem.ptr()
            << " requested=" << kRootBufferBytes << " got=" << root_mem.size()
            << "\n";

  MTLBuffer *root_buffer =
      new_mtl_buffer_no_copy(device, root_mem.ptr(), root_mem.size());
  std::cout << "root_buffer=" << root_buffer << "\n";

  run_broken_test(device, kernel_lib, root_buffer);
  // run_human_readable_test(device, kernel_lib, root_buffer);
  print_result(reinterpret_cast<int32_t *>(root_mem.ptr()));
  return 0;
}
