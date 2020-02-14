#include <metal_stdlib>
using namespace metal;

namespace {

template <typename T, typename G>
T union_cast(G g) {
  static_assert(sizeof(T) == sizeof(G), "Size mismatch");
  return *reinterpret_cast<thread const T*>(&g);
}

using byte = uchar;

struct S2 {
  // place
  constant static constexpr int stride = sizeof(int32_t);
  S2(device byte* v) : val((device int32_t*)v) {}
  device int32_t* val;
};

class S1_ch {
 private:
  device byte* addr_;
 public:
  S1_ch(device byte* a) : addr_(a) {}
  S2 get0() {
    return {addr_};
  }
  constant static constexpr int stride = S2::stride;
};

struct S1 {
  // dense
  constant static constexpr int n = 16;
  constant static constexpr int stride = S1_ch::stride * n;
  S1(device byte* a) : addr_(a) {}
  S1_ch children(int i) {
    return {addr_ + i * S1_ch::stride};
  }
 private:
  device byte* addr_;
};

class S0_ch {
 private:
  device byte* addr_;
 public:
  S0_ch(device byte* a) : addr_(a) {}
  S1 get0() {
    return {addr_};
  }
  constant static constexpr int stride = S1::stride;
};

struct S0 {
  // root
  constant static constexpr int n = 1;
  constant static constexpr int stride = S0_ch::stride * n;
  S0(device byte* a) : addr_(a) {}
  S0_ch children(int i) {
    return {addr_ + i * S0_ch::stride};
  }
 private:
  device byte* addr_;
};

}  // namespace

kernel void broken_test(
    device byte* addr [[buffer(0)]],
    device byte* global_tmps_addr [[buffer(1)]],
    const uint utid_ [[thread_position_in_grid]]) {
  // range_for, range known at compile time
  if (utid_ >= 2) return;
  int32_t tmp5(0);
  const int32_t tmp6 = 0;
  const int32_t tmp7 = 7;
  for (int tmp5_ = tmp6; tmp5_ < tmp7; tmp5_ = tmp5_ + 1) {
    int tmp5 = tmp5_;
    const int tmp9 = (static_cast<int>(utid_) + 0);
    const int32_t tmp10(tmp5);
    S0 tmp13(addr);
    auto tmp14 = 0;
    S0_ch tmp15 = tmp13.children(tmp14);
    S1 tmp16 = tmp15.get0();
    auto tmp17 = (((0 + tmp9) >> 0) & ((1 << 1) - 1));
    auto tmp18 = (((0 + tmp10) >> 0) & ((1 << 3) - 1));
    auto tmp19 = ((0 * 2 + tmp17) * 8 + tmp18);
    S1_ch tmp20 = tmp16.children(tmp19);
    device int32_t* tmp21 = tmp20.get0().val;
    int32_t tmp23 = *tmp21;
    const int32_t tmp24 = 1;
    const int32_t tmp25 = (tmp23 + tmp24);
    const int32_t tmp26 = (tmp10 + tmp24);
    auto tmp34 = (((0 + tmp26) >> 0) & ((1 << 3) - 1));
    auto tmp35 = ((0 * 2 + tmp17) * 8 + tmp34);
    S1_ch tmp36 = tmp16.children(tmp35);
    device int32_t* tmp37 = tmp36.get0().val;
    *tmp37 = tmp25;
  }
}

kernel void human_readable_test(
    device int32_t* addr [[buffer(0)]],
    const uint utid_ [[thread_position_in_grid]]) {
  // range_for, range known at compile time
  if (utid_ >= 2) return;
  for (int i = 0; i < 7; i++) {
    const int tid = static_cast<int>(utid_);
    int read_idx = tid * 8 + i;
    const int32_t read_val = addr[read_idx];
    const int32_t write_val = read_val + 1;
    int write_idx = tid * 8 + i + 1;
    addr[write_idx] = write_val;
  }
}