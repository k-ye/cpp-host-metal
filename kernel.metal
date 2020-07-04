#include <metal_stdlib>
#include <metal_compute>
using namespace metal;

using byte = uchar;

template <typename T, typename G> T union_cast(G g) { static_assert(sizeof(T) == sizeof(G), "Size mismatch"); return *reinterpret_cast<thread const T *>(&g); } inline int ifloordiv(int lhs, int rhs) { const int intm = (lhs / rhs); return (((lhs < 0) != (rhs < 0) && lhs && (rhs * intm != lhs)) ? (intm - 1) : intm); } int32_t pow_i32(int32_t x, int32_t n) { int32_t tmp = x; int32_t ans = 1; while (n) { if (n & 1) ans *= tmp; tmp *= tmp; n >>= 1; } return ans; } float fatomic_fetch_add(device float *dest, const float operand) { bool ok = false; float old_val = 0.0f; while (!ok) { old_val = *dest; float new_val = (old_val + operand); ok = atomic_compare_exchange_weak_explicit( (device atomic_int *)dest, (thread int *)(&old_val), *((thread int *)(&new_val)), metal::memory_order_relaxed, metal::memory_order_relaxed); } return old_val; } float fatomic_fetch_min(device float *dest, const float operand) { bool ok = false; float old_val = 0.0f; while (!ok) { old_val = *dest; float new_val = (old_val < operand) ? old_val : operand; ok = atomic_compare_exchange_weak_explicit( (device atomic_int *)dest, (thread int *)(&old_val), *((thread int *)(&new_val)), metal::memory_order_relaxed, metal::memory_order_relaxed); } return old_val; } float fatomic_fetch_max(device float *dest, const float operand) { bool ok = false; float old_val = 0.0f; while (!ok) { old_val = *dest; float new_val = (old_val > operand) ? old_val : operand; ok = atomic_compare_exchange_weak_explicit( (device atomic_int *)dest, (thread int *)(&old_val), *((thread int *)(&new_val)), metal::memory_order_relaxed, metal::memory_order_relaxed); } return old_val; } struct RandState { uint32_t seed; }; uint32_t metal_rand_u32(device RandState * state) { device uint *sp = (device uint *)&(state->seed); bool done = false; uint32_t nxt = 0; while (!done) { uint32_t o = *sp; nxt = o * 1103515245 + 12345; done = atomic_compare_exchange_weak_explicit( (device atomic_uint *)sp, &o, nxt, metal::memory_order_relaxed, metal::memory_order_relaxed); } return nxt * 1000000007; } int32_t metal_rand_i32(device RandState * state) { return metal_rand_u32(state); } float metal_rand_f32(device RandState *state) { return metal_rand_u32(state) * (1.0f / 4294967296.0f); }

constant constexpr int kTaichiMaxNumIndices = 8; constant constexpr int kTaichiNumChunks = 1024; struct MemoryAllocator { atomic_int next; }; struct ListgenElement { int32_t coords[kTaichiMaxNumIndices]; int32_t root_mem_offset = 0; }; struct ListManagerData { int32_t element_stride = 0; int32_t log2_num_elems_per_chunk = 0; atomic_int next; atomic_int chunks[kTaichiNumChunks]; }; struct ListManager { device ListManagerData *lm_data; device MemoryAllocator *mem_alloc; }; struct SNodeMeta { enum Type { Root = 0, Dense = 1, Bitmasked = 2, Dynamic = 3 }; int32_t element_stride = 0; int32_t num_slots = 0; int32_t mem_offset_in_parent = 0; int32_t type = 0; }; struct SNodeExtractors { struct Extractor { int32_t start = 0; int32_t num_bits = 0; int32_t acc_offset = 0; int32_t num_elements = 0; }; Extractor extractors[kTaichiMaxNumIndices]; };

using PtrOffset = int32_t; constant constexpr int kAlignment = 8; [[maybe_unused]] PtrOffset mtl_memalloc_alloc(device MemoryAllocator * ma, int32_t size) { size = ((size + kAlignment - 1) / kAlignment) * kAlignment; return atomic_fetch_add_explicit(&ma->next, size, metal::memory_order_relaxed); } [[maybe_unused]] device char *mtl_memalloc_to_ptr(device MemoryAllocator *ma, PtrOffset offs) { return reinterpret_cast<device char *>(ma + 1) + offs; } [[maybe_unused]] int num_active(thread ListManager *l) { return atomic_load_explicit(&(l->lm_data->next), metal::memory_order_relaxed); } [[maybe_unused]] void clear(thread ListManager *l) { atomic_store_explicit(&(l->lm_data->next), 0, metal::memory_order_relaxed); } [[maybe_unused]] PtrOffset mtl_listmgr_ensure_chunk(thread ListManager *l, int i) { device ListManagerData *list = l->lm_data; PtrOffset offs = 0; const int kChunkBytes = (list->element_stride << list->log2_num_elems_per_chunk); while (true) { int stored = 0; const bool is_me = atomic_compare_exchange_weak_explicit( list->chunks + i, &stored, 1, metal::memory_order_relaxed, metal::memory_order_relaxed); if (is_me) { offs = mtl_memalloc_alloc(l->mem_alloc, kChunkBytes); atomic_store_explicit(list->chunks + i, offs, metal::memory_order_relaxed); break; } else if (stored > 1) { offs = stored; break; } } return offs; } [[maybe_unused]] device char *mtl_listmgr_get_elem_from_chunk( thread ListManager *l, int i, PtrOffset chunk_ptr_offs) { device ListManagerData *list = l->lm_data; device char *chunk_ptr = reinterpret_cast<device char *>( mtl_memalloc_to_ptr(l->mem_alloc, chunk_ptr_offs)); const uint32_t mask = ((1 << list->log2_num_elems_per_chunk) - 1); return chunk_ptr + ((i & mask) * list->element_stride); } [[maybe_unused]] device char *append(thread ListManager *l) { device ListManagerData *list = l->lm_data; const int elem_idx = atomic_fetch_add_explicit( &list->next, 1, metal::memory_order_relaxed); const int chunk_idx = elem_idx >> list->log2_num_elems_per_chunk; const PtrOffset chunk_ptr_offs = mtl_listmgr_ensure_chunk(l, chunk_idx); return mtl_listmgr_get_elem_from_chunk(l, elem_idx, chunk_ptr_offs); } template <typename T> [[maybe_unused]] void append(thread ListManager *l, thread const T &elem) { device char *ptr = append(l); thread char *elem_ptr = (thread char *)(&elem); for (int i = 0; i < l->lm_data->element_stride; ++i) { *ptr = *elem_ptr; ++ptr; ++elem_ptr; } } template <typename T> [[maybe_unused]] T get(thread ListManager *l, int i) { device ListManagerData *list = l->lm_data; const int chunk_idx = i >> list->log2_num_elems_per_chunk; const PtrOffset chunk_ptr_offs = atomic_load_explicit( list->chunks + chunk_idx, metal::memory_order_relaxed); return *reinterpret_cast<device T *>( mtl_listmgr_get_elem_from_chunk(l, i, chunk_ptr_offs)); } [[maybe_unused]] int is_active(device byte *addr, SNodeMeta meta, int i) { if (meta.type == SNodeMeta::Root || meta.type == SNodeMeta::Dense) { return true; } device auto *meta_ptr_begin = reinterpret_cast<device atomic_uint *>( addr + ((meta.num_slots - i) * meta.element_stride)); if (meta.type == SNodeMeta::Dynamic) { device auto *ptr = meta_ptr_begin; uint32_t n = atomic_load_explicit(ptr, metal::memory_order_relaxed); return i < n; } device auto *ptr = meta_ptr_begin + (i / (sizeof(uint32_t) * 8)); uint32_t bits = atomic_load_explicit(ptr, metal::memory_order_relaxed); return ((bits >> (i % (sizeof(uint32_t) * 8))) & 1); } [[maybe_unused]] void activate(device byte *addr, SNodeMeta meta, int i) { if (meta.type == SNodeMeta::Root || meta.type == SNodeMeta::Dense) { return; } device auto *meta_ptr_begin = reinterpret_cast<device atomic_uint *>( addr + ((meta.num_slots - i) * meta.element_stride)); if (meta.type == SNodeMeta::Dynamic) { device auto *ptr = meta_ptr_begin; atomic_store_explicit(ptr, (uint32_t)(i + 1), metal::memory_order_relaxed); return; } device auto *ptr = meta_ptr_begin + (i / (sizeof(uint32_t) * 8)); const uint32_t mask = (1 << (i % (sizeof(uint32_t) * 8))); atomic_fetch_or_explicit(ptr, mask, metal::memory_order_relaxed); } [[maybe_unused]] void deactivate(device byte *addr, SNodeMeta meta, int i) { if (meta.type == SNodeMeta::Root || meta.type == SNodeMeta::Dense) { return; } device auto *meta_ptr_begin = reinterpret_cast<device atomic_uint *>( addr + ((meta.num_slots - i) * meta.element_stride)); if (meta.type == SNodeMeta::Dynamic) { device auto *ptr = meta_ptr_begin; atomic_store_explicit(ptr, 0u, metal::memory_order_relaxed); return; } device auto *ptr = meta_ptr_begin + (i / (sizeof(uint32_t) * 8)); const uint32_t mask = ~(1 << (i % (sizeof(uint32_t) * 8))); atomic_fetch_and_explicit(ptr, mask, metal::memory_order_relaxed); } [[maybe_unused]] void refine_coordinates( thread const ListgenElement &parent_elem, device const SNodeExtractors &child_extrators, int l, thread ListgenElement *child_elem) { for (int i = 0; i < kTaichiMaxNumIndices; ++i) { device const auto &ex = child_extrators.extractors[i]; const int mask = ((1 << ex.num_bits) - 1); const int addition = (((l >> ex.acc_offset) & mask) << ex.start); child_elem->coords[i] = (parent_elem.coords[i] | addition); } } [[maybe_unused]] int dynamic_append(device byte *addr, SNodeMeta meta, int32_t data) { device auto *n_ptr = reinterpret_cast<device atomic_int *>( addr + (meta.num_slots * meta.element_stride)); int me = atomic_fetch_add_explicit(n_ptr, 1, metal::memory_order_relaxed); *(reinterpret_cast<device int32_t *>(addr) + me) = data; return me; } [[maybe_unused]] int dynamic_length(device byte *addr, SNodeMeta meta) { device auto *n_ptr = reinterpret_cast<device atomic_int *>( addr + (meta.num_slots * meta.element_stride)); return atomic_load_explicit(n_ptr, metal::memory_order_relaxed); }

struct Runtime {
  SNodeMeta snode_metas[14];
  SNodeExtractors snode_extractors[14];
  ListManagerData snode_lists[14];
  uint32_t rand_seeds[65536];
};



struct S16 {
  // place
  constant static constexpr int stride = sizeof(float);
  S16(device byte* v) : val((device float*)v) {}
  device float* val;
};


struct S15 {
  // place
  constant static constexpr int stride = sizeof(float);
  S15(device byte* v) : val((device float*)v) {}
  device float* val;
};


struct S14 {
  // place
  constant static constexpr int stride = sizeof(float);
  S14(device byte* v) : val((device float*)v) {}
  device float* val;
};

class S13_ch {
 public:
  S13_ch(device byte* a) : addr_(a) {}
  S14 get0() {
    return {addr_};
  }

  S15 get1() {
    return {addr_ + (S14::stride)};
  }

  S16 get2() {
    return {addr_ + (S14::stride + S15::stride)};
  }

  device byte* addr() { return addr_; }

  constant static constexpr int stride = S14::stride + S15::stride + S16::stride;
 private:
  device byte* addr_;
};

struct S13 {
  // dense
  constant static constexpr int n = 262144;
  constant static constexpr int stride = S13_ch::stride * n;
  S13(device byte* a) : addr_(a) {}

  S13_ch children(int i) {
    return {addr_ + i * S13_ch::stride};
  }

 private:
  device byte* addr_;
};


struct S12 {
  // place
  constant static constexpr int stride = sizeof(float);
  S12(device byte* v) : val((device float*)v) {}
  device float* val;
};

class S11_ch {
 public:
  S11_ch(device byte* a) : addr_(a) {}
  S12 get0() {
    return {addr_};
  }

  device byte* addr() { return addr_; }

  constant static constexpr int stride = S12::stride;
 private:
  device byte* addr_;
};

struct S11 {
  // dense
  constant static constexpr int n = 1;
  constant static constexpr int stride = S11_ch::stride * n;
  S11(device byte* a) : addr_(a) {}

  S11_ch children(int i) {
    return {addr_ + i * S11_ch::stride};
  }

 private:
  device byte* addr_;
};


struct S10 {
  // place
  constant static constexpr int stride = sizeof(float);
  S10(device byte* v) : val((device float*)v) {}
  device float* val;
};

class S9_ch {
 public:
  S9_ch(device byte* a) : addr_(a) {}
  S10 get0() {
    return {addr_};
  }

  device byte* addr() { return addr_; }

  constant static constexpr int stride = S10::stride;
 private:
  device byte* addr_;
};

struct S9 {
  // dense
  constant static constexpr int n = 262144;
  constant static constexpr int stride = S9_ch::stride * n;
  S9(device byte* a) : addr_(a) {}

  S9_ch children(int i) {
    return {addr_ + i * S9_ch::stride};
  }

 private:
  device byte* addr_;
};


struct S8 {
  // place
  constant static constexpr int stride = sizeof(float);
  S8(device byte* v) : val((device float*)v) {}
  device float* val;
};

class S7_ch {
 public:
  S7_ch(device byte* a) : addr_(a) {}
  S8 get0() {
    return {addr_};
  }

  device byte* addr() { return addr_; }

  constant static constexpr int stride = S8::stride;
 private:
  device byte* addr_;
};

struct S7 {
  // dense
  constant static constexpr int n = 262144;
  constant static constexpr int stride = S7_ch::stride * n;
  S7(device byte* a) : addr_(a) {}

  S7_ch children(int i) {
    return {addr_ + i * S7_ch::stride};
  }

 private:
  device byte* addr_;
};


struct S6 {
  // place
  constant static constexpr int stride = sizeof(float);
  S6(device byte* v) : val((device float*)v) {}
  device float* val;
};

class S5_ch {
 public:
  S5_ch(device byte* a) : addr_(a) {}
  S6 get0() {
    return {addr_};
  }

  device byte* addr() { return addr_; }

  constant static constexpr int stride = S6::stride;
 private:
  device byte* addr_;
};

struct S5 {
  // dense
  constant static constexpr int n = 262144;
  constant static constexpr int stride = S5_ch::stride * n;
  S5(device byte* a) : addr_(a) {}

  S5_ch children(int i) {
    return {addr_ + i * S5_ch::stride};
  }

 private:
  device byte* addr_;
};


struct S4 {
  // place
  constant static constexpr int stride = sizeof(float);
  S4(device byte* v) : val((device float*)v) {}
  device float* val;
};

class S3_ch {
 public:
  S3_ch(device byte* a) : addr_(a) {}
  S4 get0() {
    return {addr_};
  }

  device byte* addr() { return addr_; }

  constant static constexpr int stride = S4::stride;
 private:
  device byte* addr_;
};

struct S3 {
  // dense
  constant static constexpr int n = 262144;
  constant static constexpr int stride = S3_ch::stride * n;
  S3(device byte* a) : addr_(a) {}

  S3_ch children(int i) {
    return {addr_ + i * S3_ch::stride};
  }

 private:
  device byte* addr_;
};


struct S2 {
  // place
  constant static constexpr int stride = sizeof(float);
  S2(device byte* v) : val((device float*)v) {}
  device float* val;
};

class S1_ch {
 public:
  S1_ch(device byte* a) : addr_(a) {}
  S2 get0() {
    return {addr_};
  }

  device byte* addr() { return addr_; }

  constant static constexpr int stride = S2::stride;
 private:
  device byte* addr_;
};

struct S1 {
  // dense
  constant static constexpr int n = 262144;
  constant static constexpr int stride = S1_ch::stride * n;
  S1(device byte* a) : addr_(a) {}

  S1_ch children(int i) {
    return {addr_ + i * S1_ch::stride};
  }

 private:
  device byte* addr_;
};

class S0_ch {
 public:
  S0_ch(device byte* a) : addr_(a) {}
  S1 get0() {
    return {addr_};
  }

  S3 get1() {
    return {addr_ + (S1::stride)};
  }

  S5 get2() {
    return {addr_ + (S1::stride + S3::stride)};
  }

  S7 get3() {
    return {addr_ + (S1::stride + S3::stride + S5::stride)};
  }

  S9 get4() {
    return {addr_ + (S1::stride + S3::stride + S5::stride + S7::stride)};
  }

  S11 get5() {
    return {addr_ + (S1::stride + S3::stride + S5::stride + S7::stride + S9::stride)};
  }

  S13 get6() {
    return {addr_ + (S1::stride + S3::stride + S5::stride + S7::stride + S9::stride + S11::stride)};
  }

  device byte* addr() { return addr_; }

  constant static constexpr int stride = S1::stride + S3::stride + S5::stride + S7::stride + S9::stride + S11::stride + S13::stride;
 private:
  device byte* addr_;
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



using AdStackPtr = thread byte *; inline thread uint32_t * mtl_ad_stack_n(AdStackPtr stack) { return reinterpret_cast<thread uint32_t *>(stack); } inline AdStackPtr mtl_ad_stack_data(AdStackPtr stack) { return stack + sizeof(uint32_t); } inline void mtl_ad_stack_init(AdStackPtr stack) { *mtl_ad_stack_n(stack) = 0; } inline AdStackPtr mtl_ad_stack_top_primal(AdStackPtr stack, int element_size) { const auto n = *mtl_ad_stack_n(stack); return mtl_ad_stack_data(stack) + (n - 1) * 2 * element_size; } inline AdStackPtr mtl_ad_stack_top_adjoint(AdStackPtr stack, int element_size) { return mtl_ad_stack_top_primal(stack, element_size) + element_size; } inline void mtl_ad_stack_pop(AdStackPtr stack) { thread auto &n = *mtl_ad_stack_n(stack); --n; } void mtl_ad_stack_push(AdStackPtr stack, int element_size) { thread auto &n = *mtl_ad_stack_n(stack); ++n; AdStackPtr data = mtl_ad_stack_top_primal(stack, element_size); for (int i = 0; i < element_size * 2; ++i) { data[i] = 0; } }

constant constexpr int kMetalNumBitsPerPrintMsgType = 4; constant constexpr int kMetalNumPrintMsgTypePerI32 = sizeof(int32_t) * 8 / kMetalNumBitsPerPrintMsgType; constant constexpr int kMetalPrintMsgTypeWidthMask = ((1 << kMetalNumBitsPerPrintMsgType) - 1); [[maybe_unused]] inline int mtl_compute_num_print_msg_typemasks( int num_entries) { return (num_entries + kMetalNumPrintMsgTypePerI32 - 1) / kMetalNumPrintMsgTypePerI32; } [[maybe_unused]] inline int mtl_compute_print_msg_bytes( int num_entries) { const int sz = sizeof(int32_t) * (1 + mtl_compute_num_print_msg_typemasks(num_entries) + num_entries); return sz; } class PrintMsg { public: enum Type { I32 = 1, F32 = 2, Str = 3 }; PrintMsg(device int32_t * buf, int num_entries) : mask_buf_(buf), data_buf_(buf + mtl_compute_num_print_msg_typemasks(num_entries)) { } void pm_set_i32(int i, int x) { set_entry(i, x, Type::I32); } void pm_set_f32(int i, float x) { const int32_t ix = *reinterpret_cast<thread int32_t *>(&x); set_entry(i, ix, Type::F32); } void pm_set_str(int i, int str_id) { set_entry(i, str_id, Type::Str); } Type pm_get_type(int i) { const int mask_i = i / kMetalNumPrintMsgTypePerI32; const int i_in_mask = i % kMetalNumPrintMsgTypePerI32; int mask = mask_buf_[mask_i]; mask >>= typemask_shift(i_in_mask); mask &= kMetalPrintMsgTypeWidthMask; return (Type)mask; } int32_t pm_get_data(int i) { return data_buf_[i]; } private: void set_entry(int i, int32_t x, Type ty) { const int mask_i = i / kMetalNumPrintMsgTypePerI32; const int i_in_mask = i % kMetalNumPrintMsgTypePerI32; int mask = ((int)ty & kMetalPrintMsgTypeWidthMask); mask <<= typemask_shift(i_in_mask); mask_buf_[mask_i] |= mask; data_buf_[i] = x; } inline static int typemask_shift(int i_in_mask) { return (kMetalNumPrintMsgTypePerI32 - 1 - i_in_mask) * kMetalNumBitsPerPrintMsgType; } device int32_t *mask_buf_; device int32_t *data_buf_; }; struct PrintMsgAllocator { atomic_int next; }; constant constexpr int kMetalPrintBufferSize = 2 * 1024 * 1024 - sizeof(PrintMsgAllocator); [[maybe_unused]] device int32_t * mtl_print_alloc_buf(device PrintMsgAllocator * pa, int num_entries) { const int sz = mtl_compute_print_msg_bytes(num_entries); const int cur = atomic_fetch_add_explicit(&(pa->next), sz, metal::memory_order_relaxed); if (cur + sz >= kMetalPrintBufferSize) { return (device int32_t *)0; } device byte *data_begin = reinterpret_cast<device byte *>(pa + 1); device int32_t *ptr = reinterpret_cast<device int32_t *>(data_begin + cur); *ptr = num_entries; return (ptr + 1); }

///////// mtl_k0006_linear_solver_step_c14_0_0

class mtl_k0006_linear_solver_step_c14_0_args {
 public:
  explicit mtl_k0006_linear_solver_step_c14_0_args(device byte* addr) : addr_(addr) {}
  device float* arg0() {
    // scalar, size=4 B
    return (device float*)(addr_ + 0);
  }
  
  int32_t extra_arg(int i, int j) {
    device int32_t* base = (device int32_t*)(addr_ + 4);
    return *(base + (i * 8) + j);
  }
 private:
  device byte* addr_;
};

void mtl_k0006_linear_solver_step_c14_0_0_func(
    device byte* root_addr,
    device byte* global_tmps_addr,
    device byte* ctx_addr,
    device byte* runtime_addr,
    device byte* print_addr,
    const int linear_loop_idx_) {
  device Runtime *runtime_ = reinterpret_cast<device Runtime *>(runtime_addr);
  mtl_k0006_linear_solver_step_c14_0_args kernel_ctx_(ctx_addr);
  device RandState* rand_state_ = reinterpret_cast<device RandState*>(runtime_->rand_seeds + (linear_loop_idx_ % 65536));
  device auto* print_alloc_ = reinterpret_cast<device PrintMsgAllocator*>(print_addr);
  constexpr float tmp73 = 1.0;
  constexpr float tmp69 = -0.25;
  constexpr float tmp55 = 3.8146973e-06;
  constexpr int32_t tmp49 = 512;
  constexpr int32_t tmp45 = 511;
  constexpr int32_t tmp38 = 1;
  const int tmp11 = linear_loop_idx_;
  auto tmp12 = ((tmp11 >> 9) & ((1 << 9) - 1));
  auto tmp13 = ((tmp11 >> 0) & ((1 << 9) - 1));
  const int32_t tmp14 = (tmp12 + tmp13);
  constexpr int32_t tmp15 = 2;
  const int32_t tmp16 = ifloordiv(tmp14, tmp15);
  const int32_t tmp17 = (tmp16 + tmp16);
  const int32_t tmp18 = (tmp14 - tmp17);
  constexpr int32_t tmp19 = 0;
  const int32_t tmp20 = -(tmp18 == tmp19);
  const int32_t tmp22 = (tmp20 & tmp38);
  if (tmp22) {
    const int32_t tmp25 = (tmp12 + tmp38);
    const int32_t tmp27 = ifloordiv(tmp25, tmp49);
    const int32_t tmp29 = (tmp27 * tmp49);
    const int32_t tmp30 = (tmp25 - tmp29);
    const int32_t tmp32 = (tmp12 + tmp45);
    const int32_t tmp34 = ifloordiv(tmp32, tmp49);
    const int32_t tmp36 = (tmp34 * tmp49);
    const int32_t tmp37 = (tmp32 - tmp36);
    const int32_t tmp39 = (tmp13 + tmp38);
    const int32_t tmp41 = ifloordiv(tmp39, tmp49);
    const int32_t tmp43 = (tmp41 * tmp49);
    const int32_t tmp44 = (tmp39 - tmp43);
    const int32_t tmp46 = (tmp13 + tmp45);
    const int32_t tmp48 = ifloordiv(tmp46, tmp49);
    const int32_t tmp50 = (tmp48 * tmp49);
    const int32_t tmp51 = (tmp46 - tmp50);
    S0 tmp917(root_addr);
    S0_ch tmp919 = tmp917.children(tmp19);
    S5 tmp920 = tmp919.get2();
    const int32_t tmp1005 = (tmp12 * tmp49);
    const int32_t tmp1006 = (tmp13 + tmp1005);
    S5_ch tmp924 = tmp920.children(tmp1006);
    device float* tmp925 = tmp924.get0().val;
    float tmp53 = *tmp925;
    const float tmp54 = -(tmp53);
    const float tmp56 = (tmp54 * tmp55);
    S9 tmp932 = tmp919.get4();
    auto tmp933 = ((tmp30 >> 0) & ((1 << 9) - 1));
    const int32_t tmp1013 = (tmp933 * tmp49);
    const int32_t tmp1014 = (tmp13 + tmp1013);
    S9_ch tmp936 = tmp932.children(tmp1014);
    device float* tmp937 = tmp936.get0().val;
    float tmp58 = *tmp937;
    const float tmp59 = (tmp56 - tmp58);
    auto tmp945 = ((tmp37 >> 0) & ((1 << 9) - 1));
    const int32_t tmp1021 = (tmp945 * tmp49);
    const int32_t tmp1022 = (tmp13 + tmp1021);
    S9_ch tmp948 = tmp932.children(tmp1022);
    device float* tmp949 = tmp948.get0().val;
    float tmp61 = *tmp949;
    const float tmp62 = (tmp59 - tmp61);
    auto tmp958 = ((tmp44 >> 0) & ((1 << 9) - 1));
    const int32_t tmp1030 = (tmp958 + tmp1005);
    S9_ch tmp960 = tmp932.children(tmp1030);
    device float* tmp961 = tmp960.get0().val;
    float tmp64 = *tmp961;
    const float tmp65 = (tmp62 - tmp64);
    auto tmp970 = ((tmp51 >> 0) & ((1 << 9) - 1));
    const int32_t tmp1038 = (tmp970 + tmp1005);
    S9_ch tmp972 = tmp932.children(tmp1038);
    device float* tmp973 = tmp972.get0().val;
    float tmp67 = *tmp973;
    const float tmp68 = (tmp65 - tmp67);
    const float tmp70 = (tmp68 * tmp69);
    const float tmp71 = *kernel_ctx_.arg0();
    const float tmp72 = (tmp71 * tmp70);
    const float tmp74 = (tmp73 - tmp71);
    S9_ch tmp984 = tmp932.children(tmp1006);
    device float* tmp985 = tmp984.get0().val;
    float tmp76 = *tmp985;
    const float tmp77 = (tmp74 * tmp76);
    const float tmp78 = (tmp72 + tmp77);
    *tmp985 = tmp78;
  } else {
  }
}

kernel void mtl_k0006_linear_solver_step_c14_0_0(
    device byte* root_addr [[buffer(0)]],
    device byte* global_tmps_addr [[buffer(1)]],
    device byte* ctx_addr [[buffer(2)]],
    device byte* runtime_addr [[buffer(3)]],
    device byte* print_addr [[buffer(4)]],
    const uint ugrid_size_ [[threads_per_grid]],
    const uint utid_ [[thread_position_in_grid]]) {
  // range_for, range known at compile time
  const int total_elems = 262144;
  const int begin_ = utid_ + 0;
  const int end_ = total_elems + 0;
  for (int ii = begin_; ii < end_; ii += ugrid_size_) {
    mtl_k0006_linear_solver_step_c14_0_0_func(root_addr, global_tmps_addr, ctx_addr, runtime_addr, print_addr, ii);
  }
}

///////// mtl_k0007_linear_solver_step_c14_1_0

class mtl_k0007_linear_solver_step_c14_1_args {
 public:
  explicit mtl_k0007_linear_solver_step_c14_1_args(device byte* addr) : addr_(addr) {}
  device float* arg0() {
    // scalar, size=4 B
    return (device float*)(addr_ + 0);
  }
  
  int32_t extra_arg(int i, int j) {
    device int32_t* base = (device int32_t*)(addr_ + 4);
    return *(base + (i * 8) + j);
  }
 private:
  device byte* addr_;
};

void mtl_k0007_linear_solver_step_c14_1_0_func(
    device byte* root_addr,
    device byte* global_tmps_addr,
    device byte* ctx_addr,
    device byte* runtime_addr,
    device byte* print_addr,
    const int linear_loop_idx_) {
  device Runtime *runtime_ = reinterpret_cast<device Runtime *>(runtime_addr);
  mtl_k0007_linear_solver_step_c14_1_args kernel_ctx_(ctx_addr);
  device RandState* rand_state_ = reinterpret_cast<device RandState*>(runtime_->rand_seeds + (linear_loop_idx_ % 65536));
  device auto* print_alloc_ = reinterpret_cast<device PrintMsgAllocator*>(print_addr);
  constexpr int32_t tmp1395 = 0;
  constexpr float tmp73 = 1.0;
  constexpr float tmp69 = -0.25;
  constexpr float tmp55 = 3.8146973e-06;
  constexpr int32_t tmp49 = 512;
  constexpr int32_t tmp45 = 511;
  constexpr int32_t tmp38 = 1;
  const int tmp11 = linear_loop_idx_;
  auto tmp12 = ((tmp11 >> 9) & ((1 << 9) - 1));
  auto tmp13 = ((tmp11 >> 0) & ((1 << 9) - 1));
  const int32_t tmp14 = (tmp12 + tmp13);
  constexpr int32_t tmp15 = 2;
  const int32_t tmp16 = ifloordiv(tmp14, tmp15);
  const int32_t tmp17 = (tmp16 + tmp16);
  const int32_t tmp18 = (tmp14 - tmp17);
  const int32_t tmp20 = -(tmp18 == tmp38);
  const int32_t tmp22 = (tmp20 & tmp38);
  if (tmp22) {
    const int32_t tmp25 = (tmp12 + tmp38);
    const int32_t tmp27 = ifloordiv(tmp25, tmp49);
    const int32_t tmp29 = (tmp27 * tmp49);
    const int32_t tmp30 = (tmp25 - tmp29);
    const int32_t tmp32 = (tmp12 + tmp45);
    const int32_t tmp34 = ifloordiv(tmp32, tmp49);
    const int32_t tmp36 = (tmp34 * tmp49);
    const int32_t tmp37 = (tmp32 - tmp36);
    const int32_t tmp39 = (tmp13 + tmp38);
    const int32_t tmp41 = ifloordiv(tmp39, tmp49);
    const int32_t tmp43 = (tmp41 * tmp49);
    const int32_t tmp44 = (tmp39 - tmp43);
    const int32_t tmp46 = (tmp13 + tmp45);
    const int32_t tmp48 = ifloordiv(tmp46, tmp49);
    const int32_t tmp50 = (tmp48 * tmp49);
    const int32_t tmp51 = (tmp46 - tmp50);
    S0 tmp1313(root_addr);
    S0_ch tmp1315 = tmp1313.children(tmp1395);
    S5 tmp1316 = tmp1315.get2();
    const int32_t tmp1401 = (tmp12 * tmp49);
    const int32_t tmp1402 = (tmp13 + tmp1401);
    S5_ch tmp1320 = tmp1316.children(tmp1402);
    device float* tmp1321 = tmp1320.get0().val;
    float tmp53 = *tmp1321;
    const float tmp54 = -(tmp53);
    const float tmp56 = (tmp54 * tmp55);
    S9 tmp1328 = tmp1315.get4();
    auto tmp1329 = ((tmp30 >> 0) & ((1 << 9) - 1));
    const int32_t tmp1409 = (tmp1329 * tmp49);
    const int32_t tmp1410 = (tmp13 + tmp1409);
    S9_ch tmp1332 = tmp1328.children(tmp1410);
    device float* tmp1333 = tmp1332.get0().val;
    float tmp58 = *tmp1333;
    const float tmp59 = (tmp56 - tmp58);
    auto tmp1341 = ((tmp37 >> 0) & ((1 << 9) - 1));
    const int32_t tmp1417 = (tmp1341 * tmp49);
    const int32_t tmp1418 = (tmp13 + tmp1417);
    S9_ch tmp1344 = tmp1328.children(tmp1418);
    device float* tmp1345 = tmp1344.get0().val;
    float tmp61 = *tmp1345;
    const float tmp62 = (tmp59 - tmp61);
    auto tmp1354 = ((tmp44 >> 0) & ((1 << 9) - 1));
    const int32_t tmp1426 = (tmp1354 + tmp1401);
    S9_ch tmp1356 = tmp1328.children(tmp1426);
    device float* tmp1357 = tmp1356.get0().val;
    float tmp64 = *tmp1357;
    const float tmp65 = (tmp62 - tmp64);
    auto tmp1366 = ((tmp51 >> 0) & ((1 << 9) - 1));
    const int32_t tmp1434 = (tmp1366 + tmp1401);
    S9_ch tmp1368 = tmp1328.children(tmp1434);
    device float* tmp1369 = tmp1368.get0().val;
    float tmp67 = *tmp1369;
    const float tmp68 = (tmp65 - tmp67);
    const float tmp70 = (tmp68 * tmp69);
    const float tmp71 = *kernel_ctx_.arg0();
    const float tmp72 = (tmp71 * tmp70);
    const float tmp74 = (tmp73 - tmp71);
    S9_ch tmp1380 = tmp1328.children(tmp1402);
    device float* tmp1381 = tmp1380.get0().val;
    float tmp76 = *tmp1381;
    const float tmp77 = (tmp74 * tmp76);
    const float tmp78 = (tmp72 + tmp77);
    *tmp1381 = tmp78;
  } else {
  }
}

kernel void mtl_k0007_linear_solver_step_c14_1_0(
    device byte* root_addr [[buffer(0)]],
    device byte* global_tmps_addr [[buffer(1)]],
    device byte* ctx_addr [[buffer(2)]],
    device byte* runtime_addr [[buffer(3)]],
    device byte* print_addr [[buffer(4)]],
    const uint ugrid_size_ [[threads_per_grid]],
    const uint utid_ [[thread_position_in_grid]]) {
  // range_for, range known at compile time
  const int total_elems = 262144;
  const int begin_ = utid_ + 0;
  const int end_ = total_elems + 0;
  for (int ii = begin_; ii < end_; ii += ugrid_size_) {
    mtl_k0007_linear_solver_step_c14_1_0_func(root_addr, global_tmps_addr, ctx_addr, runtime_addr, print_addr, ii);
  }
}
