const WORKGROUP_SIZE = 1024u;

override DISPATCH_COUNT:u32;
override NUM_ELEMENTS:u32;
override DUAL_DISPATCH:bool;

struct CommonData {
      outlen:  u32,       /* available space at out */
     inlen: u32,    /* available input at in */
};


@group(0) @binding(0) var<storage> in: array<u32>;
@group(0) @binding(1) var<storage,read_write> out: array<u32>;
@group(0) @binding(2) var<uniform> unidata: CommonData;
@group(0) @binding(3) var<storage,read_write> debug: array<u32>;
// Data for decoded -> store phase


@compute @workgroup_size(WORKGROUP_SIZE)
fn computeMain(  @builtin(workgroup_id) workgroup_id:vec3u,
 @builtin(local_invocation_index) local_invocation_index: u32,
@builtin(num_workgroups) num_work:vec3u) {
  let kNumElementsSrc = NUM_ELEMENTS;
  var wg_each = kNumElementsSrc/ DISPATCH_COUNT ;
  if(DUAL_DISPATCH){
    wg_each =kNumElementsSrc;
  }
  let wg_start = (0u + workgroup_id.x ) * wg_each;
  let wg_end =  (1u + workgroup_id.x )  * wg_each;
  if(DUAL_DISPATCH && workgroup_id.x != 0u){
    var total = 0u;
    for(var i = wg_start; i < wg_end;) {
     total += in[i+local_invocation_index];
      total += out[i+local_invocation_index];
      i += WORKGROUP_SIZE;
    }
    debug[1] = total;
    return;
  }

  for(var i = wg_start; i < wg_end;) {
    out[i+local_invocation_index]= in[i+local_invocation_index];
    i += WORKGROUP_SIZE;
  }
  debug[0] = 777;
}
