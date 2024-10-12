"use strict";


const WORKGROUP_SIZE = 1;

const shaderCode = `
struct CommonData {
      outlen:  u32,       /* available space at out */
     inlen: u32,    /* available input at in */
};

@group(0) @binding(0) var<storage> in: array<u32>;
  @group(0) @binding(1) var<storage,read_write> out: array<u32>;
 @group(0) @binding(2) var<uniform> ws: CommonData;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn computeMain(  @builtin(global_invocation_id) global_idx:vec3u,
@builtin(num_workgroups) num_work:vec3u) {
  for(var i = 0u ;i < ws.inlen;i++){
   out[i]= in[i];
  }
}
`;