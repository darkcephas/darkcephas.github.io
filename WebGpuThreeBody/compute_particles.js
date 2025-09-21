

const SOFT_SCALE = 0.0003;
const COARSE_RANGE = 2;
const DELTA_T = 0.0000003;
var compute_pipe;
var compute_binding;
var bindGroupLayout;

function setup_compute_particles(uniformBuffer) {

  const simShaderModule = device.createShaderModule({
    label: "ParticleAvec",
    code: `
        struct Particle {
           posi: vec2i,
           id: vec2f,
           posf: vec2f,
           vel: vec2f,
        };


        struct VisStruct
        {
            viz: array<atomic<u32>, 4>
        };
        
        @group(0) @binding(0) var<uniform> canvas_size: vec4f;
        @group(0) @binding(1) var<storage, read_write> cellStateOut: array<Particle>;
        @group(0) @binding(2) var<storage, read_write> vizBuff: array<VisStruct>;
        @group(0) @binding(3) var frame_buffer: texture_storage_2d<${canvasformat}, write>;

        @compute @workgroup_size(${WORKGROUP_SIZE})
        fn main(  @builtin(local_invocation_index) local_idx:u32,
        @builtin(	workgroup_id) wg_id:vec3u) {
            // Three bodies at the same time.
            
            const wg_size = ${WORKGROUP_SIZE};
            const int_scale_canvas = i32 (${INT_SCALE_CANVAS} );
            const float_scale_canvas = f32(int_scale_canvas);
            let width_stride = u32(canvas_size.z);
            let idx = local_idx + (wg_size * wg_id.x);
            let part_start = idx * 3; // every 3 bodies
            var min_diff = 0x3fffffff;
            var best_int_vec = cellStateOut[part_start + 0].posi;
            for(var i = 0u; i < 3u; i++){
               let diff_pos =  cellStateOut[part_start + i].posi - cellStateOut[(part_start + i+1)%3].posi;
               let curr_diff = abs(diff_pos.x) + abs(diff_pos.y);
               if(min_diff > curr_diff){
                  min_diff = curr_diff;
                  best_int_vec = cellStateOut[part_start + i].posi;
               }
            }

            // HACK
            best_int_vec = vec2i(0,0);

            var pos : array<vec2f, 3>;
            var vel : array<vec2f, 3>;
            for(var i = 0u; i < 3u; i++){
                //cellStateOut[part_start + i].id = vec2f(f32(min_diff),f32(min_diff)/5.0);
                pos[i] = cellStateOut[part_start + i].posf / float_scale_canvas +  vec2f(cellStateOut[part_start + i].posi - best_int_vec)/ float_scale_canvas;
                vel[i] = cellStateOut[part_start + i].vel;
            }
            var min_dist =  min( min(length(pos[0]-pos[1]), length(pos[1]-pos[2])) , length(pos[2]-pos[0]));
            
            var num_iter =  1000u;
            var delta_t = 0.0002/f32(num_iter);
            for(var b =0u;b <num_iter;b++){
              var force_a : array<vec2f, 3>;
              for(var i = 0u; i < 3u; i++){
                for(var j = 0u; j < 3u; j++){
                  if(i != j){
                    let diff = pos[i] - pos[j];
                    force_a[i] += - normalize(diff)/dot(diff,diff);
                  }
                }
              }

              for(var i = 0u; i < 3u; i++){
                pos[i] = pos[i] + vel[i] * delta_t + delta_t*delta_t*0.5* force_a[i];
              }

              var force_b : array<vec2f, 3>;
              for(var i = 0u; i < 3u; i++){
                for(var j = 0u; j < 3u; j++){
                  if(i != j){
                    let diff = pos[i] - pos[j];
                    force_b[i] += - normalize(diff)/dot(diff,diff);
                  }
                }
              }

              for(var i = 0u; i < 3u; i++){
                vel[i] = vel[i] +  delta_t * 0.5 * (force_a[i]+ force_b[i]);
              }
            }

            for(var i = 0u; i < 3u; i++){
                var as_int_temp = vec2i(pos[i] * float_scale_canvas);
                var as_float_temp = pos[i]* float_scale_canvas - vec2f(as_int_temp);
                var pix_pos = (pos[i]* float_scale_canvas + vec2f(0.5,0.5)) * canvas_size.xy;

                cellStateOut[part_start + i].posi = best_int_vec + as_int_temp;
                cellStateOut[part_start + i].posf = as_float_temp;
                cellStateOut[part_start + i].vel = vel[i];

                if( pix_pos.x > 0.0 && pix_pos.x < canvas_size.x
                   && pix_pos.y > 0.0 && pix_pos.y < canvas_size.y){
                     atomicAdd(&vizBuff[u32(pix_pos.x) + u32(pix_pos.y)* width_stride].viz[i], 1);
                }


        
               
            }

        }
      `
  });


  const drawShaderModule = device.createShaderModule({
    label: "ParticleAvec",
    code: `
        struct Particle {
           posi: vec2i,
           id: vec2f,
           posf: vec2f,
           vel: vec2f,
        };

        @group(0) @binding(0) var<uniform> canvas_size: vec4f;
        @group(0) @binding(1) var<storage, read_write> cellStateOut: array<Particle>;
        @group(0) @binding(2) var<storage, read_write> vizBuff: array<vec4u>;
        @group(0) @binding(3) var frame_buffer: texture_storage_2d<${canvasformat}, write>;

        @compute @workgroup_size(${WORKGROUP_SIZE})
        fn main(  @builtin(local_invocation_index) local_idx:u32,
        @builtin(	workgroup_id) wg_id:vec3u) {
            // Three bodies at the same time.
            const wg_size = ${WORKGROUP_SIZE};
            const int_scale_canvas = i32 (${INT_SCALE_CANVAS} );
            const float_scale_canvas = f32(int_scale_canvas);
            let width_stride = u32(canvas_size.z);
          
            let pix_pos = vec2u(local_idx + wg_id.x*wg_size, wg_id.y + (wg_id.z*256));
            if(pix_pos.x >= u32(canvas_size.x) || pix_pos.y >= u32(canvas_size.y)){
                // This can happen because rounding of workgroup size vs resolution
                return ;
            }
            
            let sample = vizBuff[pix_pos.x + pix_pos.y * width_stride];
            let white_color = vec4f(f32(sample.x), f32(sample.y), f32(sample.z), 1);
   
            textureStore(frame_buffer, pix_pos , white_color);
        }
      `
  });

  bindGroupLayout = device.createBindGroupLayout({
    label: "Sim",
    entries: [{
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {} // uniform
    }, {
      binding: 1,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage" }
    }, {
      binding: 2,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage" }
    }, {
      binding: 3,
      visibility: GPUShaderStage.COMPUTE,
      storageTexture: { access: "write-only", format: canvasformat }
    }]
  });


  const pipelineLayout = device.createPipelineLayout({
    label: "Sim",
    bindGroupLayouts: [bindGroupLayout],
  });

  compute_pipe = device.createComputePipeline({
    label: "Sim",
    layout: pipelineLayout,
    compute: {
      module: simShaderModule,
    }
  });


  draw_pipe = device.createComputePipeline({
    label: "Draw",
    layout: pipelineLayout,
    compute: {
      module: drawShaderModule,
    }
  });


}

function update_compute_particles(computeStorageBuffer, vizBufferStorage, encoder, step) {
  encoder.clearBuffer(vizBufferStorage);
  const computePass = encoder.beginComputePass();
  computePass.setPipeline(compute_pipe);
  let res_view = context.getCurrentTexture().createView();
  compute_binding = device.createBindGroup({
    label: "GlobalBind",
    layout: bindGroupLayout,
    entries: [{
      binding: 0,
      resource: { buffer: uniformBuffer }
    }, {
      binding: 1, // New Entry
      resource: { buffer: computeStorageBuffer },
    }, {
      binding: 2, // New Entry
      resource: { buffer: vizBufferStorage },
    },
    { binding: 3, resource: res_view },],
  });
  computePass.setBindGroup(0, compute_binding);
  const workgroupCount = Math.ceil(NUM_MICRO_SIMS / WORKGROUP_SIZE);
  computePass.dispatchWorkgroups(workgroupCount);
  computePass.setPipeline(draw_pipe);
  computePass.setBindGroup(0, compute_binding);
  const numWorkgroupPerRow = Math.floor(canvas_width_stride / WORKGROUP_SIZE);
  const numWorkgroupZ = Math.ceil(canvas_height / 256);
  const numWorkgroupY = numWorkgroupZ > 1 ? 256  : Math.ceil(canvas_height);
  computePass.dispatchWorkgroups(numWorkgroupPerRow, numWorkgroupY, numWorkgroupZ);
  computePass.end();
}

