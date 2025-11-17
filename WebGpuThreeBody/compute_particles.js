

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
            let width_stride = u32(canvas_size.z);
            let idx = local_idx + (wg_size * wg_id.x);
            let part_start = idx * 3; // every 3 bodies


            var pos : array<vec2f, 3>;
            var vel : array<vec2f, 3>;
            for(var i = 0u; i < 3u; i++){
                pos[i] = cellStateOut[part_start + i].posf;
                vel[i] = cellStateOut[part_start + i].vel;
            }


            var min_dist = 100000.0;
            var min_dist_idx = 0u;

            for(var i = 0u; i < 3u; i++){
                  var dist = length(pos[i]-pos[(i+1u)%3u]);
                  if(dist < min_dist){
                     min_dist_idx = i;
                     min_dist = dist;
                  }
            }

            
            var offset_pos = pos[min_dist_idx];
            for(var i = 0u; i < 3u; i++){
               pos[i] = pos[i] - offset_pos;
            }

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
               pos[i] = pos[i] + offset_pos;
            }

            for(var i = 0u; i < 3u; i++){
                var as_float_temp = pos[i];
                var pix_pos = (pos[i] + vec2f(0.5,0.5)) * canvas_size.xy;

                cellStateOut[part_start + i].posf = as_float_temp;
                cellStateOut[part_start + i].vel = vel[i];

                var pix_pos_int = vec2u(pix_pos);
                var pix_pos_frac = fract(pix_pos);
                
                if( pix_pos.x > 0.0 && pix_pos.x < canvas_size.x
                   && pix_pos.y > 0.0 && pix_pos.y < canvas_size.y){
                    // bilinear filtering
                    var kSubPixelMult = 16.0;
                    var x0y0 = (1-pix_pos_frac.x)* (1-pix_pos_frac.y)*kSubPixelMult;
                    var x1y0 = (pix_pos_frac.x)* (1-pix_pos_frac.y)*kSubPixelMult;
                    var x0y1 = (1-pix_pos_frac.x)* (pix_pos_frac.y)*kSubPixelMult;
                    var x1y1 = (pix_pos_frac.x)* (pix_pos_frac.y)*kSubPixelMult;


                     atomicAdd(&vizBuff[pix_pos_int.x + pix_pos_int.y * width_stride].viz[i], u32(x0y0));
                     atomicAdd(&vizBuff[pix_pos_int.x + 1 + pix_pos_int.y * width_stride].viz[i], u32(x1y0));
                     atomicAdd(&vizBuff[pix_pos_int.x + (pix_pos_int.y +1) * width_stride].viz[i], u32(x0y1));
                     atomicAdd(&vizBuff[pix_pos_int.x + 1 + (pix_pos_int.y+1) * width_stride].viz[i], u32(x1y1));

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

            let width_stride = u32(canvas_size.z);
          
            let pix_pos = vec2u(local_idx + wg_id.x*wg_size, wg_id.y + (wg_id.z*256));
            if(pix_pos.x >= u32(canvas_size.x) || pix_pos.y >= u32(canvas_size.y)){
                // This can happen because rounding of workgroup size vs resolution
                return ;
            }
            
            let sample = vizBuff[pix_pos.x + pix_pos.y * width_stride];
            let white_color = vec4f(f32(sample.x), f32(sample.y), f32(sample.z), 1)/16.0;
   
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

