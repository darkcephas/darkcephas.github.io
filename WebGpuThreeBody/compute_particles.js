
const WORKGROUP_SIZE = 256;
const SOFT_SCALE = 0.0003;
const COARSE_RANGE = 2;
const DELTA_T = 0.0000003;
var compute_pipe;
var compute_binding;

function setup_compute_particles(uniformBuffer, computeStorageBuffer) {

  const sortShaderModule = device.createShaderModule({
    label: "Particle index sort",
    code: `
        struct Particle {
           posi: vec2i,
           id: vec2u,
           posf: vec2f,
           vel: vec2f,
        };
        
        @group(0) @binding(0) var<uniform> canvas_size: vec2f;
        @group(0) @binding(1) var<storage, read_write> cellStateOut: array<Particle>;

        @compute @workgroup_size(${WORKGROUP_SIZE})
        fn main(  @builtin(local_invocation_index) local_idx:u32,
        @builtin(	workgroup_id) wg_id:vec3u) {
            // Three bodies at the same time.
            const delta_t = 0.001;
            const wg_size = ${WORKGROUP_SIZE};
            let idx = local_idx + (wg_size * wg_id.x);
            let part_start = idx * 3; // every 3 bodies
            var force_a : array<vec2f, 3>;
            for(var i = 0u; i < 3u; i++){
              for(var j = 0u; j < 3u; j++){
                if(i != j){
                  let diff = cellStateOut[part_start + i].posf -   cellStateOut[part_start + j].posf;
                  force_a[i] += - normalize(diff)/dot(diff,diff);
                }
              }
            }

            for(var i = 0u; i < 3u; i++){
              cellStateOut[part_start + i].posf = cellStateOut[part_start + i].posf + cellStateOut[i+part_start].vel * delta_t + delta_t*delta_t*0.5* force_a[i];
            }

          var force_b : array<vec2f, 3>;
           for(var i = 0u; i < 3u; i++){
              for(var j = 0u; j < 3u; j++){
                if(i != j){
                  let diff = cellStateOut[part_start + i].posf - cellStateOut[part_start + j].posf;
                  force_b[i] += - normalize(diff)/dot(diff,diff);
                }
              }
            }

            for(var i = 0u; i < 3u; i++){
              cellStateOut[part_start + i].vel = cellStateOut[part_start + i].vel +  delta_t*0.5* (force_a[i]+ force_b[i]);
            }
        }
      `
  });

  const  bindGroupLayout = device.createBindGroupLayout({
      label: "Sim",
      entries: [{
        binding: 0,
        visibility:   GPUShaderStage.COMPUTE ,
        buffer: {} // uniform
      }, {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" }
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
      module: sortShaderModule,
      entryPoint: "main",
    }
  });


  compute_binding = device.createBindGroup({
    label: "Sim",
    layout: bindGroupLayout,
    entries: [{
      binding: 0,
      resource: { buffer: uniformBuffer }
    }, {
      binding: 1, // New Entry
      resource: { buffer: computeStorageBuffer },
    }],
  });
}

function update_compute_particles(encoder, step) {
  const computePass = encoder.beginComputePass();
  computePass.setPipeline(compute_pipe);
  computePass.setBindGroup(0, compute_binding);
  const workgroupCount = Math.ceil(NUM_MICRO_SIMS / WORKGROUP_SIZE);
  computePass.dispatchWorkgroups(workgroupCount);
  computePass.end();
}

