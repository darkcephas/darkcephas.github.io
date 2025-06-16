
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
        fn computeMain(  @builtin(local_invocation_index) local_idx:u32,
        @builtin(	workgroup_id) wg_id:vec3u) {
            const wg_size = ${WORKGROUP_SIZE};
            let idx = local_idx + (wg_size * wg_id.x);
            let part_start = idx * 3; // every 3 bodies
            for(var i = 0u; i < 3u; i++){
              cellStateOut[part_start+i].posf = vec2f(0.0001f,0.0001f) +  cellStateOut[i+part_start].posf;
            }
        }
      `
  });


    // Create the bind group layout and pipeline layout.
  const  bindGroupLayout = device.createBindGroupLayout({
      label: "Cell Bind Group Layout",
      entries: [{
        binding: 0,
        visibility:   GPUShaderStage.COMPUTE ,
        buffer: {} // Grid uniform buffer
      }, {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" } // Cell state output buffer
      }]
    });
  
  
    const pipelineLayout = device.createPipelineLayout({
      label: "Cell Pipeline Layout",
      bindGroupLayouts: [bindGroupLayout],
    });

  // Create a compute pipeline that updates the game state.
  compute_pipe = device.createComputePipeline({
    label: "Simulation pipeline",
    layout: pipelineLayout,
    compute: {
      module: sortShaderModule,
      entryPoint: "computeMain",
    }
  });


  compute_binding = device.createBindGroup({
    label: "Compute renderer bind group A",
    layout: bindGroupLayout, // Updated Line
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

  // render out the stars to the buffer that will be then drawn using graphics pipe
  //encoder.clearBuffer(renderBufferStorage);
  const computePass = encoder.beginComputePass();
  computePass.setPipeline(compute_pipe);
  computePass.setBindGroup(0, compute_binding);
  const workgroupCount = Math.ceil(NUM_MICRO_SIMS / WORKGROUP_SIZE);
  computePass.dispatchWorkgroups(workgroupCount);
  computePass.end();


}

