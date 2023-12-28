var simulationShaderModule
const WORKGROUP_SIZE = 64;
var simulationPipeline;
var cellStateStorage;
var renderBufferStorage;

function setup_compute_particles(pipelineLayout) {
    
    simulationShaderModule = device.createShaderModule({
      label: "Compute simulation shader",
      code: `
        @group(0) @binding(0) var<uniform> grid: vec2f;

        struct Particle {
           pos: vec2i,
           vel: vec2f,
        };
        
        @group(0) @binding(1) var<storage> cellStateIn: array<Particle>;
        @group(0) @binding(2) var<storage, read_write> cellStateOut: array<Particle>;

        @compute @workgroup_size(${WORKGROUP_SIZE})
        fn computeMain(  @builtin(global_invocation_id) global_idx:vec3u,
        @builtin(num_workgroups) num_work:vec3u) {
          
          // Determine how many active neighbors this cell has.
          var my_pos = cellStateIn[global_idx.x].pos;
          var total_force = vec2f(0,0);
          for(var i = 0u; i < (num_work.x * u32(${WORKGROUP_SIZE})) ; i++)
          {
            if(i != global_idx.x){
              let soft_scale = 0.001;
              let vector_diff = my_pos - cellStateIn[i].pos;
              let as_float_vecf = vec2f(vector_diff)/ f32(256*256*256*64);
              var diff_length = length(as_float_vecf)+ soft_scale ;
              total_force += - (as_float_vecf) / (diff_length*diff_length*diff_length);
            }
          }

          let delta_t = 0.000002;
          let delta_v_as_int = vec2i( cellStateIn[global_idx.x].vel*delta_t * f32(256*256*256*64));
          cellStateOut[global_idx.x].pos = cellStateIn[global_idx.x].pos + delta_v_as_int;
          cellStateOut[global_idx.x].vel = cellStateIn[global_idx.x].vel + total_force*delta_t*0.05 ;
        }
      `
    });  

    renderBufferShaderModule = device.createShaderModule({
      label: "Render Buffer shader",
      code: `
        @group(0) @binding(0) var<uniform> grid: vec2f;

        struct Particle {
           pos: vec2i,
           vel: vec2f,
        };

        
        @group(0) @binding(1) var<storage> cellStateIn: array<Particle>;
        @group(0) @binding(2) var<storage, read_write> renderBufferOut: array<vec4f>;

        @compute @workgroup_size(${WORKGROUP_SIZE})
        fn computeMain(  @builtin(global_invocation_id) global_idx:vec3u,
        @builtin(num_workgroups) num_work:vec3u) {
          
          // Determine how many active neighbors this cell has.
          var my_pos = vec2f(cellStateIn[global_idx.x].pos) /  f32(256*256*256*64);
          var pixel_loc = ((my_pos+1)*0.5*512);
          var pixel_index = u32( pixel_loc.x)+  u32( pixel_loc.y) *512;
          renderBufferOut[pixel_index]= vec4(1,1,0,1);
        }
      `
    }); 

    // Create an array representing the active state of each cell.
    const cellStateArray = new Float32Array(NUM_PARTICLES_DIM * NUM_PARTICLES_DIM * 4);
    var as_int = new Int32Array(cellStateArray.buffer);
    // Create two storage buffers to hold the cell state.
    cellStateStorage = [
      device.createBuffer({
        label: "Cell State A",
        size: cellStateArray.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }),
      device.createBuffer({
        label: "Cell State B",
         size: cellStateArray.byteLength,
         usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      })
    ];
    // Mark every third cell of the first grid as active.

    for (let i = 0; i < cellStateArray.length; i+=4) {
      cellStateArray[i] =  Math.random() -0.5;
      cellStateArray[i+1] =  Math.random() -0.5;

      cellStateArray[i+2] =  cellStateArray[i+1]*25   + (Math.random() -0.5);
      cellStateArray[i+3] =- cellStateArray[i] *25 + (Math.random() -0.5);

      as_int[i] = cellStateArray[i] * (256*256*256*32);
      as_int[i+1] = cellStateArray[i+1] * (256*256*256*32);
    }
    device.queue.writeBuffer(cellStateStorage[0], 0, cellStateArray);
    device.queue.writeBuffer(cellStateStorage[1], 0, cellStateArray);
    // Create a bind group to pass the grid uniforms into the pipeline
    

    renderBufferStorage = 
      device.createBuffer({
        label: "render buffer A",
        size: 4 * 4 * canvas_width* canvas_height,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

     // Create a compute pipeline that updates the game state.
    simulationPipeline = device.createComputePipeline({
      label: "Simulation pipeline",
      layout: pipelineLayout,
      compute: {
        module: simulationShaderModule,
        entryPoint: "computeMain",
      }
    });

      // Create a compute pipeline that updates the game state.
      renderBufferPipeline = device.createComputePipeline({
      label: "Render pipeline",
      layout: pipelineLayout,
      compute: {
        module: renderBufferShaderModule,
        entryPoint: "computeMain",
      }
    });
}


function update_compute_particles(encoder,bindGroups, step)
{
  encoder.clearBuffer(renderBufferStorage);
  {
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(simulationPipeline);
    computePass.setBindGroup(0, simulationBindGroups[step % 2]);
    const workgroupCount = Math.ceil((NUM_PARTICLES_DIM* NUM_PARTICLES_DIM) / WORKGROUP_SIZE);
    computePass.dispatchWorkgroups(workgroupCount);
    computePass.end();
  }
  // render out the stars to the buffer that will be then drawn using graphics pipe
  {
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(renderBufferPipeline);
    computePass.setBindGroup(0, bindGroups[step % 2]);
    const workgroupCount = Math.ceil((NUM_PARTICLES_DIM* NUM_PARTICLES_DIM) / WORKGROUP_SIZE);
    computePass.dispatchWorkgroups(workgroupCount);
    computePass.end();
  }
}
    
