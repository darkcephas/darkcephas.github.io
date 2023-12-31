var simulationShaderModule
const WORKGROUP_SIZE = 64;
var simulationPipeline;
var cellStateStorage;
var renderBufferStorage;
var sortPipeline;
var bindGroupUniformOffset;
var massAssignBufferStorage;

function setup_compute_particles(pipelineLayout) {
    
    simulationShaderModule = device.createShaderModule({
      label: "Compute simulation shader",
      code: `
        @group(0) @binding(0) var<uniform> canvas_size: vec2f;

        struct Particle {
           pos: vec2i,
           vel: vec2f,
           id: vec2u,
        };
        
        @group(0) @binding(1) var<storage> not_used: array<Particle>;
        @group(0) @binding(2) var<storage, read_write> cellStateOut: array<Particle>;

      

        @compute @workgroup_size(${WORKGROUP_SIZE})
        fn computeMain(  @builtin(global_invocation_id) global_idx:vec3u,
        @builtin(num_workgroups) num_work:vec3u) {
          
          // Determine how many active neighbors this cell has.
          var my_pos = cellStateOut[global_idx.x].pos;
          var total_force = vec2f(0,0);
          for(var i = 0u; i < (num_work.x * u32(${WORKGROUP_SIZE})) ; i++)
          {
            if(i != global_idx.x){
              let soft_scale = 0.001;
              let vector_diff = my_pos - cellStateOut[i].pos;
              let as_float_vecf = vec2f(vector_diff)/ f32(256*256*256*64);
              var diff_length = length(as_float_vecf)+ soft_scale ;
              total_force += - (as_float_vecf) / (diff_length*diff_length*diff_length);
            }
          }

          let delta_t = 0.00002;
          let delta_v_as_int = vec2i( cellStateOut[global_idx.x].vel*delta_t * f32(256*256*256*64));
          cellStateOut[global_idx.x].pos = cellStateOut[global_idx.x].pos + delta_v_as_int;
          cellStateOut[global_idx.x].vel = cellStateOut[global_idx.x].vel + total_force*delta_t*0.02 ;
          cellStateOut[global_idx.x].id = vec2u((cellStateOut[global_idx.x].pos/i32(256*256*256))+63);
        }
      `
    });  

    renderBufferShaderModule = device.createShaderModule({
      label: "Render Buffer shader",
      code: `
        @group(0) @binding(0) var<uniform> canvas_size: vec2f;

        struct Particle {
           pos: vec2i,
           vel: vec2f,
           id: vec2u,
        };

        
        @group(0) @binding(1) var<storage> cellStateIn: array<Particle>;
        @group(0) @binding(2) var<storage, read_write> renderBufferOut: array<vec4f>;

        @compute @workgroup_size(${WORKGROUP_SIZE})
        fn computeMain(  @builtin(global_invocation_id) global_idx:vec3u,
        @builtin(num_workgroups) num_work:vec3u) {
          
          // Determine how many active neighbors this cell has.
          var my_pos = vec2f(cellStateIn[global_idx.x].pos) /  f32(256*256*256*64);
          // my pos will be -1,1 viewport in normalized
          var pixel_loc = ((my_pos+1)*0.5*canvas_size);
          var pixel_index = u32( pixel_loc.x)+  u32( pixel_loc.y) * u32(canvas_size.x);
          let linear_y  = f32(cellStateIn[global_idx.x].id.y)/128.0;
          let linear_x  = f32(cellStateIn[global_idx.x].id.x)/128.0;
          renderBufferOut[pixel_index]= vec4( 1-linear_y,linear_y,linear_x,1);
        }
      `
    }); 


    sortShaderModule = device.createShaderModule({
      label: "Particle index sort",
      code: `
        @group(0) @binding(0) var<uniform> canvas_size: vec2f;

        struct Particle {
           pos: vec2i,
           vel: vec2f,
           id: vec2u,
        };
        
        @group(0) @binding(1) var<storage> cellStateIn: array<Particle>;
        @group(0) @binding(2) var<storage, read_write> cellStateOut: array<Particle>;

        @group(1) @binding(0) var<uniform> offsets: vec4u;

        @compute @workgroup_size(${WORKGROUP_SIZE})
        fn computeMain(  @builtin(global_invocation_id) global_idx:vec3u,
        @builtin(num_workgroups) num_work:vec3u) {
          
          let a_idx = (global_idx.x/offsets.x)*(2*offsets.x) + (global_idx.x % offsets.x) + offsets.y ;
          let b_idx = a_idx + offsets.x;

          if(a_idx < offsets.z && b_idx < offsets.z)
          {
            var pos_a =  cellStateOut[a_idx].id.x + cellStateOut[a_idx].id.y *128;
            var pos_b =  cellStateOut[b_idx].id.x + cellStateOut[b_idx].id.y *128;
           
            if(pos_a > pos_b){
              var particle_saved = cellStateOut[a_idx];
              cellStateOut[a_idx] = cellStateOut[b_idx];
              cellStateOut[b_idx] = particle_saved;
            }
          }

        }
      `
    });  

    massAssignShaderModule = device.createShaderModule({
      label: "MassAssignment shader",
      code: `
        @group(0) @binding(0) var<uniform> canvas_size: vec2f;

        struct Particle {
           pos: vec2i,
           vel: vec2f,
           id: vec2u,
        };
        
        @group(0) @binding(1) var<storage> cellStateIn: array<Particle>;
        @group(0) @binding(2) var<storage, read_write> mass_array: array<vec3u>;
        @group(1) @binding(0) var<uniform> offsets: vec4u;
        @compute @workgroup_size(${WORKGROUP_SIZE})
        fn computeMain(  @builtin(global_invocation_id) global_idx:vec3u,
        @builtin(num_workgroups) num_work:vec3u) {
          
          let sel_x = global_idx.x % 128;
          let sel_y = global_idx.x / 128;
          var local_max = 0u;
          var local_min = 0xFFFFFFFFu;
          var total_mass = 0u;
          for(var i = 0u; i < (offsets.z) ; i++)
          {
            if(cellStateIn[i].id.x == sel_x && cellStateIn[i].id.y == sel_y)
            {
              local_max = max(local_max, i);
              local_min = min(local_min, i);
              total_mass++;
            }
          }
          mass_array[global_idx.x].x = total_mass; 
          mass_array[global_idx.x].y = local_min; 
          mass_array[global_idx.x].z = local_max; 
        }
      `
    }); 

    // Create an array representing the active state of each cell.
    const cellStateArray = new Float32Array(NUM_PARTICLES_DIM * NUM_PARTICLES_DIM * 6);
    var as_int = new Int32Array(cellStateArray.buffer);
    // Create two storage buffers to hold the cell state.
    cellStateStorage = 
      device.createBuffer({
        label: "Cell State A",
        size: cellStateArray.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
    // Mark every third cell of the first grid as active.

    for (let i = 0; i < cellStateArray.length; i+=6) {
      cellStateArray[i] =  Math.random() -0.5;
      cellStateArray[i+1] =  Math.random() -0.5;

      cellStateArray[i+2] =  cellStateArray[i+1]*25   + (Math.random() -0.5);
      cellStateArray[i+3] =- cellStateArray[i] *25 + (Math.random() -0.5);

      as_int[i] = cellStateArray[i] * (256*256*256*64);
      as_int[i+1] = cellStateArray[i+1] * (256*256*256*64);
    }
    device.queue.writeBuffer(cellStateStorage, 0, cellStateArray);

    // Create a bind group to pass the grid uniforms into the pipeline
    

    renderBufferStorage = 
      device.createBuffer({
        label: "render buffer A",
        size: 4 * 4 * canvas_width* canvas_height,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

     massAssignBufferStorage = 
      device.createBuffer({
        label: "mass assign storage",
        size: 4 * 4 * 128* 128,
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

      // Create a uniform buffer that describes the grid.
      const uniformOffsetsArray00 = new Int32Array([1, 0, (NUM_PARTICLES_DIM* NUM_PARTICLES_DIM),0]);
      const uniformOffsetsArray01 = new Int32Array([1, 1, (NUM_PARTICLES_DIM* NUM_PARTICLES_DIM),0]);
      const uniformOffsetsArray63_1 = new Int32Array([63, 0, (NUM_PARTICLES_DIM* NUM_PARTICLES_DIM),0]);
      const uniformOffsetsArray255_1 = new Int32Array([255, 0, (NUM_PARTICLES_DIM* NUM_PARTICLES_DIM),0]);
      let uniformOffsetsBuffers = [device.createBuffer({
        label: "Offsets 1 0",
        size: uniformOffsetsArray00.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
      device.createBuffer({
        label: "Offsets 1 1",
        size: uniformOffsetsArray01.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
      device.createBuffer({
        label: "Offsets 63 0",
        size: uniformOffsetsArray63_1.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
      device.createBuffer({
        label: "Offsets 255 0",
        size: uniformOffsetsArray255_1.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      })];
      device.queue.writeBuffer(uniformOffsetsBuffers[0], 0, uniformOffsetsArray00);
      device.queue.writeBuffer(uniformOffsetsBuffers[1], 0, uniformOffsetsArray01);
      device.queue.writeBuffer(uniformOffsetsBuffers[2], 0, uniformOffsetsArray63_1);
      device.queue.writeBuffer(uniformOffsetsBuffers[3], 0, uniformOffsetsArray255_1);
      
      let perPassBindGroupLayout = device.createBindGroupLayout({
        label: "per pass Bind Group Layout",
        entries: [{
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE |  GPUShaderStage.FRAGMENT,
          buffer: {} // Grid uniform buffer
        }, ]
      });

      bindGroupUniformOffset = [
        device.createBindGroup({
          label: "Offsets 0",
          layout: perPassBindGroupLayout, 
          entries: [{
            binding: 0,
            resource: { buffer: uniformOffsetsBuffers[0] }
          },],
        }),
        device.createBindGroup({
          label: "Offsets 1",
          layout: perPassBindGroupLayout, 
          entries: [{
            binding: 0,
            resource: { buffer: uniformOffsetsBuffers[1] }
          },],
        }),
        device.createBindGroup({
          label: "Offsets 2",
          layout: perPassBindGroupLayout, 
          entries: [{
            binding: 0,
            resource: { buffer: uniformOffsetsBuffers[2] }
          },],
        }),
        device.createBindGroup({
          label: "Offsets 3",
          layout: perPassBindGroupLayout, 
          entries: [{
            binding: 0,
            resource: { buffer: uniformOffsetsBuffers[3] }
          },],
        })]


      // Create a compute pipeline that updates the game state.
      renderBufferPipeline = device.createComputePipeline({
        label: "Render pipeline",
        layout: pipelineLayout,
        compute: {
          module: renderBufferShaderModule,
          entryPoint: "computeMain",
        }
      });

        let bindGroupLayout2 = device.createBindGroupLayout({
          label: "Bind group for uniforms",
          entries: [{
            binding: 0,
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE |  GPUShaderStage.FRAGMENT,
            buffer: {} // Grid uniform buffer
          }, ]
        });

        const pipelineLayout2 = device.createPipelineLayout({
          label: "Cell Pipeline Layout",
          bindGroupLayouts: [ bindGroupLayout, bindGroupLayout2 ],
        });
        // Create a compute pipeline that updates the game state.
    sortPipeline = device.createComputePipeline({
        label: "Particle sort pipeline",
        layout: pipelineLayout2,
        compute: {
          module: sortShaderModule,
          entryPoint: "computeMain",
        }
      });

      massAssignPipeline = device.createComputePipeline({
        label: "Mass assign pipeline",
        layout: pipelineLayout2,
        compute: {
          module: massAssignShaderModule,
          entryPoint: "computeMain",
        }
      });
}


function update_compute_particles(encoder,bindGroups, step)
{
  {
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(simulationPipeline);
    computePass.setBindGroup(0, simulationBindGroups);
    const workgroupCount = Math.ceil((NUM_PARTICLES_DIM* NUM_PARTICLES_DIM) / WORKGROUP_SIZE);
    computePass.dispatchWorkgroups(workgroupCount);
    computePass.end();
  }
  // render out the stars to the buffer that will be then drawn using graphics pipe
  encoder.clearBuffer(renderBufferStorage);
  {
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(renderBufferPipeline);
    computePass.setBindGroup(0, bindGroups[0]);
    const workgroupCount = Math.ceil((NUM_PARTICLES_DIM* NUM_PARTICLES_DIM) / WORKGROUP_SIZE);
    computePass.dispatchWorkgroups(workgroupCount);
    computePass.end();
  }
  
  for (let i = 0; i < 10; i++) {
  {
      const computePass = encoder.beginComputePass();
      computePass.setPipeline(sortPipeline);
      computePass.setBindGroup(0, simulationBindGroups);
      computePass.setBindGroup(1, bindGroupUniformOffset[3]);
      const workgroupCount = Math.ceil((NUM_PARTICLES_DIM* NUM_PARTICLES_DIM) / WORKGROUP_SIZE);
      computePass.dispatchWorkgroups(workgroupCount);
      computePass.end();
    }
    {
      const computePass = encoder.beginComputePass();
      computePass.setPipeline(sortPipeline);
      computePass.setBindGroup(0, simulationBindGroups);
      computePass.setBindGroup(1, bindGroupUniformOffset[2]);
      const workgroupCount = Math.ceil((NUM_PARTICLES_DIM* NUM_PARTICLES_DIM) / WORKGROUP_SIZE);
      computePass.dispatchWorkgroups(workgroupCount);
      computePass.end();
    }
    {
      const computePass = encoder.beginComputePass();
      computePass.setPipeline(sortPipeline);
      computePass.setBindGroup(0, simulationBindGroups);
      computePass.setBindGroup(1, bindGroupUniformOffset[0]);
      const workgroupCount = Math.ceil((NUM_PARTICLES_DIM* NUM_PARTICLES_DIM) / WORKGROUP_SIZE);
      computePass.dispatchWorkgroups(workgroupCount);
      computePass.end();
    }
    {
      const computePass = encoder.beginComputePass();
      computePass.setPipeline(sortPipeline);
      computePass.setBindGroup(0, simulationBindGroups);
      computePass.setBindGroup(1, bindGroupUniformOffset[1]);
      const workgroupCount = Math.ceil((NUM_PARTICLES_DIM* NUM_PARTICLES_DIM) / WORKGROUP_SIZE);
      computePass.dispatchWorkgroups(workgroupCount);
      computePass.end();
    }
  }
  encoder.clearBuffer(massAssignBufferStorage);
  {
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(massAssignPipeline);
    computePass.setBindGroup(0, massAssignBindGroups);
    computePass.setBindGroup(1, bindGroupUniformOffset[0]);
    const workgroupCount = Math.ceil((128*128) / WORKGROUP_SIZE);
    computePass.dispatchWorkgroups(workgroupCount);
    computePass.end();
  }
}
    
