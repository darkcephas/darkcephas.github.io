var simulationShaderModule
const WORKGROUP_SIZE = 256;
const SOFT_SCALE =  0.0003;
const COARSE_RANGE = 2;
const DELTA_T = 0.0000003;
var simulationPipeline;
var cellStateStorage;
var renderBufferStorage;
var sortPipeline;
var bindGroupUniformOffset;
var massAssignBufferStorage;
var forceIndexBufferStorage;

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

          let delta_t = 0.000002;
          let delta_v_as_int = vec2i( cellStateOut[global_idx.x].vel*delta_t * f32(256*256*256*64));
          cellStateOut[global_idx.x].pos = cellStateOut[global_idx.x].pos + delta_v_as_int;
          cellStateOut[global_idx.x].vel = cellStateOut[global_idx.x].vel + total_force*delta_t*0.02 ;
          cellStateOut[global_idx.x].id = vec2u(((cellStateOut[global_idx.x].pos + i32(256*256*256*63))
                           /i32(256*256*256)));
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
        @group(0) @binding(2) var<storage, read_write> renderBufferOut: array<atomic<u32>>;

        @compute @workgroup_size(${WORKGROUP_SIZE})
        fn computeMain(  @builtin(global_invocation_id) global_idx:vec3u,
        @builtin(num_workgroups) num_work:vec3u) {
          
          if(cellStateIn[global_idx.x].vel.x == 1000000.0)
          {
             return;
          }

          // Determine how many active neighbors this cell has.
          var my_pos = vec2f(cellStateIn[global_idx.x].pos) /  f32(256*256*256*64);
          // my pos will be -1,1 viewport in normalized
          var pixel_loc = ((my_pos+1)*0.5*canvas_size);
          var pixel_frac = fract(pixel_loc);
          var pixel_frac_m1 = vec2f(1,1) - pixel_frac;
          let int_mult = 256.0;
          {
            var pixel_index = u32( pixel_loc.x)+  u32( pixel_loc.y) * u32(canvas_size.x);
            atomicAdd(&renderBufferOut[pixel_index], u32(pixel_frac_m1.x *pixel_frac_m1.y *int_mult));
          }
          {
            var pixel_index = u32( pixel_loc.x+1.0)+  u32( pixel_loc.y) * u32(canvas_size.x);
            atomicAdd(&renderBufferOut[pixel_index], u32(pixel_frac.x *pixel_frac_m1.y *int_mult));
          }
          {
            var pixel_index = u32( pixel_loc.x)+  u32( pixel_loc.y+1.0) * u32(canvas_size.x);
            atomicAdd(&renderBufferOut[pixel_index], u32(pixel_frac_m1.x *pixel_frac.y *int_mult));
          }
          {
              var pixel_index = u32( pixel_loc.x+1.0)+  u32( pixel_loc.y+1.0) * u32(canvas_size.x);
              atomicAdd(&renderBufferOut[pixel_index], u32(pixel_frac.x *pixel_frac.y *int_mult));
          }
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
        
        @group(0) @binding(1) var<storage> not_used: array<Particle>;
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
        @group(0) @binding(2) var<storage, read_write> mass_array: array<vec4u>;
        @group(1) @binding(0) var<uniform> offsets: vec4u;
        @compute @workgroup_size(${WORKGROUP_SIZE})
        fn computeMain(@builtin(global_invocation_id) global_idx:vec3u,
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


    forceIndexShaderModule = device.createShaderModule({
      label: "Force Index shader",
      code: `
        @group(0) @binding(0) var<uniform> canvas_size: vec2f;


        @group(0) @binding(1) var<storage> massAssign: array<vec4u>;
        @group(0) @binding(2) var<storage, read_write> forceIndex: array<vec4f>;
        @group(1) @binding(0) var<uniform> offsets: vec4u;
        @compute @workgroup_size(${WORKGROUP_SIZE})
        fn computeMain(  @builtin(global_invocation_id) global_idx:vec3u,
        @builtin(num_workgroups) num_work:vec3u) {
          
          let massIdx = global_idx.x;
          // Determine how many active neighbors this cell has.
          var coarse_id = vec2i(i32(massIdx % 128), i32(massIdx / 128));
          var total_force = vec2f(0,0);
          let soft_scale = ${SOFT_SCALE};
          for(var i = 0u; i < 128 ; i++) {
            for(var j = 0u; j < 128 ; j++){
              let massSample = massAssign[i+j*128];
              let sample_id = vec2u(i,j);
              let sample_id_diff  = vec2i(sample_id)- vec2i(coarse_id);
              let accept_diff = ${COARSE_RANGE};
              let within_x = sample_id_diff.x <=accept_diff && sample_id_diff.x >= -accept_diff;
              let within_y = sample_id_diff.y <=accept_diff && sample_id_diff.y >= -accept_diff;
              if( !within_x || !within_y )
              {
                let vector_diff =  (vec2f(coarse_id) -  vec2f(sample_id));
                let as_float_vecf = vec2f(vector_diff)/ f32(64);
                var diff_length = length(as_float_vecf)+ soft_scale ;
                total_force += -(f32(massSample.x) * as_float_vecf) / (diff_length*diff_length*diff_length);
              }
            }
          }

          // update the coarse grained location for next pass
          var massingIndex = vec2f(-1,-1);
          if(massAssign[massIdx].x > 0)
          {
            massingIndex = vec2f(massAssign[massIdx].yz);
          }
          forceIndex[massIdx] = vec4f(massingIndex,total_force);
        }
      `
    }); 


    optsimShaderModule = device.createShaderModule({
      label: "Opt Compute simulation shader",
      code: `
        @group(0) @binding(0) var<uniform> canvas_size: vec2f;

        struct Particle {
           pos: vec2i,
           vel: vec2f,
           id: vec2u,
        };
        
        @group(0) @binding(1) var<storage> mass_assign: array<vec4f>;
        @group(0) @binding(2) var<storage, read_write> particleArray: array<Particle>;

      

        @compute @workgroup_size(${WORKGROUP_SIZE})
        fn computeMain(  @builtin(global_invocation_id) global_idx:vec3u,
        @builtin(num_workgroups) num_work:vec3u) {
          
          let partIdx = global_idx.x;
          if( particleArray[partIdx].vel.x== 1000000.0)
          {
            return;
          }

          // Determine how many active neighbors this cell has.
          var my_pos = particleArray[partIdx].pos;
          var coarse_id = vec2i(particleArray[partIdx].id);
          var total_force = vec2f(0,0);
        
          total_force += mass_assign[coarse_id.x + coarse_id.y *128].zw;

          let vec_coarse_range = vec2i( ${COARSE_RANGE}, ${COARSE_RANGE});
          let coarse_min = max(coarse_id - vec_coarse_range, vec2i(0,0));
          let coarse_max = min(coarse_id + vec_coarse_range, vec2i(127,127));

          for(var i = coarse_min.x; i <= coarse_max.x ; i++)
          {
            for(var j = coarse_min.y; j <= coarse_max.y ; j++)
            {
               let massSample = mass_assign[i+j*128];
                for(var k = u32(massSample.x); k <=u32(massSample.y) ; k++)
                {
                  if(k != partIdx){
                    let vector_diff = my_pos - particleArray[k].pos;
                    let as_float_vecf = vec2f(vector_diff)/ f32(256*256*256*64);
                    let soft_scale = ${SOFT_SCALE};
                    var diff_length = length(as_float_vecf)+ soft_scale ;
                    total_force += - (as_float_vecf) / (diff_length*diff_length*diff_length);
                  }
                }      
            }
          }

          let delta_t = ${DELTA_T};
          let delta_v_with_t_as_int = vec2i( particleArray[partIdx].vel*delta_t * f32(256*256*256*64));
          let force_mult = 0.03;
          particleArray[partIdx].pos = particleArray[partIdx].pos + delta_v_with_t_as_int;
          particleArray[partIdx].vel = particleArray[partIdx].vel + total_force*delta_t* force_mult;

          // update the coarse grained location for next pass
          particleArray[partIdx].id = vec2u(((particleArray[partIdx].pos + i32(256*256*256*63)) /i32(256*256*256)));
           
          if(particleArray[partIdx].pos.x <=  -i32(256*256*256*62) ||
            particleArray[partIdx].pos.x >=  i32(256*256*256*63) ||
            particleArray[partIdx].pos.y <=  -i32(256*256*256*62) ||
            particleArray[partIdx].pos.y >=  i32(256*256*256*63))
          {
            particleArray[partIdx].vel = vec2f(1000000.0,1000000.0);
          }

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
      cellStateArray[i+1] = (i/6)/ (NUM_PARTICLES_DIM*NUM_PARTICLES_DIM)-0.5;

      cellStateArray[i+2] =  cellStateArray[i+1]*30  +Math.random() -0.5;
      cellStateArray[i+3] =- cellStateArray[i] *30 +Math.random() -0.5;

      as_int[i] = cellStateArray[i] * (256*256*256*64);
      as_int[i+1] = cellStateArray[i+1] * (256*256*256*64);
    }
    device.queue.writeBuffer(cellStateStorage, 0, cellStateArray);

    // Create a bind group to pass the grid uniforms into the pipeline
    

    renderBufferStorage = 
      device.createBuffer({
        label: "render buffer A",
        size: 4 * canvas_width* canvas_height,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

     massAssignBufferStorage = 
      device.createBuffer({
        label: "mass assign storage",
        size: 4 * 4 * 128* 128,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

      forceIndexBufferStorage = 
      device.createBuffer({
        label: "Force index buffer storage",
        size: 4 * 4 * 128* 128,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });


     // Create a compute pipeline that updates the game state.
    simulationPipeline = device.createComputePipeline({
      label: "Simulation pipeline",
      layout: pipelineLayout,
      compute: {
        module: optsimShaderModule,
        entryPoint: "computeMain",
      }
    });

      // Create a uniform buffer that describes the grid.
      const uniformOffsetsArray00 = new Int32Array([1, 0, (NUM_PARTICLES_DIM* NUM_PARTICLES_DIM),0]);
      const uniformOffsetsArray01 = new Int32Array([1, 1, (NUM_PARTICLES_DIM* NUM_PARTICLES_DIM),0]);
      const uniformOffsetsArray63_1 = new Int32Array([63, 0, (NUM_PARTICLES_DIM* NUM_PARTICLES_DIM),0]);
      const uniformOffsetsArray255_1 = new Int32Array([255, 0, (NUM_PARTICLES_DIM* NUM_PARTICLES_DIM),0]);
      const uniformOffsetsArray1k_1 = new Int32Array([1023, 0, (NUM_PARTICLES_DIM* NUM_PARTICLES_DIM),0]);
      const uniformOffsetsArray4k_1 = new Int32Array([4095, 0, (NUM_PARTICLES_DIM* NUM_PARTICLES_DIM),0]);
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
      }),
      device.createBuffer({
        label: "Offsets 1k 0",
        size: uniformOffsetsArray1k_1.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
      device.createBuffer({
        label: "Offsets 4k 0",
        size: uniformOffsetsArray4k_1.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      })];
      device.queue.writeBuffer(uniformOffsetsBuffers[0], 0, uniformOffsetsArray00);
      device.queue.writeBuffer(uniformOffsetsBuffers[1], 0, uniformOffsetsArray01);
      device.queue.writeBuffer(uniformOffsetsBuffers[2], 0, uniformOffsetsArray63_1);
      device.queue.writeBuffer(uniformOffsetsBuffers[3], 0, uniformOffsetsArray255_1);
      device.queue.writeBuffer(uniformOffsetsBuffers[4], 0, uniformOffsetsArray1k_1);
      device.queue.writeBuffer(uniformOffsetsBuffers[5], 0, uniformOffsetsArray4k_1);

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
        }),        device.createBindGroup({
          label: "Offsets 4",
          layout: perPassBindGroupLayout, 
          entries: [{
            binding: 0,
            resource: { buffer: uniformOffsetsBuffers[4] }
          },],
        }),
        device.createBindGroup({
          label: "Offsets 5",
          layout: perPassBindGroupLayout, 
          entries: [{
            binding: 0,
            resource: { buffer: uniformOffsetsBuffers[5] }
          },],
        })];


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

      forceIndexPipeline = device.createComputePipeline({
        label: "force index pipeline",
        layout: pipelineLayout2,
        compute: {
          module: forceIndexShaderModule,
          entryPoint: "computeMain",
        }
      });
}

function update_compute_particles(encoder,bindGroups, step)
{

  for (let i = 0; i < 1000; i++) {
    for(let j=0;j<bindGroupUniformOffset.length;j++)
    {
      const computePass = encoder.beginComputePass();
      computePass.setPipeline(sortPipeline);
      computePass.setBindGroup(0, simulationBindGroups);
      computePass.setBindGroup(1, bindGroupUniformOffset[j]);
      const workgroupCount = Math.ceil((NUM_PARTICLES_DIM* NUM_PARTICLES_DIM) / WORKGROUP_SIZE);
      computePass.dispatchWorkgroups(workgroupCount/2);
      computePass.end();
    }
  }

  {
    encoder.clearBuffer(massAssignBufferStorage);
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(massAssignPipeline);
    computePass.setBindGroup(0, massAssignBindGroups);
    computePass.setBindGroup(1, bindGroupUniformOffset[0]);
    const workgroupCount = Math.ceil((128*128) / WORKGROUP_SIZE);
    computePass.dispatchWorkgroups(workgroupCount);
    computePass.end();
  }

  {
    encoder.clearBuffer(forceIndexBufferStorage);
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(forceIndexPipeline);
    computePass.setBindGroup(0, forceIndexBindGroups);
    computePass.setBindGroup(1, bindGroupUniformOffset[0]);
    const workgroupCount = Math.ceil((128*128) / WORKGROUP_SIZE);
    computePass.dispatchWorkgroups(workgroupCount);
    computePass.end();
  }

  for (let i = 0; i < 30; i++) {
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(simulationPipeline);
    computePass.setBindGroup(0, simulationBindGroups);
    const workgroupCount = Math.ceil((NUM_PARTICLES_DIM* NUM_PARTICLES_DIM) / WORKGROUP_SIZE);
    computePass.dispatchWorkgroups(workgroupCount);
    computePass.end();
  }

  {
    // render out the stars to the buffer that will be then drawn using graphics pipe
    encoder.clearBuffer(renderBufferStorage);
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(renderBufferPipeline);
    computePass.setBindGroup(0, bindGroups[0]);
    const workgroupCount = Math.ceil((NUM_PARTICLES_DIM* NUM_PARTICLES_DIM) / WORKGROUP_SIZE);
    computePass.dispatchWorkgroups(workgroupCount);
    computePass.end();
  }
  
}
    
