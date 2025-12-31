


const DELTA_T = 0.0000003;
var compute_pipe;
var compute_binding;
var bindGroupLayout;

function setup_compute_particles() {

  const simShaderModule = device.createShaderModule({
    label: "ParticleAvec",
    code: `
        struct Particle {
           unused_data: vec2i,
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


        fn force(pos: array<vec2f,3> ) -> array<vec2f,3> {
          var force_out : array<vec2f, 3>;
          const force_mult = 30000.0;
          for(var i = 0u; i < 3u; i++){
              for(var j = 0u; j < 3u; j++){
                 if(i != j){
                      let diff = (pos[i] - pos[j])/force_mult;
                      force_out[i] += - normalize(diff)/dot(diff,diff);
                  }
               }
          }
          return force_out;
        }

        @compute @workgroup_size(${WORKGROUP_SIZE})
        fn main(  @builtin(local_invocation_index) local_idx:u32,
        @builtin(	workgroup_id) wg_id:vec3u) {
            // Three bodies at the same time.
            
            const wg_size = ${WORKGROUP_SIZE};
            const world_scale = f32(${WORLD_SCALE});
            let width_stride = u32(canvas_size.z);
            let idx = local_idx + (wg_size * wg_id.x);
            let part_start = idx * 3; // every 3 bodies


            var pos : array<vec2f, 3>;
            var vel : array<vec2f, 3>;
            for(var i = 0u; i < 3u; i++){
                pos[i] = cellStateOut[part_start + i].posf;
                vel[i] = cellStateOut[part_start + i].vel;
            }


            for(var q = 0u; q < 20u; q++){
              var min_dist = 1000000.0;
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

              var num_iter = clamp( u32( 100.0/(min_dist)), 5u, 50u);
              var dt =  0.00002/f32(num_iter);
              const num_stages = 4u;
              const kCoeff = array(0.5, 0.5, 1.0);
              const kWeights = array(1.0/ 6.0, 2.0/ 6.0, 2.0/ 6.0, 1.0/ 6.0);
              for(var b =0u;b <num_iter;b++){

                var xk = array<array<vec2f,3>,num_stages>();
                var vk = array<array<vec2f,3>,num_stages>();
                xk[0] = vel;
                vk[0] = force(pos);

                for(var stage = 1u; stage < num_stages; stage++){
                    // Compute acceleration
                    var temp_pos = array<vec2f,3>();
                    for(var i = 0u; i < 3u; i++){
                      temp_pos[i] = pos[i] + xk[stage - 1][i] * dt * kCoeff[stage - 1];
                    }
                    var a = force(temp_pos);

                    // Compute xk and vk
                    for(var i = 0u; i < 3u; i++){
                      xk[stage][i] = vel[i] + vk[stage - 1][i] * dt * kCoeff[stage - 1]; 
                    }
                    vk[stage] = a;
                }
        
                  var dx : array<vec2f,3>;
                  var dv : array<vec2f,3>;
                  for(var stage = 0u; stage < num_stages; stage++){
                    for(var i = 0u; i < 3u; i++){
                      dx[i] += kWeights[stage] * xk[stage][i];
                      dv[i] += kWeights[stage] * vk[stage][i];
                    }
                  }

                  for(var i = 0u; i < 3u; i++){
                    pos[i] = pos[i] + dt * dx[i];
                    vel[i] = vel[i] + dt * dv[i];
                  }
              }

              for(var i = 0u; i < 3u; i++){
                pos[i] = pos[i] + offset_pos;
              }
            }

            // Anti aliased drawing of 3 particles from world to screen
            // We could actually do it stochastically!
            for(var i = 0u; i < 3u; i++){
                var as_float_temp = pos[i];
                var pix_pos = (pos[i]/world_scale + vec2f(0.5,0.5)) * canvas_size.xy;

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
      struct Triangle{
        pos0:vec3f, padd0:f32,
        pos1:vec3f, padd1:f32,
        pos2:vec3f, padd2:f32,
        col:vec3f, cpadd:f32,
      };

        @group(0) @binding(0) var<uniform> canvas_size: vec4f;
        @group(0) @binding(1) var<storage, read_write> triangles: array<Triangle>;
        @group(0) @binding(2) var<storage, read_write> accelTri: array<
                                                                  array<
                                                                   array<
                                                                    array< u32, ${ACCEL_MAX_CELL_COUNT}>
                                                                      ,${ACCEL_DIV}>
                                                                        ,${ACCEL_DIV}>
                                                                         ,${ACCEL_DIV}> ;
        @group(0) @binding(3) var frame_buffer: texture_storage_2d<${canvasformat}, write>;


        //https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
        fn ray_intersects_triangle( ray_origin:vec3f,
            ray_vector:vec3f,
            tri: Triangle) -> vec4f
        {
            var epsilon = 0.00001;

            var edge1 = tri.pos1 - tri.pos0;
            var edge2 = tri.pos2 - tri.pos0;
            var ray_cross_e2 = cross(ray_vector, edge2);
            var det = dot(edge1, ray_cross_e2);

            if (det > -epsilon && det < epsilon){
                return vec4f(0,0,0,-1);    // This ray is parallel to this triangle.
            }

            var inv_det = 1.0 / det;
            var s = ray_origin - tri.pos0;
            var u = inv_det * dot(s, ray_cross_e2);

            if ((u < 0 && abs(u) > epsilon) || (u > 1 && abs(u-1) > epsilon)){
                return vec4f(0,0,0,-1);
            }

            var s_cross_e1 = cross(s, edge1);
            var v = inv_det * dot(ray_vector, s_cross_e1);

            if ((v < 0 && abs(v) > epsilon) || (u + v > 1 && abs(u + v - 1) > epsilon)){
                return vec4f(0,0,0,-1);
            }

            // At this stage we can compute t to find out where the intersection point is on the line.
            var t = inv_det * dot(edge2, s_cross_e1);

            if (t > epsilon) // ray intersection
            {
                return  vec4f(ray_origin + ray_vector * t, t);
            }
            else // This means that there is a line intersection but not a ray intersection.
            {
                return vec4f(0,0,0,-1);
            }
        }

        var<workgroup> wg_triangle: Triangle;
        
        @compute @workgroup_size(${WORKGROUP_SIZE})
        fn main(  @builtin(local_invocation_index) local_idx:u32,
        @builtin(	workgroup_id) wg_id:vec3u) {
            const wg_size = ${WORKGROUP_SIZE};

            let width_stride = u32(canvas_size.z);

            // we have these 16x16 tiles and we have 16x16 total of them
            // for a 256 area.  65536 pixels 
            for(var tile_x = 0u; tile_x < 16u; tile_x++){
              for(var tile_y = 0u; tile_y < 16u; tile_y++){
            
                workgroupBarrier();

                var x = tile_x*16u + wg_id.x * 256u;
                var y = tile_y*16u + wg_id.y * 256u;
                if(x >= u32(canvas_size.x) || y >= u32(canvas_size.y)){  
                  // skip entire workgroup of work
                  continue;
                }

                var intra_x = local_idx % 16u;
                var intra_y = local_idx / 16u;
                x += intra_x;
                y += intra_y;


                var homo_xy = (vec2f(f32(x)/canvas_size.x,f32(y)/canvas_size.y)-vec2(0.5,0.5)) * 2.0;
                homo_xy.y = - homo_xy.y;
                var ray_orig = vec3(0,0.2, 0);
                var ray_vec = normalize(vec3f(homo_xy, 1.0));

                let s_pos = ray_vec;
                let rot = canvas_size.w*.2+3.1;
                ray_vec.x= s_pos.x * cos(rot) + s_pos.z * -sin(rot);
                ray_vec.z = s_pos.x * sin(rot) + s_pos.z * cos(rot);

                var min_t = 100000000.0;
                var color_tri = vec3f(0,0,0);

    

                for(var i =0u; i < 1000u; i++){
                     workgroupBarrier();
                    var curr_tri = triangles[i];;
                    var res = ray_intersects_triangle(ray_orig, ray_vec, curr_tri);
                    if(res.w > 0.0 && res.w <= min_t){
                        min_t = res.w;
                        color_tri = curr_tri.col;
                    }
                    workgroupBarrier();
                }
                 workgroupBarrier();

                let pix_pos = vec2u(x, y);
                  // This can happen because rounding of workgroup size vs resolution
                if(pix_pos.x < u32(canvas_size.x) || pix_pos.y < u32(canvas_size.y)){              
                  //et sample = vizBuff[pix_pos.x + pix_pos.y * width_stride];
                  let white_color = vec4f(color_tri, 1);
        
                  textureStore(frame_buffer, pix_pos , white_color);
                }
                workgroupBarrier();
              }
            }
          

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

function update_compute_particles(triStorageBuffer, triAccelBuffer, encoder, step) {
 // encoder.clearBuffer(vizBufferStorage);
  const computePass = encoder.beginComputePass();
  computePass.setPipeline(draw_pipe);
  compute_binding = device.createBindGroup({
    label: "GlobalBind",
    layout: bindGroupLayout,
    entries: [{
      binding: 0,
      resource: { buffer: uniformBuffer }
    }, {
      binding: 1, // New Entry
      resource: { buffer: triStorageBuffer },
    }, {
      binding: 2, // New Entry
      resource: { buffer: triAccelBuffer },
    },
    { binding: 3, resource: context.getCurrentTexture().createView() },],
  });
 // computePass.setBindGroup(0, compute_binding);
  //const workgroupCount = Math.ceil(NUM_MICRO_SIMS / WORKGROUP_SIZE);
  //computePass.dispatchWorkgroups(workgroupCount);
  //computePass.setPipeline(draw_pipe);
  computePass.setBindGroup(0, compute_binding);

  const dispatch_width =  Math.ceil(canvas_width / WORKGROUP_SIZE);
  const dispatch_height =  Math.ceil(canvas_height / WORKGROUP_SIZE);

  computePass.dispatchWorkgroups(dispatch_width, dispatch_height, 1);
  computePass.end();
}

