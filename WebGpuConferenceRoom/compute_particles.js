


const DELTA_T = 0.0000003;
var compute_pipe;
var compute_binding;
var bindGroupLayout;

function setup_compute_particles() {

 
  const drawShaderModule = device.createShaderModule({
    label: "ParticleAvec",
    code: `
      struct Triangle{
        pos0:vec3f, padd0:f32,
        pos1:vec3f, padd1:f32,
        pos2:vec3f, padd2:f32,
        col:vec3f, cpadd:f32,
      };

      struct Uniforms{
        canvas_size: vec2f,
        canvas_stride: f32,
        time_in:f32,
        tri_pos_min: vec4f,
        tri_pos_max: vec4f
      };

        @group(0) @binding(0) var<uniform> uni: Uniforms;
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

        @compute @workgroup_size(${WORKGROUP_SIZE})
        fn main(  @builtin(local_invocation_index) local_idx:u32,
        @builtin(	workgroup_id) wg_id:vec3u) {
            const wg_size = ${WORKGROUP_SIZE};
            // we have these 16x16 tiles and we have 16x16 total of them
            // for a 256 area.  65536 pixels 
            for(var tile_x = 0u; tile_x < 16u; tile_x++){
              for(var tile_y = 0u; tile_y < 16u; tile_y++){
            
                workgroupBarrier();

                var x = tile_x*16u + wg_id.x * 256u;
                var y = tile_y*16u + wg_id.y * 256u;
                if(x >= u32(uni.canvas_size.x) || y >= u32(uni.canvas_size.y)){  
                  // skip entire workgroup of work
                  continue;
                }

                var intra_x = local_idx % 16u;
                var intra_y = local_idx / 16u;
                x += intra_x;
                y += intra_y;


                var homo_xy = (vec2f(f32(x)/uni.canvas_size.x,f32(y)/uni.canvas_size.y)-vec2(0.5,0.5)) * 2.0;
                homo_xy.y = - homo_xy.y;
                var ray_orig = vec3(0,0.2, 0);
                var ray_vec = normalize(vec3f(homo_xy, 1.0));

                let s_pos = ray_vec;
                let rot = uni.time_in*.2+3.1;
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
                if(pix_pos.x < u32(uni.canvas_size.x) || pix_pos.y < u32(uni.canvas_size.y)){              
               
                  let white_color = vec4f(color_tri, 1);
        
                  textureStore(frame_buffer, pix_pos , white_color);
                }
                workgroupBarrier();
              }
            }
          

        }

       fn box_intersects_triangle( box_min:vec3f,
            box_max:vec3f,
            tri: Triangle) -> bool
        {
          var tri_max = max(max(tri.pos0, tri.pos1), tri.pos2);
          var tri_min = min(min(tri.pos0, tri.pos1), tri.pos2);

          // overlap test
          if(all(tri_min <= box_max) && all(tri_max >= box_min)){
            return true;
          }

          return false;
        }

        @compute @workgroup_size(${WORKGROUP_SIZE})
        fn accelmain(  @builtin(local_invocation_index) local_idx:u32,
        @builtin(	workgroup_id) wg_id:vec3u) {
            // wg_id.x is the z divisions
            // local_index will be x and y divisions
            var x = local_idx % ${ACCEL_DIV};
            var y = local_idx / ${ACCEL_DIV};
            var z = wg_id.x;

            var tri_scene_min = uni.tri_pos_min.xyz;
            var tri_scene_max = uni.tri_pos_max.xyz;
            var per_cell_delta = (tri_scene_max - tri_scene_min) / f32(${ACCEL_DIV});
            // Set zero triangles for cell
            var count_cell = 0u;
            // aabb
            var cell_min = per_cell_delta * vec3f(f32(x),f32(y),f32(z)) + tri_scene_min;
            var cell_max = per_cell_delta * (vec3f(f32(x),f32(y),f32(z)) + vec3f(1.0)) + tri_scene_min;
            for(var i =0u; i < 1000u; i++){
              workgroupBarrier();
              var curr_tri = triangles[i];
              if(box_intersects_triangle(cell_min, cell_max, curr_tri)){
                  count_cell++;
                  accelTri[z][y][x][count_cell] = i; 
              }
              workgroupBarrier();
            }
            accelTri[z][y][x][0] = count_cell;
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



  draw_pipe = device.createComputePipeline({
    label: "Draw",
    layout: pipelineLayout,
    compute: {
      module: drawShaderModule,
      entryPoint: "main"
    }
  });

  accel_pipe = device.createComputePipeline({
    label: "Draw",
    layout: pipelineLayout,
    compute: {
      module: drawShaderModule,
      entryPoint: "accelmain"
    }
  });



}

function update_compute_particles(triStorageBuffer, triAccelBuffer, encoder, step) {
 // encoder.clearBuffer(vizBufferStorage);
  const computePass = encoder.beginComputePass();
  compute_binding = device.createBindGroup({
    label: "GlobalBind",
    layout: bindGroupLayout,
    entries: [{
      binding: 0,
      resource: { buffer: uniformBuffer }
    }, {
      binding: 1, 
      resource: { buffer: triStorageBuffer },
    }, {
      binding: 2,
      resource: { buffer: triAccelBuffer },
    },
    { binding: 3, resource: context.getCurrentTexture().createView()},],
  });

  computePass.setPipeline(accel_pipe);
  computePass.setBindGroup(0, compute_binding);
  // low utilization here but oh well.
  //computePass.dispatchWorkgroups(ACCEL_DIV);

  computePass.setPipeline(draw_pipe);

  computePass.setBindGroup(0, compute_binding);

  const dispatch_width =  Math.ceil(canvas_width / WORKGROUP_SIZE);
  const dispatch_height =  Math.ceil(canvas_height / WORKGROUP_SIZE);
  computePass.dispatchWorkgroups(dispatch_width, dispatch_height, 1);
  computePass.end();
}

