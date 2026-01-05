var compute_pipe;
var compute_binding;
var bindGroupLayout;
var debuggingBufferStorage;
var kDebugArraySize = 1024*4*1024;
var wait_for_debug = false;
function setup_compute_particles() {

  

  debuggingBufferStorage =
  device.createBuffer({
    label: "debugging storage result",
    size: kDebugArraySize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
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

      struct Uniforms{
        canvas_size: vec2f,
        canvas_stride: f32,
        time_in:f32,
        tri_pos_min: vec4f,
        tri_pos_max: vec4f
      };

        @group(0) @binding(0) var<uniform> uni: Uniforms;
        @group(0) @binding(1) var<storage, read_write> triangles: array<Triangle>;
        @group(0) @binding(2) var<storage, read_write> microAccel: array<
                                                                  array<
                                                                   array<
                                                                    array< atomic<u32>, ${MICRO_ACCEL_MAX_CELL_COUNT}>
                                                                      ,${MICRO_ACCEL_DIV}>
                                                                        ,${MICRO_ACCEL_DIV}>
                                                                         ,${MICRO_ACCEL_DIV}> ;
        @group(0) @binding(3) var<storage, read_write> accelTri: array<
                                                                  array<
                                                                   array<
                                                                    array< u32, ${ACCEL_MAX_CELL_COUNT}>
                                                                      ,${ACCEL_DIV_X}>
                                                                        ,${ACCEL_DIV_Y}>
                                                                         ,${ACCEL_DIV_Z}> ;
        @group(0) @binding(4) var<storage, read_write> dbg: array<f32>;
        @group(0) @binding(5) var frame_buffer: texture_storage_2d<${canvasformat}, write>;

        //https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
        fn ray_intersects_triangle( ray_origin:vec3f,
            ray_vector:vec3f,
            tri: Triangle) -> vec4f
        {
            var epsilon = 0.000001;
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


        fn per_cell_delta() -> vec3f {
            var tri_scene_min = uni.tri_pos_min.xyz;
            var tri_scene_max = uni.tri_pos_max.xyz;
            var per_cell_delta = (tri_scene_max - tri_scene_min) / vec3f(${ACCEL_DIV_X}, ${ACCEL_DIV_Y}, ${ACCEL_DIV_Z});
            return per_cell_delta;
        }

        fn pos_to_cell( pos:vec3f) -> vec3f {
            var cell_loc_f = (pos - uni.tri_pos_min.xyz) / per_cell_delta();
            return cell_loc_f;
        }


          
        fn pos_to_micro_cell( pos:vec3f) -> vec3f {
            var tri_scene_min = uni.tri_pos_min.xyz;
            var tri_scene_max = uni.tri_pos_max.xyz;
            var micro_per_cell_delta = (tri_scene_max- tri_scene_min) / f32(${MICRO_ACCEL_DIV});
            var cell_loc_f = (pos - uni.tri_pos_min.xyz) / micro_per_cell_delta;
            return cell_loc_f;
        }

        var<workgroup> dbgIdx:u32;
        fn dbgOut(val:f32){
          dbg[dbgIdx] = val;
          dbgIdx++;
        }
        
        fn dbgOutV(val:vec3f){
          dbgOut(val.x);
          dbgOut(val.y);
          dbgOut(val.z);
        }

        @compute @workgroup_size(${WORKGROUP_SIZE})
        fn main(  @builtin(local_invocation_index) local_idx:u32,
        @builtin(	workgroup_id) wg_id:vec3u) {
            const wg_size = ${WORKGROUP_SIZE};
            // we have these 16x16 tiles and we have 16x16 total of them
            // for a 256 area.  65536 pixels 
            var tile_x = wg_id.z % 16u;
            {
              var tile_y = wg_id.z / 16u;
              {
            
                workgroupBarrier();

                var pix_x = tile_x*16u + wg_id.x * 256u;
                var pix_y = tile_y*16u + wg_id.y * 256u;
                if(pix_x >= u32(uni.canvas_size.x) || pix_y >= u32(uni.canvas_size.y)){  
                  // skip entire workgroup of work
                  return;
                }

                var intra_x = local_idx % 16u;
                var intra_y = local_idx / 16u;
                pix_x += intra_x;
                pix_y += intra_y;

                var homo_xy = (vec2f(f32(pix_x)/uni.canvas_size.x,f32(pix_y)/uni.canvas_size.y)-vec2f(0.5,0.5)) * 2.0;
                // cam transform haxz
                homo_xy *= 0.7;// fov
                homo_xy.y = - homo_xy.y;
                homo_xy.x = - homo_xy.x;
                var ray_orig = vec3(0,0.25, 0);
                var ray_vec = normalize(vec3f(homo_xy, 1.0));
                let rot =  uni.time_in *0.1;  
                {
                  let s_pos = ray_vec;
                  ray_vec.x= s_pos.x * cos(rot) + s_pos.z * -sin(rot);
                  ray_vec.z = s_pos.x * sin(rot) + s_pos.z * cos(rot);
                }

                var min_t = 111111.0;
                var color_tri = vec3f(0.2,0.2,0.0);

                var cell_loc_f = pos_to_cell(ray_orig);
                var cell_loc_i = vec3i(cell_loc_f);
                var cell_loc_remain = cell_loc_f - vec3f(cell_loc_i);

     
                // Normalize remain to be in
                //  the positive direction
                var remain_dir = select(cell_loc_remain, vec3f(1.0) - cell_loc_remain, ray_vec >= vec3f(0,0,0));

                remain_dir *= per_cell_delta(); 
                var max_cell_count = 0u;
                for(var finite_loop = 0u; finite_loop < 300; finite_loop++) {
                    var max_accel_size = vec3i(${ACCEL_DIV_X}, ${ACCEL_DIV_Y},${ACCEL_DIV_Z});
                    //cell_loc_i = vec3i(pos_to_cell(ray_orig +ray_vec*f32(finite_loop) *0.01 ));
                    if(any(cell_loc_i < vec3i(0)) || any(cell_loc_i >= max_accel_size) ){
                      break;
                    }
                    var count_cell = accelTri[cell_loc_i.z][cell_loc_i.y][cell_loc_i.x][0];
                    max_cell_count = max(max_cell_count, count_cell);
                    // cells start after zeroth

                    for(var i = 1u; i < count_cell+1; i++) {
                        var curr_tri = triangles[accelTri[cell_loc_i.z][cell_loc_i.y][cell_loc_i.x][i]];
                        var res = ray_intersects_triangle(ray_orig, ray_vec, curr_tri);
                        if(res.w > 0.0 && res.w <= min_t){
                            // WE MUST DO BOX INTERSECTION TEST or tracker for hit testing
                            if( all(cell_loc_i == vec3i(pos_to_cell(res.xyz))))
                            {
                              min_t = res.w;
                              color_tri = curr_tri.col;
                            }
                        }
                    }
                    
                    if(min_t != 111111.0 ){
                      // if we have a REAL hit we should exit this loop
                     break;
                    }
               
                    // Find next cell 
  
                    var ray_vec_abs = abs(ray_vec);
                    // when ray collision with next
                    var t_to_edge = remain_dir / ray_vec_abs;
       
                    if(t_to_edge.x <= t_to_edge.y && t_to_edge.x <= t_to_edge.z) {
                      cell_loc_i.x += i32(sign(ray_vec.x));
                      remain_dir -= t_to_edge.x * ray_vec_abs;
                      remain_dir.x = per_cell_delta().x;
                    }
                    else if(t_to_edge.y <= t_to_edge.x && t_to_edge.y <= t_to_edge.z) {
                      cell_loc_i.y += i32(sign(ray_vec.y));
                      remain_dir -= t_to_edge.y * ray_vec_abs;
                      remain_dir.y = per_cell_delta().y;
                    }
                    else {
                      cell_loc_i.z += i32(sign(ray_vec.z));
                      remain_dir -= t_to_edge.z * ray_vec_abs;
                      remain_dir.z  = per_cell_delta().z;
                    }
                    
                }

                let pix_pos = vec2u(pix_x, pix_y);
                  // This can happen because rounding of workgroup size vs resolution
                if(pix_pos.x < u32(uni.canvas_size.x) || pix_pos.y < u32(uni.canvas_size.y)){     
                // color_tri =  color_tri +vec3f(f32(max_cell_count)/128.0);   
                  textureStore(frame_buffer, pix_pos , vec4f(color_tri, 1));
                }
          
              }
            }
          

        }

       fn box_intersects_triangle( box_min:vec3f,
            box_max:vec3f,
            tri: Triangle) -> bool
        {
          var tri_max = max(max(tri.pos0, tri.pos1), tri.pos2);
          var tri_min = min(min(tri.pos0, tri.pos1), tri.pos2);

          // fast overlap test 
          if(all(tri_min <= box_max) && all(tri_max >= box_min)){
       
              // any vert inside
              // DDDDEBUG
              if(true){
                if(all(tri.pos0 <= box_max) && all(box_min <= tri.pos0)){
                  return true;
                }

                if(all(tri.pos1 <= box_max) && all(box_min <= tri.pos1)){
                  return true;
                }
              
                if(all(tri.pos2 <= box_max) && all(box_min <= tri.pos2)){
                  return true;
                }
              }


              // Triangle penetrates box (line test)
              if(true){
                var vert_array = array( tri.pos0,  tri.pos1,  tri.pos2);
                for(var line_sel = 0u; line_sel < 3u; line_sel++){
                  var ray_orig = vert_array[line_sel];
                  var ray_vec = vert_array[(line_sel+1) % 3u] - ray_orig;
                    
                  for(var max_min_sel = 0u; max_min_sel < 1u; max_min_sel++){
                    var min_max_plane = select(box_min, box_max, vec3<bool>(max_min_sel == 1u));

                    var t_each_plane = (min_max_plane - ray_orig)/ray_vec;
                    
                    for(var each_axis = 0u; each_axis < 3u; each_axis++){
                        var test_p  = ray_orig + ray_vec * t_each_plane[each_axis];
                        test_p[each_axis] = (box_min[each_axis] + box_max[each_axis]) * 0.5;
                        if( all(box_min <= test_p) && all(test_p <= box_max)){
                          return true;
                        }
                    }
                  }
                }
              }

              // Triangle cut box. (any box wire cut triangle)
              if(true){
                for(var proj_axis = 0u; proj_axis < 3u; proj_axis++) {
                  // other axis
                  var rect_axis_a = (proj_axis + 1) % 3u;
                  var rect_axis_b = (proj_axis + 2) % 3u;
                
                  for(var rect_alter_ab = 0u; rect_alter_ab < 4u; rect_alter_ab++)
                  {
                      var sel_start = vec3u(0,0,0);
                      sel_start[proj_axis] = 0;
                      sel_start[rect_axis_a] = rect_alter_ab % 2u;
                      sel_start[rect_axis_b] = rect_alter_ab / 2u;
                      var ray_orig = select(box_min, box_max, sel_start == vec3u(1u));

                      var sel_end = sel_start;
                      sel_end[proj_axis] = 1; // project as line to other side
                      var ray_end = select(box_min, box_max, sel_end == vec3u(1u));

                      var res = ray_intersects_triangle(ray_orig, ray_end - ray_orig, tri);
                      if(res.w >=0.0 && res.w <= 1.0){
                        return true;
                      }
                  }
                }
              }

              return false;
          }

          return false;
        }


        // 4k wg memory
        var<workgroup> wg_cell_count :   array<array<array<atomic<u32>, 
                                              ${ACCEL_DIV_X} / ${MICRO_ACCEL_DIV}>,
                                              ${ACCEL_DIV_Y} / ${MICRO_ACCEL_DIV}>,
                                              ${ACCEL_DIV_Z} / ${MICRO_ACCEL_DIV}> ;

        @compute @workgroup_size(${WORKGROUP_SIZE})
        fn accelmain(  @builtin(local_invocation_index) local_idx:u32,
        @builtin(	workgroup_id) wg_id:vec3u) {
            var tri_scene_min = uni.tri_pos_min.xyz;
            var per_cell_delta = per_cell_delta();
            // The microAccel has tri list at a coarse grained level
            var total_count = atomicLoad(&microAccel[wg_id.z][wg_id.y][wg_id.x][0]);
            var micro_tri_idx_thread = local_idx;

            if(local_idx == 0 && wg_id.x == 0 && wg_id.y == 0 && wg_id.z == 0){
              dbgOut(-777.0);
              for(var xx = 0u; xx < u32( ${MICRO_ACCEL_DIV} ); xx++){
                for(var yy = 0u; yy < u32(  ${MICRO_ACCEL_DIV} ); yy++){
                  for(var zz = 0u; zz < u32(  ${MICRO_ACCEL_DIV} ); zz++){
                      var total_count = atomicLoad(&microAccel[zz][yy][xx][0]);
                        dbgOut(f32(total_count));
                    }
                  }
                }
              dbgOut(-555.0);
            }

            while(micro_tri_idx_thread < total_count) {
              // Each thread loads a different triangle. +1 because of silly counter at start
              var real_tri_index =  atomicLoad(&microAccel[wg_id.z][wg_id.y][wg_id.x][micro_tri_idx_thread + 1]); 
              var curr_tri = triangles[real_tri_index];
              for(var xx = 0u; xx < u32( ${ACCEL_DIV_X} / ${MICRO_ACCEL_DIV} ); xx++){
                for(var yy = 0u; yy < u32( ${ACCEL_DIV_Y} /  ${MICRO_ACCEL_DIV} ); yy++){
                  for(var zz = 0u; zz < u32( ${ACCEL_DIV_Z} / ${MICRO_ACCEL_DIV} ); zz++){
                    var x = xx +  wg_id.x * u32( ${ACCEL_DIV_X} / ${MICRO_ACCEL_DIV} );
                    var y = yy +  wg_id.y * u32( ${ACCEL_DIV_Y} / ${MICRO_ACCEL_DIV} );
                    var z = zz +  wg_id.z * u32( ${ACCEL_DIV_Z} / ${MICRO_ACCEL_DIV} );
                    var xyz = vec3f(f32(x), f32(y), f32(z));
                    var cell_min = per_cell_delta * xyz  + tri_scene_min;
                    var cell_max = per_cell_delta * (xyz + vec3f(1.0)) + tri_scene_min;

                    if(box_intersects_triangle(cell_min, cell_max, curr_tri)){
                        var add_slot_idx = atomicAdd(&wg_cell_count[zz][yy][xx], 1);
                        if(add_slot_idx < 125) {
                          accelTri[z][y][x][add_slot_idx + 1] = real_tri_index; 
                        }
                    }
                  }
                }
              }
              micro_tri_idx_thread += ${WORKGROUP_SIZE};// next batch of triangles
            }

            workgroupBarrier();
            // Use 1 thread to assign count to first index.
            if(local_idx == 0){
              for(var xx = 0u; xx < u32( ${ACCEL_DIV_X} / ${MICRO_ACCEL_DIV} ); xx++){
                for(var yy = 0u; yy < u32( ${ACCEL_DIV_Y} /  ${MICRO_ACCEL_DIV} ); yy++){
                  for(var zz = 0u; zz < u32( ${ACCEL_DIV_Z} / ${MICRO_ACCEL_DIV} ); zz++){
                      var x = xx +  wg_id.x * u32( ${ACCEL_DIV_X} / ${MICRO_ACCEL_DIV} );
                      var y = yy +  wg_id.y * u32( ${ACCEL_DIV_Y} / ${MICRO_ACCEL_DIV} );
                      var z = zz +  wg_id.z * u32( ${ACCEL_DIV_Z} / ${MICRO_ACCEL_DIV} );
                      accelTri[z][y][x][0] = atomicLoad(&wg_cell_count[zz][yy][xx]); 
                    }
                  }
                }
            }
        }

        @compute @workgroup_size(${WORKGROUP_SIZE})
        fn microAccelmain(  @builtin(local_invocation_index) local_idx:u32,
        @builtin(	workgroup_id) wg_id:vec3u) {
            var tri_idx = local_idx + wg_id.x * ${WORKGROUP_SIZE};
            if(tri_idx >=  arrayLength(&triangles)){
              // out of bounds. Expected.
              return;
            }

            var curr_tri = triangles[tri_idx];
            var tri_scene_min = uni.tri_pos_min.xyz;
            var tri_scene_max = uni.tri_pos_max.xyz;
            var per_cell_delta = (tri_scene_max- tri_scene_min) / f32(${MICRO_ACCEL_DIV});
            for(var x = 0u; x < u32(${MICRO_ACCEL_DIV}); x++){
              for(var y = 0u; y < u32(${MICRO_ACCEL_DIV}); y++){
                for(var z = 0u; z < u32(${MICRO_ACCEL_DIV}); z++){
                var xyz = vec3f(f32(x),f32(y),f32(z));
                var cell_min = per_cell_delta * xyz  + tri_scene_min;
                var cell_max = per_cell_delta * (xyz + vec3f(1.0)) + tri_scene_min;

                if(box_intersects_triangle(cell_min, cell_max, curr_tri)){
                    var unique_idx = atomicAdd(&microAccel[z][y][x][0], 1);
                    if(unique_idx < (64*1024-10)){// saftey check
                     atomicStore(&microAccel[z][y][x][unique_idx + 1], tri_idx); 
                    }
                }
              }
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
    }, 
    {
      binding: 3,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage" }
    }, 
    {
      binding: 4,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage" }
    },{
      binding: 5,
      visibility: GPUShaderStage.COMPUTE,
      storageTexture: { access: "write-only", format: canvasformat }
    }]
  });


  const pipelineLayout = device.createPipelineLayout({
    label: "Sim",
    bindGroupLayouts: [bindGroupLayout],
  });



  draw_pipe = device.createComputePipeline({
    label: "main",
    layout: pipelineLayout,
    compute: {
      module: drawShaderModule,
      entryPoint: "main"
    }
  });

  accel_pipe = device.createComputePipeline({
    label: "accelmain",
    layout: pipelineLayout,
    compute: {
      module: drawShaderModule,
      entryPoint: "accelmain"
    }
  });

  micro_accel_pipe = device.createComputePipeline({
    label: "microAccelmain",
    layout: pipelineLayout,
    compute: {
      module: drawShaderModule,
      entryPoint: "microAccelmain"
    }
  });

}

function update_compute_particles(triStorageBuffer, triAccelBuffer, microTriAccelBuffer, numTriangles, encoder, step) {
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
      resource: { buffer: microTriAccelBuffer },
    }, 
    {
      binding: 3,
      resource: { buffer: triAccelBuffer },
    },
    {
      binding: 4,
      resource: { buffer: debuggingBufferStorage },
    },
    { binding: 5, resource: context.getCurrentTexture().createView()},
    ],
  });

  if(step <= 1){
    computePass.setPipeline(micro_accel_pipe);
    computePass.setBindGroup(0, compute_binding);
    var num_wg = Math.ceil(numTriangles/ WORKGROUP_SIZE);
    computePass.dispatchWorkgroups(num_wg);

    computePass.setPipeline(accel_pipe);
    computePass.setBindGroup(0, compute_binding);
    computePass.dispatchWorkgroups(MICRO_ACCEL_DIV, MICRO_ACCEL_DIV, MICRO_ACCEL_DIV);
  }

  computePass.setPipeline(draw_pipe);

  computePass.setBindGroup(0, compute_binding);

  const dispatch_width =  Math.ceil(canvas_width / WORKGROUP_SIZE);
  const dispatch_height =  Math.ceil(canvas_height / WORKGROUP_SIZE);
  computePass.dispatchWorkgroups(dispatch_width, dispatch_height, 256);
  computePass.end();


  var stagingBufferDebug = null;
  
  if(debug_mode){
    stagingBufferDebug = device.createBuffer({
      label: "staging buff dbg",
      size: kDebugArraySize,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    wait_for_debug = true;
    encoder.copyBufferToBuffer(
      debuggingBufferStorage,
      0, // Source offset
      stagingBufferDebug,
      0, // Destination offset
      kDebugArraySize, 
    );
  }

  return stagingBufferDebug;
}

