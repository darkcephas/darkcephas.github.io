"use strict";
var device;
var canvasformat;
var context;
var debug_mode = false;
var raster_mode = false;
var microTriAccelBuffer;
var emptyCellAccelBuff;

var compute_pipe;
var commonComputeBinding;
var cam_ray_gen_pipe;
var bindGroupLayout;
var debuggingBufferStorage;
var kDebugArraySize = 1024*4;
var wait_for_debug = false;

const WORKGROUP_SIZE = 256;
const ACCEL_DIV_X = 128;
const ACCEL_DIV_Y = 32;
const ACCEL_DIV_Z = 64;
const ACCEL_MAX_CELL_COUNT = 128;
const MICRO_ACCEL_DIV = 8;
const MICRO_ACCEL_MAX_CELL_COUNT = 1024*64; // hopefully enough space 300/512
var canvas_width;
var canvas_height;
var canvas_width_block;
var canvas_height_block;
var bindGroupLayout;
var uniformBuffer;
var simulationBindGroups;
var forceIndexBindGroups;
var rayInBufferStorage;
var rayResultBufferStorage;
var triStateStorage;
var numTriangles;
const kSizeBytesInU32 = 4;

var time_t = 0.0;
var rasterizerDepthTexture;
var triAccelBuffer;

var BIG_NUM = 100000.0;
var tri_pos_min_x = BIG_NUM;
var tri_pos_min_y = BIG_NUM;
var tri_pos_min_z = BIG_NUM;
var tri_pos_max_x = -BIG_NUM;
var tri_pos_max_y = -BIG_NUM;
var tri_pos_max_z = -BIG_NUM;
var rayTracePipeline;
var accel_pipe;
var micro_accel_pipe;
var computeCommonBindGroupLayout;
var secondaryBindGroupLayout;
var secondaryUniformBuffer;
var secondaryComputeBinding;
var bounceGenPipeline;

function UpdateUniforms() {
  // Create a uniform buffer that describes the grid.
  const uniformArray = new Float32Array([canvas_width, canvas_height, canvas_width_block, time_t,
                                        tri_pos_min_x, tri_pos_min_y, tri_pos_min_z, 0.0,
                                        tri_pos_max_x, tri_pos_max_y, tri_pos_max_z, 0.0]);
  device.queue.writeBuffer(uniformBuffer, 0, uniformArray);
}

window.onload = async function () {
  window.addEventListener('resize', onResizeCanvas, false);
  const canvas = document.querySelector("canvas");
  if (!canvas) {
    throw new Error("No canvas.");
  }

  // Your WebGPU code will begin here!
  if (!navigator.gpu) {
    throw new Error("WebGPU not supported on this browser.");
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("No appropriate GPUAdapter found.");
  }

  function onResizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    canvas_width = canvas.width;
    canvas_height = canvas.height;
    canvas_width_block  =  Math.ceil(canvas_width / 16);
    canvas_height_block  =  Math.ceil(canvas_height / 16);
    const tileBlockSize = 16*16;
    const numRayInElementBytes = 8*4; // RayIn
    const numRayInBufferTotalBytes = numRayInElementBytes * canvas_width_block * canvas_height_block * tileBlockSize;
    rayInBufferStorage =
      device.createBuffer({
        label: "RayIn buffer",
        size: numRayInBufferTotalBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

    const numRayResultElementBytes = 4*4; // RayResult
    const numRayResultBufferTotalBytes = numRayResultElementBytes * canvas_width_block * canvas_height_block * tileBlockSize;
    rayResultBufferStorage =
    device.createBuffer({
      label: "RayIn buffer",
      size: numRayResultBufferTotalBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    rasterizerDepthTexture = device.createTexture({
      size: { width: canvas_width, height: canvas_height },
      dimension: '2d',
      format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT
    });


    /**
     * Your drawings need to be inside this function otherwise they will be reset when 
     * you resize the browser window and the canvas goes will be cleared.
     */
    // drawStuff(); 
  }

  // Features requests (limits)
  {
    canvasformat = navigator.gpu.getPreferredCanvasFormat();
    var reqFeatures = canvasformat == 'bgra8unorm' ? ['bgra8unorm-storage'] : [];
    device = await adapter.requestDevice({ requiredFeatures:reqFeatures });
  }
  

  const accel_buff_size =  ACCEL_DIV_X * ACCEL_DIV_Y * ACCEL_DIV_Z * ACCEL_MAX_CELL_COUNT * kSizeBytesInU32;
  triAccelBuffer = 
    device.createBuffer({
      label: "Triangle accel index buffer",
      size: accel_buff_size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

  const micro_accel_buff_size =  MICRO_ACCEL_DIV * MICRO_ACCEL_DIV * MICRO_ACCEL_DIV * MICRO_ACCEL_MAX_CELL_COUNT * kSizeBytesInU32;
    microTriAccelBuffer = 
      device.createBuffer({
        label: "Micro list triangle index accel",
        size: micro_accel_buff_size,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });


  if(ACCEL_DIV_Y != 32){
    console.log("We assume this for bit packing!");
  }
  const empty_cell_accel_buff_size = ACCEL_DIV_Z *  ACCEL_DIV_X * kSizeBytesInU32;
  emptyCellAccelBuff = 
        device.createBuffer({
          label: "Empty cell accel",
          size: empty_cell_accel_buff_size,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
  

  context = canvas.getContext("webgpu");
  context.configure({
    device: device,
    format: canvasformat,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING, // rw from shader
  });

  onResizeCanvas();
 
  uniformBuffer = device.createBuffer({
    label: "Uniforms",
    size: 128,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  UpdateUniforms();


  const rasterizerRenderShaderModule = device.createShaderModule({
    label: 'rasterizerRenderShaderModule',
    code: `
      struct VertexInput {
        @builtin(vertex_index) instance: u32,
      };

      struct VertexOutput {
        @builtin(position) pos: vec4f,
        @location(0) prim_color: vec3f, // Not used yet
      };
      
      struct FragInput {
        @builtin(position) pos: vec4f,
        @location(0) prim_color: vec3f,
      };

      struct Triangle{
        pos: array<vec4f, 3>,
        col:vec3f, cpadd:f32,
      };

      @group(0) @binding(0) var<uniform> canvas_size: vec4f;
      @group(0) @binding(1) var<storage> renderBufferIn: array<Triangle>;
      @vertex
      fn vertexMain(input: VertexInput) -> VertexOutput {
        var output: VertexOutput;
        let which_triangle = input.instance/3u;
        var pos = renderBufferIn[which_triangle].pos[input.instance % 3].xyz;

        output.prim_color = renderBufferIn[which_triangle].col;
        //var test_array= array(vec3f(0,0,.1), vec3f(1,0,.1),vec3f(0,1,.1));
        //pos = test_array[input.instance % 3];
        let s_pos = pos;
        let rot = canvas_size.w*0.2*1.71;
        pos.x = s_pos.x * cos(rot) + s_pos.z * -sin(rot);
        pos.z = s_pos.x * sin(rot) + s_pos.z * cos(rot);
        const zNear = 0.05;
        const zFar = 1.3;
        let rangeInv = 1 / (zNear - zFar);
        pos.y -= 0.2;
        pos.x *= canvas_size.y/canvas_size.x;
        pos.x *= 1.;
        pos.y *= 1.;
        let w_per = - pos.z;
        pos.z =  zFar * rangeInv* pos.z + zNear * zFar * rangeInv;
        output.pos = vec4f(pos.xy, pos.z, w_per );
        return output;
      }

     @fragment
      fn fragmentMain(input: FragInput) -> @location(0) vec4f {
        return vec4f(input.prim_color, 1) ;
      }
    `
  });


  var rasterBindGroupLayout = device.createBindGroupLayout({
    label: "Raster render bind group",
    entries: [{
      binding: 0,
      visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
      buffer: {} // Grid uniform buffer
    }, {
      binding: 1,
      visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
      buffer: { type: "read-only-storage" }
    }]
  });

  const rasterPipelineLayout = device.createPipelineLayout({
    label: "Raster Render Pipeline Layout",
    bindGroupLayouts: [rasterBindGroupLayout],
  });

  var rasterRenderPipe = device.createRenderPipeline({
    label: "render pipeline",
    depthStencil: { depthCompare: "less", depthWriteEnabled: true, format: "depth32float" },
    layout: rasterPipelineLayout, 
    primitive: { cullMode: "front" , frontFace: "cw"},
    vertex: {
      module: rasterizerRenderShaderModule,
      entryPoint: "vertexMain",
    },
    fragment: {
      module: rasterizerRenderShaderModule,
      entryPoint: "fragmentMain",
      targets: [{
        format: canvasformat,
      }]
    }
  });

  ImportTriangleData();

  var graphicsBindGroup =
    device.createBindGroup({
      label: "raster graphics bind",
      layout: rasterBindGroupLayout, 
      entries: [{
        binding: 0,
        resource: { buffer: uniformBuffer }
      }, {
        binding: 1,
        resource: { buffer: triStateStorage }
      }],
    });




  setup_compute_particles();

  let step = 0; // Track how many simulation steps have been run        
  function updateGrid() {
  
    step++; // Increment the step count
    UpdateUniforms();
    // Start a render pass 
    const encoder = device.createCommandEncoder();

    if(raster_mode){
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: context.getCurrentTexture().createView(),
          loadOp: "clear",
          clearValue: { r:0, g:0, b:0, a:0},
          storeOp: "store",
        }],
        depthStencilAttachment: {
          depthClearValue: 1.0,
          depthLoadOp: "clear",
          depthStoreOp: "discard",
          view: rasterizerDepthTexture.createView()
        },
      });

      pass.setPipeline(rasterRenderPipe);
      pass.setBindGroup(0, graphicsBindGroup); // Updated!

        pass.draw(numTriangles*3);
    
      pass.end();
    }


   var buff_ret = null;
   if(!raster_mode){
     buff_ret = update_compute_particles( encoder, step);
   }
    const commandBuffer = encoder.finish();
    device.queue.submit([commandBuffer]);
 
    if(debug_mode)
    {
      // You can only map async after you submit.
      // It is a weird quirk of the webgpu API.
      buff_ret.mapAsync(
        GPUMapMode.READ,
        0, // Offset
        kDebugArraySize // Length
      ).then(value => 
      {
        const copyArrayBuffer = buff_ret.getMappedRange();
        const data = copyArrayBuffer.slice();
        const data_as_float = new Float32Array(data);
        console.log(data_as_float);
        buff_ret.unmap();
        wait_for_debug = false; 
      });
    }

    time_t = time_t + 0.016;
    if(!debug_mode){
      window.requestAnimationFrame(updateGrid);
    }
  }
  window.requestAnimationFrame(updateGrid);
}



function setup_compute_particles() {
  debuggingBufferStorage =
  device.createBuffer({
    label: "debugging storage result",
    size: kDebugArraySize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

 
  const drawShaderModule = device.createShaderModule({
    label: "RaytracingShaderModule",
    code: `
      struct Triangle{
        pos0:vec3f, padd0:f32,
        pos1:vec3f, padd1:f32,
        pos2:vec3f, padd2:f32,
        col:vec3f, cpadd:f32,
      };

      struct Uniforms{
        canvas_size: vec2f, canvas_stride: f32, time_in:f32,
        tri_pos_min: vec4f,
        tri_pos_max: vec4f
      };


        struct RayIn {
          ray_orig:vec3f, px:u32,
          ray_vec:vec3f, py:u32,
        };

        struct RayResult {
          tri:u32, dist_t:f32, px:u32, py:u32
        };


        const ACCEL_DIV_X =  ${ACCEL_DIV_X};
        const ACCEL_DIV_Y =  ${ACCEL_DIV_Y};
        const ACCEL_DIV_Z =  ${ACCEL_DIV_Z};
        const MICRO_ACCEL_DIV = ${MICRO_ACCEL_DIV};
        const MICRO_ACCEL_MAX_CELL_COUNT = ${MICRO_ACCEL_MAX_CELL_COUNT};
        const ACCEL_MAX_CELL_COUNT = ${ACCEL_MAX_CELL_COUNT};
        const WORKGROUP_SIZE = ${WORKGROUP_SIZE};

        @group(0) @binding(0) var<uniform> uni: Uniforms;
        @group(0) @binding(1) var<storage, read_write> triangles: array<Triangle>;
        @group(0) @binding(2) var<storage, read_write> emptyCellAccel: array<array<u32, ACCEL_DIV_X>, ACCEL_DIV_Z>;
        @group(0) @binding(3) var<storage, read_write> microAccel: array<
                                                                  array<
                                                                   array<
                                                                    array< atomic<u32>, MICRO_ACCEL_MAX_CELL_COUNT>
                                                                      ,MICRO_ACCEL_DIV>
                                                                        ,MICRO_ACCEL_DIV>
                                                                         ,MICRO_ACCEL_DIV> ;
        @group(0) @binding(4) var<storage, read_write> accelTri: array<
                                                                  array<
                                                                   array<
                                                                    array< u32, ACCEL_MAX_CELL_COUNT>
                                                                      ,ACCEL_DIV_X>
                                                                        ,ACCEL_DIV_Y>
                                                                         ,ACCEL_DIV_Z> ;
        @group(0) @binding(5) var<storage, read_write> dbg: array<f32>;

        struct AltUniforms{
           data:vec4f
        };

        // Fast changing
        @group(1) @binding(0) var<uniform> altUni: AltUniforms;
        @group(1) @binding(1) var<storage, read_write> rayIn: array<array<RayIn, WORKGROUP_SIZE>>;
        @group(1) @binding(2) var<storage, read_write> rayResult: array<array<RayResult, WORKGROUP_SIZE>>;
        @group(1) @binding(3) var frame_buffer: texture_storage_2d<${canvasformat}, write>;

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
            var per_cell_delta = (tri_scene_max - tri_scene_min) / vec3f(ACCEL_DIV_X, ACCEL_DIV_Y, ACCEL_DIV_Z);
            return per_cell_delta;
        }

        fn pos_to_cell( pos:vec3f) -> vec3f {
            var cell_loc_f = (pos - uni.tri_pos_min.xyz) / per_cell_delta();
            return cell_loc_f;
        }


          
        fn pos_to_micro_cell( pos:vec3f) -> vec3f {
            var tri_scene_min = uni.tri_pos_min.xyz;
            var tri_scene_max = uni.tri_pos_max.xyz;
            var micro_per_cell_delta = (tri_scene_max- tri_scene_min) / f32(MICRO_ACCEL_DIV);
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

        @compute @workgroup_size(WORKGROUP_SIZE)
        fn mainCameraRayGen(  @builtin(local_invocation_index) local_idx:u32,
        @builtin(	workgroup_id) wg_id:vec3u,
        @builtin( num_workgroups) num_wg:vec3u ) {
            var tile_x = local_idx % 16u;
            var tile_y = local_idx / 16u;
 
            var pix_x = tile_x + wg_id.x * 16u;
            var pix_y = tile_y + wg_id.y * 16u;

            var homo_xy = (vec2f(f32(pix_x)/uni.canvas_size.x,f32(pix_y)/uni.canvas_size.y)-vec2f(0.5,0.5)) * 2.0;
            // cam transform haxz
            homo_xy *= 0.7;// fov
            homo_xy.y = - homo_xy.y;
            homo_xy.x = - homo_xy.x;
            var ray_orig = vec3(0.4,0.3, 0);
            var ray_vec = normalize(vec3f(homo_xy, 1.0));
            let rot =  uni.time_in *0.1-1;  
            {
              let s_pos = ray_vec;
              ray_vec.x= s_pos.x * cos(rot) + s_pos.z * -sin(rot);
              ray_vec.z = s_pos.x * sin(rot) + s_pos.z * cos(rot);
            }

            rayIn[wg_id.x + num_wg.x * wg_id.y][local_idx].ray_orig = ray_orig;
            rayIn[wg_id.x + num_wg.x * wg_id.y][local_idx].ray_vec = ray_vec;
            rayIn[wg_id.x + num_wg.x * wg_id.y][local_idx].px = pix_x;
            rayIn[wg_id.x + num_wg.x * wg_id.y][local_idx].py = pix_y;
        }

        @compute @workgroup_size(WORKGROUP_SIZE)
        fn mainBounceGen(  @builtin(local_invocation_index) local_idx:u32,
        @builtin(	workgroup_id) wg_id:vec3u,
        @builtin( num_workgroups) num_wg:vec3u ) {
            var tile_x = local_idx % 16u;
            var tile_y = local_idx / 16u;

            var hit_ray_dir =  rayIn[wg_id.x + num_wg.x * wg_id.y][local_idx].ray_vec;
            var hit_ray_pos =  rayIn[wg_id.x + num_wg.x * wg_id.y][local_idx].ray_orig;
 
            var pix_x = tile_x + wg_id.x * 16u;
            var pix_y = tile_y + wg_id.y * 16u;
            var tri_idx = rayResult[wg_id.x + num_wg.x * wg_id.y][local_idx].tri;
            var curr_tri = triangles[tri_idx];
            var normal = normalize(cross(curr_tri.pos1 - curr_tri.pos0, curr_tri.pos2 - curr_tri.pos0));
            if(dot(normal, hit_ray_dir) > 0.0){
               // normal should point the opposite of the cam dir.
               // Technically we should know this but we dont because bad mesh data
               normal = -normal; 
            }

            // Back off the surface a bit 
            rayIn[wg_id.x + num_wg.x * wg_id.y][local_idx].ray_orig = hit_ray_pos - hit_ray_dir * 0.00001;
            rayIn[wg_id.x + num_wg.x * wg_id.y][local_idx].ray_vec = normal;
            rayIn[wg_id.x + num_wg.x * wg_id.y][local_idx].px = pix_x;
            rayIn[wg_id.x + num_wg.x * wg_id.y][local_idx].py = pix_y;
        }


        
        @compute @workgroup_size(WORKGROUP_SIZE)
        fn main(  @builtin(local_invocation_index) local_idx:u32,
        @builtin(	workgroup_id) wg_id:vec3u,
        @builtin( num_workgroups) num_wg:vec3u) {
            var ray_orig =  rayIn[wg_id.x + num_wg.x * wg_id.y][local_idx].ray_orig;
            var ray_vec =  rayIn[wg_id.x + num_wg.x * wg_id.y][local_idx].ray_vec;

            var min_t = 111111.0;
            var color_tri = vec3f(0.2,0.2,0.0);
            var cell_loc_f = pos_to_cell(ray_orig);
            var cell_loc_i = vec3i(cell_loc_f);
            var cell_loc_remain = cell_loc_f - vec3f(cell_loc_i);

            // Normalize remain to be in the positive direction
            var remain_dir = select(cell_loc_remain, vec3f(1.0) - cell_loc_remain, ray_vec >= vec3f(0,0,0));

            remain_dir *= per_cell_delta(); 
            var max_cell_count = 0u;
            for(var finite_loop = 0u; finite_loop < 300; finite_loop++) {
                var max_accel_size = vec3i(ACCEL_DIV_X, ACCEL_DIV_Y,ACCEL_DIV_Z);
                //cell_loc_i = vec3i(pos_to_cell(ray_orig +ray_vec*f32(finite_loop) *0.01 ));
                if(any(cell_loc_i < vec3i(0)) || any(cell_loc_i >= max_accel_size) ){
                  break;
                }
                
                if((emptyCellAccel[cell_loc_i.z][cell_loc_i.x] & (1u<<u32(cell_loc_i.y))) == 0){
                  var count_cell = accelTri[cell_loc_i.z][cell_loc_i.y][cell_loc_i.x][0];
                  max_cell_count = max(max_cell_count, count_cell);
                  // cells start after zeroth index!
                  for(var i = 1u; i < count_cell + 1; i++) {
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

            var pix_x = rayIn[wg_id.x + num_wg.x * wg_id.y][local_idx].px;
            var pix_y = rayIn[wg_id.x + num_wg.x * wg_id.y][local_idx].py;

            let pix_pos = vec2u(pix_x, pix_y);
              // This can happen because rounding of workgroup size vs resolution
            if(pix_pos.x < u32(uni.canvas_size.x) || pix_pos.y < u32(uni.canvas_size.y)){     
              //color_tri =  color_tri +vec3f(f32(max_cell_count)/128.0);   
              textureStore(frame_buffer, pix_pos , vec4f(color_tri, 1));
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
                                              ACCEL_DIV_X / MICRO_ACCEL_DIV>,
                                              ACCEL_DIV_Y / MICRO_ACCEL_DIV>,
                                              ACCEL_DIV_Z / MICRO_ACCEL_DIV> ;

        @compute @workgroup_size(WORKGROUP_SIZE)
        fn accelmain(  @builtin(local_invocation_index) local_idx:u32,
        @builtin(	workgroup_id) wg_id:vec3u) {
            let tri_scene_min = uni.tri_pos_min.xyz;
            let per_cell_delta = per_cell_delta();
            // The microAccel has tri list at a coarse grained level
            let total_count = atomicLoad(&microAccel[wg_id.z][wg_id.y][wg_id.x][0]);
            var micro_tri_idx_thread = local_idx;

            while(micro_tri_idx_thread < total_count) {
              // Each thread loads a different triangle. +1 because of silly counter at start
              var real_tri_index =  atomicLoad(&microAccel[wg_id.z][wg_id.y][wg_id.x][micro_tri_idx_thread + 1]); 
              var curr_tri = triangles[real_tri_index];
              for(var xx = 0u; xx < u32( ACCEL_DIV_X / MICRO_ACCEL_DIV ); xx++){
                for(var yy = 0u; yy < u32( ACCEL_DIV_Y /  MICRO_ACCEL_DIV ); yy++){
                  for(var zz = 0u; zz < u32( ACCEL_DIV_Z / MICRO_ACCEL_DIV ); zz++){
                    var x = xx +  wg_id.x * u32( ACCEL_DIV_X / MICRO_ACCEL_DIV );
                    var y = yy +  wg_id.y * u32( ACCEL_DIV_Y / MICRO_ACCEL_DIV );
                    var z = zz +  wg_id.z * u32( ACCEL_DIV_Z / MICRO_ACCEL_DIV );
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
              micro_tri_idx_thread += WORKGROUP_SIZE;// next batch of triangles
            }

            workgroupBarrier(); // THIS IS NO JOKE
            // Use 1 thread to assign count to first index. (no contention)
            if(local_idx == 0) {
              for(var xx = 0u; xx < u32( ACCEL_DIV_X / MICRO_ACCEL_DIV ); xx++) {
                for(var yy = 0u; yy < u32( ACCEL_DIV_Y /  MICRO_ACCEL_DIV ); yy++) {
                  for(var zz = 0u; zz < u32( ACCEL_DIV_Z / MICRO_ACCEL_DIV ); zz++) {
                      var x = xx +  wg_id.x * u32( ACCEL_DIV_X / MICRO_ACCEL_DIV );
                      var y = yy +  wg_id.y * u32( ACCEL_DIV_Y / MICRO_ACCEL_DIV );
                      var z = zz +  wg_id.z * u32( ACCEL_DIV_Z / MICRO_ACCEL_DIV );
                      let temp_count = atomicLoad(&wg_cell_count[zz][yy][xx]); 
                      accelTri[z][y][x][0] = temp_count;
                      if(temp_count == 0){
                        // bit mask to say we are empty (again single threaded)
                        emptyCellAccel[z][x] =  emptyCellAccel[z][x] | (1u<<y); 
                      }
                  }
                }
              }
            }
        }

        @compute @workgroup_size(WORKGROUP_SIZE)
        fn microAccelmain(  @builtin(local_invocation_index) local_idx:u32,
        @builtin(	workgroup_id) wg_id:vec3u) {
            let tri_idx = local_idx + wg_id.x * WORKGROUP_SIZE;
            if(tri_idx >=  arrayLength(&triangles)){
              // out of bounds. Expected.
              return;
            }

            let curr_tri = triangles[tri_idx];
            let tri_scene_min = uni.tri_pos_min.xyz;
            let tri_scene_max = uni.tri_pos_max.xyz;
            let per_cell_delta = (tri_scene_max- tri_scene_min) / f32(MICRO_ACCEL_DIV);
            for(var x = 0u; x < u32(MICRO_ACCEL_DIV); x++){
              for(var y = 0u; y < u32(MICRO_ACCEL_DIV); y++){
                for(var z = 0u; z < u32(MICRO_ACCEL_DIV); z++){
                var xyz = vec3f(f32(x),f32(y),f32(z));
                var cell_min = per_cell_delta * xyz  + tri_scene_min;
                var cell_max = per_cell_delta * (xyz + vec3f(1.0)) + tri_scene_min;

                if(box_intersects_triangle(cell_min, cell_max, curr_tri)){
                    var unique_idx = atomicAdd(&microAccel[z][y][x][0], 1);
                    if(unique_idx < (MICRO_ACCEL_MAX_CELL_COUNT - 3)){// safety check
                      atomicStore(&microAccel[z][y][x][unique_idx + 1], tri_idx); 
                    }
                }
              }
            }
           }
        }
      `
  });

  computeCommonBindGroupLayout = device.createBindGroupLayout({
    label: "compute Common bindgroup Layout",
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
    }, 
    {
      binding: 5,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage" }
    },]
  });


  secondaryBindGroupLayout = device.createBindGroupLayout({
    label: "Secondary compute bindgroup Layout",
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
      storageTexture: { access: "write-only", format: canvasformat }
    }]
  });


  const computePipelineLayout = device.createPipelineLayout({
    label: "Full pipeline layout",
    bindGroupLayouts: [computeCommonBindGroupLayout, secondaryBindGroupLayout],
  });

  commonComputeBinding = device.createBindGroup({
    label: "Common Global binding",
    layout: computeCommonBindGroupLayout,
    entries: [{
      binding: 0,
      resource: { buffer: uniformBuffer }
    }, {
      binding: 1, 
      resource: { buffer: triStateStorage },
    }, 
    {
      binding: 2,
      resource: { buffer: emptyCellAccelBuff },
    },
    {
      binding: 3,
      resource: { buffer: microTriAccelBuffer },
    }, 
    {
      binding: 4,
      resource: { buffer: triAccelBuffer },
    },
    {
      binding: 5,
      resource: { buffer: debuggingBufferStorage },
    },],
  });

  rayTracePipeline = device.createComputePipeline({
    label: "main",
    layout: computePipelineLayout,
    compute: {
      module: drawShaderModule,
      entryPoint: "main"
    }
  });

  accel_pipe = device.createComputePipeline({
    label: "accelmain",
    layout: computePipelineLayout,
    compute: {
      module: drawShaderModule,
      entryPoint: "accelmain"
    }
  });

  micro_accel_pipe = device.createComputePipeline({
    label: "microAccelmain",
    layout: computePipelineLayout,
    compute: {
      module: drawShaderModule,
      entryPoint: "microAccelmain"
    }
  });

  cam_ray_gen_pipe = device.createComputePipeline({
    label: "mainCameraRayGen",
    layout: computePipelineLayout,
    compute: {
      module: drawShaderModule,
      entryPoint: "mainCameraRayGen"
    }
  });

  bounceGenPipeline = device.createComputePipeline({
    label: "mainBounceGen",
    layout: computePipelineLayout,
    compute: {
      module: drawShaderModule,
      entryPoint: "mainBounceGen"
    }
  });
  

  secondaryUniformBuffer = device.createBuffer({
    label: "Secondary Uniforms",
    size: 128,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

}

function update_compute_particles(encoder, step) {
 // encoder.clearBuffer(rayInBufferStorage);

 {
  const uniformArray = new Float32Array([canvas_width, canvas_height, canvas_width_block, time_t,
      tri_pos_min_x, tri_pos_min_y, tri_pos_min_z, 0.0,
      tri_pos_max_x, tri_pos_max_y, tri_pos_max_z, 0.0]);
    device.queue.writeBuffer(secondaryUniformBuffer, 0, uniformArray);
 }
  const computePass = encoder.beginComputePass();
  var secondaryComputeBinding = device.createBindGroup({
    label: "Secondary alt binding",
    layout: secondaryBindGroupLayout,
    entries: [{
      binding: 0,
      resource: { buffer: secondaryUniformBuffer }
    },
    {
      binding: 1,
      resource: { buffer: rayInBufferStorage },
    },
    {
      binding: 2,
      resource: { buffer: rayResultBufferStorage },
    }, 
    { binding: 3, resource: context.getCurrentTexture().createView()},
    ],
  });

  if(step == 1){
    // Acceleration structures
    computePass.setPipeline(micro_accel_pipe);
    computePass.setBindGroup(0, commonComputeBinding);
    computePass.setBindGroup(1, secondaryComputeBinding);
    var num_wg = Math.ceil(numTriangles/ WORKGROUP_SIZE);
    computePass.dispatchWorkgroups(num_wg);

    computePass.setPipeline(accel_pipe);
    computePass.setBindGroup(0, commonComputeBinding);
    computePass.setBindGroup(1, secondaryComputeBinding);
    computePass.dispatchWorkgroups(MICRO_ACCEL_DIV, MICRO_ACCEL_DIV, MICRO_ACCEL_DIV);
  }

  // cam gen
  {
    computePass.setPipeline(cam_ray_gen_pipe);
    computePass.setBindGroup(0, commonComputeBinding);
    computePass.setBindGroup(1, secondaryComputeBinding);
    // We break everything into 16x16 tiles which also correspond to a workgroup.
    const dispatch_width =  Math.ceil(canvas_width / 16);
    const dispatch_height =  Math.ceil(canvas_height / 16);
    computePass.dispatchWorkgroups(dispatch_width, dispatch_height);
  }

  // Raytrace
  {
    computePass.setPipeline(rayTracePipeline);
    computePass.setBindGroup(0, commonComputeBinding);
    computePass.setBindGroup(1, secondaryComputeBinding);
    // We break everything into 16x16 tiles which also correspond to a workgroup.
    const dispatch_width =  Math.ceil(canvas_width / 16);
    const dispatch_height =  Math.ceil(canvas_height / 16);
    computePass.dispatchWorkgroups(dispatch_width, dispatch_height);
  }

  // Fuzzy Bounce gen (takes hits and converts to new rays(norms))
  {
    computePass.setPipeline(bounceGenPipeline);
    computePass.setBindGroup(0, commonComputeBinding);
    computePass.setBindGroup(1, secondaryComputeBinding);
    // We break everything into 16x16 tiles which also correspond to a workgroup.
    const dispatch_width =  Math.ceil(canvas_width / 16);
    const dispatch_height =  Math.ceil(canvas_height / 16);
    computePass.dispatchWorkgroups(dispatch_width, dispatch_height);
  }

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




function ImportTriangleData(){
  var meshData = cr_data;
  const dataNumFloatsPerTriangle = (4 * 3); // No w component
  const numDataTriangles = meshData.length / dataNumFloatsPerTriangle;
  const numFloatsPerTriangle = 4 * 4; // v0,v1,v2,col;
  const numLightsX = 10;
  const numLightsZ = 10;
  const trianglesPerLight = 2;
  const numTrianglesForLights = trianglesPerLight * numLightsX * numLightsZ;
  numTriangles = numTrianglesForLights + numDataTriangles;
  const triStateArray = new Float32Array(numFloatsPerTriangle * numTriangles);
  triStateStorage =
    device.createBuffer({
      label: "Triangles Buffer",
      size: triStateArray.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });


  var triDataInIdx = 0;
  var triDataPutIdx = 0;
  const scalingXYZ = 0.03;
  const bRandColor = false;
  function fAddTriData(single_data){
    triStateArray[triDataPutIdx++] = single_data;
  }
  while (triDataInIdx < meshData.length) {
    // v0, v1, v2, col (3 idx)
    for (var i = 0; i < 4; i++) {
      if(i == 3 && bRandColor){
        fAddTriData(Math.random());
        fAddTriData(Math.random());
        fAddTriData(Math.random());
        triDataInIdx+=3;
      }
      else
      {
        const localScale = i==3 ? 1.0 : scalingXYZ;
        var x = meshData[triDataInIdx++] * localScale;
        var y = meshData[triDataInIdx++] * localScale;
        var z = meshData[triDataInIdx++] * localScale;
       
        if(i!=3){
          tri_pos_max_x = Math.max(x, tri_pos_max_x);
          tri_pos_max_y = Math.max(y, tri_pos_max_y);
          tri_pos_max_z = Math.max(z, tri_pos_max_z);

          tri_pos_min_x = Math.min(x, tri_pos_min_x);
          tri_pos_min_y = Math.min(y, tri_pos_min_y);
          tri_pos_min_z = Math.min(z, tri_pos_min_z);
        }

        fAddTriData(x);
        fAddTriData(y);
        fAddTriData(z);
      }
      fAddTriData(0.0);
    }
  }

  // Extend out a tiny bit to avoid float issues
  var epsilon2 = 0.000001;
  tri_pos_max_x += epsilon2;
  tri_pos_max_y += epsilon2;
  tri_pos_max_z += epsilon2;
  tri_pos_min_x -= epsilon2;
  tri_pos_min_y -= epsilon2;
  tri_pos_min_z -= epsilon2;

  function fAddTriDataV(xx, yy,zz,ww){
    fAddTriData(xx); fAddTriData(yy); fAddTriData(zz); fAddTriData(ww);
  }

  // Manual lights for conference room
  for(var i=0; i < numLightsX; i++){
    for(var j=0; j < numLightsZ; j++){
        // 4x4 steps but inset actual rect
        var y_height = tri_pos_max_y -0.0001;
        var inset_size = 0.003;
        var step_x = (tri_pos_max_x - tri_pos_min_x)/numLightsX;
        var x_start = (step_x * i) + tri_pos_min_x + inset_size;
        var x_end = (step_x * (i+1)) + tri_pos_min_x - inset_size;

        var step_z = (tri_pos_max_z - tri_pos_min_z)/numLightsZ;
        var z_start = (step_z * j) + tri_pos_min_z + inset_size;
        var z_end = (step_z * (j+1)) + tri_pos_min_z -inset_size;

        // first triangle
        fAddTriDataV(x_start, y_height, z_start, 0);
        fAddTriDataV(x_end, y_height, z_start, 0);
        fAddTriDataV(x_start, y_height, z_end, 0);
        fAddTriDataV(1, 1, 1, 1);


        // second tri
        fAddTriDataV(x_start, y_height, z_end, 0);
        fAddTriDataV(x_end, y_height, z_start, 0);
        fAddTriDataV(x_end, y_height, z_end, 0);
        fAddTriDataV(1, 1, 1, 1);
    }
  }


  device.queue.writeBuffer(triStateStorage, 0, triStateArray);
}