"use strict";

var device;
var canvasformat;
var context;
var debug_mode = false;
var raster_mode = false;
var microTriAccelBuffer;
var emptyCellAccelBuff;

const WORKGROUP_SIZE = 256;
const ACCEL_DIV_X = 128;
const ACCEL_DIV_Y = 32;
const ACCEL_DIV_Z = 64;
const ACCEL_MAX_CELL_COUNT = 128;
const MICRO_ACCEL_DIV = 8;
const MICRO_ACCEL_MAX_CELL_COUNT = 1024*64; // The hope here is that this works :) 300/512
var canvas_width;
var canvas_height;
var canvas_width_block;
var canvas_height_block;
var bindGroupLayout;
var uniformBuffer;
var simulationBindGroups;
var forceIndexBindGroups;
var vizBufferStorage;
var empytBuffer;
var time_t = 0.0;
var depthTexture;
var triAccelBuffer;
var epsilon2 = 0.000001;
var BIG_NUM = 100000.0;
var tri_pos_min_x = BIG_NUM;
var tri_pos_min_y = BIG_NUM;
var tri_pos_min_z = BIG_NUM;
var tri_pos_max_x = -BIG_NUM;
var tri_pos_max_y = -BIG_NUM;
var tri_pos_max_z = -BIG_NUM;



function UpdateUniforms() {
  // Create a uniform buffer that describes the grid.
  const uniformArray = new Float32Array([canvas_width, canvas_height, canvas_width_block, time_t,
                                        tri_pos_min_x, tri_pos_min_y, tri_pos_min_z, 0.0 ,
                                        tri_pos_max_x, tri_pos_max_y, tri_pos_max_z, 0.0]);
  device.queue.writeBuffer(uniformBuffer, 0, uniformArray);
}
window.onload = async function () {

  window.addEventListener('resize', resizeCanvas, false);
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

  function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    canvas_width = canvas.width;
    canvas_height = canvas.height;
    canvas_width_block  =  Math.ceil(canvas_width / 16);
    canvas_height_block  =  Math.ceil(canvas_height / 16);
    var block_size = 16*16;
    const numVizBufferElementBytes = 8*4;
    const numVizBufferTotal = numVizBufferElementBytes * canvas_width_block * canvas_height_block * block_size;
    vizBufferStorage =
      device.createBuffer({
        label: "Viz buffer",
        size: numVizBufferTotal,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });


     empytBuffer =
      device.createBuffer({
        label: "Viz buffer",
        size: numVizBufferTotal,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });


    // only used by rasterizer
    depthTexture = device.createTexture({
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


  canvas_width = canvas.width;
  canvas_height = canvas.height;
  canvasformat = navigator.gpu.getPreferredCanvasFormat();

  var reqFeatures = [];
  if(canvasformat == 'bgra8unorm'){
    reqFeatures =  ['bgra8unorm-storage']
  }
  device = await adapter.requestDevice({ requiredFeatures:reqFeatures });
  
  // Moderate size structure (16mb)
  // Lists of indices
  const size_int_to_byte = 4;
  const accel_buff_size =  ACCEL_DIV_X * ACCEL_DIV_Y * ACCEL_DIV_Z * ACCEL_MAX_CELL_COUNT * size_int_to_byte;
  triAccelBuffer = 
    device.createBuffer({
      label: "Triangle accel",
      size: accel_buff_size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

  const micro_accel_buff_size =  MICRO_ACCEL_DIV * MICRO_ACCEL_DIV * MICRO_ACCEL_DIV * MICRO_ACCEL_MAX_CELL_COUNT * size_int_to_byte;
    microTriAccelBuffer = 
      device.createBuffer({
        label: "Micro Triangle accel",
        size: micro_accel_buff_size,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });


  if(ACCEL_DIV_Y != 32){
    console.log("we assume this for bit packing");
  }
  const empty_cell_accel_buff_size = ACCEL_DIV_Z *  ACCEL_DIV_X * size_int_to_byte;
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

  resizeCanvas();
  var mesh_data = cr_data;
  const numTriangles = mesh_data.length / (4 * 3);
  const numFloatsPerTriangle = 4 * 4; // v0,v1,v2,col;
  const triStateArray = new Float32Array(numFloatsPerTriangle * numTriangles);
  var triStateStorage =
    device.createBuffer({
      label: "Triangle Buffer",
      size: triStateArray.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });


  var triDataInIdx = 0;
  var triDataPutIdx = 0;
  const scalingXYZ = 0.03;
  const bRandColor = false;
  while (triDataInIdx < mesh_data.length) {
    // v0, v1, v2, col (3 idx)
    for (var i = 0; i < 4; i++) {
      if(i == 3 && bRandColor){
        triStateArray[triDataPutIdx++] = Math.random();
        triStateArray[triDataPutIdx++] = Math.random();
        triStateArray[triDataPutIdx++] = Math.random();
        triDataInIdx+=3;
      }
      else
      {
        const localScale = i==3 ? 1.0 : scalingXYZ;
        var x = mesh_data[triDataInIdx++] * localScale;
        var y = mesh_data[triDataInIdx++] * localScale;
        var z = mesh_data[triDataInIdx++] * localScale;
       
        if(i!=3){
          tri_pos_max_x = Math.max(x, tri_pos_max_x);
          tri_pos_max_y = Math.max(y, tri_pos_max_y);
          tri_pos_max_z = Math.max(z, tri_pos_max_z);

          tri_pos_min_x = Math.min(x, tri_pos_min_x);
          tri_pos_min_y = Math.min(y, tri_pos_min_y);
          tri_pos_min_z = Math.min(z, tri_pos_min_z);
        }

        triStateArray[triDataPutIdx++] = x;
        triStateArray[triDataPutIdx++] = y;
        triStateArray[triDataPutIdx++] = z;
      }
      triStateArray[triDataPutIdx++] = 0.0;
    }
  }

  tri_pos_max_x += epsilon2;
  tri_pos_max_y += epsilon2;
  tri_pos_max_z += epsilon2;

  tri_pos_min_x -= epsilon2;
  tri_pos_min_y -= epsilon2;
  tri_pos_min_z -= epsilon2;

  var dbg_zero_loc_x =  16.0 * (0.0-tri_pos_min_x)  / (tri_pos_max_x-tri_pos_min_x);
  var dbg_zero_loc_y =  16.0 * (0.2-tri_pos_min_y)  / (tri_pos_max_y-tri_pos_min_y);
  var dbg_zero_loc_z =  16.0 * (0.0-tri_pos_min_z)  / (tri_pos_max_z-tri_pos_min_z);

  device.queue.writeBuffer(triStateStorage, 0, triStateArray);


  uniformBuffer = device.createBuffer({
    label: "Uniforms",
    size: 128,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  UpdateUniforms();


  const renderShaderModule = device.createShaderModule({
    label: 'renderShaderModule',
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
        pos0:vec3f, padd0:f32,
        pos1:vec3f, padd1:f32,
        pos2:vec3f, padd2:f32,
        col:vec3f, cpadd:f32,
      };

      @group(0) @binding(0) var<uniform> canvas_size: vec4f;
      @group(0) @binding(1) var<storage> renderBufferIn: array<Triangle>;
      @vertex
      fn vertexMain(input: VertexInput) -> VertexOutput {
        var output: VertexOutput;
        let which_triangle = input.instance/3u;
        var pos = vec3f(0,0,0);
        if(input.instance % 3 == 0){
          pos = renderBufferIn[which_triangle].pos0;
        }
        else if(input.instance % 3 == 1){
          pos = renderBufferIn[which_triangle].pos1;
        }
        else if(input.instance % 3 == 2){
          pos = renderBufferIn[which_triangle].pos2;
        }
        output.prim_color = renderBufferIn[which_triangle].col;
        //var test_array= array(vec3f(0,0,.1), vec3f(1,0,.1),vec3f(0,1,.1));
        //pos = test_array[input.instance % 3];
        let s_pos = pos;
        let rot = canvas_size.w*0.2*1.71;
        pos.x= s_pos.x * cos(rot) + s_pos.z * -sin(rot);
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


  var bindGroupLayout = device.createBindGroupLayout({
    label: "render bind group",
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

  const pipelineLayout = device.createPipelineLayout({
    label: "Render Pipeline Layout",
    bindGroupLayouts: [bindGroupLayout],
  });

  var renderPipe = device.createRenderPipeline({
    label: "render pipeline",
    depthStencil: { depthCompare: "less", depthWriteEnabled: true, format: "depth32float" },
    layout: pipelineLayout, // Updated!
    primitive: { cullMode: "front" , frontFace: "cw"},
    vertex: {
      module: renderShaderModule,
      entryPoint: "vertexMain",
    },
    fragment: {
      module: renderShaderModule,
      entryPoint: "fragmentMain",
      targets: [{
        format: canvasformat,
      }]
    }
  });




  var graphicsBindGroup =
    device.createBindGroup({
      label: "graphics bind",
      layout: bindGroupLayout, // Updated Line
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

    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        loadOp: "clear",
        clearValue: { r: 0, g: 0.0, b: 0.0, a: 0.0 },
        storeOp: "store",
      }],
      depthStencilAttachment: {
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "discard",
        view: depthTexture.createView()
      },
    });

    pass.setPipeline(renderPipe);
    pass.setBindGroup(0, graphicsBindGroup); // Updated!
    if(raster_mode){
      pass.draw(numTriangles*3);
    }
    pass.end();


   var buff_ret = null;
   if(!raster_mode){
    buff_ret = update_compute_particles(triStateStorage, triAccelBuffer, microTriAccelBuffer,emptyCellAccelBuff,vizBufferStorage , numTriangles, encoder, step);
   }
    const commandBuffer = encoder.finish();
    device.queue.submit([commandBuffer]);
 
    if(debug_mode)
    {
      buff_ret.mapAsync(
        GPUMapMode.READ,
        0, // Offset
        kDebugArraySize // Length
      ).then(value => 
      {
        const copyArrayBuffer = buff_ret.getMappedRange();
        const data = copyArrayBuffer.slice();
      
        console.log(new Float32Array(data));
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
