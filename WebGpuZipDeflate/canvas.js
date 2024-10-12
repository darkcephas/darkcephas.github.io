"use strict";

var device;
var context;
var forceIndexShaderModule
const WORKGROUP_SIZE = 1;
var readOffset = 0;
var inputFileResult = null;

window.onload = async function () {
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
  device = await adapter.requestDevice();
  context = canvas.getContext("webgpu");
  var canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device: device,
    format: canvasFormat,
  });

  // This loading data is a very clever trick to make this work
  // within the web framework.
  const loaddata = document.querySelector('#loaddata');
  loaddata.addEventListener('click', () => {
    const f = document.createElement('input');
    f.type = 'file';
    f.addEventListener('change', () => {
      const file = new FileReader(f.files[0]);
      file.addEventListener('load', () => {
        // File has loaded
        inputFileResult  = new Uint8Array( file.result);
      });
      file.readAsArrayBuffer(f.files[0]);
    });
    f.click();
  });

  const decompress = document.querySelector('#decompress');
  decompress.addEventListener('click', RunDecompression);

  const savedata = document.querySelector('#savedata');
  savedata.addEventListener('click', saveDataToDisk);
}


/*
struct LocalFileHeader
{
    uint16_t header_signature[2];
    uint16_t version;
    uint16_t bit_flag;
    uint16_t compression_method;
    uint16_t last_modified_time;
    uint16_t last_modified_date;
    uint16_t crc[2];
    uint16_t compressed_size[2];
    uint16_t uncompressed_size[2];
    uint16_t file_name_num_bytes;
    uint16_t file_extra_num_bytes;
    // Additional dynamic length fields removed;
};
*/

function read8(){
  return inputFileResult[readOffset++];
}

function read16(){
  let res = inputFileResult[readOffset++];
  res = res | inputFileResult[readOffset++]<< 8;
  return res;
}

function read32(){
  let res = inputFileResult[readOffset++];
  res = res | inputFileResult[readOffset++]<< 8;
  res = res | inputFileResult[readOffset++]<< 16;
  res = res | inputFileResult[readOffset++]<< 24;
  return res;
}

function RoundTo4(val)
{
  return Math.floor((val + 3)/4)*4;// this could just be a mask
}



async function RunDecompression()  {
  let header_signature = read32();
  let version = read16();
  let bit_flag =  read16();
  let compression_method =  read16();
  let last_modified_time = read16();
  let last_modified_date =  read16();
  let crc_0 =  read32();
  let compressed_size =  read32();
  let uncompressed_size =  read32();
  let file_name_num_bytes =  read16();
  let file_extra_num_bytes = read16();

  let compressed_size_rounded = RoundTo4(compressed_size);
  let uncompressed_size_rounded =  RoundTo4(uncompressed_size);
  console.log(inputFileResult);
  // Create the bind group layout and pipeline layout.
  let bindGroupLayout = device.createBindGroupLayout({
    label: "Cell Bind Group Layout",
    entries: [{
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "read-only-storage" } // input
    }, {
      binding: 1,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage" } // output
    }, {
      binding: 2,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "uniform" } // additional data
    }]
  });

  const pipelineLayout = device.createPipelineLayout({
    label: "main Pipeline Layout",
    bindGroupLayouts: [bindGroupLayout],
  });



  forceIndexShaderModule = device.createShaderModule({
    label: "Force Index shader",
    code: `

          struct CommonData {
                outlen:  u32,       /* available space at out */
               inlen: u32,    /* available input at in */
          };
        
          @group(0) @binding(0) var<storage> in: array<u32>;
            @group(0) @binding(1) var<storage,read_write> out: array<u32>;
           @group(0) @binding(2) var<uniform> ws: CommonData;
  
          @compute @workgroup_size(${WORKGROUP_SIZE})
          fn computeMain(  @builtin(global_invocation_id) global_idx:vec3u,
          @builtin(num_workgroups) num_work:vec3u) {
            for(var i = 0u ;i < ws.inlen;i++){
             out[i]= in[i];
            }
          }
        `
  });




  // Create a compute pipeline that updates the game state.
  let renderBufferPipeline = device.createComputePipeline({
    label: "Render pipeline",
    layout: pipelineLayout,
    compute: {
      module: forceIndexShaderModule,
      entryPoint: "computeMain",
    }
  });



  // Create a uniform buffer that describes the grid.
  const uniformArrayInit = new Uint32Array([compressed_size, uncompressed_size]);
  let uniformBuffer = device.createBuffer({
    label: "Grid Uniforms",
    size: uniformArrayInit.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuffer, 0, uniformArrayInit);
  // Create an array representing the active state of each cell.



  // Create an array representing the active state of each cell.
  //const initInputDataArray = new Uint32Array(100);
  //var as_int = new Int32Array(initInputDataArray.buffer);
  //initInputDataArray[0] = 13;
  // Create two storage buffers to hold the cell state.
  let inputBufferStorage =
    device.createBuffer({
      label: "Init input array",
      size: compressed_size_rounded, // this isnt quite right...
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  // fill buffer with the init data.
  device.queue.writeBuffer(inputBufferStorage, 0, inputFileResult , readOffset, compressed_size_rounded);


  let outputBufferStorage =
    device.createBuffer({
      label: "Output result",
      size:  uncompressed_size_rounded,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });


   let commonBindGroup =
    device.createBindGroup({
      label: "Compute renderer bind group A",
      layout: bindGroupLayout, // Updated Line
      entries: [{
        binding: 0,
        resource: { buffer: inputBufferStorage }
      }, {
        binding: 1,
        resource: { buffer: outputBufferStorage }
      }, {
        binding: 2, //  uniforms
        resource: { buffer: uniformBuffer }
      }],
    });


  const stagingBuffer = device.createBuffer({
    size: uncompressed_size_rounded,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const encoder = device.createCommandEncoder();
  const computePass = encoder.beginComputePass();
  computePass.setPipeline(renderBufferPipeline);
  computePass.setBindGroup(0, commonBindGroup);
  computePass.dispatchWorkgroups(1);
  computePass.end();
  encoder.copyBufferToBuffer(
    outputBufferStorage,
    0, // Source offset
    stagingBuffer,
    0, // Destination offset
    uncompressed_size_rounded
  );
  const commandBuffer = encoder.finish();
  
  device.queue.submit([commandBuffer]);

   await stagingBuffer.mapAsync(
    GPUMapMode.READ,
    0, // Offset
    uncompressed_size_rounded // Length
   );
  const copyArrayBuffer = stagingBuffer.getMappedRange();
  const data = copyArrayBuffer.slice();
  stagingBuffer.unmap();
  console.log(new Uint8Array(data));
}

