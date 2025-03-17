"use strict";

var device;
var context;


var inputFileResult = null;
var inflated_words = null;
var querySet;
var queryBuffer;

var num_dispatch = 1;
var shader_code = null;
var loadingTextElement;
var decompressionTextElement;
var savedataTextElement;
var radioOriginalElement;
var radioByteElement;
var radioShaderLUTElement;
var radioShaderPipelineElement;
var radioShaderParallelElement;
const kDebugArraySize = 1024*256;
var output_file_name = "";
const capacity = 3;//Max number of timestamps we can store

// https://omar-shehata.medium.com/how-to-use-webgpu-timestamp-query-9bf81fb5344a
// For timestamps
async function readBuffer(device, buffer) {
  const size = buffer.size;
  const gpuReadBuffer = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const copyEncoder = device.createCommandEncoder();
  copyEncoder.copyBufferToBuffer(buffer, 0, gpuReadBuffer, 0, size);
  const copyCommands = copyEncoder.finish();
  device.queue.submit([copyCommands]);
  await gpuReadBuffer.mapAsync(GPUMapMode.READ);
  return gpuReadBuffer.getMappedRange();
}



async function loadShaderFromDisk(shader_name) {
  const f = await fetch(shader_name);
  console.log(f);
  const str_file = await f.blob();
  const array_buff = await str_file.arrayBuffer();
  const decoder = new TextDecoder();
  return decoder.decode(array_buff);
}

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
  // We dont need timestamps for this code to work but this is a prototype.
  device = await adapter.requestDevice({
    requiredFeatures: ["timestamp-query"],
    requiredLimits: {
      maxComputeInvocationsPerWorkgroup:1024,
      maxComputeWorkgroupSizeX:1024,
      maxStorageBufferBindingSize:536870912,
      maxBufferSize:536870912
    }
  });


  context = canvas.getContext("webgpu");
  var canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device: device,
    format: canvasFormat,
  });

  const decompress = document.querySelector('#decompress');
  decompress.addEventListener('click', RunDecompression);

  loadingTextElement = document.querySelector("#loadingtext");
  decompressionTextElement = document.querySelector("#decompressiontext");
  savedataTextElement = document.querySelector("#savedatatext");

}



async function RunDecompression() {
  // dynamic sided parts of header

  querySet = device.createQuerySet({
    type: "timestamp",
    count: capacity,
  });
  queryBuffer = device.createBuffer({
    size: 8 * capacity,
    usage: GPUBufferUsage.QUERY_RESOLVE
      | GPUBufferUsage.STORAGE
      | GPUBufferUsage.COPY_SRC
      | GPUBufferUsage.COPY_DST,
  });

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
    }, {
      binding: 3,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage" } // debugging
    }]
  });

  const pipelineLayout = device.createPipelineLayout({
    label: "main Pipeline Layout",
    bindGroupLayouts: [bindGroupLayout],
  });


  shader_code =  await loadShaderFromDisk('shader_parallel_decode.wgsl');

  let forceIndexShaderModule = device.createShaderModule({
    label: "Zip decode shader",
    code: shader_code,
  });

  const kNumElementsSrc =  (128*1024) * (1024);
  // Create a compute pipeline that updates the game state.
  let renderBufferPipeline = device.createComputePipeline({
    label: "Render pipeline",
    layout: pipelineLayout,
    compute: {
      module: forceIndexShaderModule,
      entryPoint: "computeMain",
      constants: {
        DISPATCH_COUNT: num_dispatch,
        NUM_ELEMENTS: kNumElementsSrc
      }
    },
  });



  // Create a uniform buffer that describes the grid.
  const uniformArrayInit = new Uint32Array([2, 3]);
  let uniformBuffer = device.createBuffer({
    label: "Grid Uniforms",
    size: uniformArrayInit.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuffer, 0, uniformArrayInit);
  // Create an array representing the active state of each cell.

  // Each element has 4 bytes as it is an integer
  const kMemCpyFullSizeBytes = 4 * kNumElementsSrc;

  const hostInitSrc = new Uint32Array(kNumElementsSrc);

  for(let i = 0; i < kNumElementsSrc;i++){
    hostInitSrc[i] = i;
  }

  // Create an array representing the active state of each cell.
  //const initInputDataArray = new Uint32Array(100);
  //var as_int = new Int32Array(initInputDataArray.buffer);
  //initInputDataArray[0] = 13;
  // Create two storage buffers to hold the cell state.
  let inputBufferStorage =
    device.createBuffer({
      label: "Init input array",
      size: kMemCpyFullSizeBytes, // this isnt quite right...
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  // fill buffer with the init data.
  const kReadOffset = 0;
  device.queue.writeBuffer(inputBufferStorage, 0, hostInitSrc, kReadOffset, kMemCpyFullSizeBytes/4);


  let outputBufferStorage =
    device.createBuffer({
      label: "Output result",
      size: kMemCpyFullSizeBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });


  let debuggingBufferStorage =
    device.createBuffer({
      label: "debugging storage result",
      size: kDebugArraySize,
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
      }, {
        binding: 3,
        resource: { buffer: debuggingBufferStorage }
      },
    ],
    });



  const encoder = device.createCommandEncoder();
  const computePass = encoder.beginComputePass({
    label: "Timing request",
    timestampWrites: { querySet: querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 },
  });

  computePass.setPipeline(renderBufferPipeline);
  computePass.setBindGroup(0, commonBindGroup);
  computePass.dispatchWorkgroups(num_dispatch);
  computePass.end();
  const stagingBuffer = device.createBuffer({
    size: kMemCpyFullSizeBytes,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  encoder.copyBufferToBuffer(
    outputBufferStorage,
    0, // Source offset
    stagingBuffer,
    0, // Destination offset
    kMemCpyFullSizeBytes
  );

  const stagingBufferDebug = device.createBuffer({
    size: kDebugArraySize,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  encoder.copyBufferToBuffer(
    debuggingBufferStorage,
    0, // Source offset
    stagingBufferDebug,
    0, // Destination offset
    kDebugArraySize
  );

  encoder.resolveQuerySet(
    querySet,
    0,// index of first query to resolve 
    capacity,//number of queries to resolve
    queryBuffer,
    0);// destination offset

  const commandBuffer = encoder.finish();

  device.queue.submit([commandBuffer]);


  // === After `commandEncoder.finish()` is called ===
  // Read the storage buffer data
  const arrayBuffer = await readBuffer(device, queryBuffer);
  // Decode it into an array of timestamps in nanoseconds
  const timingsNanoseconds = new BigInt64Array(arrayBuffer);
  const time_in_seconds = Number(timingsNanoseconds[1] - timingsNanoseconds[0]) / 1000000000.0;

  // Get result ouptut
  await stagingBuffer.mapAsync(
    GPUMapMode.READ,
    0, // Offset
    kMemCpyFullSizeBytes // Length
  );
  const copyArrayBuffer = stagingBuffer.getMappedRange();
  const data = copyArrayBuffer.slice();
  stagingBuffer.unmap();

  inflated_words = new Uint32Array(data, 0, kMemCpyFullSizeBytes/4);
 // var crc_test = crc32(inflated_bytes);
  //console.log(inflated_bytes);
 // var string = new TextDecoder().decode(inflated_bytes);
 // console.log(string);

  {
    await stagingBufferDebug.mapAsync(
      GPUMapMode.READ,
      0, // Offset
      kDebugArraySize // Length
    );
    const copyArrayBuffer = stagingBufferDebug.getMappedRange();
    const data = copyArrayBuffer.slice();
    stagingBufferDebug.unmap();
    console.log(new Uint32Array(data));
  }

  
  for(let i = 0; i < kNumElementsSrc;i++){
    if(inflated_words[i] != i){
        console.log("problem at i=" + i + " value is =" + inflated_words[i] );
        break;
    }
  }
  console.log("Result " + time_in_seconds.toFixed(4) );
  // times 2 because read and write
  const total_mem_seen_gb = (kMemCpyFullSizeBytes*2.0/1000000000)
  console.log("Result " + (total_mem_seen_gb/time_in_seconds).toFixed(2) + " GB/s");

}

