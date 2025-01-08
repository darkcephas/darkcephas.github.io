"use strict";

var device;
var context;

var readOffset = 0;
var inputFileResult = null;
var inflated_bytes = null;
var querySet;
var queryBuffer;

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


function setFileLoaded() {
  loadingTextElement.innerHTML = "Done. ";
  loadingTextElement.style.color = "green";
  decompressionTextElement.innerHTML = "Waiting for decompress";
  decompressionTextElement.style.color = "black";
}

function setFileDecompressed(passed_string) {
  decompressionTextElement.innerHTML = passed_string;
  decompressionTextElement.style.color = "green";
  savedataTextElement.innerHTML = "Decompressed data ready.";
}

function setFileDecompressedError(error_string) {
  decompressionTextElement.innerHTML = error_string;
  decompressionTextElement.style.color = "red";
  savedataTextElement.style.color = "red";
}



async function loadDemoFromDisk() {
  const f = await fetch('med_file.zip');
  console.log(f);
  const str_file = await f.blob();
  const array_buff = await str_file.arrayBuffer();
  inputFileResult = new Uint8Array(array_buff);
  setFileLoaded();
}

async function saveDataToDisk() {
  const text = inflated_bytes;
  const blob = new Blob([text], { type: 'text/plain' });
  const link = document.createElement('a');
  link.download = output_file_name;
  link.href = window.URL.createObjectURL(blob);
  link.click();
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
      maxComputeWorkgroupStorageSize:32768
    }
  });


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
    f.accept = ".zip";
    f.addEventListener('change', () => {
      const file = new FileReader(f.files[0]);
      file.addEventListener('load', () => {
        // File has loaded
        inputFileResult = new Uint8Array(file.result);
        setFileLoaded();
      });
      file.readAsArrayBuffer(f.files[0]);
    });
    f.click();
  });

  const decompress = document.querySelector('#decompress');
  decompress.addEventListener('click', RunDecompression);

  const loaddemo = document.querySelector('#loaddemo');
  loaddemo.addEventListener('click', loadDemoFromDisk);

  const savedata = document.querySelector('#savedata');
  savedata.addEventListener('click', saveDataToDisk);


  loadingTextElement = document.querySelector("#loadingtext");
  decompressionTextElement = document.querySelector("#decompressiontext");
  savedataTextElement = document.querySelector("#savedatatext");

  radioOriginalElement = document.querySelector("#decompress_radio_original");
  radioByteElement = document.querySelector("#decompress_radio_byte");
  radioShaderLUTElement = document.querySelector("#decompress_radio_lut");
  radioShaderPipelineElement = document.querySelector("#decompress_radio_pipeline");
  radioShaderParallelElement = document.querySelector("#decompress_radio_parallel");
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

function read8() {
  return inputFileResult[readOffset++];
}

function read16() {
  let res = inputFileResult[readOffset++];
  res = res | (inputFileResult[readOffset++] << 8);
  return res >>> 0;
}

function read32() {
  let res = inputFileResult[readOffset++];
  res = res | (inputFileResult[readOffset++] << 8);
  res = res | (inputFileResult[readOffset++] << 16);
  res = res | (inputFileResult[readOffset++] << 24);
  return res >>> 0;
}

function RoundTo4(val) {
  return Math.floor((val + 3) / 4) * 4;// this could just be a mask
}



async function RunDecompression() {
  readOffset = 0;
  let header_signature = read32();
  let version = read16();
  let bit_flag = read16();
  let compression_method = read16();
  let last_modified_time = read16();
  let last_modified_date = read16();
  let crc_file = read32();
  let compressed_size = read32();
  let uncompressed_size = read32();
  let file_name_num_bytes = read16();
  let file_extra_num_bytes = read16();

  let compressed_size_rounded = RoundTo4(compressed_size);
  let uncompressed_size_rounded = RoundTo4(uncompressed_size);

  // dynamic sided parts of header

  output_file_name = new TextDecoder().decode(inputFileResult.slice(readOffset, readOffset + file_name_num_bytes));
  readOffset += file_name_num_bytes;
  readOffset += file_extra_num_bytes;

  if (compression_method == 0) {
    inflated_bytes = inputFileResult.slice(readOffset, readOffset + uncompressed_size)
    setFileDecompressed("Data not compressed.");
    return;
  }

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
    }, {
      binding: 4,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage" } // debugging
    }, {
      binding: 5,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage" } // debugging
    }]
  });

  const pipelineLayout = device.createPipelineLayout({
    label: "main Pipeline Layout",
    bindGroupLayouts: [bindGroupLayout],
  });


  var shader_code = "";
  var shader_code_decompress = null;
  if(radioOriginalElement.checked){
    shader_code = shaderCode_original;
  }
  if(radioByteElement.checked){
    shader_code = shaderCode_byte;
  }
  if(radioShaderLUTElement.checked){
    shader_code =  await loadShaderFromDisk('shader_lut.wgsl');
  }
  if(radioShaderPipelineElement.checked){
    shader_code =  await loadShaderFromDisk('shader_pipeline.wgsl');
  }

  if(radioShaderParallelElement.checked){
    shader_code =  await loadShaderFromDisk('shader_parallel_decode.wgsl');
    shader_code_decompress =  await loadShaderFromDisk('shader_parallel_decompress.wgsl');
  }

  let forceIndexShaderModule = device.createShaderModule({
    label: "Zip decode shader",
    code: shader_code,
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


  let forceIndexShaderModule_decompress = null;
  let renderBufferPipeline_decompress = null;
  if(shader_code_decompress){
    forceIndexShaderModule_decompress = device.createShaderModule({
      label: "Zip decode shader fpr decompression",
      code: shader_code_decompress,
    });


    // Create a compute pipeline that updates the game state.
    renderBufferPipeline_decompress = device.createComputePipeline({
      label: "Decompression pipe",
      layout: pipelineLayout,
      compute: {
        module: forceIndexShaderModule_decompress,
        entryPoint: "computeMain",
      }
    });
  }



  // Create a uniform buffer that describes the grid.
  const uniformArrayInit = new Uint32Array([uncompressed_size, compressed_size]);
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
  device.queue.writeBuffer(inputBufferStorage, 0, inputFileResult, readOffset, compressed_size_rounded);


  let outputBufferStorage =
    device.createBuffer({
      label: "Output result",
      size: uncompressed_size_rounded,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });



  let debuggingBufferStorage =
    device.createBuffer({
      label: "debugging storage result",
      size: kDebugArraySize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });


    let decodeParallelOutBufferStorage =
    device.createBuffer({
      label: "decode parallel out storage result",
      size: 1024*1024*4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    let decodeParallelControlBufferStorage =
    device.createBuffer({
      label: "decode parallel controll storage result",
      size: 1024*4,
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
      }, {
        binding: 4,
        resource: { buffer: decodeParallelOutBufferStorage }
      }, {
        binding: 5,
        resource: { buffer: decodeParallelControlBufferStorage }
      }
    ],
    });



  const encoder = device.createCommandEncoder();
  const computePass = encoder.beginComputePass({
    label: "Timing request",
    timestampWrites: { querySet: querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 },
  });

  computePass.setPipeline(renderBufferPipeline);
  computePass.setBindGroup(0, commonBindGroup);
  computePass.dispatchWorkgroups(renderBufferPipeline_decompress ? 3 :1);
  if(false){ //renderBufferPipeline_decompress){
    computePass.setPipeline(renderBufferPipeline_decompress);
    computePass.setBindGroup(0, commonBindGroup);
    computePass.dispatchWorkgroups(1);
  }
  computePass.end();
  const stagingBuffer = device.createBuffer({
    size: uncompressed_size_rounded,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  encoder.copyBufferToBuffer(
    outputBufferStorage,
    0, // Source offset
    stagingBuffer,
    0, // Destination offset
    uncompressed_size_rounded
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
    uncompressed_size_rounded // Length
  );
  const copyArrayBuffer = stagingBuffer.getMappedRange();
  const data = copyArrayBuffer.slice();
  stagingBuffer.unmap();

  inflated_bytes = new Uint8Array(data, 0, uncompressed_size);
  var crc_test = crc32(inflated_bytes);
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
  let uncompressed_size_mb = uncompressed_size/1000000;
  if (crc_test != crc_file) {
    setFileDecompressedError("CRC does not match original!  "  + time_in_seconds.toFixed(4) + " s" + " as " + (uncompressed_size_mb/time_in_seconds).toFixed(2) + " Mb/s");
    return;
  }
  setFileDecompressed("CRC match. Decode gpu time= " + time_in_seconds.toFixed(4) + " s" + " as " + (uncompressed_size_mb/time_in_seconds).toFixed(2) + " Mb/s");
}

