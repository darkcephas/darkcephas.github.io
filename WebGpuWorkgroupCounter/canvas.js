"use strict";

var device;
var context;
var querySet;
var queryBuffer;

var testresulttext;
const kDebugArraySize = 32 * 1024 * 1024;
const kInputBufferCount = 32 * 1024 * 1024;
const kStagedReadSize = 1024;
const timestampCapacity = 16;//Max number of timestamps we can store

var maxWorkgroupSize= 256;
var maxNumIterations = 10;
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


function setRunPass(passed_string) {
  testresulttext.innerHTML = passed_string;
  testresulttext.style.color = "green";
}

function setRunError(error_string) {
  testresulttext.innerHTML = error_string;
  testresulttext.style.color = "red";
}




window.onload = async function () {
  const runthebenchmark = document.querySelector('#runtestbutton');
  runthebenchmark.addEventListener('click', RunBenchmark);
  testresulttext = document.querySelector("#testresulttext");

  var default_code =
    `
@group(0) @binding(0) var <storage, read>  _in : array < f32 >;
@group(0) @binding(1) var <storage, read_write> _out: array < f32 >;
@group(0) @binding(2) var <storage, read_write> _histo: array < atomic < u32 > >;


const kCoreActiveSlot = 1024 * 4;
const kNumIterCounterSlot = 1024 * 8;
override oDispatchSize: u32;
override oMaxWorkgroupSize: u32; 
override oMaxNumIterations: u32;

var<workgroup> wIsWaitDone:u32; // default zero

@compute @workgroup_size(oMaxWorkgroupSize)
  fn computeMain(@builtin(local_invocation_index) idx: u32,
  @builtin(workgroup_id) wg: vec3u,
  @builtin(num_workgroups)  wgs : vec3u) {

  if(idx == 0){
    var num_iter = atomicAdd(&_histo[kNumIterCounterSlot],1);
    if(num_iter >= (oMaxNumIterations-1)*oDispatchSize) {
      wIsWaitDone = 1;
    }
  }

  var isWaitDone = workgroupUniformLoad(&wIsWaitDone);
  if(isWaitDone ==0){
    return;  
  }

  let linear_idx = idx + wg.x * wgs.x;
  if(idx == 0){
    atomicAdd(& _histo[kCoreActiveSlot], 1);
  }
  workgroupBarrier();

  const kNumLoops = 200;
  var data = _in[linear_idx];
  for(var i =0u;i < kNumLoops;i++){
    data = tanh(data);
  }

  workgroupBarrier();
  if(idx == 0){
    atomicAdd(& _histo[atomicLoad(& _histo[kCoreActiveSlot])], 1);
  }

  for(var i =0u;i < kNumLoops;i++){
    data = tanh(data);
  }
  _out[linear_idx] = data;

  workgroupBarrier();
  if(idx == 0){
    atomicSub(& _histo[kCoreActiveSlot], 1);
  }
}
`;

  const paramsString = window.location.search;
  const mySearchParams = new URLSearchParams(paramsString);

  document.getElementById('shadertexta').value = default_code


  for (const [key, value] of mySearchParams) {
    if (key == 'codea') {
      document.getElementById('shadertexta').value = decodeURIComponent(value);
    }
    if (key == 'enablesubgroup') {
      document.getElementById('enablesubgroup').checked = 1;
    }
    if (key == 'dispatchsizeuseb') {
      document.getElementById('dispatchsizeuseb').checked = 1;
    }
  }

  //https://stackoverflow.com/questions/6637341/use-tab-to-indent-in-textarea
  var tabs_allow_func = function (e) {
    if (e.key == 'Tab') {
      e.preventDefault();
      var start = this.selectionStart;
      var end = this.selectionEnd;

      // set textarea value to: text before caret + tab + text after caret
      this.value = this.value.substring(0, start) +
        "\t" + this.value.substring(end);

      // put caret at right position again
      this.selectionStart =
        this.selectionEnd = start + 1;
    }
  };

  document.getElementById('shadertexta').addEventListener('keydown', tabs_allow_func);

  const getcurrenturl = document.querySelector('#getcurrenturl');
  getcurrenturl.addEventListener('click', CurrentURLToCopy);
}

async function InitGPU() {
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

  maxWorkgroupSize =  Math.min(adapter.limits.maxComputeInvocationsPerWorkgroup,adapter.limits.maxComputeWorkgroupSizeX);

  var enablesubgroupflag = document.getElementById('enablesubgroup').checked;

  // We dont need timestamps for this code to work but this is a prototype.
  //     requiredLimits: {    maxComputeInvocationsPerWorkgroup: 1024,maxComputeWorkgroupSizeX: 1024, maxComputeWorkgroupStorageSize: 32768}
  // , enablesubgroupflag ? "subgroups" : ""
  var features_list = ["timestamp-query"];
  if (enablesubgroupflag) {
    features_list.push("subgroups");
  }

  device = await adapter.requestDevice({
    requiredFeatures: features_list,
    requiredLimits: {
      maxComputeInvocationsPerWorkgroup:maxWorkgroupSize,
      maxComputeWorkgroupSizeX:maxWorkgroupSize,
    }
  });

  context = canvas.getContext("webgpu");
  var canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device: device,
    format: canvasFormat,
  });
}




async function CurrentURLToCopy() {
  //https://stackoverflow.com/questions/400212/how-do-i-copy-to-the-clipboard-in-javascript
  var full_url = window.location.origin + window.location.pathname + '?codea=' + encodeURIComponent(document.getElementById('shadertexta').value) +
    + '&dispatchcubedid=' + encodeURIComponent(document.getElementById('dispatchcubedid').value);

  if (document.getElementById('enablesubgroup').checked) {
    full_url += '&enablesubgroup=1';
  }

  navigator.clipboard.writeText(full_url).then(function () {
    console.log('Async: Copying to clipboard was successful!');
  }, function (err) {
    console.error('Async: Could not copy text: ', err);
  });
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}


async function RunBenchmark() {
  await InitGPU();
  querySet = device.createQuerySet({
    type: "timestamp",
    count: timestampCapacity,
  });
  queryBuffer = device.createBuffer({
    size: 8 * timestampCapacity,
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
      buffer: { type: "storage" } // histo
    }]
  });

  const pipelineLayout = device.createPipelineLayout({
    label: "Main Pipeline Layout",
    bindGroupLayouts: [bindGroupLayout],
  });


  var shaderCodeA = document.getElementById('shadertexta').value;

  var dispatch_cube_size = Number(document.getElementById("dispatchcubedid").value);

  let shaderModuleA = device.createShaderModule({
    label: "Benchmark compute shader",
    code: shaderCodeA,
  });

  // Create a compute pipeline that updates the game state.
  let computePipelineA = device.createComputePipeline({
    label: "Render pipeline",
    layout: pipelineLayout,
    compute: {
      module: shaderModuleA,
      entryPoint: "computeMain",
      constants: {
        oDispatchSize : dispatch_cube_size,
        oMaxWorkgroupSize: maxWorkgroupSize,
        oMaxNumIterations:maxNumIterations,
      }
    },
  });


  // fill buffer with the init data.
  const floatInitArray = new Float32Array(kInputBufferCount);
  let inputBufferStorage =
    device.createBuffer({
      label: "Init input array",
      size: floatInitArray.byteLength, // this isnt quite right...
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

  for (let i = 0; i < floatInitArray.length; i++) {
    floatInitArray[i] = Math.random();
  }
  device.queue.writeBuffer(inputBufferStorage, 0, floatInitArray, 0, floatInitArray.length);

  let debuggingBufferStorage =
    device.createBuffer({
      label: "debugging storage result",
      size: kDebugArraySize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

  let histogramBufferStorage =
    device.createBuffer({
      label: "histogram storage result",
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
        resource: { buffer: debuggingBufferStorage }
      }
        , {
        binding: 2,
        resource: { buffer: histogramBufferStorage }
      }
      ],
    });

  var time_in_seconds;
  const kNumRuns = 10;
  for(var i =0; i < kNumRuns; i++){
    const encoder = device.createCommandEncoder();
    const computePass = encoder.beginComputePass({
      label: "Timing request",
      timestampWrites: { querySet: querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 },
    });

    computePass.setPipeline( computePipelineA );
    computePass.setBindGroup(0, commonBindGroup);

    computePass.dispatchWorkgroups(dispatch_cube_size, 1, 1);
    computePass.end();

    const stagingBufferDebug = device.createBuffer({
      size: kStagedReadSize,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    encoder.copyBufferToBuffer(
      histogramBufferStorage,
      0, // Source offset
      stagingBufferDebug,
      0, // Destination offset
      kStagedReadSize
    );

    encoder.resolveQuerySet(
      querySet,
      0,// index of first query to resolve 
      timestampCapacity,//number of queries to resolve
      queryBuffer,
      0);// destination offset

    const commandBuffer = encoder.finish();

    device.queue.submit([commandBuffer]);
    const arrayBuffer = await readBuffer(device, queryBuffer);
    const timingsNanoseconds = new BigInt64Array(arrayBuffer);
    time_in_seconds = Number(timingsNanoseconds[1] - timingsNanoseconds[0]) / 1000000.0;


    await stagingBufferDebug.mapAsync(
      GPUMapMode.READ,
      0, // Offset
      kStagedReadSize // Length
    );
    const copyArrayBuffer = stagingBufferDebug.getMappedRange();
    const data = copyArrayBuffer.slice();
    stagingBufferDebug.unmap();
    const data_to_print = new Uint32Array(data);

    document.getElementById('resultstexta').value = data_to_print;
    setRunPass("Runs A time " + time_in_seconds.toFixed(5) + "ms ");
    await sleep(200);
  }

 
}

