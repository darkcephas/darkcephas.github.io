"use strict";

var device;
var context;
var querySet;
var queryBuffer;

var testresulttext;
const kDebugArraySize = 32 * 1024  ;
const kInputBufferCount = 32 * 1024  ;
const kStagedReadSize = 1024;
const timestampCapacity = 16;//Max number of timestamps we can store

var maxNumIterations = 10;
var limitsMaxWorkgroupSize = 256;
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
struct DataT
{
  x : u32,
  y : u32,
}

@group(0) @binding(3) var <uniform> _uniform: DataT ;


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
 if(_uniform.x == 0){
   _out[idx] = tan(_in[idx]);
 }
}
`;

  const paramsString = window.location.search;
  const mySearchParams = new URLSearchParams(paramsString);

  document.getElementById('shadertexta').value = default_code


  for (const [key, value] of mySearchParams) {
    if (key == 'shadertexta') {
      document.getElementById('shadertexta').value = decodeURIComponent(value);
    }
    if (key == 'enableMaxWorkgroupSize') {
      document.getElementById('enableMaxWorkgroupSize').checked = 1;
    }
    if (key == 'dispatchcubedid') {
      document.getElementById('dispatchcubedid').value = decodeURIComponent(value);
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

  var features_list = ["timestamp-query"];
  limitsMaxWorkgroupSize = Math.min(adapter.limits.maxComputeInvocationsPerWorkgroup,
    adapter.limits.maxComputeWorkgroupSizeX);
  device = await adapter.requestDevice({
    requiredFeatures: features_list,
    requiredLimits: {
      maxComputeInvocationsPerWorkgroup: limitsMaxWorkgroupSize,
      maxComputeWorkgroupSizeX: limitsMaxWorkgroupSize,
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
  var full_url = window.location.origin + window.location.pathname + '?shadertexta=' +
    encodeURIComponent(document.getElementById('shadertexta').value) +
    '&dispatchcubedid=' + encodeURIComponent(document.getElementById('dispatchcubedid').value);

  if (document.getElementById('enableMaxWorkgroupSize').checked) {
    full_url += '&enableMaxWorkgroupSize=1';
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
    }, {
      binding: 3,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "uniform" } // additional data
    }]
  });

  const pipelineLayout = device.createPipelineLayout({
    label: "Main Pipeline Layout",
    bindGroupLayouts: [bindGroupLayout],
  });


  var shaderCodeA = document.getElementById('shadertexta').value;
  let shaderModuleA = device.createShaderModule({
    label: "Benchmark compute shader",
    code: shaderCodeA,
  });

  const shaderInfo = await shaderModuleA.getCompilationInfo();

  if (shaderInfo.messages.length > 0) {
    setRunError(shaderInfo.messages[0].message + " line " + shaderInfo.messages[0].lineNum);
    return;
  }

  var enableMaxWorkgroupSize = document.getElementById('enableMaxWorkgroupSize').checked;
  var maxWorkgroupSize = 256;
  if (enableMaxWorkgroupSize) {
    maxWorkgroupSize = limitsMaxWorkgroupSize;
  }


  // Create a compute pipeline that updates the game state.
  var dispatch_cube_size = Number(document.getElementById("dispatchcubedid").value);
  let computePipelineA = device.createComputePipeline({
    label: "Render pipeline",
    layout: pipelineLayout,
    compute: {
      module: shaderModuleA,
      entryPoint: "computeMain",
      constants: {
        oDispatchSize: dispatch_cube_size,
        oMaxWorkgroupSize: maxWorkgroupSize,
        oMaxNumIterations: maxNumIterations,
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

    let debuggingBufferStorageFinal =
    device.createBuffer({
      label: "debugging storage result",
      size: kDebugArraySize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

  let histogramBufferStorageFinal =
    device.createBuffer({
      label: "histogram storage result",
      size: kDebugArraySize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });


  const uniformArrayInit = new Uint32Array([7, 5]);
  let uniformBuffer = device.createBuffer({
    label: "Grid Uniforms",
    size: uniformArrayInit.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuffer, 0, uniformArrayInit);

  const uniformArrayInitFinal = new Uint32Array([3 ,11]);
  let uniformBufferFinal = device.createBuffer({
    label: "Grid Uniforms",
    size: uniformArrayInitFinal.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBufferFinal, 0, uniformArrayInitFinal);

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
      } , {
        binding: 2,
        resource: { buffer: histogramBufferStorage }
      }, {
        binding: 3, //  uniforms
        resource: { buffer: uniformBuffer }
      }
      ],
    });

   let finalBindGroup =
    device.createBindGroup({
      label: "Compute renderer bind group B",
      layout: bindGroupLayout, // Updated Line
      entries: [{
        binding: 0,
        resource: { buffer: inputBufferStorage }
      }, {
        binding: 1,
        resource: { buffer: debuggingBufferStorageFinal }
      } , {
        binding: 2,
        resource: { buffer: histogramBufferStorageFinal }
      }, {
        binding: 3, //  uniforms
        resource: { buffer: uniformBufferFinal }
      }
      ],
    });

  var time_in_seconds;
  const kNumRuns = 10;
  for (var i = 0; i < kNumRuns; i++) {
    const encoder = device.createCommandEncoder();
    const computePass = encoder.beginComputePass({
      label: "Timing request",
      timestampWrites: { querySet: querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 },
    });

    computePass.setPipeline(computePipelineA);
    computePass.setBindGroup(0, (i== kNumRuns-1) ?finalBindGroup: commonBindGroup);

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
    await sleep(300);
  }

}

