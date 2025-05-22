"use strict";

var device;
var context;
var querySet;
var queryBuffer;

var testresulttext;
const kDebugArraySize = 1024 * 256;
const kInputBufferCount = 1024 * 256;
const timestampCapacity = 16;//Max number of timestamps we can store

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

  var enablesubgroupflag = document.getElementById('enablesubgroup').checked;
  // We dont need timestamps for this code to work but this is a prototype.
  //     requiredLimits: {    maxComputeInvocationsPerWorkgroup: 1024,maxComputeWorkgroupSizeX: 1024, maxComputeWorkgroupStorageSize: 32768}
  // , enablesubgroupflag ? "subgroups" : ""
  device = await adapter.requestDevice({
    requiredFeatures: ["timestamp-query"],

  });

  context = canvas.getContext("webgpu");
  var canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device: device,
    format: canvasFormat,
  });


  const runthebenchmark = document.querySelector('#runtestbutton');
  runthebenchmark.addEventListener('click', RunBenchmark);
  testresulttext = document.querySelector("#testresulttext");

  var default_code =
    `  
@group(0)@binding(0) var <storage,read >  _in :array< f32 >;
@group(0)@binding(1) var <storage,read_write > _out: array < f32 >;

@compute @workgroup_size(256)
fn computeMain(@builtin(local_invocation_index) idx: u32,
               @builtin(workgroup_id) wg: vec3u) {
  _out[idx] = _in[idx];
}
`;

  const paramsString = window.location.search;
  const mySearchParams = new URLSearchParams(paramsString);

  document.getElementById('shadertexta').value = default_code
  document.getElementById('shadertextb').value = default_code;

  for (const [key, value] of mySearchParams) {
    if (key == 'codea') {
      document.getElementById('shadertexta').value = decodeURIComponent(value);
    }
    if (key == 'codeb') {
      document.getElementById('shadertextb').value = decodeURIComponent(value);
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
  document.getElementById('shadertextb').addEventListener('keydown', tabs_allow_func);

  const getcurrenturl = document.querySelector('#getcurrenturl');
  getcurrenturl.addEventListener('click', CurrentURLToCopy);
}



async function CurrentURLToCopy() {
  //https://stackoverflow.com/questions/400212/how-do-i-copy-to-the-clipboard-in-javascript
  var full_url = window.location.origin + window.location.pathname + '?codea=' + encodeURIComponent(document.getElementById('shadertexta').value) +
    '&codeb=' + encodeURIComponent(document.getElementById('shadertextb').value);
  navigator.clipboard.writeText(full_url).then(function () {
    console.log('Async: Copying to clipboard was successful!');
  }, function (err) {
    console.error('Async: Could not copy text: ', err);
  });
}


async function RunBenchmark() {
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
    }]
  });

  const pipelineLayout = device.createPipelineLayout({
    label: "Main Pipeline Layout",
    bindGroupLayouts: [bindGroupLayout],
  });


  var shaderCodeA = document.getElementById('shadertexta').value;
  var shaderCodeB = document.getElementById('shadertextb').value;

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
    }
  });

  let shaderModuleB = device.createShaderModule({
    label: "Benchmark compute shader",
    code: shaderCodeB,
  });

  // Create a compute pipeline that updates the game state.
  let computePipelineB = device.createComputePipeline({
    label: "Render pipeline",
    layout: pipelineLayout,
    compute: {
      module: shaderModuleB,
      entryPoint: "computeMain",
    }
  });

  // fill buffer with the init data.
  const floatInitArray = new Float32Array(kInputBufferCount);
  let inputBufferStorage =
    device.createBuffer({
      label: "Init input array",
      size: floatInitArray.byteLength, // this isnt quite right...
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

  for (let i = 0; i < floatInitArray.length; i += 6) {
    floatInitArray[i] = Math.random();
  }
  device.queue.writeBuffer(inputBufferStorage, 0, floatInitArray, 0, floatInitArray.length);

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
        resource: { buffer: debuggingBufferStorage }
      }
      ],
    });


  var timingA = [];
  var timingB = [];
  for (var i = 0; i < 256; i++) {
    const encoder = device.createCommandEncoder();
    const computePass = encoder.beginComputePass({
      label: "Timing request",
      timestampWrites: { querySet: querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 },
    });

    computePass.setPipeline(i % 2 == 0 ? computePipelineA : computePipelineB);
    computePass.setBindGroup(0, commonBindGroup);
    var dispatch_cube_size = Number(document.getElementById("dispatchcubedid").value);
    computePass.dispatchWorkgroups(dispatch_cube_size, dispatch_cube_size, dispatch_cube_size);
    computePass.end();

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
      timestampCapacity,//number of queries to resolve
      queryBuffer,
      0);// destination offset

    const commandBuffer = encoder.finish();

    device.queue.submit([commandBuffer]);
    const arrayBuffer = await readBuffer(device, queryBuffer);
    const timingsNanoseconds = new BigInt64Array(arrayBuffer);
    const time_in_seconds = Number(timingsNanoseconds[1] - timingsNanoseconds[0]) / 1000000.0;


    await stagingBufferDebug.mapAsync(
      GPUMapMode.READ,
      0, // Offset
      kDebugArraySize // Length
    );
    const copyArrayBuffer = stagingBufferDebug.getMappedRange();
    const data = copyArrayBuffer.slice();
    stagingBufferDebug.unmap();
    //console.log(new Float32Array(data));

    if (i > 15) { // ignore first 16 runs
      if (i % 2 == 0) {
        timingA.push(time_in_seconds);
      } else {
        timingB.push(time_in_seconds);
      }
    }
  }
  timingA.sort((a, b) => a - b);
  timingB.sort((a, b) => a - b);
  console.log(timingA);
  console.log(timingB);

  setRunPass("Runs A time " + timingA[timingA.length / 2].toFixed(2) + "ms ,  time b " + timingB[timingB.length / 2].toFixed(2) + " ms");
}

