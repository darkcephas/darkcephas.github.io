const WORKGROUP_SIZE = 256;
const TOTAL_JOB_SIZE = WORKGROUP_SIZE * WORKGROUP_SIZE;



function renavigate() {
  window.location.href = document.location.search;
}


window.onload = async function () {

  
  let params = new URLSearchParams(document.location.search);

  if(params.get("renavigate_after")){
    window.setTimeout(renavigate, params.get("renavigate_after"));
  }

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
  var device = await adapter.requestDevice();
  var context = canvas.getContext("webgpu");
  let canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device: device,
    format: canvasFormat,
  });

  // Create the bind group layout and pipeline layout.
  bindGroupLayout = device.createBindGroupLayout({
    label: "Compute in/out layout",
    entries: [{
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {}
    }, {
      binding: 1,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage" }
    }]
  });

  const pipelineLayout = device.createPipelineLayout({
    label: "Compute Pipeline Layout",
    bindGroupLayouts: [bindGroupLayout],
  });

  var computeShaderModule = device.createShaderModule({
    label: "Render Buffer shader",
    code: `
        @group(0) @binding(0) var<uniform> input_var: vec2u;
        @group(0) @binding(1) var<storage, read_write> inoutbuff: array<atomic<u32>>;

        @compute @workgroup_size(${WORKGROUP_SIZE})
        fn computeMain(  @builtin(global_invocation_id) global_idx:vec3u,
        @builtin(num_workgroups) num_work:vec3u) {
          let threadId = global_idx.x;
          var num_steps = 0u;
          var x = threadId * 66889u;
          while(x != 6666666u && num_steps < input_var.x)
          {
            // https://en.wikipedia.org/wiki/Linear_congruential_generator
            x = x * 134775813u + 1u;
            num_steps++;
            
          }
          atomicAdd(&inoutbuff[threadId], num_steps);
        }
      `
  });

  inoutbufferStorage =
    device.createBuffer({
      label: "Atomic buffer ints",
      size: 4 * TOTAL_JOB_SIZE,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });


  // Create a compute pipeline that updates the game state.
  simpleComputePipeline = device.createComputePipeline({
    label: "Simple Compute pipeline",
    layout: pipelineLayout,
    compute: {
      module: computeShaderModule,
      entryPoint: "computeMain",
    }
  });


  const uniformArray = new Int32Array([(params.get("max_iter") ? params.get("max_iter") : 1000) * 1000, 10000]);
  uniformBuffer = device.createBuffer({
    label: "Grid Uniforms",
    size: uniformArray.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuffer, 0, uniformArray);

  computeBindGroup =
    device.createBindGroup({
      label: "Compute renderer bind group A",
      layout: bindGroupLayout, // Updated Line
      entries: [{
        binding: 0,
        resource: { buffer: uniformBuffer }
      }, {
        binding: 1,
        resource: { buffer: inoutbufferStorage }
      },],
    });

  
  const update_interval_ms = params.get("time_interval") ? params.get("time_interval") : 5000;

  function startSubmitLoop() {
    window.setTimeout(submitLoop, update_interval_ms);
  }


  let step = 0; // Track how many simulation steps have been run
  function submitLoop() {
    // Start a render pass 
    device.queue.writeBuffer(uniformBuffer, 0, uniformArray);
    const encoder = device.createCommandEncoder();
    let n_submits = params.get("num_submits") ? params.get("num_submits") : 1;
    console.log(" start submit  " +  performance.now());
    for (var i = 0; i < n_submits; i++) {
 
      // render out the stars to the buffer that will be then drawn using graphics pipe
      encoder.clearBuffer(inoutbufferStorage);
      const computePass = encoder.beginComputePass();
      computePass.setPipeline(simpleComputePipeline);
      computePass.setBindGroup(0, computeBindGroup);
      const workgroupCount = Math.ceil(TOTAL_JOB_SIZE / WORKGROUP_SIZE);
      computePass.dispatchWorkgroups(WORKGROUP_SIZE);
      computePass.end();

    }
    const commandBuffer = encoder.finish();
    device.queue.submit([commandBuffer]);
    step++; // Increment the step count
    device.queue.onSubmittedWorkDone().then(() => {

      console.log("All submitted commands processed  " + performance.now());
      window.setTimeout(submitLoop, update_interval_ms);
   });
    
    console.log("complete ");

    console.log("Submit " + step);
   
  }

  const wait_time_start_ms = 5000;
  window.setTimeout(startSubmitLoop, wait_time_start_ms);


}
