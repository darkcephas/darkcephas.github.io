var device;
var canvasformat;
var context;

var bindGroupLayout;
var uniformBuffer;
var simulationBindGroups;
var massAssignBindGroups;
var starGraphicsBindGroup;
var massGraphicsBindGroup;
var forceIndexBindGroups;

var forceIndexShaderModule
const WORKGROUP_SIZE = 1;

var simulationPipeline;
var cellStateStorage;
var renderBufferStorage;
var bindGroupUniformOffset;
var massAssignBufferStorage;



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
  canvasFormat = navigator.gpu.getPreferredCanvasFormat();
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
        const json = JSON.parse(file.result);
        console.log(json);
      });
      file.readAsText(f.files[0]);
    });
    f.click();
  });

  const decompress = document.querySelector('#decompress');
  decompress.addEventListener('click', RunDecompression);

  const savedata = document.querySelector('#savedata');
  savedata.addEventListener('click', saveDataToDisk);
}



async function RunDecompression()  {


  // Create the bind group layout and pipeline layout.
  bindGroupLayout = device.createBindGroupLayout({
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
        
          @group(0) @binding(0) var<storage> in: array<u32>;
            @group(0) @binding(1) var<storage,read_write> out: array<u32>;
           @group(0) @binding(2) var<uniform> test_data: u32;
  
          @compute @workgroup_size(${WORKGROUP_SIZE})
          fn computeMain(  @builtin(global_invocation_id) global_idx:vec3u,
          @builtin(num_workgroups) num_work:vec3u) {
     
            out[0]= in[0] + 666;
          }
        `
  });




  // Create a compute pipeline that updates the game state.
  renderBufferPipeline = device.createComputePipeline({
    label: "Render pipeline",
    layout: pipelineLayout,
    compute: {
      module: forceIndexShaderModule,
      entryPoint: "computeMain",
    }
  });



  // Create a uniform buffer that describes the grid.
  const uniformArrayInit = new Uint32Array([123, 345]);
  uniformBuffer = device.createBuffer({
    label: "Grid Uniforms",
    size: uniformArrayInit.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuffer, 0, uniformArrayInit);
  // Create an array representing the active state of each cell.


 


  // Create an array representing the active state of each cell.
  const initInputDataArray = new Uint32Array(100);
  var as_int = new Int32Array(initInputDataArray.buffer);
  initInputDataArray[0] = 13;
  // Create two storage buffers to hold the cell state.
  inputBufferStorage =
    device.createBuffer({
      label: "Init input array",
      size: initInputDataArray.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  // fill buffer with the init data.
  device.queue.writeBuffer(inputBufferStorage, 0, initInputDataArray);


  outputBufferStorage =
    device.createBuffer({
      label: "Output result",
      size:  initInputDataArray.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });


    commonBindGroup =
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
    size: 666,
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
    4
  );
  const commandBuffer = encoder.finish();
  
  device.queue.submit([commandBuffer]);

   await stagingBuffer.mapAsync(
    GPUMapMode.READ,
    0, // Offset
    4 // Length
   );
  const copyArrayBuffer =
    stagingBuffer.getMappedRange(0, 4);
  const data = copyArrayBuffer.slice();
  stagingBuffer.unmap();
  console.log(new Uint32Array(data));
}

