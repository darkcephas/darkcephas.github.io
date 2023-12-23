var device;
var canvasformat;
var context;
const GRID_SIZE =16;
    
window.onload =  async  function () {
  const canvas = document.querySelector("canvas");

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
    

    
    // Create the bind group layout and pipeline layout.
    const bindGroupLayout = device.createBindGroupLayout({
      label: "Cell Bind Group Layout",
      entries: [{
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE |  GPUShaderStage.FRAGMENT,
        buffer: {} // Grid uniform buffer
      }, {
        binding: 1,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE |  GPUShaderStage.FRAGMENT,
        buffer: { type: "read-only-storage"} // Cell state input buffer
      }, {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage"} // Cell state output buffer
      }]
    });
    
    const pipelineLayout = device.createPipelineLayout({
      label: "Cell Pipeline Layout",
      bindGroupLayouts: [ bindGroupLayout ],
    });
    
   setup_render_particles(pipelineLayout);
    setup_compute_particles(pipelineLayout);
 

            // Create a uniform buffer that describes the grid.
    const uniformArray = new Float32Array([GRID_SIZE, GRID_SIZE]);
    const uniformBuffer = device.createBuffer({
      label: "Grid Uniforms",
      size: uniformArray.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(uniformBuffer, 0, uniformArray);
    // Create an array representing the active state of each cell.
 
 
    const bindGroups = [
      device.createBindGroup({
        label: "Cell renderer bind group A",
        layout: bindGroupLayout, // Updated Line
        entries: [{
          binding: 0,
          resource: { buffer: uniformBuffer }
        }, {
          binding: 1,
          resource: { buffer: cellStateStorage[0] }
        }, {
          binding: 2, // New Entry
          resource: { buffer: cellStateStorage[1] }
        }],
      }),
      device.createBindGroup({
        label: "Cell renderer bind group B",
        layout: bindGroupLayout, // Updated Line

        entries: [{
          binding: 0,
          resource: { buffer: uniformBuffer }
        }, {
          binding: 1,
          resource: { buffer: cellStateStorage[1] }
        }, {
          binding: 2, // New Entry
          resource: { buffer: cellStateStorage[0] }
        }],
      }),
    ];
    


    const UPDATE_INTERVAL = 200; // Update every 200ms (5 times/sec)
    let step = 0; // Track how many simulation steps have been run
        
    function updateGrid() {
      step++; // Increment the step count
      
      // Start a render pass 
      const encoder = device.createCommandEncoder();
      draw_particles(encoder,bindGroups, step);
      update_compute_particles(encoder,bindGroups, step)
      const commandBuffer = encoder.finish();
      device.queue.submit([commandBuffer]);
    }

setInterval(updateGrid, UPDATE_INTERVAL);
}
