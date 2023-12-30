var device;
var canvasformat;
var context;
const NUM_PARTICLES_DIM = 128;
var canvas_width;
var canvas_height;
var bindGroupLayout;
var uniformBuffer;
var simulationBindGroups;
var massAssignBindGroups;

window.onload =  async  function () {
  const canvas = document.querySelector("canvas");
  if (!canvas) {
    throw new Error("No canvas.");
  }
  canvas_width = canvas.width;
  canvas_height = canvas.height;
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
     bindGroupLayout = device.createBindGroupLayout({
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
    const uniformArray = new Float32Array([canvas_width, canvas_height]);
    uniformBuffer = device.createBuffer({
      label: "Grid Uniforms",
      size: uniformArray.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(uniformBuffer, 0, uniformArray);
    // Create an array representing the active state of each cell.
 

    commonBindGroup = [
      device.createBindGroup({
        label: "Compute renderer bind group A",
        layout: bindGroupLayout, // Updated Line
        entries: [{
          binding: 0,
          resource: { buffer: uniformBuffer }
        }, {
          binding: 1,
          resource: { buffer: cellStateStorage[1] }
        }, {
          binding: 2, // New Entry
          resource: { buffer: renderBufferStorage }
        }],
      }),
      device.createBindGroup({
        label: "Compute renderer bind group B",
        layout: bindGroupLayout, // Updated Line

        entries: [{
          binding: 0,
          resource: { buffer: uniformBuffer }
        }, {
          binding: 1,
          resource: { buffer: cellStateStorage[0] }
        }, {
          binding: 2, // New Entry
          resource: { buffer: renderBufferStorage }
        }],
      }),
    ];
 

    graphicsBindGroup = 
      device.createBindGroup({
        label: "Compute renderer bind group A",
        layout: bindGroupLayout, // Updated Line
        entries: [{
          binding: 0,
          resource: { buffer: uniformBuffer }
        }, {
          binding: 1,
          resource: { buffer: renderBufferStorage }
        }, {
          binding: 2, // New Entry
          resource: { buffer:  cellStateStorage[0]}
        }],
      });
 

    simulationBindGroups =  
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
      });

      massAssignBindGroups =  
      device.createBindGroup({
        label: "mass assign",
        layout: bindGroupLayout, // Updated Line
        entries: [{
          binding: 0,
          resource: { buffer: uniformBuffer }
        }, {
          binding: 1,
          resource: { buffer: cellStateStorage[0] }
        }, {
          binding: 2, // New Entry
          resource: { buffer: massAssignBufferStorage }
        }],
      });


    const UPDATE_INTERVAL = 1; // 
    let step = 0; // Track how many simulation steps have been run
        
    function updateGrid() {
      step++; // Increment the step count
      
      // Start a render pass 
      const encoder = device.createCommandEncoder();
      update_compute_particles(encoder, commonBindGroup, step);
      draw_particles(encoder, graphicsBindGroup, step);
      const commandBuffer = encoder.finish();
      device.queue.submit([commandBuffer]);
      window.requestAnimationFrame(updateGrid);
    }

    //window.setInterval(updateGrid, UPDATE_INTERVAL);
    window.requestAnimationFrame(updateGrid);
}
