var device;
var canvasformat;
var context;
const NUM_PARTICLES_MAX = 256;
var canvas_width;
var canvas_height;
var bindGroupLayout;
var uniformBuffer;
var simulationBindGroups;
var massAssignBindGroups;
var starGraphicsBindGroup;
var massGraphicsBindGroup;
var forceIndexBindGroups;

window.onload = async function () {

  window.addEventListener('resize', resizeCanvas, false);
  const canvas = document.querySelector("canvas");
  function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    /**
     * Your drawings need to be inside this function otherwise they will be reset when 
     * you resize the browser window and the canvas goes will be cleared.
     */
    // drawStuff(); 
  }

  resizeCanvas();


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



    // Create an array representing the active state of each cell.
    const cellStateArray = new Float32Array(NUM_PARTICLES_MAX * 6);
    var as_int = new Int32Array(cellStateArray.buffer);
    // Create two storage buffers to hold the cell state.
    var cellStateStorage =
      device.createBuffer({
        label: "Cell State A",
        size: cellStateArray.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
    // Mark every third cell of the first grid as active.
  
    
  // Create a bind group to pass the grid uniforms into the pipeline

    for (let i = 0; i < cellStateArray.length; i += 6) {
      cellStateArray[i] = Math.random() - 0.5;
      cellStateArray[i + 1] = (i / 6) / (NUM_PARTICLES_MAX) - 0.5;
  
      cellStateArray[i + 2] = cellStateArray[i + 1] * 30 + Math.random() - 0.5;
      cellStateArray[i + 3] = - cellStateArray[i] * 30 + Math.random() - 0.5;
  
      as_int[i] = cellStateArray[i] * (256 * 256 * 256 * 64);
      as_int[i + 1] = cellStateArray[i + 1] * (256 * 256 * 256 * 64);
    }
    device.queue.writeBuffer(cellStateStorage, 0, cellStateArray);

    
  // Create a uniform buffer that describes the grid.
  const uniformArray = new Float32Array([canvas_width, canvas_height]);
  uniformBuffer = device.createBuffer({
    label: "Grid Uniforms",
    size: uniformArray.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuffer, 0, uniformArray);


    
  setup_compute_particles(uniformBuffer, cellStateStorage);
  setup_render_particles(uniformBuffer, cellStateStorage);

  let step = 0; // Track how many simulation steps have been run        
  function updateGrid() {
    step++; // Increment the step count

    // Start a render pass 
    const encoder = device.createCommandEncoder();
    update_compute_particles(encoder, step);
    draw_particles(encoder, step);
    const commandBuffer = encoder.finish();
    device.queue.submit([commandBuffer]);
    window.requestAnimationFrame(updateGrid);
  }
  window.requestAnimationFrame(updateGrid);
}
