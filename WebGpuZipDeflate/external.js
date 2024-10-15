"use strict";

//https://stackoverflow.com/questions/18638900/javascript-crc32
var crc32 = (function () {
  var table = new Uint32Array(256);

  // Pre-generate crc32 polynomial lookup table
  // http://wiki.osdev.org/CRC32#Building_the_Lookup_Table
  // ... Actually use Alex's because it generates the correct bit order
  //     so no need for the reversal function
  for (var i = 256; i--;) {
    var tmp = i;

    for (var k = 8; k--;) {
      tmp = tmp & 1 ? 3988292384 ^ tmp >>> 1 : tmp >>> 1;
    }

    table[i] = tmp;
  }

  // crc32b
  // Example input        : [97, 98, 99, 100, 101] (Uint8Array)
  // Example output       : 2240272485 (Uint32)
  return function (data) {
    var crc = -1; // Begin with all bits set ( 0xffffffff )

    for (var i = 0, l = data.length; i < l; i++) {
      crc = crc >>> 8 ^ table[crc & 255 ^ data[i]];
    }

    return (crc ^ -1) >>> 0; // Apply binary NOT
  };

})();

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




