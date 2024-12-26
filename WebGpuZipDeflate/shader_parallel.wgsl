struct CommonData {
      outlen:  u32,       /* available space at out */
     inlen: u32,    /* available input at in */
};

struct ThreadState {
     outcnt: u32,       /* bytes written to out so far */
    /* input state */
     incnt:u32,        /* bits read so far */
     bitbuf:u32,                 /* bit buffer */
     err:i32,
     decode_to_store:u32, // useful as a type of global
     decode_len: u32, // 1 if val n if copy
} ;

var<private> ts : ThreadState;

const MAXBITS=15 ;             /* maximum bits in a code */
const MAXLCODES=286 ;          /* maximum number of literal/length codes */
const MAXDCODES=30 ;           /* maximum number of distance codes */
const MAXCODES=(MAXLCODES+MAXDCODES);  /* maximum codes lengths to read */
const FIXLCODES=288;           /* number of fixed literal/length codes */

const WORKGROUP_SIZE = 128;

var<workgroup>  lengths:array<i32, MAXCODES>;            /* descriptor code lengths */
var<workgroup>  lencnt:array<u32, MAXBITS + 1>;
var<workgroup>  lensym:array<u32, FIXLCODES>;
var<workgroup>  distcnt:array<u32, MAXBITS + 1>;
 // Length should be MAXDCODES but is FIXLCODES to use same fixed sized pointer
var<workgroup>  distsym:array<u32, FIXLCODES>;

var<workgroup> lenLut:array<u32, 1024>;
var<workgroup> distLut:array<u32, 1024>;

// Stream out ring buffer.
struct RingData {
    byte_pos:  u32,
    code: u32,   
};

const DECODED_RING_SIZE = WORKGROUP_SIZE*2;
var<workgroup> decoded_ring:array<RingData, DECODED_RING_SIZE>;
var<workgroup> decode_head:atomic<u32>;
var<workgroup> decode_tail:atomic<u32>;

// read encoded stream serializer
var<workgroup> g_incnt:atomic<u32>;

// allocation of bytes to write
var<workgroup> g_write_tracker:atomic<u32>;

// We do backreference copy so we need to always update what is complete
var<workgroup> g_write_complete:atomic<u32>;

@group(0) @binding(0) var<storage> in: array<u32>;
@group(0) @binding(1) var<storage,read_write> out: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> unidata: CommonData;
@group(0) @binding(3) var<storage,read_write> debug: array<u32>;

var<workgroup> ws : CommonData;



const ERROR_OUTPUT_OVERFLOW = 2;
const ERROR_NO_MATCH_COMPLEMENT = 3;
const ERROR_INPUT_OVERFLOW = 4;
const ERROR_INPUT_BITS_OVERFLOW = 5;
const ERROR_STORE_NOT_SUPPORTED = 7;
const ERROR_RAN_OUT_OF_CODES = -10;
const ERROR_INCOMPLETE_CODE_SINGLE = -8;
const ERROR_NO_END_BLOCK_CODE = -9;
const ERROR_NO_LAST_LENGTH = -5;
const ERROR_INVALID_SYMBOL=-7;
const ERROR_BAD_COUNTS = -3;
const ERROR_REQUIRED_COMPLETE_CODE=-4;
const ERROR_TOO_MANY_LENGTHS= -6;

// Const data access is very fast ...  no worries!
const kLens= array<u32,29> ( /* Size base for length codes 257..285 */
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31,
    35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258 );
const kLext= array<u32,29> ( /* Extra bits for length codes 257..285 */
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
    3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0 );
const kDists= array<u32,30> ( /* Offset base for distance codes 0..29 */
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193,
    257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145,
    8193, 12289, 16385, 24577 );
const  kDext= array<u32,30> ( /* Extra bits for distance codes 0..29 */
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
    7, 7, 8, 8, 9, 9, 10, 10, 11, 11,
    12, 12, 13, 13 );

var<private> debug_idx:u32 = 20;

var<workgroup> atomic_idx:atomic<u32>;

fn DebugWrite(val:u32){
    debug[atomicAdd(&atomic_idx,1)]= val;
}

fn ReportError(error_code:i32){
    if(ts.err==0){
        ts.err = error_code;
    }
}

// prepare 32 bits of buffered data at bit index ts.inct
fn  Read32() {
    var reverse_offset  = ts.incnt % 32;
    // The forward offset of zero causes issues in the shift... not sure why
    if(reverse_offset == 0){
        ts.bitbuf = in[ts.incnt/32];
        return;
    }
    var forward_offset = 32 - reverse_offset;
    ts.bitbuf = (in[ts.incnt/32] >> reverse_offset) | (in[(ts.incnt/32)+1] << forward_offset);
}

fn PeekByteOut( rev_offset_in_bytes:u32, byte_loc:u32) -> u32
{
    var offset:u32 = byte_loc - rev_offset_in_bytes;
    var sub_index:u32 = offset % 4;
    var  val:u32 = atomicLoad( &out[offset / 4]);
    //var temp = extractBits(val, (8 * sub_index), 8);
    var temp = (val >> (8 * sub_index)) & 0xff;
    return temp;
}

fn StreamWriteByteOut( val:u32)
{
    ts.decode_to_store = val;
    ts.decode_len = 1;
}

fn WriteByteOut(val:u32, byte_loc:u32)
{
    var sub_index:u32 = byte_loc % 4;
    // this only works because there is zero in the original byte
    atomicOr(&out[byte_loc/4],   val << (sub_index * 8u));

    if (byte_loc + 1 > ws.outlen) {
        ReportError(ERROR_OUTPUT_OVERFLOW);
        // webgpu handles any buffer out of bounds!
    }
}


fn StreamCopyBytes( dist:u32, len:u32) 
{
    var val = (1<<31) | (len << 16) | dist;
    ts.decode_to_store = val;
    ts.decode_len = len;
    return;
}


fn CopyBytes( dist:u32, len:u32,  byte_start:u32) 
{
    var len_tmp = len;
    var counter_local:u32 = 0;
    while (len_tmp != 0) {
        len_tmp--;
        var val:u32 = PeekByteOut(dist, byte_start+ counter_local);
        WriteByteOut(val, byte_start+ counter_local);
        counter_local++;
    }
    return;
}

fn bits_local( need:u32 ) -> u32
{
    // return need bits, zeroing the bits above that
    var out_val =  u32(ts.bitbuf & ((1u << need) - 1u));
    ts.bitbuf = ts.bitbuf >> need;
    ts.incnt += need;
    return out_val;
}


fn bits( need:u32 ) -> u32
{
    Read32();
    ts.incnt += need;
    // return need bits, zeroing the bits above that
    var bits_out =  u32(ts.bitbuf & ((1u << need) - 1u));
    return bits_out;
}

fn  stored() 
{
     ReportError(ERROR_STORE_NOT_SUPPORTED);   
     return;

    // discard leftover bits from current byte (assumes ts.bitcnt < 8) 
    //ts.bitbuf = 0;
    //ts.bitcnt = 0;

    // get length and check against its one's complement 
    // length of stored block 
    //var len :u32 = ReadByteIn() | (ReadByteIn() << 8);
    //if( ReadByteIn() != (~len & 0xff) ||
   //     ReadByteIn() != ((~len >> 8) & 0xff)) {
   //     ReportError(ERROR_NO_MATCH_COMPLEMENT);  
   // }
    //while (len !=0) {
      //  len--;
       // var val:u32 = ReadByteIn();
      // WriteByteOut(val);
    //}
}

struct DecodeRtn {
  symbol: u32,
  cnt: u32, // zero means could not decode
}  

fn decode(ptr_array_cnt: ptr<workgroup, array<u32,  MAXBITS + 1>> , ptr_array_sym: ptr<workgroup, array<u32, FIXLCODES>>,
            bitbuf_in:u32, left_in:u32) -> DecodeRtn
{
    var bitbuf:u32  = bitbuf_in;
    var left:u32  = left_in;
    var code:i32 = 0; // len bits being decoded
    var first:i32 = 0;  // first code of length len 
    var index:i32 = 0; // index of first code of length len in symbol table 
    var len:u32 = 1; // current number of bits in code
    // This code is making a tree from lower(left) to larger(right) bits
    // each bit adds another level to the tree ;increasing the available codes
    // the count for this level removes these codes 
    while (true) {
        while (left !=0) {
            code |= i32(bitbuf & 1);
            bitbuf >>= 1;
            // number of codes of length len 
            var count:i32 = i32(ptr_array_cnt[len]);
            if (code - count < first) { /* if length len, return symbol */
                var local_inded:i32 = index + (code - first);
                return  DecodeRtn(ptr_array_sym[local_inded], len);
            }
            // else update for next length
            index += count;             
            first += count;
            first <<= 1;
            code <<= 1;
            len++;
            left--;
        }
        return DecodeRtn(0u, 0u);

    }
    return DecodeRtn(0u, 0u);
}

fn decode_mutate(ptr_array_cnt: ptr<workgroup, array<u32,  MAXBITS + 1>> , ptr_array_sym: ptr<workgroup, array<u32, FIXLCODES>>) -> u32
{
    var decode_res:DecodeRtn = decode(ptr_array_cnt, ptr_array_sym, ts.bitbuf, 32);
    if(decode_res.cnt == 0){
        ReportError(ERROR_RAN_OUT_OF_CODES);
    }
    ts.bitbuf = ts.bitbuf >> decode_res.cnt;
    ts.incnt = ts.incnt + decode_res.cnt;
    return decode_res.symbol;
}

fn GenLut(local_invocation_index:u32)
{
    for(var i:u32 = local_invocation_index ; i < 1024;i+= WORKGROUP_SIZE){
        var decode_res:DecodeRtn = decode(&lencnt, &lensym, i, 10);
         if(decode_res.cnt == 0){
             lenLut[i] = 0;
         }
         else{
            lenLut[i] =  (decode_res.cnt << 25) | decode_res.symbol;
            // is a len/dist copy?
            if(decode_res.symbol > 256) {
                var offset:u32 = decode_res.symbol - 257;
                var num_bits_needed:u32 = kLext[offset];
                if( (decode_res.cnt + num_bits_needed) > 10 ){
                     lenLut[i] = 0; // FAILED !
                }
                else {
                    var masked_bits:u32 = i >> decode_res.cnt;
                    decode_res.cnt += num_bits_needed;
                    var masked_bits_len:u32 = masked_bits & ((1u << num_bits_needed) - 1u);
                    lenLut[i] =  (decode_res.cnt << 25) | decode_res.symbol | ((kLens[offset] + masked_bits_len) << 9);
                }
            }
         }
    }

    for(var i:u32 = local_invocation_index ; i < 1024;i+=WORKGROUP_SIZE){
        var decode_res:DecodeRtn = decode(&distcnt, &distsym, i, 10);
         if(decode_res.cnt == 0){
             distLut[i] = 0;
         }
         else{
            distLut[i] = (decode_res.cnt << 24)  | ( kDists[decode_res.symbol]<< 8) | kDext[decode_res.symbol] ;
         }
    }
}

fn construct_code(ptr_array_cnt: ptr<workgroup, array<u32,  MAXBITS + 1>>, 
        ptr_array_sym: ptr<workgroup, array<u32, FIXLCODES>>,
         offset:i32,  n:i32) -> i32 
{
    var  offs:array<i32, MAXBITS + 1>;        /* offsets in symbol table for each length */

    /* count number of codes of each length */
    for (var len:i32 = 0; len <= MAXBITS; len++) {
        ptr_array_cnt[len] = 0;
    }
    /* current symbol when stepping through length[] */
    for (var symbol:i32 = 0; symbol < n; symbol++) {
        (ptr_array_cnt[lengths[symbol+offset]])++;   /* assumes lengths are within bounds */
    }

    if (i32(ptr_array_cnt[0]) == n) {              /* no codes! */
        return 0;                       /* complete, but decode() will fail */
    }

    /* check for an over-subscribed or incomplete set of lengths */
    var left:i32 = 1;                           /* one possible code of zero length */
     /* current length when stepping through h->count[] */
    for (var len:i32 = 1; len <= MAXBITS; len++) {
        left <<= 1;                     /* one more bit, double codes left */
        left -= i32(ptr_array_cnt[len]);          /* deduct count from possible codes */
        if (left < 0) {
            return left;                /* over-subscribed--return negative */
        }
    }                                   /* left > 0 means incomplete */

    /* generate offsets into symbol table for each length for sorting */
    offs[1] = 0;
    for (var len:i32 = 1; len < MAXBITS; len++) {
        offs[len + 1] = offs[len] + i32(ptr_array_cnt[len]);
    }

    /*
     * put symbols in table sorted by length, by symbol order within each
     * length
     */
    for (var symbol:i32 = 0; symbol < n; symbol++) {
        if (lengths[symbol+offset] != 0) {
            ptr_array_sym[offs[lengths[symbol+offset]]] = u32(symbol);
            offs[lengths[symbol+offset]]++;
        }
    }

    /* return zero for complete set, positive for incomplete set */
    return left;
}


var<workgroup> decode_done:u32;
fn codes(local_invocation_index:u32)
{
    // everyone sets?
     workgroupBarrier();
     
    if(local_invocation_index == 0){
    decode_done = 0;
    }
     workgroupBarrier();

    var counter = 0;
    // Each invocation is going to start at staggered offset
    ts.incnt = atomicLoad(&g_incnt);
    ts.incnt += local_invocation_index;
      workgroupBarrier();
    // decode literals and length/distance pairs 
    while(true) {

        // bits from stream 
        Read32();
        let start_bit_offset = ts.incnt;
        var lut_len_res:u32 = lenLut[ts.bitbuf & 0x3FF];
        var invocation_hit_end_of_block = false;
        if (lut_len_res == 0)
        { 
            // SLOW PATH none LUT
            var symbol:u32 = decode_mutate(&lencnt, &lensym);

            if (symbol < 256) { // literal: symbol is the byte 
                StreamWriteByteOut(symbol); // write out the literal 
            }
            else if (symbol == 256){  
                // end of block symbol 
                invocation_hit_end_of_block = true;
            } 
            else if (symbol > 256) {     
                symbol -= 257;  // length and distance codes get and compute length 
                if (symbol >= 29) {
                    ReportError(ERROR_RAN_OUT_OF_CODES);       
                }
                var len:u32 = kLens[symbol] + bits_local(kLext[symbol]);
                symbol = decode_mutate(&distcnt, &distsym);
                // distance for copy 
                var dist:u32 = kDists[symbol] + bits_local(kDext[symbol]);
                // copy length bytes from distance bytes back
                StreamCopyBytes(dist, len);
            }
        }
        else {
            
            var symbol:u32 = 0x1FF & lut_len_res;
            let temp_cnt:u32 = lut_len_res >> 25; 
            ts.bitbuf = ts.bitbuf >> temp_cnt;
            ts.incnt  = ts.incnt + temp_cnt;

            if (symbol < 256) { // literal: symbol is the byte 
                StreamWriteByteOut(symbol); // write out the literal 
            }
            else if (symbol == 256){ 
                // end of block symbol 
                invocation_hit_end_of_block = true;
            } 
            else if (symbol > 256) {     
                symbol -= 257;  // length and distance codes get and compute length 
                if (symbol >= 29) {
                    ReportError(ERROR_RAN_OUT_OF_CODES);       
                }

                let len:u32 = (lut_len_res >> 9) & 0xFFFF;

                var lut_dist_res:u32 = distLut[ts.bitbuf & 0x3FF];
                var dist:u32 =  0;
                if(lut_dist_res == 0)
                 {
                    symbol = decode_mutate(&distcnt, &distsym);
                    // distance for copy 
                    dist = kDists[symbol] + bits_local(kDext[symbol]);
                }
                else {
                    let temp_cnt:u32 = lut_dist_res >> 24;
                    ts.bitbuf = ts.bitbuf >> temp_cnt;
                      ts.incnt  = ts.incnt + temp_cnt;
                    let dist_kdist:u32 = (lut_dist_res >> 8) & 0xFFFF;
                    let dist_kdext:u32 = (lut_dist_res & 0xFF);
                    dist = dist_kdist + bits_local(dist_kdext);
                }
        
                // copy length bytes from distance bytes back
                StreamCopyBytes(dist, len);
            }

        }

        //if(ts.err != 0){      
          //  return;
       // }

        // Either push into decompression queue or confirm failure

        while(true){
            var curr_read_incnt = atomicLoad(&g_incnt);
            if(start_bit_offset == curr_read_incnt ){
                if(invocation_hit_end_of_block){
                    decode_done = ts.incnt;
                    // All others will be forced to be speculative with this number
                    atomicStore(&g_incnt, 0xFFFFFFFF);
                    // This decode has no data but we are done now.
                    break;
                }

                // grab an index;
                var decode_head_mod = atomicAdd(&decode_head, 1) % DECODED_RING_SIZE;
                var old_write_tracker = atomicAdd(&g_write_tracker, ts.decode_len);
                decoded_ring[decode_head_mod].byte_pos = old_write_tracker;
                decoded_ring[decode_head_mod].code = ts.decode_to_store;
                atomicStore(&g_incnt, ts.incnt);
                break;
            }
            else if(curr_read_incnt > start_bit_offset) 
            {
                // speculative failure
                break;
            }
        }  

        workgroupBarrier();
        if(local_invocation_index == 0){
            while(atomicLoad(&decode_tail) != atomicLoad(&decode_head) ){
                var next_tail_mod = atomicLoad(&decode_tail) % DECODED_RING_SIZE;
                var ring_data = decoded_ring[next_tail_mod];
                atomicAdd(&decode_tail,1);
                var data_to_write = ring_data.code;
                var byte_pos_start = ring_data.byte_pos;
                if( (data_to_write & (1<<31)) !=0 ){
                    var dist = data_to_write & 0xFFFF;
                    var len = (data_to_write>>16) & 0x7FFF;
                    CopyBytes(dist, len, byte_pos_start);
                }
                else
                {
                    WriteByteOut(data_to_write, byte_pos_start);
                }
            }
        }



        workgroupBarrier();

        ts.incnt = atomicLoad(&g_incnt);
        ts.incnt += local_invocation_index;
       
        if(workgroupUniformLoad(&decode_done) != 0){
          atomicStore(&g_incnt, decode_done);
          ts.incnt = atomicLoad(&g_incnt);
          ts.incnt += local_invocation_index;
          workgroupBarrier();
          break;
        }
    }
}
    
    
fn fixed()
{
    // build fixed huffman tables if first call
    var symbol:u32;
    // literal/length table 
    for (symbol = 0; symbol < 144; symbol++) {
        lengths[symbol] = 8;
    }
    for (; symbol < 256; symbol++) {
        lengths[symbol] = 9;
    }
    for (; symbol < 280; symbol++) {
        lengths[symbol] = 7;
    }
    for (; symbol < FIXLCODES; symbol++) {
        lengths[symbol] = 8;
    }
    construct_code(&lencnt, &lensym, 0, FIXLCODES);

    // distance table
    for (symbol = 0; symbol < MAXDCODES; symbol++) {
        lengths[symbol] = 5;
    }
    construct_code(&distcnt, &distsym, 0, MAXDCODES);
}


const  kOrder = array<u32,19>(     /* permutation of code length codes */
 16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15 );

fn dynamic()
{           
    var index:u32;                          /* index of lengths[] */

    /* get number of lengths in each table, check lengths */
      /* number of lengths in descriptor */
    var nlen:u32 = bits(5) + 257u;
    var ndist:u32 = bits(5) + 1u;
    var ncode:u32 = bits(4) + 4u;
    if (nlen > MAXLCODES || ndist > MAXDCODES) {
        /* bad counts */
        ReportError(ERROR_BAD_COUNTS);
        return;
    }

    /* read code length code lengths (really), missing lengths are zero */
    for (index = 0; index < ncode; index++) {
        lengths[kOrder[index]] = i32(bits(3));
    }
    for (; index < 19; index++) {
        lengths[kOrder[index]] = 0;
    }

    /* build huffman table for code lengths codes (use lencode temporarily) */
    if (construct_code(&lencnt, &lensym, 0 , 19) != 0) {
        /* require complete code set here */
        ReportError(ERROR_REQUIRED_COMPLETE_CODE);
        return;
    }

    /* read length/literal and distance code length tables */
    index = 0;
    while (index < nlen + ndist) {
        var symbol:u32;             /* decoded value */
        var len:u32;                /* last length to repeat */

        Read32();
        symbol = decode_mutate(&lencnt, &lensym);
        if (symbol < 0) {
            /* invalid symbol */
            ReportError(ERROR_INVALID_SYMBOL);
            return;    
        }
        if (symbol < 16) {              /* length in 0..15 */
            lengths[index] = i32(symbol);
            index++;
        }
        else {                          /* repeat instruction */
            len = 0;                    /* assume repeating zeros */
            if (symbol == 16) {         /* repeat last length 3..6 times */
                if (index == 0) {
                    ReportError(ERROR_NO_LAST_LENGTH);
                    return;      
                }
                len = u32(lengths[index - 1]);       /* last length */
                symbol = 3u + bits(2);
            }
            else if (symbol == 17) {     /* repeat zero 3..10 times */
                symbol = 3u + bits(3);
            }
            else {                       /* == 18, repeat zero 11..138 times */
                symbol = 11u + bits(7);
            }
            if (index + symbol > nlen + ndist) {
                /* too many lengths! */
                ReportError(ERROR_TOO_MANY_LENGTHS);
                return;            
            }
            while (symbol !=0) {            /* repeat last or zero symbol times */
                symbol--;
                lengths[index] = i32(len);
                index++;
            }
        }
    }

    /* check for end-of-block code -- there better be one! */
    if (lengths[256] == 0) {
        ReportError(ERROR_NO_END_BLOCK_CODE); 
        return;
    }

    /* build huffman table for literal/length codes */
    var err:i32 = construct_code(&lencnt, &lensym, 0, i32(nlen));
    if (err !=0  && (err < 0 || nlen != u32(lencnt[0] + lencnt[1]) )) {
        // incomplete code ok only for single length 1 code 
        ReportError(ERROR_INCOMPLETE_CODE_SINGLE);
        return;     
    }

    /* build huffman table for distance codes */
    err = construct_code(&distcnt, &distsym, i32(nlen),i32(ndist));
    if (err !=0 && (err < 0 || ndist != u32(distcnt[0] + distcnt[1]) )) {
        // incomplete code ok only for single length 1 code 
        ReportError(ERROR_INCOMPLETE_CODE_SINGLE);
        return;   
    }
}


var<workgroup> last_block:u32;

fn puff( dictlen:u32,         // length of custom dictionary
    destlen:u32,        /* amount of output space */
    sourcelen:u32, /* amount of input available */
    local_invocation_index:u32)     
    -> i32
{
   
    ts.err = 0;                    /* return value */

    /* initialize output state */
    ws.outlen = destlen;                /* ignored if dest is NIL */
    ts.outcnt = dictlen;

    /* initialize input state */
    ws.inlen = sourcelen;
    ts.incnt = 0;
    ts.bitbuf = 0;


    while(true) {
        //if(ts.err != 0){
       //     break;
       // }

        // Only the first invocation does all the init table work 
        if(local_invocation_index == 0){
            Read32();
            var last:u32 = bits(1);         /* one if last block */
            var type_now:u32 = bits(2);         /* block type_now 0..3 */
            var is_store = false;
            if (type_now == 0) {
                debug[3]++;
                stored();
                is_store = true; // not supported anyway
            }
            else
            {
                if (type_now == 1) {
                    debug[1]++;
                    fixed();
                
                }
                else if (type_now == 2) {
                    debug[2]++;
                    dynamic();
                }
                else {
                    // type_now == 3, invalid
                    ts.err = -1;
                }
            }

            last_block =  last;
            atomicStore(&g_incnt, ts.incnt);
        }


        if(true){
            workgroupBarrier();
            GenLut(local_invocation_index);
            workgroupBarrier();
            codes(local_invocation_index);
            //last_block = 1;
            workgroupBarrier();
        }
        
       // if (ts.err != 0) {
       //     break;                  /* return with error */
       // }

        if(workgroupUniformLoad(&last_block) != 0){
          break;
        }
    } 

    if (ts.err <= 0) {
         // update the lengths and return 
       // *destlen = ts.outcnt - dictlen;
        //*sourcelen = ts.incnt;
    }
    return ts.err;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn computeMain(  @builtin(global_invocation_id) global_idx:vec3u,
 @builtin(local_invocation_index) local_invocation_index: u32,
@builtin(num_workgroups) num_work:vec3u) {
    atomicStore(&atomic_idx, 20);
    puff(0,unidata.outlen, unidata.inlen, local_invocation_index);
}
