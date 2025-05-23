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
     decode_is_copy:bool,
     invocation_hit_end_of_block:bool,
     is_output_dispatch: bool
} ;

var<private> ts : ThreadState;



const MAXBITS=15 ;             /* maximum bits in a code */
const MAXLCODES=286 ;          /* maximum number of literal/length codes */
const MAXDCODES=30 ;           /* maximum number of distance codes */
const MAXCODES=(MAXLCODES+MAXDCODES);  /* maximum codes lengths to read */
const FIXLCODES=288;           /* number of fixed literal/length codes */

const WORKGROUP_SIZE = 1u;

var<private>  lengths:array<i32, MAXCODES>;            /* descriptor code lengths */
var<private>  lencnt:array<u32, MAXBITS + 1>;
var<private>  lensym:array<u32, FIXLCODES>;
var<private>  distcnt:array<u32, MAXBITS + 1>;
 // Length should be MAXDCODES but is FIXLCODES to use same fixed sized pointer
var<private>  distsym:array<u32, FIXLCODES>;


@group(0) @binding(0) var<storage> in: array<u32>;
@group(0) @binding(1) var<storage,read_write> out: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> unidata: CommonData;
@group(0) @binding(3) var<storage,read_write> debug: array<u32>;
// Data for decoded -> store phase
const MAX_NUMBER_DEFLATE_BLOCKS = 1024*1024;
@group(0) @binding(4) var<storage, read_write>  d_start_inc_and_bytes:array<u32, MAX_NUMBER_DEFLATE_BLOCKS>;      
// cleanest way to get all this atomics into one storage.. :(      
@group(0) @binding(5) var<storage, read_write> d_head_tail_complete_useless:array<atomic<u32>, 1024>;   

// avoid contented atomics memory
const D_HEAD_INDEX = 0;
const D_TAIL_INDEX = 128;
const D_COMPLETE_INDEX = 256;
const D_USELESS_INDEX = 256+128;

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
    ts.decode_is_copy = false;
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
    ts.decode_is_copy = true;
    ts.decode_to_store = dist;
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

fn decode_dist(bitbuf_in:u32, left_in:u32) -> DecodeRtn
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
            var count:i32 = i32(distcnt[len]);
            if (code - count < first) { /* if length len, return symbol */
                var local_inded:i32 = index + (code - first);
                return  DecodeRtn(distsym[local_inded], len);
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

fn decode_len(bitbuf_in:u32, left_in:u32) -> DecodeRtn
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
            var count:i32 = i32(lencnt[len]);
            if (code - count < first) { /* if length len, return symbol */
                var local_inded:i32 = index + (code - first);
                return  DecodeRtn(lensym[local_inded], len);
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

fn decode_mutate_dist() -> u32
{
    var decode_res:DecodeRtn = decode_dist(ts.bitbuf, 32);
    if(decode_res.cnt == 0){
        ReportError(ERROR_RAN_OUT_OF_CODES);
    }
    ts.bitbuf = ts.bitbuf >> decode_res.cnt;
    ts.incnt = ts.incnt + decode_res.cnt;
    return decode_res.symbol;
}

fn decode_mutate_len() -> u32
{
    var decode_res:DecodeRtn = decode_len(ts.bitbuf, 32);
    if(decode_res.cnt == 0){
        ReportError(ERROR_RAN_OUT_OF_CODES);
    }
    ts.bitbuf = ts.bitbuf >> decode_res.cnt;
    ts.incnt = ts.incnt + decode_res.cnt;
    return decode_res.symbol;
}

fn construct_code_dist(offset:i32,  n:i32) -> i32 
{
    var  offs:array<i32, MAXBITS + 1>;        /* offsets in symbol table for each length */

    /* count number of codes of each length */
    for (var len:i32 = 0; len <= MAXBITS; len++) {
        distcnt[len] = 0;
    }
    /* current symbol when stepping through length[] */
    for (var symbol:i32 = 0; symbol < n; symbol++) {
        (distcnt[lengths[symbol+offset]])++;   /* assumes lengths are within bounds */
    }

    if (i32(distcnt[0]) == n) {              /* no codes! */
        return 0;                       /* complete, but decode() will fail */
    }

    /* check for an over-subscribed or incomplete set of lengths */
    var left:i32 = 1;                           /* one possible code of zero length */
     /* current length when stepping through h->count[] */
    for (var len:i32 = 1; len <= MAXBITS; len++) {
        left <<= 1;                     /* one more bit, double codes left */
        left -= i32(distcnt[len]);          /* deduct count from possible codes */
        if (left < 0) {
            return left;                /* over-subscribed--return negative */
        }
    }                                   /* left > 0 means incomplete */

    /* generate offsets into symbol table for each length for sorting */
    offs[1] = 0;
    for (var len:i32 = 1; len < MAXBITS; len++) {
        offs[len + 1] = offs[len] + i32(distcnt[len]);
    }

    /*
     * put symbols in table sorted by length, by symbol order within each
     * length
     */
    for (var symbol:i32 = 0; symbol < n; symbol++) {
        if (lengths[symbol+offset] != 0) {
            distsym[offs[lengths[symbol+offset]]] = u32(symbol);
            offs[lengths[symbol+offset]]++;
        }
    }

    /* return zero for complete set, positive for incomplete set */
    return left;
}

fn construct_code_len(offset:i32,  n:i32) -> i32 
{
    var  offs:array<i32, MAXBITS + 1>;        /* offsets in symbol table for each length */

    /* count number of codes of each length */
    for (var len:i32 = 0; len <= MAXBITS; len++) {
        lencnt[len] = 0;
    }
    /* current symbol when stepping through length[] */
    for (var symbol:i32 = 0; symbol < n; symbol++) {
        (lencnt[lengths[symbol+offset]])++;   /* assumes lengths are within bounds */
    }

    if (i32(lencnt[0]) == n) {              /* no codes! */
        return 0;                       /* complete, but decode() will fail */
    }

    /* check for an over-subscribed or incomplete set of lengths */
    var left:i32 = 1;                           /* one possible code of zero length */
     /* current length when stepping through h->count[] */
    for (var len:i32 = 1; len <= MAXBITS; len++) {
        left <<= 1;                     /* one more bit, double codes left */
        left -= i32(lencnt[len]);          /* deduct count from possible codes */
        if (left < 0) {
            return left;                /* over-subscribed--return negative */
        }
    }                                   /* left > 0 means incomplete */

    /* generate offsets into symbol table for each length for sorting */
    offs[1] = 0;
    for (var len:i32 = 1; len < MAXBITS; len++) {
        offs[len + 1] = offs[len] + i32(lencnt[len]);
    }

    /*
     * put symbols in table sorted by length, by symbol order within each
     * length
     */
    for (var symbol:i32 = 0; symbol < n; symbol++) {
        if (lengths[symbol+offset] != 0) {
            lensym[offs[lengths[symbol+offset]]] = u32(symbol);
            offs[lengths[symbol+offset]]++;
        }
    }

    /* return zero for complete set, positive for incomplete set */
    return left;
}

fn codex()
{
    // bits from stream 
    Read32();
    // SLOW PATH none LUT
    var symbol:u32 = decode_mutate_len();

    if (symbol < 256) { // literal: symbol is the byte 
        StreamWriteByteOut(symbol); // write out the literal 
    }
    else if (symbol == 256){  
        // end of block symbol 
        ts.invocation_hit_end_of_block = true;
        ts.decode_len = 0;
    } 
    else if (symbol > 256) {     
        symbol -= 257;  // length and distance codes get and compute length 
        if (symbol >= 29) {
            ReportError(ERROR_RAN_OUT_OF_CODES);       
        }
        var len:u32 = kLens[symbol] + bits_local(kLext[symbol]);
        symbol = decode_mutate_dist();
        // distance for copy 
        var dist:u32 = kDists[symbol] + bits_local(kDext[symbol]);
        // copy length bytes from distance bytes back
        StreamCopyBytes(dist, len);
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
    construct_code_len(0, FIXLCODES);

    // distance table
    for (symbol = 0; symbol < MAXDCODES; symbol++) {
        lengths[symbol] = 5;
    }
    construct_code_dist(0, MAXDCODES);
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
    if (construct_code_len(0 , 19) != 0) {
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
        symbol = decode_mutate_len();
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
    var err:i32 = construct_code_len(0, i32(nlen));
    if (err !=0  && (err < 0 || nlen != u32(lencnt[0] + lencnt[1]) )) {
        // incomplete code ok only for single length 1 code 
        ReportError(ERROR_INCOMPLETE_CODE_SINGLE);
        return;     
    }

    /* build huffman table for distance codes */
    err = construct_code_dist(i32(nlen),i32(ndist));
    if (err !=0 && (err < 0 || ndist != u32(distcnt[0] + distcnt[1]) )) {
        // incomplete code ok only for single length 1 code 
        ReportError(ERROR_INCOMPLETE_CODE_SINGLE);
        return;   
    }
}


var<workgroup> last_block:u32;

var<workgroup> g_start_idx:u32;
var<workgroup> g_start_count:u32;


fn codes_decode(local_invocation_index:u32)
{
    ts.invocation_hit_end_of_block = false;
    var counter = 0u;
    while(true) {
        counter++;
        codex();
        if(ts.invocation_hit_end_of_block){
            break;
        }
        else if(ts.decode_is_copy){
            CopyBytes( ts.decode_to_store, ts.decode_len, ts.outcnt);
            ts.outcnt += ts.decode_len;
        }
        else{
            WriteByteOut(ts.decode_to_store, ts.outcnt);
            ts.outcnt += ts.decode_len;
        }
    }
}
    
fn puff_decode( dictlen:u32,         // length of custom dictionary
    destlen:u32,        /* amount of output space */
    sourcelen:u32, /* amount of input available */
    local_invocation_index:u32)     
    -> i32
{
    if(local_invocation_index >= g_start_count){
        return 0; //keep only active threads
    }
   
    ts.err = 0;                    /* return value */

    /* initialize output state */
    ws.outlen = destlen;                /* ignored if dest is NIL */

    /* initialize input state */
    ws.inlen = sourcelen;
    ts.bitbuf = 0;
    ts.incnt = d_start_inc_and_bytes[(g_start_idx + local_invocation_index) * 2];
    ts.outcnt = d_start_inc_and_bytes[(g_start_idx + local_invocation_index) * 2 + 1];
    DebugWrite(ts.incnt );
    DebugWrite(  ts.outcnt);
    // This only does 1 block per invocation. 
    Read32();
    var last:u32 = bits(1);         /* one if last block */
    var type_now:u32 = bits(2);         /* block type_now 0..3 */
    var is_store = false;
    if (type_now == 0) {
        stored();
        is_store = true; // not supported anyway
        ts.err = -1;
    }
    else
    {
        if (type_now == 1) {
            fixed();
            debug[7]++;
        }
        else if (type_now == 2) {
            dynamic();
            debug[8]++;
        }
        else {
            // type_now == 3, invalid
            ts.err = -1;
        }
    }
  
    codes_decode(local_invocation_index);
    debug[9] = u32( ts.err);
    return ts.err;
}


@compute @workgroup_size(WORKGROUP_SIZE)
fn computeMain(  @builtin(workgroup_id) workgroup_id:vec3u,
 @builtin(local_invocation_index) local_invocation_index: u32,
@builtin(num_workgroups) num_work:vec3u) {

    ts.is_output_dispatch = workgroup_id.x >= 0; // only first one plays a role
    
    atomicStore(&atomic_idx, 100);
    last_block = 0;
    while(workgroupUniformLoad(&last_block) == 0) {
        workgroupBarrier();
        storageBarrier();
        var main_dispatch_complete = 0u;
        if(local_invocation_index == 0){
            g_start_count = 0;
            // order here is important
            main_dispatch_complete = atomicLoad(&d_head_tail_complete_useless[D_COMPLETE_INDEX]);
        }
        storageBarrier();
        var head_read = 0u;
        if(local_invocation_index == 0){
            head_read = atomicLoad(&d_head_tail_complete_useless[D_HEAD_INDEX]);
        }
        storageBarrier();
        if(local_invocation_index == 0){
            var tail_read = atomicLoad(&d_head_tail_complete_useless[D_TAIL_INDEX]);
            if(head_read > tail_read){
                // lets see if we can grab as many as possible
                 var acquired_count = min(WORKGROUP_SIZE, head_read- tail_read);
                //var acquired_count = min(1, head_read- tail_read);
                var new_tail = acquired_count + tail_read;
                var cas = atomicCompareExchangeWeak(&d_head_tail_complete_useless[D_TAIL_INDEX], tail_read, new_tail);
                if(cas.exchanged){
                    g_start_idx = tail_read;
                    g_start_count = acquired_count;
                    DebugWrite(8888888);
                    DebugWrite(g_start_idx);
                    DebugWrite(g_start_count);
                }
            } else if(head_read == tail_read && main_dispatch_complete != 0){
                last_block = 1;
            }
        }

        puff_decode(0, unidata.outlen, unidata.inlen, local_invocation_index);

        workgroupBarrier();
        atomicAdd(&d_head_tail_complete_useless[D_USELESS_INDEX], 1);
    }
}
