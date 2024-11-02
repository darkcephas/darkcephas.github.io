"use strict";
const WORKGROUP_SIZE = 1;


const shaderCode_original = `
struct CommonData {
      outlen:  u32,       /* available space at out */
     inlen: u32,    /* available input at in */
};

struct ThreadState {
     outcnt: u32,       /* bytes written to out so far */
    /* input state */
     incnt:u32,        /* bytes read so far */
     bitbuf:u32,                 /* bit buffer */
     bitcnt:u32,                 /* number of bits in bit buffer */
     err:i32,
} ;

var<private> ts : ThreadState;




const MAXBITS=15 ;             /* maximum bits in a code */
const MAXLCODES=286 ;          /* maximum number of literal/length codes */
const MAXDCODES=30 ;           /* maximum number of distance codes */
const MAXCODES=(MAXLCODES+MAXDCODES);  /* maximum codes lengths to read */
const FIXLCODES=288;           /* number of fixed literal/length codes */


var<workgroup>  lengths:array<i32, MAXCODES>;            /* descriptor code lengths */
var<workgroup>  lencnt:array<i32, MAXBITS + 1>;
var<workgroup>  lensym:array<i32, FIXLCODES>;
var<workgroup>  distcnt:array<i32, MAXBITS + 1>;
var<workgroup>  distsym:array<i32, MAXDCODES>;


var<workgroup> debug_counter:atomic<u32>;

@group(0) @binding(0) var<storage> in: array<u32>;
@group(0) @binding(1) var<storage,read_write> out: array<u32>;
@group(0) @binding(2) var<uniform> unidata: CommonData;
@group(0) @binding(3) var<storage,read_write> debug: array<u32>;

var<workgroup> ws : CommonData;


const ERROR_OUTPUT_OVERFLOW = 2;
const ERROR_NO_MATCH_COMPLEMENT = 2;
const ERROR_RAN_OUT_OF_CODES = -10;
const ERROR_INCOMPLETE_CODE_SINGLE = -8;
const ERROR_NO_END_BLOCK_CODE = -9;
const ERROR_NO_LAST_LENGTH = -5;
const ERROR_INVALID_SYMBOL=-7;
const ERROR_BAD_COUNTS = -3;
const ERROR_REQUIRED_COMPLETE_CODE=-4;
const ERROR_TOO_MANY_LENGTHS= -6;

fn  ReadByteIn() -> u32
{
    if (ts.incnt + 1 > ws.inlen) {
        ts.err = ERROR_OUTPUT_OVERFLOW;
        return 0;
    }

    var val:u32 = in[ts.incnt/4];

    var sub_index:u32 = ts.incnt % 4;
    val = (val >> (8 * sub_index)) & 0xff;

    ts.incnt++;
    return val;
}

fn PeekByteOut( rev_offset_in_bytes:u32) -> u32
{
    if (ts.outcnt + 1 > ws.outlen) {
        ts.err = ERROR_OUTPUT_OVERFLOW;
        return 0;
    }

    var  offset:u32 = ts.outcnt - rev_offset_in_bytes;
    var  val:u32 = out[offset / 4];

    var sub_index:u32 = offset % 4;
    val = (val >> (8 * sub_index)) & 0xff;
    return val;
}


fn WriteByteOut( val:u32)
{
    if (ts.outcnt + 1 > ws.outlen) {
        ts.err = ERROR_OUTPUT_OVERFLOW;
        return;
    }

    var curr_val:u32 = out[ts.outcnt/4];

    var  sub_index:u32 = ts.outcnt % 4;
    // mask out the byte
    curr_val = curr_val & ~(0xffu << (sub_index*8u));
    curr_val = curr_val | ( (val&0xffu) << (sub_index * 8u));

    out[ts.outcnt/4] = curr_val;
    ts.outcnt++;
}


fn bits( need:u32) ->u32
{
    /* bit accumulator (can use up to 20 bits) */
    /* load at least need bits into val */
    var val :u32 = ts.bitbuf;
    while (ts.bitcnt < need) {
        var val2: u32 = ReadByteIn();
        val |= val2 << ts.bitcnt;  /* load eight bits */
        ts.bitcnt += 8;
    }

    /* drop need bits and update buffer, always zero to seven bits left */
    ts.bitbuf = val >> need;
    ts.bitcnt -= need;

    /* return need bits, zeroing the bits above that */
    return u32(val & ((1u << need) - 1u));
}

fn  stored() 
{
    /* discard leftover bits from current byte (assumes ts.bitcnt < 8) */
    ts.bitbuf = 0;
    ts.bitcnt = 0;

    /* get length and check against its one's complement */
    /* length of stored block */
    var len :u32 = ReadByteIn() | (ReadByteIn() << 8);
    if( ReadByteIn() != (~len & 0xff) ||
        ReadByteIn() != ((~len >> 8) & 0xff)) {
        ts.err = ERROR_NO_MATCH_COMPLEMENT;  
    }

    while (len !=0) {
        len--;
        var val:u32 = ReadByteIn();
        WriteByteOut(val);
    }
}




fn  decode_lencode() -> i32
{
     /* bits from stream */
    var bitbuf:i32 = i32(ts.bitbuf);
    /* bits left in next or left to process */
    var left:i32 =i32(ts.bitcnt);
     // len bits being decoded 
    var code:i32 = 0;
    // first code of length len 
    var first:i32 = 0;
     // index of first code of length len in symbol table 
    var index:i32 = 0;
     // current number of bits in code 
    var len:i32 = 1;
      // next number of codes 
    var next:i32 = 1;
    while (true) {
        while (left !=0) {
            left--;
            code |= bitbuf & 1;
            bitbuf >>= 1;
            // number of codes of length len 
            var count:i32 =  lencnt[next];
            next++;
            if (code - count < first) { 
                // if length len, return symbol
                ts.bitbuf = u32(bitbuf);
                ts.bitcnt = (ts.bitcnt - u32(len)) & 0x7u;
                var local_inded:i32 = index + (code - first);
                return  lensym[local_inded];
            }
            // else update for next length
            index += count;            
            first += count;
            first <<= 1;
            code <<= 1;
            len++;
           
        }
        left = (MAXBITS + 1) - len;
        if (left == 0) {
            break;
        }

        bitbuf = i32(ReadByteIn());
        if (left > 8) {
            left = 8;
        }
    }
    ts.err = ERROR_RAN_OUT_OF_CODES;
    return ERROR_RAN_OUT_OF_CODES;                        
}

fn decode_distcode() -> i32
{
     /* bits from stream */
    var bitbuf:i32 = i32(ts.bitbuf);
    /* bits left in next or left to process */
    var left:i32 = i32(ts.bitcnt);
     var code:i32 = 0; // len bits being decoded
    var first:i32 = 0;  // first code of length len 
    var index:i32 = 0; // index of first code of length len in symbol table 
    var len:i32 = 1; // current number of bits in code
     var next:i32 = 1;    /* next number of codes */
    while (true) {
        while (left !=0) {
            code |= bitbuf & 1;
            bitbuf >>= 1;
             /* number of codes of length len */
             var count:i32 =  distcnt[next];
            next++;
            if (code - count < first) { /* if length len, return symbol */
                ts.bitbuf = u32(bitbuf);
                ts.bitcnt = (ts.bitcnt - u32(len)) & 0x7u;
                var local_inded:i32 = index + (code - first);
                return  distsym[local_inded];
            }
            index += count;             /* else update for next length */
            first += count;
            first <<= 1;
            code <<= 1;
            len++;
            left--;
        }
        left = (MAXBITS + 1) - len;
        if (left == 0) {
            break;
        }

        bitbuf = i32(ReadByteIn());
        if (left > 8) {
            left = 8;
        }
    }
    ts.err = ERROR_RAN_OUT_OF_CODES;
    return ERROR_RAN_OUT_OF_CODES;
}




fn construct_lencode(offset:i32, n:i32) -> i32 
{       
    var  offs:array<i32, MAXBITS + 1>;        /* offsets in symbol table for each length */

    /* count number of codes of each length */
    for (var len:i32 = 0; len <= MAXBITS; len++) {
        lencnt[len] = 0;
    }
    /* current symbol when stepping through length[] */
    for (var symbol:i32 = 0; symbol < n; symbol++) {
        (lencnt[lengths[symbol+ offset]])++;   /* assumes lengths are within bounds */
    }

    if (lencnt[0] == n) {              /* no codes! */
        return 0;                       /* complete, but decode() will fail */
    }

    /* check for an over-subscribed or incomplete set of lengths */
     var left:i32 = 1;                           /* one possible code of zero length */
     /* current length when stepping through h->count[] */
    for (var len:i32 = 1; len <= MAXBITS; len++) {
        left <<= 1;                     /* one more bit, double codes left */
        left -= lencnt[len];          /* deduct count from possible codes */
        if (left < 0) {
            return left;                /* over-subscribed--return negative */
        }
    }                                   /* left > 0 means incomplete */

    /* generate offsets into symbol table for each length for sorting */
    offs[1] = 0;
    for (var len:i32 = 1; len < MAXBITS; len++) {
        offs[len + 1] = offs[len] + lencnt[len];
    }

    /*
     * put symbols in table sorted by length, by symbol order within each
     * length
     */
    for (var symbol:i32 = 0; symbol < n; symbol++) {
        if (lengths[symbol+ offset] != 0) {
            lensym[offs[lengths[symbol+ offset]]] = i32(symbol);
            offs[lengths[symbol+ offset]]++;
        }
    }

    /* return zero for complete set, positive for incomplete set */
    return left;
}


fn construct_distcode( offset:i32,  n:i32) -> i32 
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

    if (distcnt[0] == n) {              /* no codes! */
        return 0;                       /* complete, but decode() will fail */
    }

    /* check for an over-subscribed or incomplete set of lengths */
    var left:i32 = 1;                           /* one possible code of zero length */
     /* current length when stepping through h->count[] */
    for (var len:i32 = 1; len <= MAXBITS; len++) {
        left <<= 1;                     /* one more bit, double codes left */
        left -= distcnt[len];          /* deduct count from possible codes */
        if (left < 0) {
            return left;                /* over-subscribed--return negative */
        }
    }                                   /* left > 0 means incomplete */

    /* generate offsets into symbol table for each length for sorting */
    offs[1] = 0;
    for (var len:i32 = 1; len < MAXBITS; len++) {
        offs[len + 1] = offs[len] + distcnt[len];
    }

    /*
     * put symbols in table sorted by length, by symbol order within each
     * length
     */
    for (var symbol:i32 = 0; symbol < n; symbol++) {
        if (lengths[symbol+offset] != 0) {
            distsym[offs[lengths[symbol+offset]]] = i32(symbol);
            offs[lengths[symbol+offset]]++;
        }
    }

    /* return zero for complete set, positive for incomplete set */
    return left;
}

// Const data access is very fast
const lens= array<u32,29> ( /* Size base for length codes 257..285 */
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31,
    35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258 );
const lext= array<u32,29> ( /* Extra bits for length codes 257..285 */
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
    3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0 );
const dists= array<u32,30> ( /* Offset base for distance codes 0..29 */
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193,
    257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145,
    8193, 12289, 16385, 24577 );
const  dext= array<u32,30> ( /* Extra bits for distance codes 0..29 */
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
    7, 7, 8, 8, 9, 9, 10, 10, 11, 11,
    12, 12, 13, 13 );



fn  codes() -> i32
{
    /* decode literals and length/distance pairs */
    while(true) {
        var symbol:i32  = decode_lencode();
        if (symbol < 0) {
            return ERROR_INVALID_SYMBOL;
        }

        if (symbol < 256) {             /* literal: symbol is the byte */
            /* write out the literal */
            WriteByteOut(u32(symbol));
        }
        else if (symbol > 256) {        
            // length and distance codes
            // get and compute length 
            symbol -= 257;
            if (symbol >= 29) {
                // invalid fixed code
                return ERROR_RAN_OUT_OF_CODES;           
            }
            // length for copy 
            var len:u32;           
            len = lens[symbol] + bits(lext[symbol]);

            // get and check distance 
            symbol = decode_distcode();
            if (symbol < 0) {
                return ERROR_INVALID_SYMBOL;        
            }
            // distance for copy 
            var dist:u32 =  dists[symbol] + bits(dext[symbol]);

            // copy length bytes from distance bytes back
            while (len != 0) {
                len--;
                var val:u32 = PeekByteOut(dist);
                WriteByteOut(val);
            }
        }

         /* end of block symbol */
        if (symbol == 256){
          return 0;
        }          
    }

    /* done with a valid fixed or dynamic block */
    return 0;
}
    
    
fn fixed()
{
    /* build fixed huffman tables if first call (may not be thread safe) */
    var symbol:u32;
    /* literal/length table */
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
    construct_lencode(0, FIXLCODES);

    /* distance table */
    for (symbol = 0; symbol < MAXDCODES; symbol++) {
        lengths[symbol] = 5;
    }
    construct_distcode(0, MAXDCODES);
}


const  order = array<i32,19>(     /* permutation of code length codes */
 16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15 );

fn dynamic()
{           
    var index:u32;                          /* index of lengths[] */
    var err: i32;                            /* construct() return value */

    /* get number of lengths in each table, check lengths */
      /* number of lengths in descriptor */
    var nlen:u32 = bits(5) + 257u;
    var ndist:u32 = bits(5) + 1u;
    var ncode:u32 = bits(4) + 4u;
    if (nlen > MAXLCODES || ndist > MAXDCODES) {
        /* bad counts */
        err = ERROR_BAD_COUNTS;
        return;
    }

    /* read code length code lengths (really), missing lengths are zero */
    for (index = 0; index < ncode; index++) {
        lengths[order[index]] = i32(bits(3));
    }
    for (; index < 19; index++) {
        lengths[order[index]] = 0;
    }

    /* build huffman table for code lengths codes (use lencode temporarily) */
    err = construct_lencode(0, 19);
    if (err != 0) {
        /* require complete code set here */
        err = ERROR_REQUIRED_COMPLETE_CODE;
        return;
    }

    /* read length/literal and distance code length tables */
    index = 0;
    while (index < nlen + ndist) {
        var symbol:u32;             /* decoded value */
        var len:u32;                /* last length to repeat */

        symbol = u32(decode_lencode());
        if (symbol < 0) {
            /* invalid symbol */
            err = i32(symbol);
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
                    err = ERROR_NO_LAST_LENGTH;
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
                err = ERROR_TOO_MANY_LENGTHS;// -6;
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
        err = ERROR_NO_END_BLOCK_CODE; 
        return;
    }

    /* build huffman table for literal/length codes */
    err = construct_lencode(0, i32(nlen));
    if (err !=0  && (err < 0 || nlen != u32(lencnt[0] + lencnt[1]) )) {
        // incomplete code ok only for single length 1 code 
        err = ERROR_INCOMPLETE_CODE_SINGLE;
        return;     
    }

    /* build huffman table for distance codes */
    err = construct_distcode(i32(nlen),i32(ndist));
    if (err !=0 && (err < 0 || ndist != u32(distcnt[0] + distcnt[1]) )) {
        // incomplete code ok only for single length 1 code 
        err = ERROR_INCOMPLETE_CODE_SINGLE;
        return;   
    }
}


fn puff( dictlen:u32,         // length of custom dictionary
    destlen:u32,        /* amount of output space */
    sourcelen:u32)      /* amount of input available */
    -> i32
{
   
    ts.err=0;                    /* return value */

    /* initialize output state */
    ws.outlen = destlen;                /* ignored if dest is NIL */
    ts.outcnt = dictlen;

    /* initialize input state */
    ws.inlen = sourcelen;
    ts.incnt = 0;
    ts.bitbuf = 0;
    ts.bitcnt = 0;


    /* process blocks until last block or error */
    var last:u32 =0;             /* block information */
    while(true) {
        last = bits(1);         /* one if last block */
        var type_now:u32 = bits(2);         /* block type_now 0..3 */
        if (type_now == 0) {
         debug[3]++;
            stored();
        }
        else
        {
            if (type_now == 1) {
                debug[1]++;
                fixed();
                ts.err |= codes();
            }
            else if (type_now == 2) {
                debug[2]++;
                dynamic();
                ts.err |= codes();

            }
            else {
                // type_now == 3, invalid
                ts.err = -1;
            }
        }
          
        if (ts.err != 0) {
            break;                  /* return with error */
        }

        if(last != 0){
          break;
        }
    } 
    

    /* update the lengths and return */
    if (ts.err <= 0) {
       // *destlen = ts.outcnt - dictlen;
        //*sourcelen = ts.incnt;
    }
    return ts.err;
}

@compute @workgroup_size(256)
fn computeMain(  @builtin(global_invocation_id) global_idx:vec3u,
 @builtin(local_invocation_index) local_invocation_index: u32,
@builtin(num_workgroups) num_work:vec3u) {
  if(local_invocation_index != 0)
  {
    if(local_invocation_index == 63){
        for(var i =0;i <10000000;i++){
            atomicAdd(&debug_counter,1);
        }
    }
    return;
  }

  puff(0,unidata.outlen, unidata.inlen);

  debug[0] = 776;

}
`;