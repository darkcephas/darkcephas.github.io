"use strict";


const WORKGROUP_SIZE = 1;

const shaderCode = `
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



@group(0) @binding(0) var<storage> in: array<u32>;
  @group(0) @binding(1) var<storage,read_write> out: array<u32>;
 @group(0) @binding(2) var<uniform> ws: CommonData;



fn  ReadByteIn() -> u32
{
    if (ts.incnt + 1 > ws.inlen) {
        ts.err = 2;
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
        ts.err = 2;
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
        ts.err = 2;
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

fn  stored() -> i32
{
    /* discard leftover bits from current byte (assumes ts.bitcnt < 8) */
    ts.bitbuf = 0;
    ts.bitcnt = 0;

    /* get length and check against its one's complement */
    /* length of stored block */
    var len :u32 = ReadByteIn() | (ReadByteIn() << 8);
    if( ReadByteIn() != (~len & 0xff) ||
        ReadByteIn() != ((~len >> 8) & 0xff)) {
        ts.err = -2;  /* didn't match complement! */
    }

    while (len !=0) {
        var val:u32 = ReadByteIn();
        WriteByteOut(val);
        len--;
    }

    /* done with a valid stored block */
    return 0;
}




fn  decode_lencode() -> i32
{
    var len:i32;            /* current number of bits in code */
    var code:i32;           /* len bits being decoded */
    var first:i32;          /* first code of length len */
    var count:i32;          /* number of codes of length len */
    var index:i32;          /* index of first code of length len in symbol table */               
    var next:i32;        /* next number of codes */

     /* bits from stream */
    var bitbuf:i32 = i32(ts.bitbuf);
    /* bits left in next or left to process */
    var left:i32 =i32(ts.bitcnt);
    code = 0;
    first = 0;
    index = 0;
    len = 1;
    next = 1;
    while (true) {
        while (left !=0) {
            code |= bitbuf & 1;
            bitbuf >>= 1;
            count =  lencnt[next];
            next++;
            if (code - count < first) { /* if length len, return symbol */
                ts.bitbuf = u32(bitbuf);
                ts.bitcnt = (ts.bitcnt - u32(len)) & 0x7u;
                var local_inded:i32 = index + (code - first);
                return  lensym[local_inded];
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
    ts.err = -10;
    return -10;                         /* ran out of codes */
}

fn decode_distcode() -> i32
{
    var len:i32;            /* current number of bits in code */
    var code:i32;           /* len bits being decoded */
    var first:i32;          /* first code of length len */
    var count:i32;          /* number of codes of length len */
    var index:i32;          /* index of first code of length len in symbol table */
    var next:i32;        /* next number of codes */

     /* bits from stream */
    var bitbuf:i32 = i32(ts.bitbuf);
    /* bits left in next or left to process */
    var left:i32 = i32(ts.bitcnt);
    code = 0;
    first = 0;
    index = 0;
    len = 1;
    next = 1;
    while (true) {
        while (left !=0) {
            code |= bitbuf & 1;
            bitbuf >>= 1;
            count =  distcnt[next];
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
    ts.err = -10;
    return -10;                         /* ran out of codes */
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


@compute @workgroup_size(${WORKGROUP_SIZE})
fn computeMain(  @builtin(global_invocation_id) global_idx:vec3u,
@builtin(num_workgroups) num_work:vec3u) {
  for(var i = 0u ;i < ws.inlen;i++){
   out[i]= in[i];
  }
}
`;