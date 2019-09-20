#include <stdio.h>
#include <string.h>
#include <math.h>
#include "pulp.h"
#include "rt/rt_api.h"

#define NOGPIO
// #define BYTE


#define ITERATIONS 100
#define ENABLE_CYCLE_TRACE 0

#define ALIM_1_VOLT 0
#define FREQ_FC (250*1000000)
#define FREQ_CL (175*1000000)


#define Wi  112
#define Hi  112
#define Wic 100
#define Hic 100
#define Wil 1024
#define Hil 16

#define Wxor    112
#define Hxor    112
#define Cxor	32

char tests_names[][50] = {

{"5x5 Convolutions"},
{"Xnor Conv 5x5"},

{"Parallel 5x5 Convolution"},
{"Parallel Xnor Conv 5x5"},

//The benchmark names added by Shien Zhu on 2019-9-3
{"Standard Full Precison Conv"}
{"Scaled XNOE-Net Conv"}
{"Optimized Scaled XNOE-Net Conv"}

{"Parallel Standard Full Precison Conv"}
{"Parallel Scaled XNOE-Net Conv"}
{"Parallel Optimized Scaled XNOE-Net Conv"}
};

int test_input_w[] = { Wi,Wi,Wi,Wil,Wxor };
int test_input_h[] = { Hi,Hi,Hi,Hil,Hxor };


int tests_ops[10] = {
	((Wi / 2)*(Hi / 2)),
	((Wi / 2)*(Hi / 2)),
	(Wic - 4)*(Hic - 4),
	(Wil*Hil),
	(Wic - 4)*(Hic - 4)
};

int tests_input[][2] = {
	{Wi , Hi },
	{Wi , Hi },
	{Wic, Hic},
	{Wil, Hil},
	{Wic, Hic}
};


char tests_titles[][50] = {
{"Sequential"},
{"Parallel"}
};



#define ALIGN(Value, Size)      (((Value)&((1<<(Size))-1))?((((Value)>>(Size))+1)<<(Size)):(Value))

#ifndef RISCV
#define Min(a, b)       __builtin_pulp_minsi((a), (b))
#define Max(a, b)       __builtin_pulp_maxsi((a), (b))
#else
#define Min(a, b)       (((a)<(b))?(a):(b))
#define Max(a, b)       (((a)>(b))?(a):(b))
#endif

#ifdef __EMUL__
#define plp_cluster_fetch(a)
#define plp_cluster_wait(a)
#endif



#ifdef NOGPIO
#define WriteGpio(a, b)
#else
#define WriteGpio(a, b) rt_gpio_set_pin_value(0, a, b)
#endif

#ifdef RISCV
#define TOT_TEST 1
int test_num[TOT_TEST] = { 5 };

#ifndef __EMUL__
#endif


#define L2_MEM                          __attribute__((section(".heapl2ram")))
#define L1_CL_MEM                       __attribute__((section(".heapsram")))
#define L1_FC_MEM                       __attribute__((section(".fcTcdm")))
/* HW timer */
#define ARCHI_FC_TIMER_ADDR                     ( ARCHI_FC_PERIPHERALS_ADDR + ARCHI_FC_TIMER_OFFSET  )
#define PLP_TIMER_VALUE_LO                      0x08
#define PLP_TIMER_CFG_REG_LO                    0x00
#define PLP_TIMER_ENABLE_BIT            0
#define PLP_TIMER_RESET_BIT             1
#define PLP_TIMER_IRQ_ENABLE_BIT        2
#define PLP_TIMER_IEM_BIT               3
#define PLP_TIMER_CMP_CLR_BIT           4
#define PLP_TIMER_ONE_SHOT_BIT          5
#define PLP_TIMER_PRESCALER_ENABLE_BIT  6
#define PLP_TIMER_CLOCK_SOURCE_BIT      7
#define PLP_TIMER_PRESCALER_VALUE_BIT   8
#define PLP_TIMER_PRESCALER_VALUE_BITS  8
#define PLP_TIMER_64_BIT                31

#define plp_timer_conf_get(a,b,c,d,e,f,g,h,i)      ((a << PLP_TIMER_ENABLE_BIT) \
        | (b << PLP_TIMER_RESET_BIT) \
        | (c << PLP_TIMER_IRQ_ENABLE_BIT) \
        | (d << PLP_TIMER_IEM_BIT) \
        | (e << PLP_TIMER_CMP_CLR_BIT) \
        | (f << PLP_TIMER_ONE_SHOT_BIT) \
        | (g << PLP_TIMER_PRESCALER_ENABLE_BIT) \
        | (h << PLP_TIMER_PRESCALER_VALUE_BIT) \
        | (i << PLP_TIMER_64_BIT) \
        )
#define gap8_resethwtimer()                     pulp_write32(ARCHI_FC_TIMER_ADDR + PLP_TIMER_CFG_REG_LO, plp_timer_conf_get(1,1,0,0,0,0,0,0,0))
#define gap8_readhwtimer()                      pulp_read32(ARCHI_FC_TIMER_ADDR + PLP_TIMER_VALUE_LO)

#else
#define TOT_TEST 4
int test_num[TOT_TEST] = { 5,4,5,4 };
#include "Gap8.h"

static int CoreCountDynamic = 0;
static int ActiveCore = 8;

static inline unsigned int __attribute__((always_inline)) ChunkSize(unsigned int X)

{
	unsigned int NCore;
	unsigned int Log2Core;
	unsigned int Chunk;

	if (CoreCountDynamic) NCore = ActiveCore; else NCore = gap8_ncore();
	Log2Core = gap8_fl1(NCore);
	Chunk = (X >> Log2Core)  + ((X&(NCore - 1)) != 0);
	return Chunk;
}
#endif

#define STACK_SIZE      2048
#define MOUNT           1
#define UNMOUNT         0
#define CID             0


typedef struct ClusterArg {
	int test_num;
	int Iter;
	int Trace;
	char Mode[10];
	int Iter_operations;
} ClusterArg_t;

ClusterArg_t Arg;


char str[100];
static char *float_to_string(float in) {

	int d1 = in;
	float f2 = in - d1;
	int d2 = trunc(f2 * 10000);

	sprintf(str, "%d.%04d", d1, d2);
	return str;
}

#ifndef RISCV
#define B_ins(dst, src, size, off)      gap8_bitinsert(dst, src, size, off)
#define B_ins_r(dst, src, size, off)    gap8_bitinsert_r(dst, src, size, off)
#define B_ext(x, size, off)             gap8_bitextract(x, size, off)
#define B_extu(x, size, off)            gap8_bitextractu(x, size, off)
#define B_ext_r(x, size, off)           gap8_bitextract_r(x, size, off)
#define B_extu_r(x, size, off)          gap8_bitextractu_r(x, size, off)
#define B_popc(src)                     __builtin_popcount((src))
#else
static __attribute__((always_inline)) unsigned int bitcount32(unsigned int b)
{
	b -= (b >> 1) & 0x55555555;
	b = (b & 0x33333333) + ((b >> 2) & 0x33333333);
	b = (b + (b >> 4)) & 0x0f0f0f0f;
	return (b * 0x01010101) >> 24;
}
#define B_ins(dst, src, size, off)      (((dst) & ~(((1<<(size))-1)<<(off))) | (((src) & ((1<<(size))-1))<<(off)))
#define B_ins_r(dst, src, size, off)    (((dst) & ~(((1<<(size))-1)<<(off))) | (((src) & ((1<<(size))-1))<<(off)))
#define B_ext(x, size, off)             (((((x)>>(off))&((unsigned int)(1<<(size))-1))<<(32-(size)))>>(32-(size)))
#define B_extu(x, size, off)            (((x)>>(off))&((unsigned int)(1<<(size))-1))
#define B_ext_r(x, size, off)           (((((x)>>(off))&((unsigned int)(1<<(size))-1))<<(32-(size)))>>(32-(size)))
#define B_extu_r(x, size, off)          (((x)>>(off))&((unsigned int)(1<<(size))-1))
#define B_popc(src)                     bitcount32((src))
#endif

#define VSOC	1000
#define GPIO	17

#ifdef BYTE
typedef signed char Ty;
#else
typedef short int Ty;
#endif

#ifdef BYTE
#define MAX_MEM	55000
#else
#define MAX_MEM	(55000/2)
#endif
Ty L1_CL_MEM Mem[MAX_MEM];

typedef struct {
	Ty *__restrict__ In;
	int W;
	int H;
	Ty *__restrict__ Filter;
	Ty *__restrict__ Out;
	int Norm;
} ArgConvT;

typedef struct {
    Ty *__restrict__ In;
    int W;
    int H;
    int C;
    Ty *__restrict__ Filter;
    Ty *__restrict__ Out;
    int Norm;
} ArgConvTnew;

typedef struct {

	unsigned int InBit;
	signed char *__restrict__ Out;
	unsigned int FilterBit;
	int W;
	int H;
} ArgConvBT;

typedef struct {

    unsigned int InBit;
    signed char *__restrict__ Out;
    signed char *__restrict__ OutK;
    unsigned int FilterBit;
    int W;
    int H;
    int C;
} ArgConvBTnew;

void CheckMem(int Size)

{
	if (Size > MAX_MEM) {
		printf("Memory Overflow (%d>%d). Aborting\n", Size, MAX_MEM); exit(0);
	}
}

#ifndef RISCV
v4s L1_CL_MEM LinearMask[4] = { (v4s)0, (v4s)0xFF, (v4s)0xFFFF, (v4s)0xFFFFFF };
#ifdef BYTE
// Vector
#else
// Same Vector
#endif


// Parallel 5X5 Conv
void __attribute__((noinline)) ParAdditive5x5Convolution(ArgConvT *Arg)

{
	Ty *__restrict__ In = Arg->In;
	Ty *__restrict__ Filter = Arg->Filter;
	Ty *__restrict__ Out = Arg->Out;
	int W = Arg->W, H = Arg->H, Norm = Arg->Norm;
	int FH = 5, FW = 5;
	unsigned int CoreId = gap8_coreid();
	unsigned int Chunk = ChunkSize(W - 4);
	unsigned int First = Chunk*CoreId;
	unsigned int Last = Min(First + Chunk, W - 4);
	for (unsigned int c = First; c < Last; c++) {
		for (int l = 0; l < (H - 4); l++) {
			int R = Out[l*W + c] << Norm;
			for (int kl = 0; kl < FH; kl++) {
				R += Filter[kl*FW + 0] * In[(l + kl)*W + c + 0];
				R += Filter[kl*FW + 1] * In[(l + kl)*W + c + 1];
				R += Filter[kl*FW + 2] * In[(l + kl)*W + c + 2];
				R += Filter[kl*FW + 3] * In[(l + kl)*W + c + 3];
				R += Filter[kl*FW + 4] * In[(l + kl)*W + c + 4];
				/*
												for (int kc=0; kc<FW; kc++) {
														R += Filter[kl*FW+kc]*In[(l+kl)*W + c+kc];
												}
				*/
			}
			Out[l*W + c] = R >> Norm;
		}
	}
	gap8_waitbarrier(0);
}


// Parallel XNOR Conv 5X5
void __attribute__((noinline)) ParXnorConv5x5(ArgConvBT *Arg)

{
	unsigned int InBit = Arg->InBit;
	signed char *__restrict__ Out = Arg->Out;
	unsigned int FilterBit = Arg->FilterBit;
	int W = Arg->W;
	int H = Arg->H;

	int Wo = W - 4;
	int Ho = H - 4;
	unsigned int CoreId = gap8_coreid();
	unsigned int Chunk = ChunkSize(Wo);
	unsigned int First = Chunk*CoreId;
	unsigned int Last = Min(First + Chunk, Wo);
	int Wo_F = First; int Wo_L = Last;
	int Ho_F = 0; int Ho_L = Ho;
	unsigned int Stride = 1, K = 2;
	unsigned int C = *((unsigned int *)(FilterBit / 8)) >> (FilterBit % 8);
	signed char *PtO1 = Out + Wo*Ho_F + Wo_F;
	unsigned char *PtByte;
	unsigned int PtBit;
	unsigned int ExtMask = 5 << 5;
	unsigned int CC = C;
	CC = B_ins(CC, B_ext(C, 5, 5), 5, 6);
	CC = B_ins(CC, B_ext(C, 5, 10), 5, 12);
	CC = B_ins(CC, B_ext(C, 5, 15), 5, 18);
	CC = B_ins(CC, B_ext(C, 5, 20), 5, 24);

	int Iter = Wo_L - Wo_F;
	for (int i = 0, w = Wo_F; (i < (Iter / 2)); i++, w += 2) {
		unsigned int V, N;
		PtBit = InBit + (Ho_F*Stride)*W + (w*Stride);
		PtByte = (unsigned char *)(PtBit / 8);
		char *PtO = (char *)PtO1;
		V = 0;
		N = B_extu_r(*(unsigned short int *)PtByte, 6, (PtBit % 8));
		V = B_ins(V, N, 6, 0); PtBit += W; PtByte = (unsigned char *)(PtBit / 8);
		N = B_extu_r(*(unsigned short int *)PtByte, 6, (PtBit % 8));
		V = B_ins(V, N, 6, 6); PtBit += W; PtByte = (unsigned char *)(PtBit / 8);
		N = B_extu_r(*(unsigned short int *)PtByte, 6, (PtBit % 8));
		V = B_ins(V, N, 6, 12); PtBit += W; PtByte = (unsigned char *)(PtBit / 8);
		N = B_extu_r(*(unsigned short int *)PtByte, 6, (PtBit % 8));
		V = B_ins(V, N, 6, 18); PtBit += W; PtByte = (unsigned char *)(PtBit / 8);

		for (int h = Ho_F; h < Ho_L; h++) {
			N = B_extu_r(*(unsigned short int *)PtByte, 6, (PtBit % 8));
			V = B_ins(V, N, 6, 24); PtBit += W; PtByte = (unsigned char *)(PtBit / 8);
			int R0 = B_popc((~(V^CC)) & 0x1F7DF7DF),
				R1 = B_popc((~((V >> 1) ^ CC)) & 0x1F7DF7DF);
			unsigned int V1 = R0 | (R1 << 8);
			v4s Val = (v4s)(unsigned int)(*(unsigned short int *) PtO);
			Val = __builtin_pulp_add4(Val, (v4s)V1);
			*(unsigned short int *)PtO = (unsigned short int)(unsigned int)Val; PtO += Wo;
			V = V >> ((2 * K + 1) + 1);
		}
		PtO1 += 2;
	}
	PtO1 = Out + Wo*Ho_F + Wo_F + 2 * (Iter / 2);
	for (int w = Wo_F + 2 * (Iter / 2); w < Wo_L; w++) {
		unsigned int V, N;
		PtBit = InBit + (Ho_F*Stride)*W + (w*Stride);
		PtByte = (unsigned char *)(PtBit / 8);
		char *PtO = (char *)PtO1;
		V = 0;
		N = B_extu_r(*(unsigned short int *)PtByte, 5, (PtBit % 8));
		V = B_ins(V, N, 5, 0); PtBit += W; PtByte = (unsigned char *)(PtBit / 8);
		N = B_extu_r(*(unsigned short int *)PtByte, 5, (PtBit % 8));
		V = B_ins(V, N, 5, 5); PtBit += W; PtByte = (unsigned char *)(PtBit / 8);
		N = B_extu_r(*(unsigned short int *)PtByte, 5, (PtBit % 8));
		V = B_ins(V, N, 5, 10); PtBit += W; PtByte = (unsigned char *)(PtBit / 8);
		N = B_extu_r(*(unsigned short int *)PtByte, 5, (PtBit % 8));
		V = B_ins(V, N, 5, 15); PtBit += W; PtByte = (unsigned char *)(PtBit / 8);
		for (int h = Ho_F; h < Ho_L; h++) {
			N = B_extu_r(*(unsigned short int *)PtByte, 5, (PtBit % 8));
			V = B_ins(V, N, 5, 20); PtBit += W; PtByte = (unsigned char *)(PtBit / 8);
			*PtO += B_popc((~(V^C)) & 0x1FFFFFF); PtO += Wo;
			V = V >> (2 * K + 1);
		}
		PtO1++;
	}
}

#endif


// 3X3 Conv
void __attribute__((noinline)) My3X3Convolution(float *__restrict__ In, int H, int W, int C, int Filternum,
        float *__restrict__ Filter, float *__restrict__ Out)
{
    int Wo=W-2;
    int Ho=H-2;
    int KW=3, KH=3;

    for (int iw = 0; iw < (W - 2); iw++) {
        for (int ih = 0; ih < (H - 2); ih++) {
            for (int fn = 0; fn < Filternum; fn++) {
                int R = Out[fn * Wo * Ho + ih * Wo + iw] = 0;
                // Use C, H, W format for data locality
                for (int kc = 0; kc < C; kc++) {
                    // This is the H, W, C format
                    //R += Filter[(0 * KW + 0)*C + kc] * In[((ih+0) * W + iw+0)*C + kc];
                    // We use the C, H, W format
                    R +=Filter[fn * C * KW * KH + kc * KH * KW + 0 * KW + 0]*In[kc * W * H + (ih + 0) * W + iw + 0];
                    R +=Filter[fn * C * KW * KH + kc * KH * KW + 0 * KW + 1]*In[kc * W * H + (ih + 0) * W + iw + 1];
                    R +=Filter[fn * C * KW * KH + kc * KH * KW + 0 * KW + 2]*In[kc * W * H + (ih + 0) * W + iw + 2];
                    R +=Filter[fn * C * KW * KH + kc * KH * KW + 1 * KW + 0]*In[kc * W * H + (ih + 1) * W + iw + 0];
                    R +=Filter[fn * C * KW * KH + kc * KH * KW + 1 * KW + 1]*In[kc * W * H + (ih + 1) * W + iw + 1];
                    R +=Filter[fn * C * KW * KH + kc * KH * KW + 1 * KW + 2]*In[kc * W * H + (ih + 1) * W + iw + 2];
                    R +=Filter[fn * C * KW * KH + kc * KH * KW + 2 * KW + 0]*In[kc * W * H + (ih + 2) * W + iw + 0];
                    R +=Filter[fn * C * KW * KH + kc * KH * KW + 2 * KW + 1]*In[kc * W * H + (ih + 2) * W + iw + 1];
                    R +=Filter[fn * C * KW * KH + kc * KH * KW + 2 * KW + 2]*In[kc * W * H + (ih + 2) * W + iw + 2];
                }
                Out[fn * Wo * Ho + ih * Wo + iw] = R;
            }
        }
    }
}


// Parallel 3X3 Conv
void __attribute__((noinline)) MyP3X3Convolution(float *__restrict__ In, int H, int W, int C, int Filternum,
        float *__restrict__ Filter, float *__restrict__ Out)
{
    int Wo=W-2;
    int Ho=H-2;
    unsigned int CoreId = gap8_coreid();
    unsigned int Chunk = ChunkSize(Wo);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = Min(First + Chunk, Wo);
    int Wo_F = First; int Wo_L = Last;
    int Ho_F = 0; int Ho_L = Ho;
    int KW=3, KH=3;

    // direct conv
    for (int iw = Wo_F; iw < Wo_L; iw++) {
        for (int ih = Ho_F; ih < Ho_L; ih++) {
            for (int fn = 0; fn < Filternum; fn++) {
                int R = Out[fn * Wo * Ho + ih * Wo + iw] = 0;
                // Use C, H, W format for data locality
                for (int kc = 0; kc < C; kc++) {
                    // This is the H, W, C format
                    //R += Filter[(0 * KW + 0)*C + kc] * In[((ih+0) * W + iw+0)*C + kc];
                    // We use the C, H, W format
                    R +=Filter[fn * C * KW * KH + kc * KH * KW + 0 * KW + 0]*In[kc * W * H + (ih + 0) * W + iw + 0];
                    R +=Filter[fn * C * KW * KH + kc * KH * KW + 0 * KW + 1]*In[kc * W * H + (ih + 0) * W + iw + 1];
                    R +=Filter[fn * C * KW * KH + kc * KH * KW + 0 * KW + 2]*In[kc * W * H + (ih + 0) * W + iw + 2];
                    R +=Filter[fn * C * KW * KH + kc * KH * KW + 1 * KW + 0]*In[kc * W * H + (ih + 1) * W + iw + 0];
                    R +=Filter[fn * C * KW * KH + kc * KH * KW + 1 * KW + 1]*In[kc * W * H + (ih + 1) * W + iw + 1];
                    R +=Filter[fn * C * KW * KH + kc * KH * KW + 1 * KW + 2]*In[kc * W * H + (ih + 1) * W + iw + 2];
                    R +=Filter[fn * C * KW * KH + kc * KH * KW + 2 * KW + 0]*In[kc * W * H + (ih + 2) * W + iw + 0];
                    R +=Filter[fn * C * KW * KH + kc * KH * KW + 2 * KW + 1]*In[kc * W * H + (ih + 2) * W + iw + 1];
                    R +=Filter[fn * C * KW * KH + kc * KH * KW + 2 * KW + 2]*In[kc * W * H + (ih + 2) * W + iw + 2];
                }
                Out[fn * Wo * Ho + ih * Wo + iw] = R;
            }
        }
    }
}


// XNOR Conv 3X3
void __attribute__((noinline)) XnorConv3X3(float *__restrict__ Input,  int H, int W, int C, int Filternum,
        unsigned int *__restrict__ Filter, float *__restrict__ Ffactor, float *__restrict__ Out)
{
    /* input matrix: Input
     * input size: H,W,C
     * output matrix: Out
     * filter padded: Filter
     * filter scaling factor: Ffactor
     * filter number: Filternum
     * */

    // The kernel size
    int KW=3;
    int KH=3;
    // the output size
    int Wo=W-2;
    int Ho=H-2;

    // This is the padded input matrix which is the sign(X)
    unsigned int PaddedInput[H*W*C/32];
    // This is the scaling factor matrix K
    float Inputsum[H*W];
    float Infactor[Ho*Wo];
    // the padded channel num
    int paddedC=C/32;
    // the number of bits that do a popc(XOR)
    int xornum=9 * paddedC * 32;


    // Calculate the scaling factor for the input.
    for (int iw = 0; iw < W; iw++) {
        for (int ih = 0; ih < H; ih++) {
            float sum=0;
            for (int ic = 0; ic < C; ic++){
                sum=sum+Input[(ih * W + iw)*C + ic];
            }
            // for the first layer with optimized algorithm
            Inputsum[ih * W + iw]=sum;
        }
    }
    for (int iw = 0; iw < Wo; iw++) {
        for (int ih = 0; ih < Ho; ih++) {
            float sum=0;
            for (int kw = 0; kw < KW; kw++){
                for (int kh = 0; kh < KH; kh++){
                    sum=sum+Inputsum[(ih + kh) * W + iw + kw];
                }
            }
            Infactor[ih * Wo + iw]=sum/KW/KH;
        }
    }


    // Pad the input.
    for (int iw = 0; iw < W; iw++) {
        for (int ih = 0; ih < H; ih++) {
            unsigned int pad;
            for (int ic = 0; ic < C; ic++){
                if ((ic%32)==0){
                    PaddedInput[(ih * W + iw)*paddedC+ic/32-1]=pad;
                    pad=0;
                }
                if (Input[(ih * W + iw)*C + ic] <0) {
                    // pad=pad + ((32-ic%32)<<1);
                    B_ins_r(pad, 0b1, 1, ic%32);
                }
            }
            PaddedInput[(ih * W + iw)*paddedC+paddedC-1]=pad;
        }
    }

    // XNOR-Net Conv
    for (int iw = 0; iw < Wo; iw++) {
        for (int ih = 0; ih < Ho; ih++) {
            for (int fn = 0; fn < Filternum; fn++) {
                // Use H, W, C format for data locality
                int R = 0;
                for (int kc = 0; kc < paddedC; kc++) {
                    // This is the H, W, C format
                    //R += Filter[(0 * KW + 0)*C + kc] * In[((ih+0) * W + iw+0)*C + kc];
                    // XOR, '1' shands for -1
                    R +=B_popc(Filter[fn * 9 * paddedC+ (0 * KW + 0) * paddedC + kc]^PaddedInput[((ih + 0) * W + iw + 0 ) * paddedC + kc]);
                    R +=B_popc(Filter[fn * 9 * paddedC+ (0 * KW + 1) * paddedC + kc]^PaddedInput[((ih + 0) * W + iw + 1 ) * paddedC + kc]);
                    R +=B_popc(Filter[fn * 9 * paddedC+ (0 * KW + 2) * paddedC + kc]^PaddedInput[((ih + 0) * W + iw + 2 ) * paddedC + kc]);
                    R +=B_popc(Filter[fn * 9 * paddedC+ (1 * KW + 0) * paddedC + kc]^PaddedInput[((ih + 1) * W + iw + 0 ) * paddedC + kc]);
                    R +=B_popc(Filter[fn * 9 * paddedC+ (1 * KW + 1) * paddedC + kc]^PaddedInput[((ih + 1) * W + iw + 1 ) * paddedC + kc]);
                    R +=B_popc(Filter[fn * 9 * paddedC+ (1 * KW + 2) * paddedC + kc]^PaddedInput[((ih + 1) * W + iw + 2 ) * paddedC + kc]);
                    R +=B_popc(Filter[fn * 9 * paddedC+ (2 * KW + 0) * paddedC + kc]^PaddedInput[((ih + 2) * W + iw + 0 ) * paddedC + kc]);
                    R +=B_popc(Filter[fn * 9 * paddedC+ (2 * KW + 1) * paddedC + kc]^PaddedInput[((ih + 2) * W + iw + 1 ) * paddedC + kc]);
                    R +=B_popc(Filter[fn * 9 * paddedC+ (2 * KW + 2) * paddedC + kc]^PaddedInput[((ih + 2) * W + iw + 2 ) * paddedC + kc]);
                }
                R = xornum - 2 * R;
                // get a complete output point
                Out[fn * Wo * Ho + ih * Wo + iw] = R * Ffactor[fn] * Infactor[ih * Wo + iw];
            }
        }
    }
}


// Optimized XNOR Conv 3X3
void __attribute__((noinline)) MyXnorConv3X3(float *__restrict__ Input, int H, int W, int C, int Filternum, int layer,
        unsigned int *__restrict__ Filter, float *__restrict__ Ffactor, float *__restrict__ Out, float * Kmatrix)
{
    /* input matrix: Input
     * input size: H,W,C
     * output matrix: Out
     * filter padded: Filter
     * filter scaling factor: Ffactor
     * filter number: Filternum
     * the layer type: layer: 0: only produce K, 1: *K and produce K, 2: *K and get final output
     * */

    // The kernel size
    int KW=3;
    int KH=3;
    // the output size
    int Wo=W-2;
    int Ho=H-2;

    // This is the padded input matrix which is the sign(X)
    unsigned int PaddedInput[H*W*C/32];
    // This is the scaling factor matrix K
    float Inputsum[H*W];
    float Infactor[Ho*Wo];
    // the padded channel num
    int paddedC=C/32;
    // the number of bits that do a popc(XOR)
    int xornum=9 * paddedC * 32;


    // Calculate the average across the channel for the input.
    if (layer==0){
        for (int iw = 0; iw < W; iw++) {
            for (int ih = 0; ih < H; ih++) {
                float sum=0;
                for (int ic = 0; ic < C; ic++){
                    sum=sum+Input[(ih * W + iw)*C + ic];
                }
                // for the first layer with optimized algorithm
                    Inputsum[ih * W + iw]=sum;
            }
        }
    }
    else{
        for (int iw = 0; iw < W; iw++) {
            for (int ih = 0; ih < H; ih++) {
                float sum=0;
                for (int ic = 0; ic < C; ic++){
                    sum=sum+Input[(ih * W + iw)*C + ic];
                }
                // for the continued layer with optimized algorithm
                    Inputsum[ih * W + iw]=sum*Kmatrix[ih * W + iw];
            }
        }
    }

    // calculate the input scaling factor
    for (int iw = 0; iw < Wo; iw++) {
        for (int ih = 0; ih < Ho; ih++) {
            float sum=0;
            for (int kw = 0; kw < KW; kw++){
                for (int kh = 0; kh < KH; kh++){
                    sum=sum+Inputsum[(ih + kh) * W + iw + kw];
                }
            }
            Kmatrix[ih * Wo + iw]=Infactor[ih * Wo + iw]=sum;
        }
    }

    // Pad the input.
    for (int iw = 0; iw < W; iw++) {
        for (int ih = 0; ih < H; ih++) {
            unsigned int pad=0;
            for (int ic = 0; ic < C; ic++){
                if ((ic>0) && ((ic%32)==0)){
                    PaddedInput[(ih * W + iw)*paddedC+ic/32-1]=pad;
                    pad=0;
                }
                if (Input[(ih * W + iw)*C + ic] <0) {
                    // pad=pad + ((32-ic%32)<<1);
                    B_ins_r(pad, 0b1, 1, ic%32);
                }
            }
            PaddedInput[(ih * W + iw)*paddedC+paddedC-1]=pad;
        }
    }

    // XNOR-Net Conv
    if (layer==2){
        for (int iw = 0; iw < Wo; iw++) {
            for (int ih = 0; ih < Ho; ih++) {
                for (int fn = 0; fn < Filternum; fn++) {
                    // Use H, W, C format for data locality
                    int R = 0;
                    for (int kc = 0; kc < paddedC; kc++) {
                        // This is the H, W, C format
                        //R += Filter[(0 * KW + 0)*C + kc] * In[((ih+0) * W + iw+0)*C + kc];
                        // XOR, '1' stands for -1
                        R +=B_popc(Filter[fn * 9 * paddedC+ (0 * KW + 0) * paddedC + kc]^PaddedInput[((ih + 0) * W + iw + 0 ) * paddedC + kc]);
                        R +=B_popc(Filter[fn * 9 * paddedC+ (0 * KW + 1) * paddedC + kc]^PaddedInput[((ih + 0) * W + iw + 1 ) * paddedC + kc]);
                        R +=B_popc(Filter[fn * 9 * paddedC+ (0 * KW + 2) * paddedC + kc]^PaddedInput[((ih + 0) * W + iw + 2 ) * paddedC + kc]);
                        R +=B_popc(Filter[fn * 9 * paddedC+ (1 * KW + 0) * paddedC + kc]^PaddedInput[((ih + 1) * W + iw + 0 ) * paddedC + kc]);
                        R +=B_popc(Filter[fn * 9 * paddedC+ (1 * KW + 1) * paddedC + kc]^PaddedInput[((ih + 1) * W + iw + 1 ) * paddedC + kc]);
                        R +=B_popc(Filter[fn * 9 * paddedC+ (1 * KW + 2) * paddedC + kc]^PaddedInput[((ih + 1) * W + iw + 2 ) * paddedC + kc]);
                        R +=B_popc(Filter[fn * 9 * paddedC+ (2 * KW + 0) * paddedC + kc]^PaddedInput[((ih + 2) * W + iw + 0 ) * paddedC + kc]);
                        R +=B_popc(Filter[fn * 9 * paddedC+ (2 * KW + 1) * paddedC + kc]^PaddedInput[((ih + 2) * W + iw + 1 ) * paddedC + kc]);
                        R +=B_popc(Filter[fn * 9 * paddedC+ (2 * KW + 2) * paddedC + kc]^PaddedInput[((ih + 2) * W + iw + 2 ) * paddedC + kc]);
                    }
                    R = xornum - 2 * R;
                    // get a complete output point for the final layer
                    Out[fn * Wo * Ho + ih * Wo + iw] = R * Ffactor[fn] * Infactor[ih * Wo + iw];
                }
            }
        }
    }
    else{
        for (int iw = 0; iw < Wo; iw++) {
            for (int ih = 0; ih < Ho; ih++) {
                for (int fn = 0; fn < Filternum; fn++) {
                    // Use H, W, C format for data locality
                    int R = 0;
                    for (int kc = 0; kc < paddedC; kc++) {
                        // This is the H, W, C format
                        //R += Filter[(0 * KW + 0)*C + kc] * In[((ih+0) * W + iw+0)*C + kc];
                        // XOR, '1' stands for -1
                        R +=B_popc(Filter[fn * 9 * paddedC+ (0 * KW + 0) * paddedC + kc]^PaddedInput[((ih + 0) * W + iw + 0 ) * paddedC + kc]);
                        R +=B_popc(Filter[fn * 9 * paddedC+ (0 * KW + 1) * paddedC + kc]^PaddedInput[((ih + 0) * W + iw + 1 ) * paddedC + kc]);
                        R +=B_popc(Filter[fn * 9 * paddedC+ (0 * KW + 2) * paddedC + kc]^PaddedInput[((ih + 0) * W + iw + 2 ) * paddedC + kc]);
                        R +=B_popc(Filter[fn * 9 * paddedC+ (1 * KW + 0) * paddedC + kc]^PaddedInput[((ih + 1) * W + iw + 0 ) * paddedC + kc]);
                        R +=B_popc(Filter[fn * 9 * paddedC+ (1 * KW + 1) * paddedC + kc]^PaddedInput[((ih + 1) * W + iw + 1 ) * paddedC + kc]);
                        R +=B_popc(Filter[fn * 9 * paddedC+ (1 * KW + 2) * paddedC + kc]^PaddedInput[((ih + 1) * W + iw + 2 ) * paddedC + kc]);
                        R +=B_popc(Filter[fn * 9 * paddedC+ (2 * KW + 0) * paddedC + kc]^PaddedInput[((ih + 2) * W + iw + 0 ) * paddedC + kc]);
                        R +=B_popc(Filter[fn * 9 * paddedC+ (2 * KW + 1) * paddedC + kc]^PaddedInput[((ih + 2) * W + iw + 1 ) * paddedC + kc]);
                        R +=B_popc(Filter[fn * 9 * paddedC+ (2 * KW + 2) * paddedC + kc]^PaddedInput[((ih + 2) * W + iw + 2 ) * paddedC + kc]);
                    }
                    R = xornum - 2 * R;
                    // get a complete output point for the first or a continuous layer
                    Out[fn * Wo * Ho + ih * Wo + iw] = R * Ffactor[fn];
                }
            }
        }
    }
}


void __attribute__ ((noinline)) Additive5x5Convolution(Ty *__restrict__ In, int W, int H, Ty *__restrict__ Filter, Ty *__restrict__ Out, int Norm)

{
    int FH=5,FW=5;
    for (int c=0; c<(W-4); c++) {
        for (int l=0; l<(H-4); l++) {
            int R = Out[l*W+c]<<Norm;
            for (int kl=0; kl<FH; kl++) {
                R += Filter[kl*FW+0]*In[(l+kl)*W + c+0];
                R += Filter[kl*FW+1]*In[(l+kl)*W + c+1];
                R += Filter[kl*FW+2]*In[(l+kl)*W + c+2];
                R += Filter[kl*FW+3]*In[(l+kl)*W + c+3];
                R += Filter[kl*FW+4]*In[(l+kl)*W + c+4];
/*
                                for (int kc=0; kc<FW; kc++) {
                                        R += Filter[kl*FW+kc]*In[(l+kl)*W + c+kc];
                                }
*/
            }
            Out[l*W+c] = R>>Norm;
        }
    }
}


// XNOR Conv 5X5
void __attribute__((noinline)) XnorConv5x5(
	unsigned int InBit,
	signed char *__restrict__ Out,
	unsigned int FilterBit,
	int W,
	int H
)

{
	int Wo = W - 4; int Ho = H - 4;
	int Wo_F = 0; int Wo_L = Wo;
	int Ho_F = 0; int Ho_L = Ho;

	unsigned int Stride = 1, K = 2;
	unsigned int C = *((unsigned int *)(FilterBit / 8)) >> (FilterBit % 8);
	signed char *PtO1 = Out + Wo*Ho_F + Wo_F;
	unsigned char *PtByte;
	unsigned int PtBit;
	unsigned int ExtMask = 5 << 5;
	unsigned int CC = C;
	CC = B_ins(CC, B_ext(C, 5, 5), 5, 6);
	CC = B_ins(CC, B_ext(C, 5, 10), 5, 12);
	CC = B_ins(CC, B_ext(C, 5, 15), 5, 18);
	CC = B_ins(CC, B_ext(C, 5, 20), 5, 24);

	int Iter = Wo_L - Wo_F;
	for (int i = 0, w = Wo_F; (i < (Iter / 2)); i++, w += 2) {
		unsigned int V, N;
		PtBit = InBit + (Ho_F*Stride)*W + (w*Stride);
		PtByte = (unsigned char *)(PtBit / 8);
		char *PtO = (char *)PtO1;
		V = 0;
		N = B_extu_r(*(unsigned short int *)PtByte, 6, (PtBit % 8));
		V = B_ins(V, N, 6, 0); PtBit += W; PtByte = (unsigned char *)(PtBit / 8);
		N = B_extu_r(*(unsigned short int *)PtByte, 6, (PtBit % 8));
		V = B_ins(V, N, 6, 6); PtBit += W; PtByte = (unsigned char *)(PtBit / 8);
		N = B_extu_r(*(unsigned short int *)PtByte, 6, (PtBit % 8));
		V = B_ins(V, N, 6, 12); PtBit += W; PtByte = (unsigned char *)(PtBit / 8);
		N = B_extu_r(*(unsigned short int *)PtByte, 6, (PtBit % 8));
		V = B_ins(V, N, 6, 18); PtBit += W; PtByte = (unsigned char *)(PtBit / 8);

		for (int h = Ho_F; h < Ho_L; h++) {
			N = B_extu_r(*(unsigned short int *)PtByte, 6, (PtBit % 8));
			V = B_ins(V, N, 6, 24); PtBit += W; PtByte = (unsigned char *)(PtBit / 8);
			int R0 = B_popc((~(V^CC)) & 0x1F7DF7DF),
				R1 = B_popc((~((V >> 1) ^ CC)) & 0x1F7DF7DF);
#ifdef RISCV
			unsigned int Val = (*(unsigned short int *) PtO);
			R0 += Val & 0xFF; R1 += (Val >> 8); Val = R0 | (R1 << 8);
#else
			unsigned int V1 = R0 | (R1 << 8);
			v4s Val = (v4s)(unsigned int)(*(unsigned short int *) PtO);
			Val = __builtin_pulp_add4(Val, (v4s)V1);
#endif
			*(unsigned short int *)PtO = (unsigned short int)(unsigned int)Val; PtO += Wo;
			V = V >> ((2 * K + 1) + 1);
		}
		PtO1 += 2;
	}
	PtO1 = Out + Wo*Ho_F + Wo_F + 2 * (Iter / 2);
	for (int w = Wo_F + 2 * (Iter / 2); w < Wo_L; w++) {
		unsigned int V, N;
		PtBit = InBit + (Ho_F*Stride)*W + (w*Stride);
		PtByte = (unsigned char *)(PtBit / 8);
		char *PtO = (char *)PtO1;
		V = 0;
		N = B_extu_r(*(unsigned short int *)PtByte, 5, (PtBit % 8));
		V = B_ins(V, N, 5, 0); PtBit += W; PtByte = (unsigned char *)(PtBit / 8);
		N = B_extu_r(*(unsigned short int *)PtByte, 5, (PtBit % 8));
		V = B_ins(V, N, 5, 5); PtBit += W; PtByte = (unsigned char *)(PtBit / 8);
		N = B_extu_r(*(unsigned short int *)PtByte, 5, (PtBit % 8));
		V = B_ins(V, N, 5, 10); PtBit += W; PtByte = (unsigned char *)(PtBit / 8);
		N = B_extu_r(*(unsigned short int *)PtByte, 5, (PtBit % 8));
		V = B_ins(V, N, 5, 15); PtBit += W; PtByte = (unsigned char *)(PtBit / 8);
		for (int h = Ho_F; h < Ho_L; h++) {
			N = B_extu_r(*(unsigned short int *)PtByte, 5, (PtBit % 8));
			V = B_ins(V, N, 5, 20); PtBit += W; PtByte = (unsigned char *)(PtBit / 8);
			*PtO += B_popc((~(V^C)) & 0x1FFFFFF); PtO += Wo;
			V = V >> (2 * K + 1);
		}
		PtO1++;
	}
}

void RunTest(int Which, int Iter, int Trace, char *Mode, int * num_ops)

{
	unsigned int Ti;
	ArgConvT Arg;
	ArgConvBT Arg1;
	Ty *IN, *OUT, *FILTER;
	float *In, *Out, *Filter, *Ffactor, *K;
	unsigned int *XNORFilter;
	int H=56, W=56, C=32, KW=3,KH=3,Filternum=32,layer=0;
  	switch (Which) {
	case 0:
		IN = Mem; OUT = Mem + Wic*Hic; FILTER = Mem + Wic*Hic + (Wic - 4)*(Hic - 4); CheckMem(Wic*Hic + (Wic - 4)*(Hic - 4) + 5 * 5);
		gap8_resethwtimer();
		WriteGpio(GPIO, 1);
		Ti = gap8_readhwtimer();
		for (int i = 0; i < Iter; i++) Additive5x5Convolution(IN, Wic, Hic, FILTER, OUT, 6);
		Ti = gap8_readhwtimer() - Ti;
		WriteGpio(GPIO, 0);
		*num_ops = Ti;
		if (Trace) printf("[%2d][%s] %7d %35s: %8d cycles\n", Which, Mode, (Wic - 4)*(Hic - 4)*Iter, "5x5 Convolutions", Ti);
		break;
	case 1:
	{
		unsigned int In = (unsigned int)Mem;
		signed char *Out = (signed char *)Mem + ((Wxor*Hxor + 7) / 8);
		unsigned int Filter = (unsigned int)Mem + ((Wxor*Hxor + 7) / 8) * 8 + (Wxor - 4)*(Hxor - 4) * 8;
		gap8_resethwtimer();
		WriteGpio(GPIO, 1);
		Ti = gap8_readhwtimer();
		for (int i = 0; i < Iter; i++) XnorConv5x5(In, Out, Filter, Wxor, Hxor);
		Ti = gap8_readhwtimer() - Ti;
		WriteGpio(GPIO, 0);
		*num_ops = Ti;
	}
	if (Trace) printf("[%2d][%s] %7d %35s: %8d cycles\n", Which, Mode, (Wxor - 4)*(Hxor - 4)*Iter, "Xnor Conv 5x5", Ti);
	break;
#ifndef RISCV
	case 2:
		IN = Mem; OUT = Mem + Wic*Hic; FILTER = Mem + Wic*Hic + (Wic - 4)*(Hic - 4); CheckMem(Wic*Hic + (Wic - 4)*(Hic - 4) + 5 * 5);
		Arg.In = IN; Arg.W = Wic; Arg.H = Hic; Arg.Filter = FILTER; Arg.Out = OUT; Arg.Norm = 6;
		gap8_resethwtimer();
		WriteGpio(GPIO, 1);
		Ti = gap8_readhwtimer();
		for (int i = 0; i < Iter; i++) rt_team_fork(gap8_ncore(), (void *)ParAdditive5x5Convolution, (void *)&Arg);
		Ti = gap8_readhwtimer() - Ti;
		WriteGpio(GPIO, 0);
		*num_ops = Ti;
		if (Trace) printf("[%2d][%s] %7d %35s: %8d cycles, %1d Cores\n", Which, Mode, (Wic - 4)*(Hic - 4)*Iter, "Parallel 5x5 Convolution", Ti, gap8_ncore());
		break;
	case 3:
	{
		unsigned int In = (unsigned int)Mem;
		signed char *Out = (signed char *)Mem + ((Wxor*Hxor + 7) / 8);
		unsigned int Filter = (unsigned int)Mem + ((Wxor*Hxor + 7) / 8) * 8 + (Wxor - 4)*(Hxor - 4) * 8;
		Arg1.InBit = In; Arg1.Out = Out; Arg1.FilterBit = Filter; Arg1.W = Wxor; Arg1.H = Hxor;
		gap8_resethwtimer();
		WriteGpio(GPIO, 1);
		Ti = gap8_readhwtimer();
		for (int i = 0; i < Iter; i++) rt_team_fork(gap8_ncore(), (void *)ParXnorConv5x5, (void *)&Arg1);
		Ti = gap8_readhwtimer() - Ti;
		WriteGpio(GPIO, 0);
		*num_ops = Ti;
	}
	if (Trace) printf("[%2d][%s] %7d %35s: %8d cycles, %1d Cores\n", Which, Mode, (Wxor - 4)*(Hxor - 4)*Iter, "Parallel Xnor Conv 5x5", Ti, gap8_ncore());
	break;
#endif
        case 4:
            In = Mem; Out = Mem+H*W*C; Filter=Mem+H*W*C+(H-2)*(W-2)*Filternum; CheckMem(H*W*C+(H-2)*(W-2)*Filternum+KW*KH*C*Filternum);
            gap8_resethwtimer();
            WriteGpio(GPIO, 1);
            Ti = gap8_readhwtimer();
            for (int i=0; i<Iter; i++) My3X3Convolution(In, H, W, C, Filternum, Filter, Out);
            Ti = gap8_readhwtimer() - Ti;
            WriteGpio(GPIO, 0);
            *num_ops = Ti;
            if (Trace) printf("[%2d][%s] %7d %35s: %8d cycles\n", Which, Mode, (H)*(W)*Iter, "My3X3Convolution", Ti);
            break;
        case 5:
            In = Mem; Out = Mem+H*W*C;
            XNORFilter=Mem+H*W*C+(H-2)*(W-2)*Filternum;Ffactor=Mem+H*W*C+(H-2)*(W-2)*Filternum+KH*KW*C*Filternum;
            CheckMem(Wi*Hi+(Wi/2)*(Hi/2));
            gap8_resethwtimer();
            WriteGpio(GPIO, 1);
            Ti = gap8_readhwtimer();
            for (int i=0; i<Iter; i++) XnorConv3X3(In, H, W, C, Filternum, XNORFilter, Ffactor, Out);
            Ti = gap8_readhwtimer() - Ti;
            WriteGpio(GPIO, 0);
            *num_ops = Ti;
            if (Trace) printf("[%2d][%s] %7d %35s: %8d cycles\n", Which, Mode, (Wi/2)*(Hi/2)*Iter, "XnorConv3X3", Ti);
            break;
        case 6:
            In = Mem; Out = Mem+H*W*C; Filter=Mem+H*W*C+(H-2)*(W-2)*Filternum; CheckMem(H*W*C+(H-2)*(W-2)*Filternum+KW*KH*C*Filternum);
            gap8_resethwtimer();
            WriteGpio(GPIO, 1);
            Ti = gap8_readhwtimer();
            for (int i=0; i<Iter; i++) MyXnorConv3X3(In, H, W, C, Filternum, layer, XNORFilter, Ffactor, Out, K);
            Ti = gap8_readhwtimer() - Ti;
            WriteGpio(GPIO, 0);
            *num_ops = Ti;
            if (Trace) printf("[%2d][%s] %7d %35s: %8d cycles\n", Which, Mode, (H)*(W)*Iter, "MyXnorConv3X3", Ti);
            break;
	}
}

int benchmarks(ClusterArg_t * ArgC) {

	int   test_num = ArgC->test_num;
	char* Mode = ArgC->Mode;
	int   Trace = ArgC->Trace;
	int   NumIter = ArgC->Iter;

	RunTest(test_num, NumIter, Trace, Mode, &(ArgC->Iter_operations));

	return 0;
}


int main()
{
	long start_time, end_time;
	long int tot_time, op_num, kernel_op_num;
	float res, res_kernel;
	int cur_test = 0;

#if !ALIM_1_VOLT
	PMU_set_voltage(1150, 0);
	PMU_set_voltage(1200, 0);
#else
	PMU_set_voltage(1000, 0);
#endif

#ifndef NOGPIO
	rt_padframe_profile_t *profile_gpio = rt_pad_profile_get("hyper_gpio");

	if (profile_gpio == NULL) {
		printf("pad config error\n");
		return 1;
	}
	rt_padframe_set(profile_gpio);
	// GPIO initialization
	rt_gpio_init(0, GPIO);
	rt_gpio_set_dir(0, 1 << GPIO, RT_GPIO_IS_OUT);

#endif

	printf("\n\n");
	printf("                      --------------------------------------------------------\n");
	printf("                      --------------------------------------------------------\n");
	printf("                      ---------------   GAP8 benchmarks   --------------------\n");
	printf("                      --------------------------------------------------------\n");
	printf("                      --------------------------------------------------------\n\n\n");


	printf("Gap8 Input Voltage    : %s\n", ALIM_1_VOLT ? "1.0 Volt" : "1.2 Volts");
	printf("Fabric Controller Freq: %d MhZ\n", FREQ_FC / 1000000);
	printf("Cluster  Freq         : %d MhZ\n\n\n", FREQ_CL / 1000000);

	printf("Number of iterations for each benchmark: %d\n\n\n", ITERATIONS);


	if (rt_event_alloc(NULL, 8)) return -1;

	rt_cluster_mount(MOUNT, CID, 0, NULL);

	//Set Fabric Controller and Cluster Frequencies
	rt_freq_set(RT_FREQ_DOMAIN_FC, FREQ_FC);
	rt_freq_set(RT_FREQ_DOMAIN_CL, FREQ_CL);

	for (int j = 0; j < TOT_TEST; j++) {
		printf("\n                      ---------------   %15s   ---------------\n", tests_titles[j]);
		for (int i = 0; i < test_num[j]; i++) {

			Arg.test_num = cur_test++;
			Arg.Iter = ITERATIONS;
			Arg.Trace = ENABLE_CYCLE_TRACE;
#ifdef BYTE
			strcpy(Arg.Mode, "Byte");
#else
			strcpy(Arg.Mode, "Short");
#endif

			start_time = rt_time_get_us();
			rt_cluster_call(NULL, CID, (void *)benchmarks, &Arg, NULL, 0, 0, 8, NULL);
			end_time = rt_time_get_us();

			tot_time = end_time - start_time;
			op_num = Arg.Iter_operations;

			printf("%30s Input: %dx%d (x%d iterations) Time: %10ld uSec. Cycles: %10ld\n", tests_names[i], test_input_w[i], test_input_h[i], ITERATIONS, tot_time, op_num);

		}
	}


	rt_cluster_mount(UNMOUNT, CID, 0, NULL);

	return 0;
}