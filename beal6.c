#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include <time.h>

// gcc beal6.c -o beal6.bin -O3 -Wall -march=native -fopenmp -lOpenCL

#ifndef alignof 
  #define alignof _Alignof
#endif
#define U128 unsigned __int128
#define MAXN 112
#define MINN 8

typedef union {
  U128 num;
  uint64_t hilo[2]; // 0 lo, 1 hi
} U128_2xU64_t;

U128 gcd(U128 a, U128 b) {
  U128 temp;
  if (a == 0) return 0;
  if (b == 0) return 0;
  while (1) {
    if (b > a) {
      temp = a;
      a = b;
      b = temp;
    }
    // a >= b
    a = (a % b);
    if (a == 0) return b;
    if (a == 1) return 1;
  }
}
//============================================================================================================
// OpenCL code:
//============================================================================================================

#define NAMES_LENGTH 255
#define ARGSSIZE 2
#define MAXRETURNEDIXSPERWKITEM 4
#define MAXZIXSKIPS 3
#define YIXOFFSETMAX 10000 // Must be even
#define CL_TARGET_OPENCL_VERSION 120
#define DEFAULT_LOCAL_DIM 192
#define DEFAULTWKITEMS 100000
#define KERNEL_COUNT 1
#define ulong uint64_t
#define uint uint32_t
#define bool _Bool
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
typedef struct {
  char DeviceName[NAMES_LENGTH]; 
  char PlatformName[NAMES_LENGTH]; 
  cl_kernel kernel;
  size_t kernelcount;
  size_t LocalWkgrpsize[3];
  size_t GlobalWkgrpsize[3];
  cl_platform_id PlatformId;
  cl_device_id DeviceId;
  cl_uint DeviceMaxCUs;
  cl_context ContextId;
  cl_command_queue CommandQueueId;
  cl_program ProgramId;
  cl_uint MaxWkgrpdims;
  cl_ulong MaxAlloc;
  cl_ulong MaxGlobalMem;
} GPUinfo_t;


void printf_cl_error(cl_int res) {
  if (res == CL_INVALID_MEM_OBJECT) printf("CL_INVALID_MEM_OBJECT\n");
  if (res == CL_INVALID_SAMPLER) printf("CL_INVALID_SAMPLER\n");
  if (res == CL_INVALID_KERNEL) printf("CL_INVALID_KERNEL\n");
  if (res == CL_INVALID_ARG_INDEX) printf("CL_INVALID_ARG_INDEX\n");
  if (res == CL_INVALID_ARG_VALUE) printf("CL_INVALID_ARG_VALUE\n");
  if (res == CL_INVALID_ARG_SIZE) printf("CL_INVALID_ARG_SIZE\n");
  if (res == CL_INVALID_COMMAND_QUEUE) printf("CL_INVALID_COMMAND_QUEUE\n");
  if (res == CL_INVALID_CONTEXT) printf("CL_INVALID_CONTEXT\n");
  if (res == CL_INVALID_MEM_OBJECT) printf("CL_INVALID_MEM_OBJECT\n");
  if (res == CL_INVALID_VALUE) printf("CL_INVALID_VALUE\n");
  if (res == CL_INVALID_EVENT_WAIT_LIST) printf("CL_INVALID_EVENT_WAIT_LIST\n");
  if (res == CL_MEM_OBJECT_ALLOCATION_FAILURE) printf("CL_MEM_OBJECT_ALLOCATION_FAILURE\n");
  if (res == CL_OUT_OF_HOST_MEMORY) printf("CL_OUT_OF_HOST_MEMORY\n");
  if (res == CL_INVALID_PROGRAM_EXECUTABLE) printf("CL_INVALID_PROGRAM_EXECUTABLE\n");
  if (res == CL_INVALID_KERNEL_ARGS) printf("CL_INVALID_KERNEL_ARGS\n");
  if (res == CL_INVALID_WORK_DIMENSION) printf("CL_INVALID_WORK_DIMENSION\n");
  if (res == CL_INVALID_GLOBAL_WORK_SIZE) printf("CL_INVALID_GLOBAL_WORK_SIZE\n");
  if (res == CL_INVALID_WORK_GROUP_SIZE) printf("CL_INVALID_WORK_GROUP_SIZE\n");
  if (res == CL_INVALID_WORK_ITEM_SIZE) printf("CL_INVALID_WORK_ITEM_SIZE\n");
  if (res == CL_INVALID_GLOBAL_OFFSET) printf("CL_INVALID_GLOBAL_OFFSET\n");
  if (res == CL_OUT_OF_RESOURCES) printf("CL_OUT_OF_RESOURCES\n");
  if (res == CL_INVALID_OPERATION) printf("CL_INVALID_OPERATION\n");
  if (res == CL_BUILD_PROGRAM_FAILURE) printf("CL_BUILD_PROGRAM_FAILURE\n");
  if (res == CL_COMPILER_NOT_AVAILABLE) printf("CL_COMPILER_NOT_AVAILABLE\n");
  if (res == CL_INVALID_BUILD_OPTIONS) printf("CL_INVALID_BUILD_OPTIONS\n");
  if (res == CL_INVALID_BINARY) printf("CL_INVALID_BUILD_OPTIONS\n");
  if (res == CL_INVALID_DEVICE) printf("CL_INVALID_DEVICE\n");
}
 
_Bool oclinit(GPUinfo_t *GPUinfo) {
  //Choose OpenCL device with the most CUs.  
  cl_uint platformCount = 0;
  cl_uint deviceCount = 0;
  cl_uint DeviceMaxCUs = 0;
  cl_uint MaxWkgrpdims = 0;
  GPUinfo->DeviceMaxCUs = 0;
  printf_cl_error(clGetPlatformIDs(0, NULL, &platformCount));
  printf("Detected %i OpenCL Platforms.\n", platformCount);
  if (platformCount == 0) return false;
  cl_platform_id platforms[platformCount];
  printf_cl_error(clGetPlatformIDs(platformCount, platforms, NULL));
  for (int i=0; i<platformCount; i++) {
    printf_cl_error(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount));
    printf("Detected %i Devices In Platform.\n", deviceCount);
    if (deviceCount) {
      cl_device_id devices[deviceCount];
      printf_cl_error(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL));
      for (int j=0; j<deviceCount; j++) {
        printf_cl_error(clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(DeviceMaxCUs), &DeviceMaxCUs, NULL));
        printf_cl_error(clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(MaxWkgrpdims), &MaxWkgrpdims, NULL));
        if ((DeviceMaxCUs > GPUinfo->DeviceMaxCUs) && (MaxWkgrpdims == 3)) {      
          GPUinfo->MaxWkgrpdims = MaxWkgrpdims;   
          GPUinfo->DeviceMaxCUs = DeviceMaxCUs;
          printf_cl_error(clGetDeviceInfo(devices[j], CL_DEVICE_NAME, NAMES_LENGTH, GPUinfo->DeviceName, NULL));
          printf_cl_error(clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(GPUinfo->MaxGlobalMem), &GPUinfo->MaxGlobalMem, NULL));
          printf_cl_error(clGetDeviceInfo(devices[j], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(GPUinfo->MaxAlloc), &GPUinfo->MaxAlloc, NULL));
          printf_cl_error(clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, NAMES_LENGTH, GPUinfo->PlatformName, NULL));
          GPUinfo->PlatformId = platforms[i];
          GPUinfo->DeviceId = devices[j];
        }
      }
    }
  }
  if (GPUinfo->DeviceMaxCUs > 0) {
    printf("Selected device %s in platform %s.\n", GPUinfo->DeviceName, GPUinfo->PlatformName);
    printf("Maximum GPU memory allocation is %f MiB.\n", GPUinfo->MaxAlloc/(1048576.0));
    printf("Global GPU memory is %f MiB.\n", GPUinfo->MaxGlobalMem/(1048576.0));
    return true;
  } else {  
    printf("No GPU selected.\n");
    return false;
  }
}
char kernel0[] = "\
bool geEmU128(ulong *a, ulong *b) {\
  bool res = *(a+1) > *(b+1);\
  bool eqres = (*(a+1) == *(b+1));\
  bool lores = *a >= *b;\
  return res | (eqres & lores); \
}\
bool gtEmU128(ulong *a, ulong *b) {\
  bool res = *(a+1) > *(b+1);\
  bool eqres = (*(a+1) == *(b+1));\
  bool lores = *a > *b;\
  return res | (eqres & lores);\
}\
bool eqEmU128(ulong *a, ulong *b) {\
  return (*(a+1) == *(b+1)) && (*a == *b);\
}\
void addEmU128(ulong *a, ulong *b, ulong *c) {\
  bool carry = false;\
  ulong hi,lo;\
  lo = *a + *b;\
  carry = lo < *b;\
  hi = *(a+1) + *(b+1) + carry;\
  *c = lo;\
  *(c+1) = hi;\
}\
\
__kernel void getsolutionsGPU(__global ulong *arr, __global  ulong *restrict args, __global  ulong *restrict retixs) { \
  __private ulong privretixs[1+MAXRETURNEDIXSPERWKITEM] = {0};\
  __private uint i;\
  __private ulong workitemno = ((get_global_id(2) - get_global_offset(2)) * get_global_size(1) * get_global_size(0)) + ((get_global_id(1) - get_global_offset(1)) * get_global_size(0)) + (get_global_id(0) - get_global_offset(0));\
  __private ulong xtoa[2];\
  __private ulong xtoaplusytob[2];\
  __private ulong xix = 2*(args[0] + workitemno);\
  xix = xix > ARRSIZE ? ARRSIZE : xix;\
  __private ulong zix = xix;\
  __private ulong yix = 2*args[0];\
  __private ulong yixoffset;\
  __private uint save;\
  __private bool skipserror = false;\
  __private uint privretixsix = 0;\
  xtoa[0] = arr[xix];\
  xtoa[1] = arr[xix + 1];\
  addEmU128(xtoa, arr+yix, xtoaplusytob);\
  xix = 0;\
  zix = ARRSIZE-2;\
  for (i=0; i<args[1]; i++) {\
    yix = (xix + zix)/2;\
    yix = ((yix >> 1) << 1);\
    save = geEmU128(arr+yix, xtoaplusytob);\
    xix = (save ? xix : yix);\
    zix = (save ? yix : zix);\
  }\
  zix = yix;\
  xix = 2*(args[0] + workitemno);\
  xix = xix > ARRSIZE ? ARRSIZE : xix;\
  for(yix = 2*args[0];yix<ARRSIZE;yix+=YIXOFFSETMAX+2) {\
    for(yixoffset=0; (yixoffset <= YIXOFFSETMAX) && (yix+yixoffset < ARRSIZE); yixoffset+=2) {\
      addEmU128(xtoa, arr+yix+yixoffset, xtoaplusytob);\
      for(i=0;i<MAXZIXSKIPS + (args[0] <= 9325);i++) {\
        zix += 2*gtEmU128(xtoaplusytob, arr+zix);\
        zix = zix > ARRSIZE ? ARRSIZE : zix;\
      }\
      skipserror |= (zix < ARRSIZE) && gtEmU128(xtoaplusytob, arr+zix);\
      save = (zix < ARRSIZE) && eqEmU128(xtoaplusytob, arr+zix) && (yix+yixoffset >= xix);\
      privretixs[privretixsix] = (save ? 1+yix+yixoffset : privretixs[privretixsix]);\
      privretixsix += save;\
      privretixsix = privretixsix - (privretixsix > MAXRETURNEDIXSPERWKITEM);\
    }\
    barrier(CLK_GLOBAL_MEM_FENCE);\
  }\
  for(i=0;i<=MAXRETURNEDIXSPERWKITEM; i++) retixs[(workitemno * (1+MAXRETURNEDIXSPERWKITEM)) + i] = privretixs[i];\
  retixs[(workitemno * (1+MAXRETURNEDIXSPERWKITEM))] = (skipserror ? -1 : retixs[(workitemno * (1+MAXRETURNEDIXSPERWKITEM))]);\
}";

uint64_t getsolutionsGPU(U128_2xU64_t *arr, uint64_t count, _Bool printsolutions, GPUinfo_t *GPUinfo) {
  uint64_t solutionscount = 0;
  uint64_t args[ARGSSIZE] = {0};
  args[1] = 1; // to become >=ceil(log_2(count)) for binary search loops in kernel.
  solutionscount = count;
  while (solutionscount) {
    solutionscount >>= 1;
    args[1]++;
  }
  solutionscount = 0;
  size_t LWkgrpsize[] = {DEFAULT_LOCAL_DIM,1,1};
  size_t GWkgrpsize[] = {DEFAULT_LOCAL_DIM,10,100};
  size_t workitems = GWkgrpsize[0]*GWkgrpsize[1]*GWkgrpsize[2];
  char ocloptions[512];
  sprintf(ocloptions, "-D MAXRETURNEDIXSPERWKITEM=%u -D ARRSIZE=%lu -D MAXZIXSKIPS=%u -D YIXOFFSETMAX=%u", MAXRETURNEDIXSPERWKITEM, count*2, MAXZIXSKIPS, YIXOFFSETMAX);
  cl_int res;
  GPUinfo->ContextId = clCreateContext(NULL, 1, &GPUinfo->DeviceId, NULL, NULL, &res);
  printf_cl_error(res);
  GPUinfo->CommandQueueId = clCreateCommandQueue(GPUinfo->ContextId, GPUinfo->DeviceId, 0, &res);
  printf_cl_error(res);
  const char *kernels[KERNEL_COUNT];
  size_t kernelsizes[KERNEL_COUNT];
  kernels[0] = kernel0;
  kernelsizes[0] = strlen(kernel0);
  GPUinfo->ProgramId = clCreateProgramWithSource(GPUinfo->ContextId, KERNEL_COUNT, (const char **)kernels, (const size_t*)kernelsizes, &res);
  printf_cl_error(res);
  printf_cl_error(clBuildProgram(GPUinfo->ProgramId, 1, &GPUinfo->DeviceId, ocloptions, NULL, NULL));
  uint64_t *retixs = aligned_alloc(4096, workitems*(1+MAXRETURNEDIXSPERWKITEM)*sizeof(uint64_t));
  memset(retixs,0,workitems*(1+MAXRETURNEDIXSPERWKITEM)*sizeof(uint64_t));
  cl_mem retixs_mem_obj = clCreateBuffer(GPUinfo->ContextId, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, workitems*(1+MAXRETURNEDIXSPERWKITEM)*sizeof(uint64_t), retixs, &res);
  printf_cl_error(res);
  printf_cl_error(clEnqueueWriteBuffer(GPUinfo->CommandQueueId, retixs_mem_obj, CL_TRUE, 0, workitems*(1+MAXRETURNEDIXSPERWKITEM)*sizeof(uint64_t), (const void *)retixs, 0, NULL, NULL));
  cl_mem arr_mem_obj = clCreateBuffer(GPUinfo->ContextId, CL_MEM_READ_ONLY, (1+count)*sizeof(U128_2xU64_t), NULL, &res);
  printf_cl_error(res);
  cl_mem args_mem_obj = clCreateBuffer(GPUinfo->ContextId, CL_MEM_READ_ONLY, ARGSSIZE*sizeof(unsigned long), NULL, &res);
  printf_cl_error(res);
  GPUinfo->kernel = clCreateKernel(GPUinfo->ProgramId, "getsolutionsGPU", &res);
  printf_cl_error(res);
  printf_cl_error(clSetKernelArg(GPUinfo->kernel, 0, sizeof(cl_mem), (void *)&arr_mem_obj));
  printf_cl_error(clSetKernelArg(GPUinfo->kernel, 1, sizeof(cl_mem), (void *)&args_mem_obj));
  printf_cl_error(clSetKernelArg(GPUinfo->kernel, 2, sizeof(cl_mem), (void *)&retixs_mem_obj));
  printf_cl_error(clEnqueueWriteBuffer(GPUinfo->CommandQueueId, arr_mem_obj, CL_TRUE, 0, (1+count)*sizeof(U128_2xU64_t), (const void *)arr, 0, NULL, NULL)); // Allow 'index 'overflow' with extra dummy element
  while (args[0] < count) {
    printf_cl_error(clEnqueueWriteBuffer(GPUinfo->CommandQueueId, args_mem_obj, CL_TRUE, 0, ARGSSIZE*sizeof(unsigned long), (const void *)args, 0, NULL, NULL));
    printf_cl_error(clEnqueueNDRangeKernel(GPUinfo->CommandQueueId, GPUinfo->kernel, GPUinfo->MaxWkgrpdims, NULL, GWkgrpsize, LWkgrpsize, 0, NULL, NULL));
    printf_cl_error(clEnqueueReadBuffer(GPUinfo->CommandQueueId, retixs_mem_obj, CL_TRUE, 0, workitems*(1+MAXRETURNEDIXSPERWKITEM)*sizeof(uint64_t), (void *)retixs, 0, NULL, NULL));
    for (uint64_t i=0; i<workitems; i++) {
      if ((retixs[(1+MAXRETURNEDIXSPERWKITEM)*i] == 0xffffffffffffffffULL) && (args[0] + i < count)) {
        fprintf(stderr, "Calculation Failed!! (ZIXSKIPERROR - Increase MAXZIXSKIPS from %u)\n", MAXZIXSKIPS);
        return 0;
      }
      if ((retixs[(1+MAXRETURNEDIXSPERWKITEM)*i + MAXRETURNEDIXSPERWKITEM] > 0) && (args[0] + i < count)) {
        fprintf(stderr, "Calculation Failed!! (MAXRETURNEDIXSPERWKITEMERROR - Increase MAXRETURNEDIXSPERWKITEM from %u)\n", MAXRETURNEDIXSPERWKITEM);
        return 0;
      }
    }
    for (uint64_t i=0; i<workitems; i++) {
      for (uint64_t j=0; j<MAXRETURNEDIXSPERWKITEM; j++) {
        uint64_t ix = retixs[i*(1+MAXRETURNEDIXSPERWKITEM) + j];
        if ((ix) && (args[0] + i < count)) {
          solutionscount++;
          ix--;
          U128 temp = arr[i+args[0]].num + arr[ix/2].num;
          if (gcd(arr[i+args[0]].num, arr[ix/2].num) == 1) {
              printf("JACKPOT!!!\n0x%012lx%016lx + 0x%012lx%016lx = 0x%012lx%016lx\n", (uint64_t)(arr[i+args[0]].num >> 64), (uint64_t)arr[i+args[0]].num, (uint64_t)(arr[ix/2].num >> 64), (uint64_t)arr[ix/2].num, (uint64_t)(temp >> 64), (uint64_t)temp);
          } else {
            if (printsolutions) {
              printf("0x%012lx%016lx + 0x%012lx%016lx = 0x%012lx%016lx\n", (uint64_t)(arr[i+args[0]].num >> 64), (uint64_t)arr[i+args[0]].num, (uint64_t)(arr[ix/2].num >> 64), (uint64_t)arr[ix/2].num, (uint64_t)(temp >> 64), (uint64_t)temp);
            }
          }
        } else break;
      }
    }
    args[0] += workitems;
  }
  free(retixs);
  return solutionscount;
}
//============================================================================================================


int myqsort(uint64_t size, U128_2xU64_t *data, int level) {
  // In Place Quicksort Variant
  assert(level < 1000);
  U128 temp;
  if (size <= 1) return 0;
  if (size == 2) {
    if (data[0].num > data[1].num) {
      temp = data[0].num;
      data[0].num = data[1].num;
      data[1].num = temp;
    }
    return 0;
  }
  U128 imean, sum, imax = 0, imin = (((U128)0xffffffffffffffffULL) << 64) + 0xffffffffffffffffULL;
  uint64_t i,j, ltimean;
  for (i=0; i<size; i++) {
    imax = (imax > data[i].num ? imax : data[i].num);
    imin = (imin < data[i].num ? imin : data[i].num);
  }
  sum = imax + imin;
  imean = sum / 2;
  if (sum < imax) imean += ((U128)1) << 127;
  ltimean = 0;
  while (ltimean == 0) {
    for (i=0; i<size; i++) {
      ltimean += (data[i].num < imean);
    }
    if (ltimean == 0) imean++;
  }
  i = 0;
  j = ltimean;
  while (1) {
    // Find next element to swap in low partition
    while ((data[i].num < imean) && (i < ltimean)) i++;
    if (i >= ltimean) break;
    // Find next element to swap in high partition
    while (data[j].num >= imean) j++;
    temp = data[i].num;
    data[i].num = data[j].num;
    data[j].num = temp;
  }
  if (ltimean == size) return 0;
  if (myqsort(ltimean, data, level+1) != 0) return 1;
  if (myqsort(size - ltimean, data+ltimean, level+1) != 0) return 1;
  return 0;
}


int qsortcmp(const void *a, const void *b) {
    const U128_2xU64_t *c = a;
    const U128_2xU64_t *d = b;
    return c->num == d->num ? 0 : (c->num < d->num  ? -1 : 1);
}

uint64_t deduplicatearrayu128(U128_2xU64_t *array, uint64_t arraysize) {
  // Removes duplicates from sorted ascending array.
  if (arraysize == 0) return 0;
  uint64_t i1 = 0;
  for (uint64_t i2 = 1; i2 < arraysize; i2++) {
    if (array[i1].num != array[i2].num) {
      i1++;
      array[i1].num = array[i2].num;
    }
  }  
  return ++i1;
}

uint64_t getpowerarrayinitcount(int n) {
  U128 twoton = ((U128)1) << n;
  U128 xto3 = 8;  
  uint64_t initarrix = 1;  
  for(uint64_t x = 2; xto3 < twoton; ) {
    U128 xpow = xto3;
    while (true) {
      initarrix++;
      if (xpow >= twoton/x) break;
      xpow *= x;
    }
    x++;
    xto3 = ((((U128)x)*x)*x);
  }  
  return 1+initarrix; // Add 1 for extra dummy element so index can 'overflow'
}

U128_2xU64_t *getpowerarray(int n, uint64_t initarrsize, uint64_t *countret) {
  // Creates EmU128 array of all x^y < 2^n with y>=4
  //printf("Creating initial space for %lu powers (~%f MB)\n", initarrsize, initarrsize*sizeof(U128)/(1000000.0));
  U128 twoton = ((U128)1) << n;
  U128_2xU64_t *initarr = aligned_alloc(alignof(U128_2xU64_t), initarrsize*sizeof(U128_2xU64_t));
  if (initarr == NULL) return NULL;
  initarr[initarrsize-1].num = 1; // Dummy element
  initarr[0].num = 1;
  U128 xto3 = 8;  
  uint64_t initarrix = 1;  
  for(uint64_t x = 2; xto3 < twoton; ) {
    U128 xpow = xto3;
    while (true) {
      assert(initarrix < initarrsize);
      initarr[initarrix++].num = xpow;
      if (xpow >= twoton/x) break;
      xpow *= x;
    }
    x++;
    xto3 = ((((U128)x)*x)*x);
  }  
  assert(myqsort(initarrix, initarr, 0) == 0);
  uint64_t count = deduplicatearrayu128(initarr, initarrix);
  *countret = count;
  return initarr;
}


int main(int argc, char **argv){
  _Bool inputvalid = true;
  _Bool printsolutions = false;
  int n = 16;
  uint32_t maxRAMMB = 2000;
  GPUinfo_t GPUinfo;
  _Bool GPUavailable = oclinit(&GPUinfo);
  uint64_t initarrsize;
  if (argc < 2) {
    inputvalid = false;
  } else {
    n = atoi(argv[1]);
    if ((n > MAXN) || (n < MINN)) inputvalid = false;
    if ((inputvalid) && (argc > 2)) {
      maxRAMMB = atoi(argv[2]);
    }
    if (inputvalid) {
      initarrsize = getpowerarrayinitcount(n);  
      if ((2*initarrsize*sizeof(U128_2xU64_t))/1000000.0f > maxRAMMB) {
        printf("%f MB RAM required.\n", (2*initarrsize*sizeof(U128_2xU64_t))/1000000.0f);
        inputvalid = false;
      }        
    }
  }
  if (argc > 3) {
    if (strcmp(argv[3], "--printsolutions") == 0) printsolutions = true;
  }
  if (!inputvalid) {
    printf("This program finds all positive integer solutions to x^a + y^b = z^c for x,y,z >= 1, a,b,c >= 3, and z^c < 2^n for %i <= n <= %i.\nUsage:- %s n [maxRAMMB] [--printsolutions]\n", MINN, MAXN, argv[0]);
    exit(0);
  }
  uint64_t count;
  uint64_t solutionscount;
  U128_2xU64_t *arr = getpowerarray(n, initarrsize, &count);
  if (arr == NULL) {
    fprintf(stderr, "Could not allocate RAM.\n");
    exit(1);
  }
  printf("There are %lu perfect powers of 3+ < 2^%i\n", count, n);
  uint64_t starttime, endtime;
  if (GPUavailable) {
    starttime = time(0);
    solutionscount = getsolutionsGPU(arr, count, printsolutions, &GPUinfo);
    endtime = time(0);
    if (endtime > starttime+5) {
      printf("Found %lu solutions in %lu seconds. (~%f million comparisons /s)\n", solutionscount, endtime-starttime, ((double)count*count)/(2000000.0*(endtime-starttime)));
    } else {
      printf("Found %lu solutions.\n", solutionscount);
    }
  } 
  free(arr);
}
