/* 
@author: Nian Wu and Miaomiao Zhang
This file is a revision of the FLASH(C++) codebase, designed to interact with NeurEPDiff. Specifically, it receives predicted geodesic shooting results from NeurEPDiff and optimizes the registration energy accordingly.
This file is going to deal with 2D bull eyes' registration presented in paper NeurEPDiff https://arxiv.org/pdf/2303.07115.pdf.
*/



#include "Python.h"
#include <numpy/arrayobject.h>
#include "WnShooting.h"
#include "MPIlib.h"
#include "Vec2D.h"
#include<sys/timeb.h>
using namespace std;

long long systemtime()
{
    timeb t;
    ftime(&t);
    return t.time*1000+t.millitm;
}

void init_numpy(){
    import_array();
}



int main(int argc, char** argv)
{
  int argi = 1; 
  int idx = atoi(argv[argi++]);
  char src[100];     
  char tar[100];
  char outPrefix[100];
  strcpy(src, argv[argi++]);
  strcpy(tar, argv[argi++]);
  strcpy(outPrefix, argv[argi++]);

  int truncX = atoi(argv[argi++]);
  int truncY = atoi(argv[argi++]);
  int truncZ = atoi(argv[argi++]);
  int numStep = atoi(argv[argi++]);
  int maxIter = atoi(argv[argi++]);
  float stepSizeGD = atof(argv[argi++]);
  float alpha = atof(argv[argi++]);
  float gamma = atof(argv[argi++]);
  int lpower = atoi(argv[argi++]);
  float sigma = atof(argv[argi++]);
  float prevEnergy = 1e20;
  char temPath[100];
  int memType = 0;
  cout << fixed;
  cout<< stepSizeGD <<"  "<< truncX <<"  "<< truncY <<"  "<< truncZ <<"  "<< alpha <<"  " << gamma <<"  " << lpower <<"  "<< sigma <<"  " << endl;
  MemoryType mType;
  // runs on CPU or GPU
  if (memType == 0)
      mType = MEM_HOST;
  else 
      mType = MEM_DEVICE;


  // read data
  Image3D *I0, *I1;
  I0 = new Image3D(mType);I1 = new Image3D(mType);
  ITKFileIO::LoadImage(*I0, src); ITKFileIO::LoadImage(*I1, tar);
  GridInfo grid = I0->grid(); Vec3Di mSize = grid.size();
  int fsx = mSize.x;
  int fsy = mSize.y;
  int fsz = mSize.z;
  int nImVox = fsx*fsy*fsz;
  // precalculate low frequency location
  if (truncX % 2 == 0) truncX -= 1; // set last dimension as zero if it is even
  if (truncY % 2 == 0) truncY -= 1; // set last dimension as zero if it is even

  FftOper *fftOper = new FftOper(alpha, gamma, lpower, grid, truncX, truncY, truncZ); 
  fftOper->FourierCoefficient();

  FieldComplex3D *prevV0 = new FieldComplex3D(truncX, truncY, truncZ);
  FieldComplex3D *v0 = new FieldComplex3D(truncX, truncY, truncZ);
  Field3D *v0Spatial = new Field3D(grid, mType);
  FieldComplex3D *gradv = new FieldComplex3D(truncX, truncY, truncZ);

  GeodesicShooting *geodesicshooting = new GeodesicShooting(fftOper, mType, numStep);
  Opers::Copy(*(geodesicshooting->I0), *I0);
  Opers::Copy(*(geodesicshooting->I1), *I1);
  geodesicshooting->sigma = sigma;
  
  
  

  /////////////  Interact with NeurEPDiff and optimize registration energy   ////////////////
  Py_Initialize();
  if ( !Py_IsInitialized() ) {
          return -1;
  }
  init_numpy(); //Initialize NumPy Python module
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("print(sys.version)");
  PyRun_SimpleString("sys.path.append('/NeurDPDiff/Test')");
  PyObject* pRet = NULL;
  PyObject* ArgArray=NULL, *PySrc=NULL, *PyTar=NULL;
  PyObject* pModule = PyImport_ImportModule("NeruEPDiff_2D_Optimize");  //Import Python script for 2D NeurEPDiff
//   PyObject* pModule = PyImport_ImportModule("NeruEPDiff_3D_Optimize_2");  //Import Python script for 3D NeurEPDiff
//   PyObject* pModule = PyImport_ImportModule("NeruEPDiff_3D_Optimize");  //Import Python script for 3D NeurEPDiff



  if(!pModule){
      printf("Fail to import Python module...\n");
      return -1;
  }
  PyObject* pFunc = PyObject_GetAttrString(pModule, "initfun");   //import Python Funtion


  // set some temporary variables //
  size_t size = fsz*fsx*fsy*sizeof(float);
  float *V0X = (float *)malloc(size);
  float *V0Y = (float *)malloc(size);
  float *V0Z = (float *)malloc(size);
  npy_intp Dims[3] = {fsz, fsx, fsy};
  PyObject *PyV0X, *PyV0Y, *PyV0Z;
  PyArrayObject *VX, *VY, *VZ;
  int k;
  long long t = systemtime();
  long long t1;
  // set some temporary variables //
  
  // start to optimize registration energy //
  // storeV[]: save the sequence of velocity from v0->vt //
  for (int cnt=0;cnt<maxIter;cnt++){
      // PyV0X  PyV0Y  PyV0Z: varibles passed on to NeruEPDiff
      Copy_FieldComplex(*geodesicshooting->storeV[0], *v0);
      fftOper->fourier2spatial(*v0Spatial, *v0);
      memcpy(V0X,v0Spatial->getX(),nImVox*sizeof(float));
      memcpy(V0Y,v0Spatial->getY(),nImVox*sizeof(float));
      memcpy(V0Z,v0Spatial->getZ(),nImVox*sizeof(float));
      PyV0X  = PyArray_SimpleNewFromData(3, Dims, NPY_FLOAT, V0X);
      PyV0Y  = PyArray_SimpleNewFromData(3, Dims, NPY_FLOAT, V0Y);
      PyV0Z  = PyArray_SimpleNewFromData(3, Dims, NPY_FLOAT, V0Z);
      ArgArray = PyTuple_New(4);
      PyTuple_SetItem(ArgArray, 0, PyV0X);
      PyTuple_SetItem(ArgArray, 1, PyV0Y);
      PyTuple_SetItem(ArgArray, 2, PyV0Z);
      PyTuple_SetItem(ArgArray, 3, Py_BuildValue("i",cnt+1));

      // interact with NeruEPDiff: pass on the initial velocity(ArgArray) to NeruEPDiff //
      pRet = PyObject_CallObject(pFunc, ArgArray); 

      
      for (int i = 1; i <=numStep; i++){
          // Analyze the predictions from NeruEPDiff
          k = i-1;
          VX = (PyArrayObject *)PyTuple_GetItem(pRet, k*3);  //PyTuple_SetItem
          VY = (PyArrayObject *)PyTuple_GetItem(pRet, k*3+1);  //PyTuple_SetItem
          VZ = (PyArrayObject *)PyTuple_GetItem(pRet, k*3+2);  //PyTuple_SetItem
          memcpy(v0Spatial->getX(), VX->data, nImVox*sizeof(float));
          memcpy(v0Spatial->getY(), VY->data, nImVox*sizeof(float));
          memcpy(v0Spatial->getZ(), VZ->data, nImVox*sizeof(float));
          //save the NeruEPDiff predictions in storeV[]
          fftOper->spatial2fourier(*geodesicshooting->storeV[i], *v0Spatial);
      }
      

      // forward shooting with storeV[] 
      geodesicshooting->ImageMatching(*v0, *gradv, 1, stepSizeGD, prevEnergy, *prevV0);
      cout<< cnt+1 << "  " <<geodesicshooting->TotalEnergy << "  " <<geodesicshooting->IEnergy <<"   "<< geodesicshooting->VEnergy<<endl;

  }
  

  // t = systemtime() - t;
  // printf("It took me (%f seconds).\n",((float)t)/1000);


  /// Save results ///
  sprintf(temPath, "%s/deformIm.mhd", outPrefix); 
  ITKFileIO::SaveImage(*(geodesicshooting->deformIm), temPath);
  for (int i = 0; i <= numStep; i++)
  {
    fftOper->fourier2spatial(*v0Spatial, *geodesicshooting->storeV[i]);
    sprintf(temPath, "%s/v0%d_%d.mhd", outPrefix, idx, i); 
    ITKFileIO::SaveField(*v0Spatial, temPath);    
  }



  Py_Finalize();
  delete fftOper;
  delete geodesicshooting;
  delete I0;
  delete I1;
  delete v0;
  delete v0Spatial;
  delete gradv;
  delete prevV0; prevV0 = NULL;
}
