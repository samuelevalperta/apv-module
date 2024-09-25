#include <Python.h>
#include <math.h>
#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>
#include <stdio.h>
#include <sys/mman.h>

#define ALFA 1
#define BETA 1
#define GAMMA 1
#define OMEGA 1
#define MU 1
#define THETA 1
#define EPS 1

#define VAR_NUMBER 3
#define dadt(a, p) ((ALFA * a) - EPS * p);
#define dpdt(a, p, v) ((THETA * p * a) - (MU * p) - (OMEGA * p * v))
#define dvdt(p, v) ((BETA * p * v) - (GAMMA * v))

void f(double a, double p, double v, double *results) {
  *results = dadt(a, p);
  *(results + 1) = dpdt(a, p, v);
  *(results + 2) = dvdt(p, v);
}

void next(double *x, double h) {
  double increments[] = {0, 0, 0};
  f(*(x), *(x + 1), *(x + 2), increments);
  for (int i = 0; i < VAR_NUMBER; i++) {
    *((x + i) + VAR_NUMBER) = *(x + i) + h * (*(increments + i));
  }
}

static PyObject *getAPVMatrix(PyObject *Py_UNUSED(self), PyObject *args) {
  double tf, t0, h, a_zero, p_zero, v_zero;

  if (!PyArg_ParseTuple(args, "dddddd", &tf, &t0, &h, &a_zero, &p_zero,
                        &v_zero)) // "dddddd" beacuse we are expecting 6 doubles
    return NULL;

  int n_steps = (int)floor((tf - t0) / h);
  npy_intp dims[2] = {n_steps + 2, 3}; // + 2 beacuse of t0 and tf

  PyObject *numpy_array = PyArray_SimpleNew(2, dims, NPY_FLOAT64);
  if (numpy_array == NULL) {
    return NULL;
  }

  // double *x = (double *)malloc((n_steps)*VAR_NUMBER * sizeof(double));
  // if (x == NULL) {
  //   return NULL;
  // }

  double *x = (double *)PyArray_DATA((PyArrayObject *)numpy_array);

  *x++ = a_zero;
  *x++ = p_zero;
  *x++ = v_zero;

  for (int i = 0; i < n_steps; i++) {
    next(x, h);
  }

  next(x, tf - (t0 + h * n_steps));

  return numpy_array;
}

static PyMethodDef methods[] = {
    {"getAPVMatrix", getAPVMatrix, METH_VARARGS, "doc"}, {NULL, NULL, 0, NULL}};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT, // always required
    "APV",                 // module name
    "Speed up APV model matrix generation with Euler's Method", // description
    -1,      // module size (-1 indicates we don't use this feature)
    methods, // method table
};

PyMODINIT_FUNC PyInit_APV() {
  printf("Initialization\n");
  return PyModule_Create(&module_def);
}

// PyMODINIT_FUNC initfoo(void) {
//   import_array(); // enable NumPy C API
// }
