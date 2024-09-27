#include <Python.h>
#include <math.h>
#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>
#include <stdio.h>
#include <sys/mman.h>

#define ALPHA 0.1
#define EPSILON 0.01
#define THETA 0.015
#define MU 0.05
#define OMEGA 0.01
#define BETA 0.015
#define GAMMA 0.15

#define VAR_NUMBER 3
#define dadt(a, p) ((ALPHA * a) - (EPSILON * p * a));
#define dpdt(a, p, v) ((THETA * p * a) - (MU * p) - (OMEGA * p * v))
#define dvdt(p, v) ((BETA * p * v) - (GAMMA * v))

void f(double a, double p, double v, double increments[]) {
  increments[0] = dadt(a, p);
  increments[1] = dpdt(a, p, v);
  increments[2] = dvdt(p, v);
}

double *next(double *x, double h) {
  double increments[VAR_NUMBER];
  f(x[0], x[1], x[2], increments);
  for (int i = 0; i < VAR_NUMBER; i++) {
    x[VAR_NUMBER + i] = x[i] + (h * increments[i]);
  }
  return x + VAR_NUMBER;
}

static PyObject *getAPVMatrix(PyObject *Py_UNUSED(self), PyObject *args) {
  double ti, tf, h, a_zero, p_zero, v_zero;

  if (!PyArg_ParseTuple(args, "dddddd", &ti, &tf, &h, &a_zero, &p_zero,
                        &v_zero)) // "dddddd" beacuse we are expecting 6 doubles
    return NULL;

  unsigned int n_steps = (unsigned int)(tf - ti) / h;
  npy_intp dims[2] = {n_steps, 3};

  PyObject *numpy_array = PyArray_SimpleNew(2, dims, NPY_FLOAT64);
  if (numpy_array == NULL) {
    return NULL;
  }

  double *x = (double *)PyArray_DATA((PyArrayObject *)numpy_array);

  x[0] = a_zero;
  x[1] = p_zero;
  x[2] = v_zero;

  for (unsigned int i = 0; i < n_steps; i++) {
    x = next(x, h);
  }

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
  import_array();
  return PyModule_Create(&module_def);
}
