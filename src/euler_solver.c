#include "config.h"
#include "numpy/ndarraytypes.h"
#include "numpy/npy_common.h"
#include <Python.h>
#include <math.h>
#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>
#include <stdio.h>
#include <sys/mman.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

double *next(double *x) {
  double dx[VAR_NUMBER];
  f(x, dx);
  for (int i = 0; i < VAR_NUMBER; i++) {
    x[VAR_NUMBER + i] = x[i] + (dt * dx[i]);
  }
  return x + VAR_NUMBER;
}

static PyObject *getSolutionMatrix(PyObject *Py_UNUSED(self), PyObject *args) {

  PyArrayObject *x_zero_array = NULL;

  // Parse x0 array from args
  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &x_zero_array)) {
    return NULL;
  }

  x_zero_array = (PyArrayObject *)PyArray_FROM_OTF(
      (PyObject *)x_zero_array, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

  if (x_zero_array == NULL) {
    PyErr_SetString(PyExc_TypeError,
                    "Input must be a 3D NumPy array of npy_double (float64)");
    return NULL;
  }

  npy_intp *dimss = PyArray_DIMS(x_zero_array);

  if ((int)PyArray_Size((PyObject *)x_zero_array) != VAR_NUMBER) {
    PyErr_SetString(PyExc_ValueError, "Input array must have 3 dimensions");
    return NULL;
  }

  double *x_zero_data = (double *)PyArray_DATA(x_zero_array);

  // npy_intp dims[2] = {n_steps, VAR_NUMBER};
  unsigned int n_steps = (unsigned int)(tf - ti) / dt;
  npy_intp dims[2] = {VAR_NUMBER, n_steps};

  // Create x npy_array
  PyObject *npy_array = PyArray_SimpleNew(2, dims, NPY_FLOAT64);
  if (npy_array == NULL) {
    return NULL;
  }

  double *x = (double *)PyArray_DATA((PyArrayObject *)npy_array);

  // Set x0 condition
  for (int i = 0; i < VAR_NUMBER; i++)
    x[i] = x_zero_data[i];

  // Iterate over all steps
  for (unsigned int i = 0; i < n_steps; i++)
    x = next(x);

  return npy_array;
}

static PyMethodDef methods[] = {
    {"getSolutionMatrix", getSolutionMatrix, METH_VARARGS, "doc"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT, // always required
    "euler_solver",        // module name
    "A Python extension module for solving ODE using Euler method", // description
    -1,      // module size (-1 indicates we don't use this feature)
    methods, // method table
};

PyMODINIT_FUNC PyInit_euler_solver() {
  // Initialize the NumPy API
  import_array();
  Py_Initialize();
  return PyModule_Create(&module_def);
}
