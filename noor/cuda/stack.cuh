/*
MIT License

Copyright (c) 2015-2018 Ardavan Kanani

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#ifndef CUDASTACK_CUH
#define CUDASTACK_CUH

extern __shared__ uint shstack[];
template<class T>
class CudaStack {
    T* _s;
    int _top;
    //int _sentinel;
public:
    //__device__ CudaStack( T* s ) : _s( s ), _top( -1 ), _sentinel( -1 ) {};
    __device__ CudaStack( T* s ) : _s( s ), _top( -1 ) {};
    __device__ bool isEmpty() { return _top == -1; }
    __device__ int top_index() { return _top; }
    __device__ T top() { return _s[_top]; }
    __device__ T pop() { T element = _s[_top--];  return element; }
    __device__ void push( T element ) { _s[++_top] = element; }
    __device__ void reset() { _top = -1; }
};
#endif /* STACK_CUH */