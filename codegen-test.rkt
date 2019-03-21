#|
 | Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
 |
 | Redistribution and use in source and binary forms, with or without 
 | modification, are permitted provided that the following conditions are met:
 |
 | 1. Redistributions of source code must retain the above copyright notice, 
 | this list of conditions and the following disclaimer.
 |
 | 2. Redistributions in binary form must reproduce the above copyright notice, 
 | this list of conditions and the following disclaimer in the documentation 
 | and/or other materials provided with the distribution.
 |
 | THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 | AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 | IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 | ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
 | LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 | CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 | SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
 | INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 | CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 | ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 | POSSIBILITY OF SUCH DAMAGE.
 |#

#lang racket

(require "codegen.rkt")

(define struct-size 2)

(define func
  '(define (AOS-loadsh-sketch-fan threadId blockID blockDim I O a b c)
   (define I-cached (create-matrix-local (x-y-z struct-size)))
   (define localId (modulo (get-x threadId) 32))
   (define offset
     (* struct-size (- (+ (* blockID blockDim) (get-x threadId)) localId)))
   (global-to-local
    I
    I-cached
    (x-y-z 1)
    offset
    (x-y-z (* warpSize struct-size))
    #f
    #:round
    struct-size
    #:shfl
    (lambda (localId i)
      (fan localId warpSize 2 16 32 -1 i struct-size 0 1 17)))
   (define O-cached
     (permute-vector
      I-cached
      struct-size
      (lambda (i) (fan i struct-size 1 2 2 1 localId warpSize 0 16 1))))
   (local-to-global
    O-cached
    O
    (x-y-z 1)
    offset
    (x-y-z (* warpSize struct-size))
    #f
    #:round
    struct-size
    #:shfl
    (lambda (localId i)
      (fan localId warpSize 0 1 32 -1 i struct-size 15 1 16)))))
  

(print-cuda (racket2cuda func 1 #:const-map (hash 'struct-size struct-size 'warpSize 32 'n 64)))
