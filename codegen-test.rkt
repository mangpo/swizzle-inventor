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
;(print-cuda (convert-statement loop))
;(print-cuda (convert-statement fan))