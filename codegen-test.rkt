#lang racket

(require "codegen.rkt")

(define func
  '(define (conv1d-sketch threadId blockID blockDim I O I-sizes O-sizes)
   (define I-cached (create-matrix-local (x-y-z 2)))
   (define warpID (get-warpId threadId))
   (define offset (+ (* blockID blockDim) (* warpID warpSize)))
   (define gid (get-global-threadId threadId blockID))
   (global-to-local
    I
    I-cached
    (x-y-z 1)
    (+ (* blockID blockDim) (* warpID warpSize))
    (x-y-z (+ warpSize 2))
    #f
    #:round
    2)
   (define localId (get-idInWarp threadId))
   (define o (create-accumulator (list +) (lambda (x) (/ x 3)) blockDim))
   (for
    ((i 3))
    (let* ((index (ite (>= localId 1) 0 (ite (>= localId 2) 1 2)))
           (index2 (ite (>= i 1) 0 (ite (>= i 2) 1 2)))
           (lane i)
           (x (shfl (get I-cached index index2) lane)))
      (accumulate o x #:pred (= (@dup i) (@dup i)))))
   (reg-to-global (accumulate-final o) O gid)))
  

(print-cuda (racket2cuda func 1))
;(print-cuda (convert-statement loop))
;(print-cuda (convert-statement fan))