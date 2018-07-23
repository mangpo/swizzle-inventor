#lang racket

(require "codegen.rkt")

(define func
  '(define (AOS-loadhsh3* threadId blockID blockDim I O a b c)
  (define I-cached (create-matrix-local (x-y-z struct-size)))
  (define warpID (get-warpId threadId))
  (define offset
    (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))
  (define gid (get-global-threadId threadId blockID))
  (global-to-local
   I
   I-cached
   (x-y-z 1)
   offset
   (x-y-z (* warpSize struct-size))
   #f #:round struct-size
   #:shfl (lambda (localId i) (fan localId warpSize 0 1 32 1 i struct-size 31 1)))
  (define localId (get-idInWarp threadId))
  (define O-cached (permute-vector I-cached struct-size
                                   (lambda (i) (fan i struct-size 2 3 3 1 localId warpSize 0 1))))
  (local-to-global
   O-cached
   O
   (x-y-z 1)
   offset
   (x-y-z (* warpSize struct-size))
   #f #:round struct-size
   #:shfl (lambda (localId i)
            (fan localId warpSize 11 32 32 1 i struct-size 20 1)))
  ))
  

(print-cuda (racket2cuda func 1))
;(print-cuda (convert-statement loop))
;(print-cuda (convert-statement fan))