#lang racket

(require "codegen.rkt")

(define func
  '(define (conv2d threadId blockID blockDim I O I-sizes O-sizes)
  (define gid (+ (* blockID blockDim) threadId))
  (define gx (get-x gid))
  (define gy (get-y gid))
  (define id (modulo (get-x threadId) warpSize))
  (define warp-col (modulo id W))
  (define warp-row (quotient id W))

  (define offset-x (* (quotient gx warpSize) W))
  (define offset-y (* gy H))

  (define I-cached (create-matrix-local (x-y-z 2 2)))
  (global-to-local I I-cached
                 (x-y-z 1 1)
                 (lov2vol (x-y-z offset-x offset-y))
                 (+ warp-shape 2) #f
                 #:warp-shape (x-y-z W H) #:round (x-y-z 2 2) #:size N)

  (define o (create-accumulator (list +) (lambda (x) (/ x 9)) blockDim))
  
  (for* ([ky 3] [kx 3])
    (let* ([index-j (ite (< warp-row ky) 1 0)]
           [index-i (ite (< warp-col kx) 1 0)]
           [lane-x (fan warp-col W 1 W W 1
                        kx 3 1 3)]
           [lane-y (fan warp-row H 1 H H 1
                        ky 3 1 3)]
           [lane (+ (* lane-y W) lane-x)]
           [x (shfl (get I-cached index-i index-j) lane)])
      (accumulate o x)
      ))
  (reg-to-global (accumulate-final o) O
                 (lov2vol (x-y-z (+ offset-x warp-col) (+ offset-y warp-row))) #:size (- N 2))
  ))
  

(print-cuda (racket2cuda func 1))
;(print-cuda (convert-statement loop))
;(print-cuda (convert-statement fan))