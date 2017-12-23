#lang rosette

(require "util.rkt" "cuda.rkt")

(define sizes (x-y-z 10))
(define I (create-matrix sizes
                         (lambda () (define-symbolic* x integer?) x)))
(define O (create-matrix sizes))

(define (conv1d threadId blockID blockDim I)
  (define I-cached (create-matrix (x-y-z 2)))
  (define warpID (get-warpId threadId blockDim))
  (define offset (+ (* blockID blockDim) (* warpID warpSize)))  ;; warpID = (threadIdy * blockDimx + threadIdx)/warpSize
  (pretty-display `(threadId ,threadId))
  (pretty-display `(warpID ,warpID))
  (pretty-display `(offset ,offset))
  (global-to-reg I I-cached
                 (x-y-z 1)
                 offset (x-y-z (+ warpSize 2)) blockDim #f)
  (pretty-display `(I ,I))
  (pretty-display `(I-cached ,I-cached))

  (define localId (get-idInWarp threadId blockDim))
  (pretty-display `(localId ,localId))
  ;;(define o (accumulate-variable '(*) '/
  (for ([i 3])
    (let* ([index (if (< localId i) 1 0)]
           [x (shfl (get I-cached index) (modulo (+ i localId) warpSize))])
      (pretty-display `(x ,x)))
    )
  )

(run-kernel conv1d (x-y-z 8) (x-y-z 1) I)