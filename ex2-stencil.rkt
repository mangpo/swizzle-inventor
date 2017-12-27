#lang rosette

(require "util.rkt" "cuda.rkt")
(require (only-in racket [sort %sort] [< %<]))

(define sizes (x-y-z 6))
(define I (create-matrix sizes
                         (lambda () (define-symbolic* x integer?) x)))
(define O (create-matrix sizes))
(define O* (create-matrix sizes))

(define /3 (lambda (x) (/ x 3)))

(define (conv1d-spec I O o-sizes)
  (for ([i (get-x o-sizes)])
    (let ([o (create-accumulator o (list +) /3)])
      (for ([j 3])
        (accumulate o (get I (+ i j))))
      (normalize-accumulator o)
      (set O i o))))

(define (conv1d threadId blockID blockDim I O)
  (define I-cached (create-matrix (x-y-z 2)))
  (define warpID (get-warpId threadId blockDim))
  (define offset (+ (* blockID blockDim) (* warpID warpSize)))  ;; warpID = (threadIdy * blockDimx + threadIdx)/warpSize
  (pretty-display `(threadId ,threadId))
  (pretty-display `(warpID ,warpID))
  (pretty-display `(offset ,offset))
  (global-to-warp-reg I I-cached
                 (x-y-z 1)
                 offset (x-y-z (+ warpSize 2)) blockDim sizes #f)
  (pretty-display `(I ,I))
  (pretty-display `(I-cached ,I-cached))

  (define localId (get-idInWarp threadId blockDim))
  (pretty-display `(localId ,localId))
  (define o (create-accumulator o (list +) /3 blockDim))
  (for ([i 3])
    (let* ([index (if (< localId i) 1 0)]
           [x (shfl (get I-cached index) (modulo (+ i localId) warpSize))])
      (pretty-display `(x ,x))
      (accumulate o x)
      (normalize-accumulator o)
      ))
  (reg-to-global o O threadId blockDim (- sizes 2))
  )


(run-kernel conv1d (x-y-z 8) (x-y-z 1) I O*)
(pretty-display `(O* ,O*))
(for ([i 4])
  (pretty-display `(O* ,i ,(get-accumulator-val (get O* i)))))

(conv1d-spec I O (- sizes 2))
(pretty-display `(O ,O))
(for ([i 4])
  (pretty-display `(O ,i ,(get-accumulator-val (get O i)))))

(pretty-display `(equal? ,(equal? O O*)))