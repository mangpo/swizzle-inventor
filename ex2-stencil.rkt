#lang rosette

(require "util.rkt" "cuda.rkt" "cuda-synth.rkt")
(require (only-in racket [sort %sort] [< %<]))


(define sizes (x-y-z 6))
;(define I (create-matrix sizes (lambda () (define-symbolic* x integer?) x)))
(define I (create-matrix sizes gen-uid))
(define O (create-matrix sizes))
(define O* (create-matrix sizes))

(define /3 (lambda (x) (/ x 3)))

(define (conv1d-spec I O o-sizes)
  (for ([i (get-x o-sizes)])
    (let ([o (create-accumulator o (list +) /3)])
      (for ([j 3])
        (accumulate o (get I (+ i j))))
      (set O i o))))

(define (conv1d threadId blockID blockDim I O)
  (define I-cached (create-matrix (x-y-z 2)))
  (define warpID (get-warpId threadId))
  (define offset (+ (* blockID blockDim) (* warpID warpSize)))  ;; warpID = (threadIdy * blockDimx + threadIdx)/warpSize
  (global-to-warp-reg I I-cached
                 (x-y-z 1)
                 offset (x-y-z (+ warpSize 2)) sizes #f)

  (define localId (get-idInWarp threadId))
  (define o (create-accumulator o (list +) /3 blockDim))
  (for ([i 3])
    (let* ([index (ite (< localId i) 1 0)]
           [lane (+ i localId)]
           [x (shfl (get I-cached index) lane)])
      (accumulate o x)
      ))
  (reg-to-global o O threadId (- sizes 2))
  )

(define (conv1d-sketch threadId blockID blockDim I O)
  (define I-cached (create-matrix (x-y-z 2)))
  (define warpID (get-warpId threadId))
  (define offset (+ (* blockID blockDim) (* warpID warpSize)))  ;; warpID = (threadIdy * blockDimx + threadIdx)/warpSize
  (global-to-warp-reg I I-cached
                 (x-y-z 1)
                 offset (x-y-z (+ warpSize 2)) sizes #f)

  (define localId (get-idInWarp threadId))
  (define o (create-accumulator o (list +) /3 blockDim))
    
  (for ([i 3])
    (let* ([index (?index localId (@dup i) 1)]
           [lane (?lane localId (@dup i) 1)]
           [x (shfl (get I-cached index) lane)])
      (accumulate o x #:pred (?cond localId (@dup i)))
      ))
  (reg-to-global o O threadId (- sizes 2))
  )


(run-kernel conv1d-sketch (x-y-z 8) (x-y-z 1) I O*)
#;(pretty-display `(O* ,O*))
#;(for ([i 4])
  (pretty-display `(O* ,i ,(get-accumulator-val (get O* i)))))

(conv1d-spec I O (- sizes 2))
#;(pretty-display `(O ,O))
#;(for ([i 4])
  (pretty-display `(O ,i ,(get-accumulator-val (get O i)))))

;(pretty-display `(acc-equal? ,(acc-equal? O O*)))
;(verify #:guarantee (assert (acc-equal? O O*)))

;; 1s (conc)
(pretty-display "solving...")
(define sol
  (time
   (synthesize
    #:forall (symbolics I)
    #:guarantee (assert (acc-equal? O O*)))))
(print-forms sol)