#lang rosette

(require "util.rkt" "cuda.rkt" "cuda-synth.rkt")
(require (only-in racket [sort %sort] [< %<]))


(define sizes (x-y-z 10))
;(define I (create-matrix sizes (lambda () (define-symbolic* x integer?) x)))
(define I (create-matrix sizes gen-uid))
(define O (create-matrix (- sizes 2)))
(define O* (create-matrix (- sizes 2)))

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

  (for/bounded ([i (??)])
    (let* ([index (?index localId (@dup i) 1)]
           [lane (?lane localId (@dup i) 1)]
           [x (shfl (get I-cached index) lane)])
      (accumulate o x #:pred (?cond localId (@dup i)))
      ))
  
  (reg-to-global o O threadId (- sizes 2))
  )

(define (synthesis)
  (run-kernel conv1d-sketch (x-y-z 8) (x-y-z 1) I O*)
  #;(pretty-display `(O* ,O*))
  #;(for ([i 8])
      (pretty-display `(O* ,i ,(get-accumulator-val (get O* i)))))
  
  (conv1d-spec I O (- sizes 2))
  #;(pretty-display `(O ,O))
  (for ([i 8])
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
  )
;(synthesis)

(define (load-synth)
  ;; Store
  (define (conv1d-store threadId blockId blockDim O)
    (define warpID (get-warpId threadId))
    (define o
      (for/vector ([w  warpID]
                   [t threadId])
        (ID t w blockId)))
    (reg-to-global o O (get-global-threadId threadId blockId) (- sizes 2))
    )
  
  ;; Run spec
  (conv1d-spec I O (- sizes 2))
  
  ;; Collect IDs
  (define IDs (create-matrix (- sizes 2)))
  (run-kernel conv1d-store (x-y-z 8) (x-y-z 1) IDs)
  (define-values (threads warps blocks) (get-grid-storage))
  (collect-inputs O IDs threads warps blocks)
  (define n-regs (num-regs warps I))
  (pretty-display `(n-regs ,n-regs))
  
  ;; Load
  (define (conv1d-load threadId blockId blockDim I warp-input-spec)
    (define warpId (get-warpId threadId))
    ;; sketch starts
    (define I-cached (create-matrix (x-y-z n-regs)))
    (global-to-warp-reg I I-cached
                        (x-y-z (??)) ;; stride
                        (x-y-z (?warp-offset [(get-x blockId) (get-x blockDim)] [warpId warpSize])) ;; offset
                        (x-y-z (?warp-size warpSize 1)) ;; load size
                        sizes #f)
    #;(global-to-warp-reg I I-cached
                        (x-y-z (choose 1 2 3)) ;; stride
                        (+ (* blockId blockDim) (* warpId warpSize)) ;; offset
                        (x-y-z (+ warpSize 2)) ;; load size
                        sizes #f)
    ;; sketch ends
    (check-warp-input warp-input-spec I I-cached warpId blockId)
    )
  
  (run-kernel conv1d-load (x-y-z 8) (x-y-z 1) I warps)
  (define sol
    (time
     (synthesize
      #:forall (symbolics I)
      #:guarantee (assert #t))))
  (when (sat? sol)
    (print-forms sol)
    (define sol-hash (match sol [(model m) m]))
    (for ([key-val (hash->list sol-hash)])
      (let ([key (car key-val)]
            [val (cdr key-val)])
        (when (string-contains? (format "~a" key) "stencil:115") ;; stride
          (assert (not (equal? key val)))
          (pretty-display `(v ,key ,val ,(string-contains? (format "~a" key) "stencil:113")))))
      ))
  
  (define sol2
    (time
     (synthesize
      #:forall (symbolics I)
      #:guarantee (assert #t))))
  (when (sat? sol2)
    (print-forms sol2))
  )
(load-synth)