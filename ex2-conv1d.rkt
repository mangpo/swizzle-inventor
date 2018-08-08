#lang rosette

(require "util.rkt" "cuda.rkt" "cuda-synth.rkt")

(define n-block 2)
(define /3 (lambda (x) (/ x 3)))

(define (create-IO warpSize)
  (set-warpSize warpSize)
  (define block-size (* 2 warpSize))
  (define I-sizes (x-y-z (* 2 block-size)))
  (define O-sizes (- I-sizes 2))
  (define W (create-matrix (list 3) gen-uid))
  (define I (create-matrix I-sizes gen-uid))
  (define O (create-matrix O-sizes))
  (define O* (create-matrix O-sizes))
  (values block-size I-sizes O-sizes I O O* W))

(define (run-with-warp-size spec kernel w)
  (define-values (block-size I-sizes O-sizes I O O* W)
    (create-IO w))

  (spec I O W O-sizes)
  (run-kernel kernel (x-y-z block-size) (x-y-z n-block) I O* W I-sizes O-sizes)
  ;(acc-print O*)
  (acc-equal? O O*)
  )

(define (conv1d-spec I O W o-sizes)
  (for ([i (get-x o-sizes)])
    (let ([o (create-accumulator (list * +) identity)])
      (for ([j 3])
        (accumulate o (list (get W j) (get I (+ i j)))))
      (set O i o))))

(define my-lane-spec
  (vector
   (inst 0 (vector 1 2))))
(define (conv1d threadId blockID blockDim I O W I-sizes O-sizes)
  (define I-cached (create-matrix-local (x-y-z 2)))
  (define warpID (get-warpId threadId))
  (define offset (+ (* blockID blockDim) (* warpID warpSize)))  ;; warpID = (threadIdy * blockDimx + threadIdx)/warpSize
  (define gid (get-global-threadId threadId blockID))
  (global-to-local I I-cached
                 (x-y-z 1)
                 (+ (* blockID blockDim) (* warpID warpSize))
                 (x-y-z (+ warpSize 2)) #f #:round 2)

  (define localId (get-idInWarp threadId))
  (define o (create-accumulator (list * +) identity blockDim))
  (for ([i 3])
    (let* ([index (ite (< localId i) 1 0)]
           [lane (+ i localId)]
           ;[lane (interpret-lane my-lane-spec (vector localId (@dup i)) (vector))] ;;(+ i localId)
           [x (shfl (get I-cached index) lane)]
           [w (@dup (get W i))]
           )
      ;(pretty-display `(lane ,i ,localId ,lane))
      (accumulate o (list w x))
      ))
  (reg-to-global o O gid)
  )


;(define my-lane (gen-lane? 1)) ;;(+ i localId)
(define (conv1d-sketch threadId blockID blockDim I O W I-sizes O-sizes)
  (define I-cached (create-matrix-local (x-y-z 2)))
  ;(define warpID (get-warpId threadId))
  ;(define offset (+ (* blockID blockDim) (* warpID warpSize)))  ;; warpID = (threadIdy * blockDimx + threadIdx)/warpSize
  ;(define gid (get-global-threadId threadId blockID))
  (define gid (+ (* blockID blockDim) threadId))
  (define localId (get-idInWarp threadId))
  (define offset (- gid localId))
  (global-to-local I I-cached
                 (x-y-z 1)
                 offset
                 (x-y-z (+ warpSize 2)) #f #:round 2)

  (define o (create-accumulator (list * +) identity blockDim))

  (for/bounded ([i 3])
    (let* ([index (ite (?cond (@dup i) localId) (@dup 0) (@dup 1))]
           [lane (?fan localId warpSize
                            i warpSize [])] 
           [x (shfl (get I-cached index) lane)]
           [w (@dup (get W i))])
      (accumulate o (list w x) #:pred (?cond localId (@dup i))) ; (?cond localId (@dup i))
      ))
  
  (reg-to-global o O gid)
  )


(define (test)
  (for ([w (list 32)])
    (let ([ret (run-with-warp-size conv1d-spec conv1d w)])
      (pretty-display `(test ,w ,ret))))
  )
;(test)

(define (synthesis)
  (pretty-display "solving...")
  (define sol
    (time (solve
           (assert (andmap
                    (lambda (w) (run-with-warp-size conv1d-spec conv1d-sketch w))
                    (list 32))))))
  (print-forms sol)
  ;(print-lane 'lane (evaluate my-lane sol) '#(localId i) '#())
  )
(define t0 (current-seconds))
(synthesis)
(define t1 (current-seconds))
(- t1 t0)

(define (load-synth)
  (define-values (block-size I-sizes O-sizes I O O*)
    (create-IO 4))
  
  ;; Store
  (define (conv1d-store threadId blockId blockDim O)
    (define warpID (get-warpId threadId))
    (define o
      (for/vector ([w  warpID]
                   [t threadId])
        (ID t w blockId)))
    (reg-to-global o O (get-global-threadId threadId blockId))
    )
  
  ;; Run spec
  (conv1d-spec I O O-sizes)
  
  ;; Collect IDs
  (define IDs (create-matrix O-sizes))
  (run-kernel conv1d-store (x-y-z block-size) (x-y-z n-block) IDs)
  (define-values (threads warps blocks) (get-grid-storage))
  (collect-inputs O IDs threads warps blocks)
  (define n-regs (num-regs warps I))
  (pretty-display `(n-regs ,n-regs))
  
  ;; Load
  (define (conv1d-load threadId blockId blockDim I warp-input-spec)
    (define warpId (get-warpId threadId))
    ;; sketch starts
    (define I-cached (create-matrix-local (x-y-z n-regs)))
    (global-to-local I I-cached
                        (x-y-z (??)) ;; stride
                        (x-y-z (?warp-offset [(get-x blockId) (get-x blockDim)] [warpId warpSize])) ;; offset
                        (x-y-z (?warp-size warpSize 1)) ;; load size
                        #f)
    
    ;; sketch ends
    (check-warp-input warp-input-spec I I-cached warpId blockId)
    )
  
  (run-kernel conv1d-load (x-y-z block-size) (x-y-z n-block) I warps)
  (define sol (time (solve (assert #t))))
  (when (sat? sol)
    (print-forms sol)
    #;(define sol-hash (match sol [(model m) m]))
    #;(for ([key-val (hash->list sol-hash)])
      (let ([key (car key-val)]
            [val (cdr key-val)])
        (when (string-contains? (format "~a" key) "stencil:115") ;; stride
          (assert (not (equal? key val)))
          (pretty-display `(v ,key ,val ,(string-contains? (format "~a" key) "stencil:113")))))
      ))
  )
;(load-synth)