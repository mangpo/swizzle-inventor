#lang rosette

(require "util.rkt" "cuda.rkt" "cuda-synth.rkt")

(set-warpSize 4)
(define struct-size 2)
(define n-block 2)
(define block-size (* 2 warpSize))
(define array-size (* n-block block-size))
(define I-sizes (x-y-z (* array-size struct-size)))
(define O-sizes (x-y-z array-size))
(define I (create-matrix I-sizes gen-uid))
(define O (create-matrix O-sizes))
(define O* (create-matrix O-sizes))

(define c (gcd struct-size warpSize))
(define a (/ struct-size c))
(define b (/ warpSize c))

(define (AOS-sum-spec I O O-sizes)
  (for ([i array-size])
    (let ([o (create-accumulator o (list +) identity)])
      (for ([j struct-size])
        (accumulate o (get I (+ (* i struct-size) j))))
      (set O i o)))
  )

(define (AOS-sum-slow threadId blockID blockDim I O)
  (define I-cached (create-matrix (x-y-z struct-size)))
  (define warpID (get-warpId threadId))
  (define offset (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))  ;; warpID = (threadIdy * blockDimx + threadIdx)/warpSize
  (define gid (get-global-threadId threadId blockID))
  (global-to-warp-reg I I-cached
                 (x-y-z struct-size)
                 offset (x-y-z (* warpSize struct-size)) I-sizes #f)

  (define localId (get-idInWarp threadId))
  (define o (create-accumulator o (list +) identity blockDim))
  (for ([i struct-size])
    (let* ([index (@dup i)]
           [lane localId]
           [x (shfl (get I-cached index) lane)])
      (accumulate o x)
      ))
  (reg-to-global o O gid O-sizes)
  )

(define (AOS-sum-fast threadId blockID blockDim I O)
  (define I-cached (create-matrix (x-y-z struct-size)))
  (define warpID (get-warpId threadId))
  (define offset (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))  ;; warpID = (threadIdy * blockDimx + threadIdx)/warpSize
  (define gid (get-global-threadId threadId blockID))
  (global-to-warp-reg I I-cached
                 (x-y-z 1)
                 offset (x-y-z (* warpSize struct-size)) I-sizes #f)
  (pretty-display `(I-cached ,I-cached))

  (define localId (get-idInWarp threadId))
  (define o (create-accumulator o (list +) identity blockDim))
  (for ([i struct-size])
    (let* ([index (modulo (+ localId i) struct-size)]
           [lane (+ (modulo (+ i (quotient localId b)) struct-size) (* localId struct-size))]
           [x (shfl (get I-cached index) lane)])
      (pretty-display `(lane ,lane))
      (accumulate o x)
      ))
  (reg-to-global o O gid O-sizes)
  )

(define (test)
  (AOS-sum-spec I O O-sizes)
  (run-kernel AOS-sum-slow (x-y-z block-size) (x-y-z n-block) I O*)
  ;(run-kernel AOS-sum-fast (x-y-z block-size) (x-y-z n-block) I O*)
  (pretty-display (acc-equal? O O*)))


;; index depth 1, lane depth 4, ??: > 5 min
;; index depth 1, lane depth 4, fixed constants (a b c): 8 s
;; warpsize 6: 527 s
;; warpsize 8: ?
(define (AOS-sum-sketch threadId blockID blockDim I O)
  (define I-cached (create-matrix (x-y-z struct-size)))
  (define warpID (get-warpId threadId))
  (define offset (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))  ;; warpID = (threadIdy * blockDimx + threadIdx)/warpSize
  (define gid (get-global-threadId threadId blockID))
  (global-to-warp-reg I I-cached
                 (x-y-z 1) ;(x-y-z struct-size)
                 offset (x-y-z (* warpSize struct-size)) I-sizes #f)

  (define localId (get-idInWarp threadId))
  (define o (create-accumulator o (list +) identity blockDim))
  (for/bounded ([i (??)])
    (let* ([index (modulo (?index localId (@dup i) 1) struct-size)]  ;(+ localId i)
           [lane (?lane localId (@dup i) [a b c struct-size] 4)]  ; (+ (modulo (+ i (quotient localId 2)) 2) (* localId 2))
           [x (shfl (get I-cached index) lane)])
      (accumulate o x #:pred (?cond localId (@dup i)))
      ))
  (reg-to-global o O gid O-sizes)
  )

;(define (synthesis)
  (AOS-sum-spec I O O-sizes)
  (run-kernel AOS-sum-sketch (x-y-z block-size) (x-y-z n-block) I O*)
  
  (pretty-display "solving...")
  (define sol
    (time
     (synthesize
      #:forall (symbolics I)
      #:guarantee (assert (acc-equal? O O*)))))
  (print-forms sol)
  ;)
;(synthesis)

(define (load-synth)
  ;; Store
  (define (AOS-sum-store threadId blockId blockDim O)
    (define warpID (get-warpId threadId))
    (define o
      (for/vector ([w  warpID]
                   [t threadId])
        (ID t w blockId)))
    (reg-to-global o O (get-global-threadId threadId blockId) O-sizes)
    )
  
  ;; Run spec
  (AOS-sum-spec I O O-sizes)
  
  ;; Collect IDs
  (define IDs (create-matrix O-sizes))
  (run-kernel AOS-sum-store (x-y-z block-size) (x-y-z n-block) IDs)
  (define-values (threads warps blocks) (get-grid-storage))
  (collect-inputs O IDs threads warps blocks)
  (define n-regs (num-regs warps I))
  (pretty-display `(n-regs ,n-regs))

  ;; Load
  (define (AOS-sum-load threadId blockId blockDim I warp-input-spec)
    (define warpId (get-warpId threadId))
    ;; sketch starts
    (define I-cached (create-matrix (x-y-z n-regs)))
    (global-to-warp-reg I I-cached
                        (x-y-z 1) ;; stride
                        (x-y-z (?warp-offset [(get-x blockId) (get-x blockDim)] [warpId warpSize])) ;; offset
                        (x-y-z (?warp-size warpSize 1)) ;; load size
                        I-sizes #f)
    ;; sketch ends
    (check-warp-input warp-input-spec I I-cached warpId blockId)
    )

  (run-kernel AOS-sum-load (x-y-z block-size) (x-y-z n-block) I warps)
  (define sol
    (time
     (synthesize
      #:forall (symbolics I)
      #:guarantee (assert #t))))
  (when (sat? sol)
    (print-forms sol))
  )
;(load-synth)