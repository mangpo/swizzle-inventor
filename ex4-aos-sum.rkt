#lang rosette

(require "util.rkt" "cuda.rkt" "cuda-synth.rkt")

(define struct-size 4)
(define n-block 2)

(define (create-IO warpSize)
  (set-warpSize warpSize)
  (define block-size (* 2 warpSize))
  (define array-size (* n-block block-size))
  (define I-sizes (x-y-z (* array-size struct-size)))
  (define O-sizes (x-y-z array-size))
  (define I (create-matrix I-sizes gen-uid))
  (define O (create-matrix O-sizes))
  (define O* (create-matrix O-sizes))
  (values block-size I-sizes O-sizes I O O*))

(define (run-with-warp-size spec kernel w)
  (define-values (block-size I-sizes O-sizes I O O*)
    (create-IO w))
  
  (define c (gcd struct-size warpSize))
  (define a (/ struct-size c))
  (define b (/ warpSize c))

  (spec I O O-sizes)
  (run-kernel kernel (x-y-z block-size) (x-y-z n-block) I O* I-sizes O-sizes a b c)
  ;(acc-print O*)
  (acc-equal? O O*))

(define (AOS-sum-spec I O O-sizes)
  (for ([i (get-x O-sizes)])
    (let ([o (create-accumulator o (list +) identity)])
      (for ([j struct-size])
        (accumulate o (get I (+ (* i struct-size) j))))
      (set O i o)))
  )

(define (AOS-sum-slow threadId blockID blockDim I O I-sizes O-sizes a b c)
  (define I-cached (create-matrix (x-y-z struct-size)))
  (define warpID (get-warpId threadId))
  (define offset (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))  ;; warpID = (threadIdy * blockDimx + threadIdx)/warpSize
  (define gid (get-global-threadId threadId blockID))
  (global-to-warp-reg I I-cached
                 (x-y-z struct-size)
                 offset (x-y-z (* warpSize struct-size)) #f)

  (define localId (get-idInWarp threadId))
  (define o (create-accumulator o (list +) identity blockDim))
  (for ([i struct-size])
    (let* ([index (@dup i)]
           [lane localId]
           [x (shfl (get I-cached index) lane)])
      (accumulate o x)
      ))
  (reg-to-global o O gid)
  )

(define (AOS-sum-fast threadId blockID blockDim I O I-sizes O-sizes a b c)
  (define I-cached (create-matrix (x-y-z struct-size)))
  (define warpID (get-warpId threadId))
  (define offset (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))  ;; warpID = (threadIdy * blockDimx + threadIdx)/warpSize
  (define gid (get-global-threadId threadId blockID))
  (global-to-warp-reg I I-cached
                 (x-y-z 1)
                 offset (x-y-z (* warpSize struct-size)) #f)
  ;(pretty-display `(I-cached ,I-cached))

  (define localId (get-idInWarp threadId))
  (define o (create-accumulator o (list +) identity blockDim))
  (for ([i struct-size])
    (let* ([index (modulo (+ i localId) struct-size)]
           [lane (+ (modulo (+ i (quotient localId b)) struct-size) (* localId struct-size))]
           [x (shfl (get I-cached index) lane)])
      ;(pretty-display `(lane ,lane))
      (accumulate o x)
      ))
  (reg-to-global o O gid)
  )

(define (AOS-sum-test2 threadId blockID blockDim I O I-sizes O-sizes a b c)
   (define I-cached (create-matrix (x-y-z struct-size)))
   (define warpID (get-warpId threadId))
   (define offset
     (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))
   (define gid (get-global-threadId threadId blockID))
   (global-to-warp-reg
    I
    I-cached
    (x-y-z 1)
    offset
    (x-y-z (* warpSize struct-size))
    #f)
   (define localId (get-idInWarp threadId))
   (define o (create-accumulator o (list +) identity blockDim))
   (for/bounded
    ((i 2))
    (let* ((index (modulo (- (@dup i) localId) struct-size))
           (lane
            (+
             (quotient
              (* (+ localId localId) (@dup struct-size))
              (@dup struct-size))
             (modulo
              (- (modulo (@dup i) (@dup b)) (quotient localId (@dup b)))
              (@dup struct-size))))
           (x (shfl (get I-cached index) lane)))
      (accumulate o x #:pred (@dup #t))))
   (reg-to-global o O gid))

(define (AOS-sum-test3 threadId blockID blockDim I O I-sizes O-sizes a b c)
   (define I-cached (create-matrix (x-y-z struct-size)))
   (define warpID (get-warpId threadId))
   (define offset
     (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))
   (define gid (get-global-threadId threadId blockID))
   (global-to-warp-reg
    I
    I-cached
    (x-y-z 1)
    offset
    (x-y-z (* warpSize struct-size))
    #f)
   (define localId (get-idInWarp threadId))
   (define o (create-accumulator o (list +) identity blockDim))
   (for/bounded
    ((i 3))
    (let* ((index
            (modulo
             (- (* (@dup i) (@dup warpSize)) (* localId (@dup warpSize)))
             struct-size))
           (lane
            (+
             (-
              (+ (quotient (@dup i) (@dup warpSize)) (@dup i))
              (-
               (quotient localId (@dup c))
               (quotient (@dup i) (@dup warpSize))))
             (+
              (+ (* (@dup i) (@dup b)) (* localId (@dup struct-size)))
              (+ (* localId (@dup warpSize)) (* localId (@dup c))))))
           (x (shfl (get I-cached index) lane)))
      (accumulate o x #:pred (= localId localId))))
   (reg-to-global o O gid))

(define (test)
  (for ([w (list 5 32)])
    (let ([ret (run-with-warp-size AOS-sum-spec AOS-sum-test3 w)])
      (pretty-display `(test ,w ,ret))))
  )
(test)

;; index depth 1, lane depth 4, ??: > 5 min
;; index depth 1, lane depth 4, fixed constants (a b c): 19 s
;; m = 3, warpsize 4 & 5 + warpSize choice: 73 s
;; change ?lane: 20 s
;; synthesize both load and shuffle: 30 s (8 bit), 50 s (16 bit)
(define (AOS-sum-sketch threadId blockID blockDim I O I-sizes O-sizes a b c)
  (define I-cached (create-matrix (x-y-z struct-size)))
  (define warpID (get-warpId threadId))
  (define offset (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))  ;; warpID = (threadIdy * blockDimx + threadIdx)/warpSize
  (define gid (get-global-threadId threadId blockID))
  (global-to-warp-reg I I-cached
                        (x-y-z 1) ;; stride
                        (x-y-z (?warp-offset [(get-x blockID) (get-x blockDim)] [warpID warpSize])) ;; offset
                        (x-y-z (?warp-size warpSize 1)) ;; load size
                        #f)

  (define localId (get-idInWarp threadId))
  (define o (create-accumulator o (list +) identity blockDim))
  (for/bounded ([i (??)])
    (let* ([index (modulo (?index localId (@dup i) [a b c struct-size warpSize] 2) struct-size)]  ; (?index localId (@dup i) 1)
           [lane (?lane localId (@dup i) [a b c struct-size warpSize] 4)]  ; (+ (modulo (+ i (quotient localId 2)) 2) (* localId 2))
           [x (shfl (get I-cached index) lane)])
      (accumulate o x #:pred (?cond localId (@dup i)))
      ))
  (reg-to-global o O gid)
  )

(define (synthesis)
  (pretty-display "solving...")
  (define sol (time (solve (assert (andmap (lambda (w) (run-with-warp-size AOS-sum-spec AOS-sum-sketch w))
                                     (list 4))))))
  (print-forms sol)
  )
(synthesis)

(define (load-synth)
  (define-values (block-size I-sizes O-sizes I O O*)
    (create-IO 4))
  
  ;; Store
  (define (AOS-sum-store threadId blockId blockDim O)
    (define warpID (get-warpId threadId))
    (define o
      (for/vector ([w  warpID]
                   [t threadId])
        (ID t w blockId)))
    (reg-to-global o O (get-global-threadId threadId blockId))
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
                        #f)
    ;; sketch ends
    (check-warp-input warp-input-spec I I-cached warpId blockId)
    )

  (run-kernel AOS-sum-load (x-y-z block-size) (x-y-z n-block) I warps)
  (define sol (time (solve (assert #t))))
    #;(time
     (synthesize
      #:forall (symbolics I)
      #:guarantee (assert #t)))
  (when (sat? sol)
    (print-forms sol))
  )
;(load-synth)