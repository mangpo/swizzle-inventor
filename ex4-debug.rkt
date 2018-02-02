#lang rosette

(require "util.rkt" "cuda.rkt" "cuda-synth.rkt")

(define struct-size 3)
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

  (reset-cost)
  (spec I O O-sizes)
  (pretty-display `(spec-cost ,(get-cost)))
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

(define (AOS-sum-sketch threadId blockID blockDim I O I-sizes O-sizes a b c)
  (define I-cached (create-matrix (x-y-z struct-size)))
  (define warpID (get-warpId threadId))
  (define offset (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))  ;; warpID = (threadIdy * blockDimx + threadIdx)/warpSize
  (define gid (get-global-threadId threadId blockID))
  (global-to-warp-reg I I-cached
                        (x-y-z 1) ;; stride
                        (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize))
                        (x-y-z (* warpSize struct-size))
                        #f)

  (define localId (get-idInWarp threadId))
  (define o (create-accumulator o (list +) identity blockDim))
  (for/bounded ([i struct-size])
    (let* ([index (?lane-mod2 (@dup i) localId [a b c struct-size warpSize] 0)]
           [lane (?lane-mod2 (@dup i) localId [a b c struct-size warpSize] 0)]
           [x (shfl (get I-cached index) lane)])
      (accumulate o x #:pred (?cond localId (@dup i)))
      ))
  
  (reg-to-global o O gid)
  )


(pretty-display "solving...")
(assert
 (andmap (lambda (w) (run-with-warp-size AOS-sum-spec AOS-sum-sketch w))
         (list 32)))
(define cost (get-cost))

;;;;;;;;;;;;;;; slow ;;;;;;;;;;;;;;;
(define solver (current-solver))
(solver-assert solver (asserts))
;(solver-minimize (current-solver) (list cost))
;(define sol (solver-check solver))

;;;;;;;;;;;;;;; fast ;;;;;;;;;;;;;;;
(define sol (time (solve (assert #t))))

;;;;;;;;;;;;;;; solution ;;;;;;;;;;;;;;;
(define this-cost (evaluate cost sol))
(print-forms sol)
(pretty-display `(cost ,this-cost))

(displayln "---- optimizing ----")
(define opt-sol (time (optimize #:minimize (list cost) #:guarantee (assert #t))))
(define opt-cost (evaluate cost opt-sol))
(print-forms opt-sol)
(pretty-display `(cost ,opt-cost))