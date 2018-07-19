#lang racket

(require "codegen.rkt")

(define func
  '(define (mult64-sketch threadId blockID blockDim A B C n)
   (define globalID (+ threadId (* blockID blockDim)))
   (define a-cached (create-matrix-local (x-y-z 2 1)))
   (define b-cached (create-matrix-local (x-y-z 2 1)))
   (global-to-local
    A
    a-cached
    (x-y-z 1 1)
    (* (quotient globalID (x-y-z warpSize 1)) (x-y-z n 1))
    (x-y-z n 1)
    #f
    #:warp-shape
    (x-y-z warpSize 1)
    #:round
    (x-y-z 2 1))
   (global-to-local
    B
    b-cached
    (x-y-z 1 1)
    (* (quotient globalID (x-y-z warpSize 1)) (x-y-z n 1))
    (x-y-z n 1)
    #f
    #:warp-shape
    (x-y-z warpSize 1)
    #:round
    (x-y-z 2 1))
   (define tidx (get-idInWarp threadId))
   (define acc1 (create-accumulator (list bvand bvxor) identity blockDim))
   (define acc2 (create-accumulator (list bvand bvxor) identity blockDim))
   (define acc3 (create-accumulator (list bvand bvxor) identity blockDim))
   (define acc4 (create-accumulator (list bvand bvxor) identity blockDim))
   (for
    ((i warpSize))
    (let* ((lane-a1 (fan tidx warpSize 0 warpSize warpSize 1 i warpSize 1 warpSize #:offset 0))
           (lane-a2 (fan tidx warpSize 0 warpSize warpSize 1 i warpSize 0 1 #:offset 0))
           (lane-b1 (fan tidx warpSize 0 1 warpSize 1 i warpSize -1 warpSize #:offset 0))
           (lane-b2 (fan tidx warpSize 0 1 warpSize 1 i warpSize -1 warpSize #:offset 0))
           (idx-a1 (ite (= (@dup i) (+ 0 (@dup i))) (@dup 0) (@dup 1)))
           (idx-a2 (ite #f (@dup 0) (@dup 1)))
           (idx-b1 (ite (< tidx (- warpSize (@dup i))) (@dup 0) (@dup 1)))
           (idx-b2 (ite (>= (@dup i) (- warpSize tidx)) (@dup 0) (@dup 1)))
           (a1 (shfl (get a-cached idx-a1 (@dup 0)) lane-a1))
           (a2 (shfl (get a-cached idx-a2 (@dup 0)) lane-a2))
           (b1 (shfl (get b-cached idx-b1 (@dup 0)) lane-b1))
           (b2 (shfl (get b-cached idx-b2 (@dup 0)) lane-b2)))
      (accumulate acc1 (list a1 b1) #:pred (< (@dup i) (+ 1 tidx)))
      (accumulate acc3 (list a1 b1) #:pred (> (@dup i) (+ 0 tidx)))
      (accumulate acc2 (list a1 b2) #:pred (<= tidx (+ 1 tidx)))
      (accumulate acc4 (list a1 b2) #:pred #f)
      (accumulate acc2 (list a2 b1) #:pred (> tidx (+ -1 (@dup i))))
      (accumulate acc4 (list a2 b1) #:pred (>= (@dup i) (+ 1 tidx)))
      (accumulate acc1 (list a2 b2) #:pred #f)
      (accumulate acc3 (list a2 b2) #:pred (< (@dup i) (+ warpSize tidx)))))
   (reg-to-global acc1 C globalID)
   (reg-to-global acc2 C (+ globalID (@dup (x-y-z warpSize 0))))
   (reg-to-global acc3 C (+ globalID (@dup (x-y-z (* 2 warpSize) 0))))
   (reg-to-global acc4 C (+ globalID (@dup (x-y-z (* 3 warpSize) 0))))))
  

(print-cuda (racket2cuda func 2))
;(print-cuda (convert-statement loop))
;(print-cuda (convert-statement fan))