#lang racket

(require "codegen.rkt")

(define func
  '(define (mult64 threadId blockID blockDim A B C n)
  (define block-offset (* blockID blockDim))
  (define globalID (+ threadId block-offset))
  (define warpId (get-warpId threadId))
  (define a-cached (create-matrix-local (x-y-z 2 1)))
  (define b-cached (create-matrix-local (x-y-z 2 1)))
  (global-to-local A a-cached
                   (x-y-z 1 1) ;; stride
                   (* (quotient globalID (x-y-z warpSize 1))
                      (x-y-z n 1))
                   (x-y-z n 1)
                   #f #:warp-shape (x-y-z warpSize 1) #:round (x-y-z 2 1) #:size n)
  (global-to-local B b-cached
                   (x-y-z 1 1) ;; stride
                   (* (quotient globalID (x-y-z warpSize 1))
                      (x-y-z n 1))
                   (x-y-z n 1)
                   #f #:warp-shape (x-y-z warpSize 1) #:round (x-y-z 2 1) #:size n)
  
  (define tidx (get-idInWarp threadId))
  (define acc1 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc2 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc3 (create-accumulator (list bvand bvxor) identity blockDim))
  (define acc4 (create-accumulator (list bvand bvxor) identity blockDim))

  (for ([i warpSize])
    (let* ([lane-a1 (fan tidx warpSize 0 warpSize warpSize 1
                        i warpSize 1 warpSize)]
           [lane-a2 (fan tidx warpSize 0 warpSize warpSize 1
                        i warpSize 1 warpSize)]
           [lane-b1 (fan tidx warpSize 1 warpSize warpSize 1
                        i warpSize (- warpSize 1) warpSize)]
           [lane-b2 (fan tidx warpSize 1 warpSize warpSize 1
                        i warpSize (- warpSize 1) warpSize)]
           [a1 (shfl (get a-cached (@dup 0) (@dup 0)) lane-a1)]
           [a2 (shfl (get a-cached (@dup 1) (@dup 0)) lane-a2)]
           [b1 (shfl (get b-cached (@dup 0) (@dup 0)) lane-b1)]
           [b2 (shfl (get b-cached (@dup 1) (@dup 0)) lane-b2)]
          )
      (accumulate acc1 (list a1 b1) #:pred (<= i tidx))
      
      (accumulate acc2 (list a1 b1) #:pred (> i tidx))
      (accumulate acc2 (list a1 b2) #:pred (<= i tidx))
      (accumulate acc2 (list a2 b1) #:pred (<= i tidx))
      
      (accumulate acc3 (list a1 b2) #:pred (> i tidx))
      (accumulate acc3 (list a2 b1) #:pred (> i tidx))
      (accumulate acc3 (list a2 b2) #:pred (<= i tidx))
      
      (accumulate acc4 (list a2 b2) #:pred (> i tidx))
      ))

  (reg-to-global acc1 C (+ block-offset threadId))
  (reg-to-global acc2 C (+ block-offset (@dup (x-y-z warpSize 0)) threadId))
  (reg-to-global acc3 C (+ block-offset (@dup (x-y-z (* 2 warpSize) 0)) threadId))
  (reg-to-global acc4 C (+ block-offset (@dup (x-y-z (* 3 warpSize) 0)) threadId))
  ))
  
(define loop
  '(for
    ((i struct-size))
    (let* ((lane1
            (+ (* (quotient localId 4) 4)
               (modulo (- localId i) 4)
             ))
           (x (shfl (get I-cached (@dup i)) lane1)))
      (set temp (@dup i) x))))

(define fan
  '(define lane (fan i struct-size 0 1 2 1 localId warpSize 0 1 #:offset 0)))

(print-cuda (racket2cuda func 2))
;(print-cuda (convert-statement loop))
;(print-cuda (convert-statement fan))