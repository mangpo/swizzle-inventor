#lang rosette

(require rosette/lib/synthax)
(require "cuda-small.rkt")

(define index-spec
  #(#(1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0)
    #(0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3)
    #(3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2)
    #(2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3 0 1)
    ))


(define struct-size 4)
(define c (gcd struct-size warpSize))
(define a (/ struct-size c))
(define b (/ warpSize c))

(define-synthax ?const
  ([(?const c ...)
    (choose 0 1 -1 c ...)])
  )

(define-synthax (?lane x ... [c ...] depth)
  #:base (@modulo (@+ (@* x (?const c ...)) ...
                      (@quotient x (?const c ...)) ...
                      (?const c ...))
                  (?const c ...))
  #:else (@+ (?lane x ... [c ...] 0)
             (?lane x ... [c ...] (- depth 1))))

(define (index-compute localId i)
  ;; Sat
  #;(@modulo (@+ (@* (@dup i) (?const a b c struct-size warpSize)) (@* localId (?const a b c struct-size warpSize))
               (@quotient (@dup i) (?const a b c struct-size warpSize)) (@quotient localId (?const a b c struct-size warpSize))
               (?const a b c struct-size warpSize))
           (?const a b c struct-size warpSize))
  ;; Unsat
  (?lane (@dup i) localId [a b c struct-size warpSize] 0)
  )

(define localId (for/vector ([i warpSize]) i))
(define sol (time
             (solve (assert
                     (andmap (lambda (i) (equal? (@modulo (index-compute localId i) 32)
                                                 (@modulo (vector-ref index-spec i) 32)))
                             (range 4))))))
(print-forms sol)
  