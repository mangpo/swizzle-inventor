#lang rosette

(require rosette/lib/synthax)
(require "cuda.rkt")
(provide ?index ?lane ?cond print-forms choose)

(define-synthax ?cond
  ([(?cond x ...)
    (choose (@dup #t)
            ((choose < <= > >= =) (choose x ...) (choose x ...)))])
  )

(define-synthax (?index x ... depth)
 #:base (choose x ... (@dup (??)))
 #:else (choose
         x ... (@dup (??))
         (ite (?cond x ...)
              (?index x ... (- depth 1))
              (?index x ... (- depth 1)))))

(define-synthax (?lane x ... depth)
 #:base (choose x ... (@dup 1))
 #:else (choose
         x ... (@dup 1)
         ((choose + -)
          (?lane x ... (- depth 1))
          (?lane x ... (- depth 1)))))
