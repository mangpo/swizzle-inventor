#lang rosette

(require rosette/lib/synthax)
(require "util.rkt" "cuda.rkt")
(provide ?? ?index ?lane ?cond
         ?warp-size ?warp-offset
         print-forms choose
         ID get-grid-storage collect-inputs check-warp-input num-regs vector-list-append)

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

(define-synthax (?warp-size x ... depth)
 #:base (choose x ... (??))
 #:else (choose
         x ... (??)
         ((choose + -)
          (?warp-size x ... (- depth 1))
          (?warp-size x ... (- depth 1)))))

(define-synthax ?warp-offset
  ([(?warp-offset [id size] ...)
    (+ (??) (* (??) id size) ...)])
  )

;;;;;;;;;;;;;;;;; load synthesis ;;;;;;;;;;;;;;;;;;;;
(struct ID (thread warp block))

(define (get-grid-storage)
  (define blocks (create-matrix (get-gridDim) list))
  (define warps (create-matrix (cons (/ blockSize warpSize) (get-gridDim)) list))
  (define threads (create-matrix (append (get-blockDim) (get-gridDim)) list))
  (values threads warps blocks))

(define (update-val M i v)
  (define current (get* M i))
  (define update (append v current))
  (set* M i update))

(define (collect-inputs O IDs threads warps blocks)
  (define (f o id)
    (cond
      [(vector? id)
       (for ([oi o] [idi id]) (f oi idi))]

      [(accumulator? o)
       (define vals (flatten (accumulator-val o)))
       (update-val blocks (ID-block id) vals)
       (update-val warps (cons (ID-warp id) (ID-block id)) vals)
       (update-val threads (append (ID-thread id) (ID-block id)) vals)]))
  (f O IDs))

(define (num-regs warps I)
  (define all-inputs (list->set (to-list I)))
  (define max-num 0)
  (define (f x)
    (cond
      [(vector? x) (for ([xi x]) (f xi))]
      [else
       (define n (set-count (set-intersect (list->set x) all-inputs)))
       (when (> n max-num) (set! max-num n))]))
  (f warps)
  (+ (quotient (- max-num 1) warpSize) 1))

(define (to-list x)
  (cond
    [(or (vector? x) (list? x)) (for/list ([xi x]) (to-list xi))]
    [else x]))

(define (vector-list-append x y)
  (cond
    [(and (vector? x) (vector? y)) (for/vector ([xi x] [yi y]) (vector-list-append xi yi))]
    [else (append x y)]))

(define (check-warp-input warp-input-spec I I-cached warpId blockId)
  (define all-inputs (to-list I))
  (define n (/ (vector-length warpId) warpSize))
  (define warp-input (create-matrix (list n) list))
  ;(pretty-display `(I-cached ,I-cached))
  (for ([my-input I-cached]
        [wid warpId])
    (let* ([current (get warp-input wid)]
           [update (append (to-list my-input) current)])
      (set warp-input wid update)))
  ;(pretty-display `(warp-input ,warp-input))
  
  (for ([i n]
        [my-input warp-input])
    (let ([spec (list->set (get* warp-input-spec (cons i blockId)))])
      (for ([x spec])
        (when (member x all-inputs)
          (assert (member x my-input))))
    )))
