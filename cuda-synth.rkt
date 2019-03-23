#|
 | Copyright (c) 2018-2019, University of California, Berkeley.
 | Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
 |
 | Redistribution and use in source and binary forms, with or without 
 | modification, are permitted provided that the following conditions are met:
 |
 | 1. Redistributions of source code must retain the above copyright notice, 
 | this list of conditions and the following disclaimer.
 |
 | 2. Redistributions in binary form must reproduce the above copyright notice, 
 | this list of conditions and the following disclaimer in the documentation 
 | and/or other materials provided with the distribution.
 |
 | THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 | AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 | IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 | ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
 | LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 | CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 | SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
 | INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 | CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 | ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 | POSSIBILITY OF SUCH DAMAGE.
 |#

#lang rosette

(require rosette/lib/synthax)
(require "util.rkt" "cuda.rkt")
(provide ?? ?lane ?lane-mod
         ?sw-xform ?sw-xform-easy ?sw-xform-extra
         ?cond ?cond-easy
         ?warp-size ?warp-offset
         print-forms choose
         ID get-grid-storage collect-inputs check-warp-input num-regs vector-list-append
         unique unique-warp unique-list)

;; Condition swizzle (easy template, smaller search space)
(define-synthax ?cond-easy
  ([(?cond-easy x ...)
    (choose #t #f
            ((choose < <= > >= =) (choose x ...) (choose x ...)))]

   ))

;; Condition swizzle (full template)
(define-synthax ?cond
  ([(?cond x ... #:mod mod)
    (choose #t #f
            ((choose < <= > >= =)
             (choose x ...)
             (modulo (+ (* (??) (choose x ...)) (??)) mod)
             ))]

   [(?cond x ... [c ...])
    (choose #t #f
            ((choose < <= > >= =) (choose x ...)
                                  ((choose + -) (?const warpSize c ...) (choose x ...))))]
   
   [(?cond x ...)
    (choose #t #f
            ((choose < <= > >= =) (choose x ...)
                                  ((choose + -) (?const warpSize) (choose x ...))))]

   ))

(define-synthax (?lane-c x ... [c ...] depth)
 #:base (choose x ... (@dup (?const c ...)))
 #:else (choose
         x ... (@dup (?const c ...))
         ((choose quotient modulo *) (?lane x ... [c ...] (- depth 1)) (?const c ...))
         ((choose + -)
          (?lane-c x ... [c ...] (- depth 1))
          (?lane-c x ... [c ...] (- depth 1)))))

(define-synthax (?lane x ... depth)
 #:base (choose x ... (@dup (??)))
 #:else (choose
         x ... (@dup (??))
         ((choose quotient modulo *) (?lane x ... (- depth 1)) (??))
         ((choose + -)
          (?lane x ... (- depth 1))
          (?lane x ... (- depth 1)))))

;; Naive template for transformation index swizzle
(define-synthax ?lane-mod
  ([(?lane-mod x ... depth n [c ...])
    (modulo (?lane-c x ... [c ...] depth) n)]
  
   [(?lane-mod x ... depth n)
    (modulo (?lane x ... depth) n)]

   ))

;; Proposed template for transformation index swizzle (easy template, smaller search space)
(define-synthax ?sw-xform-easy
  ([(?sw-xform-easy eid n k m)
    (sw-xform eid n (??) n n 1 ;(choose 1 -1)
         k m (??) m (??))]

   [(?sw-xform-easy eid n k m #:fw conf-fw)
    (sw-xform eid n (??) n n conf-fw
         k m (??) m (??))]

   [(?sw-xform-easy eid n k m [c ...])
    (sw-xform eid n (?const c ...) n n 1 ;(choose 1 -1)
         k m (?const c ...) m (?const m c ...))]

   [(?sw-xform-easy eid n k m [c ...] #:fw conf-fw)
    (sw-xform eid n (?const c ...) n n conf-fw
         k m (?const c ...) m (?const m c ...))]
   )
  )

;; Proposed template for transformation index swizzle (full template)
(define-synthax ?sw-xform
  ([(?sw-xform eid n k m)
    (sw-xform eid n (??) (??) (??) (choose 1 -1)
         k m (??) (??) (??))]

   [(?sw-xform eid n k m #:fw conf-fw)
    (sw-xform eid n (??) (??) (??) conf-fw
         k m (??) (??) (??))]

   [(?sw-xform eid n k m [c ...])
    (sw-xform eid n (?const c ...) (?const n c ...) (?const n c ...) (choose 1 -1)
         k m (?const c ...) (?const m c ...) (?const m c ...))]

   [(?sw-xform eid n k m [c ...] #:fw conf-fw)
    (sw-xform eid n (?const c ...) (?const n c ...) (?const n c ...) conf-fw
         k m (?const c ...) (?const m c ...) (?const m c ...))]
   )
  )

;; Proposed template for transformation index swizzle (advanced template, bigger search space)
(define-synthax ?sw-xform-extra
  ([(?sw-xform-extra eid n k m)
    (sw-xform eid n (??) (??) (??) (choose 1 -1)
         k m (??) (??) (??)
         #:gcd (??) #:ecr (??) #:ec (??)
         )]))

(define-synthax ?const
  ([(?const c ...)
    (choose 0 1 -1 c ...)])
  )

(define-synthax ?const-
  ([(?const c ...)
    (choose 0 1 c ... -1 (- 0 c) ...)])
  )

(define-synthax (?warp-size-const x ... depth)
 #:base (choose x ... (??))
 #:else (choose
         x ... (??)
         ((choose + -)
          (?warp-size-const x ... (- depth 1))
          (?warp-size-const x ... (- depth 1)))))

(define-synthax (?warp-size x ... depth)
 #:base (choose x ...)
 #:else (choose
         x ...
         ((choose + -)
          (?warp-size x ... (- depth 1))
          (?warp-size-const x ... (- depth 1)))
         (-
          (?warp-size-const x ... (- depth 1))
          (?warp-size x ... (- depth 1)))
         (* (??) (?warp-size x ... (- depth 1)))
         ))

(define-synthax ?warp-offset
  ([(?warp-offset [id size] ...)
    (+ (??) (* (??) id size) ...)])
  )


;;;;;;;;;;;;;;;;; for data loading synthesis ;;;;;;;;;;;;;;;;;;;;
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

      [(and (accumulator? o) (ID? id))
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
          ;(pretty-display `(check ,i ,n ,x ,(list? (member x my-input))))
          (assert (member x my-input))))
    )))


;;;;;;;;;;;;;;;;;;;;;;;;;;; constraint ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(define (unique-warp lane)
  ;(pretty-display `(unique-warp ,lane))
  (define len (vector-length lane))
  (for ([o (quotient len warpSize)])
    (let ([offset (* o warpSize)])
      (let ([l (for/list ([i warpSize])
                 (vector-ref lane (+ offset i)))])
        (apply distinct? l)))))

(define (unique lane)
  ;(pretty-display `(unique ,lane))
  (define len (vector-length lane))
  (for ([i len])
    (let ([x (vector-ref lane i)])
      (for ([j (range (add1 i) len)])
        (let ([y (vector-ref lane j)])
          ;(pretty-display `(xy ,(+ offset i) ,(+ offset j) ,x ,y))
          (assert (not (= x y))))))))

(define (unique-list l)
  (apply distinct? l))
