#|
 | Copyright (c) 2018-2019, University of California, Berkeley.
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
(provide ?? ?index ?lane ?lane-log ?lane-log-bv ?lane-mod1 ?lane-mod2 ?lane-mod3 ?fan
         ?cond ?ite ?const ?const32
         ?warp-size ?warp-offset
         print-forms choose
         ID get-grid-storage collect-inputs check-warp-input num-regs vector-list-append
         gen-lane? interpret-lane print-lane inst unique unique-warp unique-list)

   
(define-synthax (?ite x ... depth)
 #:base (choose x ... (@dup (??)))
 #:else (choose
         x ... (@dup (??))
         ;;((choose + -) (?index x ... (- depth 1)) (?index x ... (- depth 1)))
         (ite (?cond x ...)
              (?ite x ... (- depth 1))
              (?ite x ... (- depth 1)))))

(define-synthax (?lane-log x ... [c ...] depth)
 #:base (choose x ... (@dup (bv 1 BW)))
 #:else (choose
         x ... (@dup (bv 1 BW))
         ((choose bvshl bvlshr extract) (?lane-log x ... [c ...] (- depth 1)) (choose c ...))
         ((choose bvadd bvsub)
          (?lane-log x ... [c ...] (- depth 1))
          (?lane-log x ... [c ...] (- depth 1)))))

(define-synthax (?lane-log-bv x ... [c ...] depth)
 #:base (choose x ... (@dup (bv 1 BW)))
 #:else (choose
         x ... (@dup (bv 1 BW))
         ((choose bvshl bvlshr extract) (?lane-log-bv x ... [c ...] (- depth 1)) (choose (bv c BW) ...))
         ((choose bvadd bvsub)
          (?lane-log-bv x ... [c ...] (- depth 1))
          (?lane-log-bv x ... [c ...] (- depth 1)))))

(define-synthax (?lane x ... [c ...] depth)
 #:base (choose x ... (@dup 1))
 #:else (choose
         x ... (@dup 1)
         ((choose quotient modulo *) (?lane x ... [c ...] (- depth 1)) (choose (@dup c) ...))
         ((choose + -)
          (?lane x ... [c ...] (- depth 1))
          (?lane x ... [c ...] (- depth 1)))))

(define-synthax ?cond
  ([(?cond x ... #:mod mod)
    (choose #t #f
            ((choose < <= > >= =)
             (choose x ...)
             (modulo (+ (* (??) (choose x ...)) (??)) mod)
             ))]
   [(?cond x ...)
    (choose #t #f
            ((choose < <= > >= =) (choose x ...) (choose x ...)))]

   ))

(define-synthax ?fan
  ([(?fan eid n k m)
    (fan eid n (??) (??) (??) (choose 1 -1)
         k m (??) (??) #:offset (??) #:dg (??))]

   [(?fan eid n k m #:fw conf-fw)
    (fan eid n (??) (??) (??) conf-fw
         k m (??) (??) #:offset (??) #:dg (??))]

   [(?fan eid n k m [c ...])
    (fan eid n (?const c ...) (?const n c ...) (?const n c ...) (choose 1 -1)
         k m (?const c ...) (?const m c ...) #:offset (?const m c ...))]

   [(?fan eid n k m [c ...] #:fw conf-fw)
    (fan eid n (?const c ...) (?const n c ...) (?const n c ...) conf-fw
         k m (?const c ...) (?const m c ...) #:offset (?const m c ...))]
   )
  )

(define-synthax ?index
  ([(?index x ... [c ...] depth)
    (choose (?ite x ... depth)
            (?lane x ... [c ...] depth))])
  )

(define-synthax ?const
  ([(?const c ...)
    (choose 0 1 -1 c ...)])
  )

(define-synthax ?const-
  ([(?const c ...)
    (choose 0 1 c ... -1 (- 0 c) ...)])
  )

(define-synthax ?const32
  ([(?const32)
    (choose 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32)])
  )

#|
(define-synthax (?lane-mod1 x1 [c ...] depth)
  #:base (modulo (+ (+ (* x1 (?const- c ...)) (* (?const- c ...) (quotient x1 (?const c ...))))
                    (?const- c ...))
                 (?const c ...))
  #:else (+ (?lane-mod1 x1 [c ...] 0)
            (?lane-mod1 x1 [c ...] (- depth 1))))

(define-synthax (?lane-mod2 x1 x2 [c ...] depth)
  #:base (modulo (+ (+ (* x1 (?const- c ...)) (* (?const- c ...)(quotient x1 (?const c ...))))
                    (+ (* x2 (?const- c ...)) (* (?const- c ...)(quotient x2 (?const c ...))))
                    (?const- c ...))
                 (?const c ...))
  #:else (+ (?lane-mod2 x1 x2 [c ...] 0)
            (?lane-mod2 x1 x2 [c ...] (- depth 1))))

(define-synthax (?lane-mod3 x1 x2 x3 [c ...] depth)
  #:base (modulo (+ (+ (* x1 (?const- c ...)) (* (?const- c ...)(quotient x1 (?const c ...))))
                    (+ (* x2 (?const- c ...)) (* (?const- c ...)(quotient x2 (?const c ...))))
                    (+ (* x3 (?const- c ...)) (* (?const- c ...)(quotient x3 (?const c ...))))
                    (?const- c ...))
                 (?const c ...))
  #:else (+ (?lane-mod3 x1 x2 x3 [c ...] 0)
            (?lane-mod3 x1 x2 x3 [c ...] (- depth 1))))
|#

(define-synthax (?lane-mod1 x1 [c ...] depth)
  #:base (modulo (+ (+ (* x1 (?const c ...)) (quotient x1 (?const c ...)))
                    (?const c ...))
                 (?const c ...))
  #:else (+ (?lane-mod1 x1 [c ...] 0)
            (?lane-mod1 x1 [c ...] (- depth 1))))

(define-synthax (?lane-mod2 x1 x2 [c ...] depth)
  #:base (modulo (+ (+ (* x1 (?const c ...)) (quotient x1 (?const c ...)))
                    (+ (* x2 (?const c ...)) (quotient x2 (?const c ...)))
                    (?const c ...))
                 (?const c ...))
  #:else (+ (?lane-mod2 x1 x2 [c ...] 0)
            (?lane-mod2 x1 x2 [c ...] (- depth 1))))

(define-synthax (?lane-mod3 x1 x2 x3 [c ...] depth)
  #:base (modulo (+ (+ (* x1 (?const c ...)) (quotient x1 (?const c ...)))
                    (+ (* x2 (?const c ...)) (quotient x2 (?const c ...)))
                    (+ (* x3 (?const c ...)) (quotient x3 (?const c ...)))
                    (?const c ...))
                 (?const c ...))
  #:else (+ (?lane-mod3 x1 x2 x3 [c ...] 0)
            (?lane-mod3 x1 x2 x3 [c ...] (- depth 1))))

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

;;;;;;;;;;;;;;;;; alternate encoding ;;;;;;;;;;;;;;;;;;;

(struct inst (op args))
#;(define opcodes '#((+ v cc)
                   (+ v v)
                   (+ y cc)
                   (+ y v)
                   (+ y x)

                   (- v cc)
                   (- cc v)
                   (- v v)
                   (- y cc)
                   (- cc y)
                   (- y v)
                   (- v y)
                   (- y x)
                   (- x y)

                   (* v c)
                   (* y c)
                   (quotient v c)
                   (quotient y c)
                   (modulo v c)
                   (modulo y c)))
(define opcodes '#(+:vv +:vc
                        -:vv -:vc -:cv
                        *:vc quotient:vc modulo:vc))
(define default-consts '#(1))

(define (gen-lane? n)
  (define (gen-sym)
    (define-symbolic* x integer?)
    x)
  (define (gen-inst) (inst (gen-sym) (vector (gen-sym) (gen-sym))))
  (for/vector ([i n]) (gen-inst)))

(define (print-lane name prog vars consts)
  (define len (vector-length prog))
  (define consts-ext (vector-append consts default-consts))

  (define (print-inst i p)
    (define inst (vector-ref opcodes (inst-op p)))
    (define op (car inst))
    (define args-type (cdr inst))
    (define args (inst-args p))

    (define args-str
      (for/list ([type args-type]
                 [arg args])
        (cond
            [(equal? type 'c)  (vector-ref consts arg)]
            [(equal? type 'cc) (vector-ref consts-ext arg)]
            [(equal? type 'v)  (vector-ref vars arg)]
            [(equal? type 'y)  (string->symbol (format "x~a" (sub1 i)))]
            [(equal? type 'x)  (string->symbol (format "x~a" arg))])))

    (pretty-display (format "[x~a ~a]" i (cons op args-str)))
    )

  (pretty-display (format "~a:" name))
  (for ([p prog] [i (in-naturals)]) (print-inst i p))
  (pretty-display (format "[~a x~a]" name (sub1 len)))
  )

#;(define (interpret-lane prog vars consts)
  (define len (vector-length prog))
  (define consts-ext (vector-append consts default-consts))
  (define intermediates (make-vector len #f))

  (define (interpret-inst i p)
    (pretty-display `(interpret-inst ,i ,p))
    (define inst (vector-ref opcodes (inst-op p)))
    (define op (car inst))
    (define args-type (cdr inst))
    (define args (inst-args p))
    (pretty-display `(op ,op))

    (define (exe f)
      (pretty-display `(exe ,f))
      (define operands
        (for/list ([type args-type]
                   [arg args])
          ;(pretty-display `(operand ,type ,arg))
          (cond
            [(equal? type 'c)  (@dup (vector-ref consts arg))]
            [(equal? type 'cc) (@dup (vector-ref consts-ext arg))]
            [(equal? type 'v)  (vector-ref vars arg)]
            [(equal? type 'y)  (vector-ref intermediates (sub1 i))]
            [(equal? type 'x)  (if (< arg i) (vector-ref intermediates arg) (assert #f))])))
      (pretty-display `(before apply ,operands))
      ;;(define res (apply f operands))
      (define res
      (for*/all ([x (first operands)] [y (second operands)])
        (f x y)))
      ;;(define res (f (first operands) (second operands)))
      (pretty-display `(res ,i))
      (vector-set! intermediates i res)
      ;(pretty-display `(intermediates ,intermediates))
      )

    (define-syntax op-eq
      (syntax-rules ()
        ((op-eq x) (equal? x op))
        ((op-eq a b ...) (or (inst-eq a) (inst-eq b) ...))))
    
    (cond
      [(op-eq '+) (exe +)]
      [(op-eq '-) (exe -)]
      [(op-eq '*) (exe *)]
      [(op-eq 'quotient) (exe quotient)]
      [(op-eq 'modulo)   (exe modulo)]
      [else (assert #f)]
      )
    )

  (for ([p prog] [i (in-naturals)])
    (interpret-inst i p))
  (define ret (vector-ref intermediates (sub1 len)))
  ;(pretty-display `(ret ,ret ,intermediates ,(sub1 len)))
  ret
  )

(define (interpret-lane prog vars consts)
  (define len (vector-length prog))
  (define consts-ext (vector-append consts default-consts))
  (define intermediates (vector-append (make-vector len #f) vars))

  (define (interpret-inst i p)
    ;(pretty-display `(interpret-inst ,i ,p))
    (define op (vector-ref opcodes (inst-op p)))
    (define args (inst-args p))
    ;(pretty-display `(op ,op))

    (define (vv f)
      (define a (vector-ref intermediates (vector-ref args 0)))
      (define b (vector-ref intermediates (vector-ref args 1)))
      (f a b))

    (define (vc f)
      (define a (vector-ref intermediates (vector-ref args 0)))
      (define b (vector-ref consts-ext (vector-ref args 1)))
      (f a b))

    (define (cv f)
      (define a (vector-ref consts-ext (vector-ref args 0)))
      (define b (vector-ref intermediates (vector-ref args 1)))
      (f a b))

    (define (exe compute f)
      (vector-set! intermediates i (compute f)))

    (define-syntax op-eq
      (syntax-rules ()
        ((op-eq x) (equal? x op))
        ((op-eq a b ...) (or (inst-eq a) (inst-eq b) ...))))
    
    (cond
      [(op-eq '+:vv) (exe vv +)]
      [(op-eq '+:vc) (exe vc +)]
      [(op-eq '-:vv) (exe vv -)]
      [(op-eq '-:vc) (exe vc -)]
      [(op-eq '-:cv) (exe cv -)]
      [(op-eq '*:vc) (exe vc *)]
      [(op-eq 'quotient:vc) (vc quotient)]
      [(op-eq 'modulo:vc)   (vc modulo)]
      [else (assert #f)]
      )
    )

  (for ([p prog] [i (in-naturals)])
    (interpret-inst i p))
  (define ret (vector-ref intermediates (sub1 len)))
  ;(pretty-display `(ret ,ret ,intermediates ,(sub1 len)))
  ret
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
#;(define (unique-warp lane)
  ;(pretty-display `(unique-warp ,lane))
  (define len (vector-length lane))
  (for ([o (quotient len warpSize)])
    (let ([offset (* o warpSize)])
      (for ([i warpSize])
        (let ([x (vector-ref lane (+ offset i))])
          (for ([j (range (add1 i) warpSize)])
            (let ([y (vector-ref lane (+ offset j))])
              ;(pretty-display `(xy ,(+ offset i) ,(+ offset j) ,x ,y))
              (assert (not (= x y))))))))))

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
