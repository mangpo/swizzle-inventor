#lang rosette

(require rosette/lib/synthax)
(require "util.rkt" "cuda.rkt")
(provide ?? ?index ?lane ?cond ?ite
         ?warp-size ?warp-offset
         print-forms choose
         ID get-grid-storage collect-inputs check-warp-input num-regs vector-list-append
         gen-lane? interpret-lane print-lane inst)

(define-synthax ?cond
  ([(?cond x ...)
    (choose (@dup #t)
            ((choose < <= > >= =) (choose x ...) (choose x ...)))])
  )
   
(define-synthax (?ite x ... depth)
 #:base (choose x ... (@dup (??)))
 #:else (choose
         x ... (@dup (??))
         ;;((choose + -) (?index x ... (- depth 1)) (?index x ... (- depth 1)))
         (ite (?cond x ...)
              (?ite x ... (- depth 1))
              (?ite x ... (- depth 1)))))

;; old
(define-synthax (?lane x ... [c ...] depth)
 #:base (choose x ... (@dup 1))
 #:else (choose
         x ... (@dup 1)
         ((choose quotient modulo *) (?lane x ... [c ...] (- depth 1)) (choose (@dup c) ...))
         ((choose + -)
          (?lane x ... [c ...] (- depth 1))
          (?lane x ... [c ...] (- depth 1)))))

#;(define-synthax (?lane x ... [c ...] depth)
 #:base (choose x ... (@dup 1)
                ((choose quotient *) (choose x ...) (choose (@dup c) ...)))
 #:else (choose
         x ... (@dup 1)
         ((choose quotient *) (choose x ...) (choose (@dup c) ...))
         (modulo (?lane x ... [c ...] (- depth 1)) (choose (@dup c) ...))
         ((choose + -)
          (?lane x ... [c ...] (- depth 1))
          (?lane x ... [c ...] (- depth 1)))))

(define-synthax ?index
  ([(?index x ... [c ...] depth)
    (choose (?ite x ... depth)
            (?lane x ... [c ...] depth))])
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

;;;;;;;;;;;;;;;;; alternate encoding ;;;;;;;;;;;;;;;;;;;

(struct inst (op args))
(define opcodes '#((+ v cc)
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

(define (interpret-lane prog vars consts)
  (define len (vector-length prog))
  (define consts-ext (vector-append consts default-consts))
  (define intermediates (make-vector len #f))

  (define (interpret-inst i p)
    ;(pretty-display `(interpret-inst ,i ,p))
    (define inst (vector-ref opcodes (inst-op p)))
    (define op (car inst))
    (define args-type (cdr inst))
    (define args (inst-args p))
    ;(pretty-display `(op ,op ,args-type ,args))

    (define (exe f)
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
      (define res (apply f operands))
      ;(pretty-display `(res ,i ,res ,intermediates))
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
