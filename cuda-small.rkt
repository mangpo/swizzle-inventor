#lang rosette

(require rosette/lib/synthax)
(provide (all-defined-out)
         #;(rename-out [@+ +] [@- -] [@* *] [@modulo modulo] [@quotient quotient] [@< <] [@<= <=] [@> >] [@>= >=] [@= =]))

(current-bitwidth 8)

(define warpSize 32)
(define (@dup x) (for/vector ([i warpSize]) x))

(define (iterate x y op)
  (define (f x y)
    (cond
      [(and (vector? x) (vector? y)) (for/vector ([i (vector-length x)]) (f (vector-ref x i) (vector-ref y i)))]
      [(vector? x) (for/vector ([i (vector-length x)]) (f (vector-ref x i) y))]
      [(vector? y) (for/vector ([i (vector-length y)]) (f x (vector-ref y i)))]
      [(and (list? x) (list? y)) (map f x y)]
      [(list? x) (map (lambda (xi) (f xi y)) x)]
      [(list? y) (map (lambda (yi) (f x yi)) y)]
      [else (op x y)]))
  (f x y))

(define-syntax-rule (define-operator my-op @op op)
  (begin
    (define (@op l)
      (cond
        [(= (length l) 1) (car l)]
        [(= (length l) 2)
         ;(when (equal? `@op `$<) (pretty-display `(@op ,l)))
         (iterate (first l) (second l) op)]
        [else (iterate (first l) (@op (cdr l)) op)]))
    (define my-op (lambda l (@op l)))
    ))

(define-operator @+ $+ +)
(define-operator @- $- -)
(define-operator @* $* *)
(define-operator @> $> >)
(define-operator @>= $>= >=)
(define-operator @< $< <)
(define-operator @<= $<= <=)
(define-operator @= $= =)
(define-operator @modulo $modulo modulo)
(define-operator @quotient $quotient quotient)
