#lang rosette

(require "util.rkt")

(define (drop@ name) 
  (if (regexp-match? #rx"^@.+$" name)
      (regexp-replace #rx"@" name "")
      name))

;(require (only-in rosette [+ p+] [* p*] [modulo p-modulo] [< p<] [<= p<=] [> p>] [>= p>=] [= p=] [if p-if]))
(require (only-in racket [sort %sort] [< %<]))
(provide (rename-out [@+ +] [@- -] [@* *] [@modulo modulo] [@< <] [@<= <=] [@> >] [@>= >=] [@= =] [@ite ite])
         define-shared
         global-to-shared shared-to-global global-to-warp-reg global-to-reg reg-to-global
         warpSize get-warpId get-idInWarp
         shfl
         accumulator create-accumulator accumulate get-accumulator-val normalize-accumulator
         run-kernel)


(define warpSize 4)

;;;;;;;;;;;;;;;;;;;;;;;;;;; lifted operations ;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define-syntax-rule (define-shared x exp) (define x exp))

(define (iterate x y op)
  (define (f x y)
    (cond
      [(and (vector? x) (vector? y)) (for/vector ([xi x] [yi y]) (f xi yi))]
      [(vector? x) (for/vector ([xi x]) (f xi y))]
      [(vector? y) (for/vector ([yi y]) (f x yi))]
      [(and (list? x) (list? y)) (for/list ([xi x] [yi y]) (f xi yi))]
      [(list? x) (for/list ([xi x]) (f xi y))]
      [(list? y) (for/list ([yi y]) (f x yi))]
      [else (op x y)]))
  (f x y))

(define (@ite c x y) ;; TODO
  (define (f c x y)
    (cond
      [(and (vector? c) (vector? x) (vector? y)) (for/vector ([ci c] [xi x] [yi y]) (f ci xi yi))]
      [(and (vector? c) (vector? x)) (for/vector ([ci c] [xi x]) (f ci xi y))]
      [(and (vector? c) (vector? y)) (for/vector ([ci c] [yi y]) (f ci x yi))]
      [(and (vector? x) (vector? y)) (for/vector ([xi x] [yi y]) (f c xi yi))]
      [(and (vector? c)) (for/vector ([ci c]) (f ci x y))]
      [(and (vector? x)) (for/vector ([xi x]) (f c xi y))]
      [(and (vector? y)) (for/vector ([yi y]) (f c x yi))]
      [else (if c x y)]))
  (f c x y))

(define-syntax-rule (define-operator my-op @op op)
  (begin
    (define (@ l)
      (cond
        [(= (length l) 1) (car l)]
        [(= (length l) 2) (iterate (first l) (second l) op)]
        [else (iterate (first l) (@ (cdr l)) op)]))
    (define my-op (lambda l (@ l)))
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


;(define-syntax-rule (> x y) (iterate x y p>))
;(define-syntax-rule (>= x y) (iterate x y p>=))
;(define-syntax-rule (< x y) (iterate x y p<))
;(define-syntax-rule (<= x y) (iterate x y p<=))
;(define-syntax-rule (= x y) (iterate x y p=))
;(define-syntax-rule (modulo x y) (iterate x y p-modulo))

;;;;;;;;;;;;;;;;;;;;;;;;;;; memory operations ;;;;;;;;;;;;;;;;;;;;;;;;;;;
(define (global-to-shared I I-shared pattern offset sizes
                          #:bounds [bounds (for/list ([i (length sizes)]) 1000000000)]
                          #:transpose [transpose #f])
  (cond
    [(= (length offset) 1)
     (let ([size-x (get-x sizes)]
           [bound-x (get-x bounds)]
           [offset-x (get-x offset)])
       (for ([i size-x])
         (when (< (+ offset-x i) bound-x)
           (set I-shared i (get I (+ offset-x i))))))]
    
    [(= (length offset) 2)
     (let ([size-x (get-x sizes)]
           [size-y (get-y sizes)]
           [bound-x (get-x bounds)]
           [bound-y (get-y bounds)]
           [offset-x (get-x offset)]
           [offset-y (get-y offset)])
       (for* ([y size-y] [x size-x])
         (when (and (< (+ offset-x x) bound-x) (< (+ offset-y y) bound-y))
           (if transpose
               (set I-shared y x (get I (+ offset-x x) (+ offset-y y)))
               (set I-shared x y (get I (+ offset-x x) (+ offset-y y)))))))]
    
    [(= (length offset) 3)
     (let ([size-x (get-x sizes)]
           [size-y (get-y sizes)]
           [size-z (get-z sizes)]
           [bound-x (get-x bounds)]
           [bound-y (get-y bounds)]
           [bound-z (get-z bounds)]
           [offset-x (get-x offset)]
           [offset-y (get-y offset)]
           [offset-z (get-z offset)])
       (for* ([z size-z] [y size-y] [x size-x])
         (when (and (< (+ offset-x x) bound-x) (< (+ offset-y y) bound-y) (< (+ offset-z z) bound-z))
           (if transpose
               (set I-shared z y x (get I (+ offset-x x) (+ offset-y y) (+ offset-z z)))
               (set I-shared x y z (get I (+ offset-x x) (+ offset-y y) (+ offset-z z)))))))]
    ))

(define (shared-to-global I-shared I pattern offset sizes 
                          #:bounds [bounds (for/list ([i (length sizes)]) 1000000000)]
                          #:transpose [transpose #f])
  (cond
    [(= (length offset) 1)
     (let ([size-x (get-x sizes)]
           [bound-x (get-x bounds)]
           [offset-x (get-x offset)])
       (for ([i size-x])
         (when (< (+ offset-x i) bound-x)
           (set I (+ offset-x i) (get I-shared i)))))]
    
    [(= (length offset) 2)
     (let ([size-x (get-x sizes)]
           [size-y (get-y sizes)]
           [bound-x (get-x bounds)]
           [bound-y (get-y bounds)]
           [offset-x (get-x offset)]
           [offset-y (get-y offset)])
       (for* ([y size-y] [x size-x])
         (when (and (< (+ offset-x x) bound-x) (< (+ offset-y y) bound-y))
           (if transpose
               (set I (+ offset-x y) (+ offset-y x) (get I-shared x y))
               (set I (+ offset-x x) (+ offset-y y) (get I-shared x y))))))]
    
    [(= (length offset) 3)
     (let ([size-x (get-x sizes)]
           [size-y (get-y sizes)]
           [size-z (get-z sizes)]
           [bound-x (get-x bounds)]
           [bound-y (get-y bounds)]
           [bound-z (get-z bounds)]
           [offset-x (get-x offset)]
           [offset-y (get-y offset)]
           [offset-z (get-z offset)])
       (for* ([z size-z] [y size-y] [x size-x])
         (when (and (< (+ offset-x x) bound-x) (< (+ offset-y y) bound-y) (< (+ offset-z z) bound-z))
           (if transpose
               (set I (+ offset-x z) (+ offset-y y) (+ offset-z x) (get I-shared x y z))
               (set I (+ offset-x x) (+ offset-y y) (+ offset-z z) (get I-shared x y z))))))]
    ))

(define-syntax-rule
  (global-to-reg I I-reg offset bounds)
  (let* ([blockSize (vector-length offset)]
         [new-I-reg (make-vector blockSize #f)])
    (for ([t blockSize])
      (set new-I-reg t (clone I-reg)))
    (set! I-reg new-I-reg)
    (for ([i blockSize]
          [global-i offset])
      (when (for/and ([b bounds] [i global-i]) (< i b))
        (set I-reg i (get* I global-i))))))

(define-syntax-rule
  (reg-to-global I-reg I offset bounds)
  (let* ([blockSize (vector-length offset)])
    (for ([i blockSize]
          [global-i offset])
      (when (for/and ([b bounds] [i global-i]) (< i b))
        (set* I global-i (get I-reg i))))))

(define-syntax-rule 
  (global-to-warp-reg I I-reg pattern offset sizes blockDim bounds transpose)
  (cond
    [(= (length blockDim) 1)
     (let* ([size-x (get-x sizes)]
            [bound-x (get-x bounds)]
            [stride-x (get-x pattern)]
            [blockSize (apply * blockDim)]
            [iter-x (add1 (quotient (sub1 size-x) (* warpSize stride-x)))]
            [I-len (vector-length I)]
            [new-I-reg (make-vector blockSize #f)])
       (for ([t blockSize])
         (set new-I-reg t (clone I-reg)))
       (set! I-reg new-I-reg)
       (for* ([warp (quotient blockSize warpSize)])
         (let ([offset-x (get-x (get offset (* warp warpSize)))]
               [global-x 0])
           (pretty-display `(warp ,warp ,offset-x ,size-x))
           (for* ([it iter-x]
                  [t warpSize]
                  [my-i stride-x])
             (when (and (< global-x size-x)
                        (< (+ offset-x global-x) I-len)
                        (< (+ offset-x global-x) bound-x))
               (set (get I-reg (+ t (* warp warpSize))) ;; thead in a block
                    (+ my-i (* it stride-x)) ;; local index
                    (get I (+ offset-x global-x))))
             (set! global-x (add1 global-x))
             ))))]

    ;; TODO
    [else (raise "unimplemented")]
    ))

;;;;;;;;;;;;;;;;;;;;;;;;;;; shuffle operations ;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (shfl val lane)
  (define len (vector-length val))
  (define res (make-vector len #f))
  (define lane-vec
    (if (vector? lane)
        (for/vector ([l lane]) (modulo l warpSize))
        (for/vector ([i len]) (modulo lane warpSize))))
  (for ([iter (quotient (vector-length val) warpSize)])
    (let ([offset (* iter warpSize)])
      (for ([i warpSize])
        (let ([i-dest (+ offset i)]
              [i-src (+ offset (get lane-vec (+ offset i)))])
        (set res i-dest (get val i-src))))))
  res)

;;;;;;;;;;;;;;;;;;;;;;;;;;; special accumulators ;;;;;;;;;;;;;;;;;;;;;;;;;;;
(define (acc=? x y recursive-equal?)
  (and (equal? (accumulator-val x) (accumulator-val y))
       (equal? (accumulator-oplist x) (accumulator-oplist y))
       (equal? (accumulator-opfinal x) (accumulator-opfinal y))))

(define (acc-hash-1 x recursive-equal-hash)
    (+ (* 10007 (equal-hash-code (accumulator-val x)))
       (* 101 (equal-hash-code (accumulator-oplist x)))
       (* 3 (equal-hash-code (accumulator-opfinal x)))))

(define (acc-hash-2 x recursive-equal-hash)
    (+ (* 101 (equal-secondary-hash-code (accumulator-val x)))
       (* 3 (equal-secondary-hash-code (accumulator-oplist x)))
       (* 10007 (equal-secondary-hash-code (accumulator-opfinal x)))))

(struct accumulator (val oplist opfinal veclen) #:mutable
  #:methods gen:equal+hash
  [(define equal-proc acc=?)
   (define hash-proc  acc-hash-1)
   (define hash2-proc acc-hash-2)])

(define-syntax create-accumulator
  (syntax-rules ()
    ((create-accumulator o op-list final-op)
     (accumulator (list) op-list final-op #f))
    ((create-accumulator o op-list final-op blockDim)
     (build-vector (apply * blockDim)
                   (lambda (i) (accumulator (list) op-list final-op (apply * blockDim)))))))

(define-syntax-rule (get-accumulator-val x)
  (if (vector? x)
      (for/vector ([xi x]) (accumulator-val xi))
      (accumulator-val x)))

(define (normalize-accumulator x)
  (if (accumulator? x)
      (set-accumulator-val!
       x
       (%sort (accumulator-val x) (lambda (x y) (string<? (format "~a" x) (format "~a" y)))))
      (for ([acc x]) (normalize-accumulator acc))))

(define (vector-of-list l veclen)
  (for/vector ([i veclen])
    (let ([each (map (lambda (x) (if (vector? x) (get x i) x)) l)])
      (%sort each (lambda (x y) (string<? (format "~a" x) (format "~a" y)))))))

(define (accumulate x val-list #:pred [pred #t])
  (define (f val-list op-list veclen)
    (if (= (length op-list) 1)
        (begin
          (assert (or (number? val-list) (vector? val-list)))
          (if (or (vector? val-list) (equal? veclen #f))
              val-list
              (for/vector ([i veclen]) val-list)))
        (let ([l (for/list ([val val-list])
                   (f val (cdr op-list) veclen))])
          (if veclen
              (vector-of-list l veclen)
              l))))

  (cond
    [(vector? x)
     (define veclen (accumulator-veclen (get x 0)))
     (define addition (f val-list (accumulator-oplist (get x 0)) veclen))
     (define pred-vec (if (vector? pred) pred (for/vector ([i veclen]) pred)))
     (for ([acc x]
           [add addition]
           [p pred-vec])
       (when p
         (set-accumulator-val! acc (cons add (accumulator-val acc)))))]

    [pred
     (define add (f val-list (accumulator-oplist x) #f))
     (set-accumulator-val! x (cons add (accumulator-val x)))
     ]))

;;;;;;;;;;;;;;;;;;;;;;;;;;; run kernel ;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (get-warpId threadID blockDim)
  (if (list? threadID)
      (let ([sum 0])
        (for ([id (reverse threadID)]
              [dim (cdr (reverse (cons 1 blockDim)))])
          (set! sum (+ sum (* id dim))))
        (quotient sum warpSize))
      (for/vector ([id threadID])
        (get-warpId id blockDim))))

(define (get-idInWarp threadID blockDim)
  (if (list? threadID)
      (let ([sum 0])
        (for ([id (reverse threadID)]
              [dim (cdr (reverse (cons 1 blockDim)))])
          (set! sum (+ sum (* id dim))))
        (modulo sum warpSize)) ;; differ only here
      (for/vector ([id threadID])
        (get-idInWarp id blockDim))))

(define (get-threadId sizes)
  (define ret (list))
  (define (rec id sizes)
    (if (empty? sizes)
        (set! ret (cons id ret))
        (for ([i (car sizes)])
          (rec (cons i id) (cdr sizes)))))
  (rec (list) (reverse sizes))
  (list->vector (reverse ret)))

(define (run-grid kernel gridDim blockDim threadIds args)
  (define (f blockID sizes)
    (if (empty? sizes)
        (begin
          (pretty-display `(blockID ,blockID ,blockDim ,threadIds))
          (apply kernel (append (list threadIds blockID blockDim) args)))
        (for ([i (car sizes)])
          (f (cons i blockID) (cdr sizes)))))
  (f (list) (reverse gridDim)))
    

(define-syntax-rule (run-kernel kernel blockDim gridDim x ...)
  (let ([Ids (get-threadId blockDim)])
    (run-grid kernel gridDim blockDim Ids (list x ...))))


(define (test-transpose1)
  (define I (create-matrix (x-y-z 4 4) (lambda () 0)))
  (for* ([y 4] [x 4]) (set I x y (+ x (* 10 y))))
  (define I-shared (create-matrix (x-y-z 3 2) (lambda () 0)))
  (global-to-shared I I-shared #f (x-y-z 2 1) (x-y-z 2 3) #:transpose #t)
  (pretty-display `(I ,I))
  (pretty-display `(I-shared ,I-shared))
  (assert (equal? I-shared #(#(12 22 32) #(13 23 33))) 'test-transpose1)
  )

(define (test-transpose2)
  (define I-shared #(#(12 22 32) #(13 23 33)))
  (define I (create-matrix (x-y-z 4 4) (lambda () 0)))
  (shared-to-global I-shared I #f (x-y-z 2 1) (x-y-z 3 2) #:transpose #t)
  (pretty-display `(I ,I))
  (pretty-display `(I-shared ,I-shared))
  (assert (equal? I #(#(0 0 0 0) #(0 0 12 13) #(0 0 22 23) #(0 0 32 33))) 'test-transpose2)
  )

;(test-transpose2)
     
  