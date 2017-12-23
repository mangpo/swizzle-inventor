#lang rosette

(require "util.rkt")

(define (drop@ name) 
  (if (regexp-match? #rx"^@.+$" name)
      (regexp-replace #rx"@" name "")
      name))

(require (only-in rosette [+ p+] [* p*] [modulo p-modulo] [< p<] [<= p<=] [> p>] [>= p>=] [= p=] [if p-if]))
(provide define-shared + * modulo < <= > >= = if
         global-to-shared shared-to-global
         global-to-reg
         warpSize get-warpId get-idInWarp
         shfl
         run-kernel)


(define warpSize 4)

;;;;;;;;;;;;;;;;;;;;;;;;;;; lifting operations ;;;;;;;;;;;;;;;;;;;;;;;;;;;

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

(define (if c x y)
  (define (f c x y)
    (cond
      [(and (vector? c) (vector? x) (vector? y)) (for/vector ([ci c] [xi x] [yi y]) (f ci xi yi))]
      [(and (vector? c) (vector? x)) (for/vector ([ci c] [xi x]) (f ci xi y))]
      [(and (vector? c) (vector? y)) (for/vector ([ci c] [yi y]) (f ci x yi))]
      [(and (vector? x) (vector? y)) (for/vector ([xi x] [yi y]) (f c xi yi))]
      [(and (vector? c)) (for/vector ([ci c]) (f ci x y))]
      [(and (vector? x)) (for/vector ([xi x]) (f c xi y))]
      [(and (vector? y)) (for/vector ([yi y]) (f c x yi))]
      [else (p-if c x y)]))
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

(define-operator + @+ p+)
(define-operator * @* p*)
(define-operator > @> p>)
(define-operator >= @>= p>=)
(define-operator < @< p<)
(define-operator <= @<= p<=)
#|
(define-operator = @= p=)
(define-operator modulo @modulo p-modulo)
|#

;(define-syntax-rule (> x y) (iterate x y p>))
;(define-syntax-rule (>= x y) (iterate x y p>=))
;(define-syntax-rule (< x y) (iterate x y p<))
;(define-syntax-rule (<= x y) (iterate x y p<=))
(define-syntax-rule (= x y) (iterate x y p=))
(define-syntax-rule (modulo x y) (iterate x y p-modulo))

;;;;;;;;;;;;;;;;;;;;;;;;;;; memory operations ;;;;;;;;;;;;;;;;;;;;;;;;;;;
(define (global-to-shared I I-shared pattern offset sizes #:transpose [transpose #f])
  (cond
    [(= (length offset) 1)
     (let ([size-x (get-x sizes)]
           [offset-x (get-x offset)])
       (for ([i size-x])
         (set I-shared i (get I (p+ offset-x i)))))]
    
    [(= (length offset) 2)
     (let ([size-x (get-x sizes)]
           [size-y (get-y sizes)]
           [offset-x (get-x offset)]
           [offset-y (get-y offset)])
       (for* ([y size-y] [x size-x])
         (p-if transpose
             (set I-shared y x (get I (p+ offset-x x) (p+ offset-y y)))
             (set I-shared x y (get I (p+ offset-x x) (p+ offset-y y))))))]
    
    [(= (length offset) 3)
     (let ([size-x (get-x sizes)]
           [size-y (get-y sizes)]
           [size-z (get-z sizes)]
           [offset-x (get-x offset)]
           [offset-y (get-y offset)]
           [offset-z (get-z offset)])
       (for* ([z size-z] [y size-y] [x size-x])
         (p-if transpose
             (set I-shared z y x (get I (p+ offset-x x) (p+ offset-y y) (p+ offset-z z)))
             (set I-shared x y z (get I (p+ offset-x x) (p+ offset-y y) (p+ offset-z z))))))]
    ))

(define (shared-to-global I-shared I pattern offset sizes #:transpose [transpose #f])
  (cond
    [(= (length offset) 1)
     (let ([size-x (get-x sizes)]
           [offset-x (get-x offset)])
       (for ([i size-x])
         (set I (p+ offset-x i) (get I-shared i))))]
    
    [(= (length offset) 2)
     (let ([size-x (get-x sizes)]
           [size-y (get-y sizes)]
           [offset-x (get-x offset)]
           [offset-y (get-y offset)])
       (for* ([y size-y] [x size-x])
         (p-if transpose
             (set I (p+ offset-x y) (p+ offset-y x) (get I-shared x y))
             (set I (p+ offset-x x) (p+ offset-y y) (get I-shared x y)))))]
    
    [(= (length offset) 3)
     (let ([size-x (get-x sizes)]
           [size-y (get-y sizes)]
           [size-z (get-z sizes)]
           [offset-x (get-x offset)]
           [offset-y (get-y offset)]
           [offset-z (get-z offset)])
       (for* ([z size-z] [y size-y] [x size-x])
         (p-if transpose
             (set I (p+ offset-x z) (p+ offset-y y) (p+ offset-z x) (get I-shared x y z))
             (set I (p+ offset-x x) (p+ offset-y y) (p+ offset-z z) (get I-shared x y z)))))]
    ))

(define-syntax-rule
  (global-to-reg I I-reg pattern offset sizes blockDim transpose)
  ;; 1. 1-level vector of size blockSize
  ;; 2. get-warpId
  (cond
    [(= (length blockDim) 1)
     (let* ([size-x (get-x sizes)]
            [stride-x (get-x pattern)]
            [blockSize (apply p* blockDim)]
            [iter-x (add1 (quotient (sub1 size-x) (* warpSize stride-x)))]
            [I-len (vector-length I)]
            [new-I-reg (make-vector blockSize #f)])
       (for ([t blockSize])
         (set new-I-reg t (clone I-reg)))
       (set! I-reg new-I-reg)
       ;;(for ([my-i (vector-length I-reg)])
       ;;  (set I-reg my-i (make-vector blockSize #f)))
       (for* ([warp (quotient blockSize warpSize)])
         (let ([offset-x (get-x (get offset (* warp warpSize)))]
               [global-x 0])
           (pretty-display `(warp ,warp ,offset-x ,size-x))
           (for* ([it iter-x]
                  [t warpSize]
                  [my-i stride-x])
             (when (and (< global-x size-x)
                        (< (+ offset-x global-x) I-len))
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
  (for ([iter (quotient (vector-length val) warpSize)])
    (let ([offset (* iter warpSize)])
      (for ([i warpSize])
        (let ([i-dest (p+ offset i)]
              [i-src (p+ offset (get lane (p+ offset i)))])
        (set res i-dest (get val i-src))))))
  res)

;;;;;;;;;;;;;;;;;;;;;;;;;;; run kernel ;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (get-warpId threadID blockDim)
  (p-if (list? threadID)
      (let ([sum 0])
        (for ([id (reverse threadID)]
              [dim (cdr (reverse (cons 1 blockDim)))])
          (set! sum (+ sum (* id dim))))
        (quotient sum warpSize))
      (for/vector ([id threadID])
        (get-warpId id blockDim))))

(define (get-idInWarp threadID blockDim)
  (p-if (list? threadID)
      (let ([sum 0])
        (for ([id (reverse threadID)]
              [dim (cdr (reverse (cons 1 blockDim)))])
          (set! sum (+ sum (* id dim))))
        (p-modulo sum warpSize)) ;; differ only here
      (for/vector ([id threadID])
        (get-idInWarp id blockDim))))

(define (get-threadId sizes)
  (define ret (list))
  (define (rec id sizes)
    (p-if (empty? sizes)
        (set! ret (cons id ret))
        (for ([i (car sizes)])
          (rec (cons i id) (cdr sizes)))))
  (rec (list) (reverse sizes))
  (list->vector (reverse ret)))

(define (run-grid kernel gridDim blockDim threadIds args)
  (define (f blockID sizes)
    (p-if (empty? sizes)
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
  (for* ([y 4] [x 4]) (set I x y (+ x (p* 10 y))))
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
     
  