#lang racket

(define data-type "int")
(define index-type "int")

(define dims 1)
(define thread-id #f)
(define block-id #f)
(define block-size #f)

(define env (make-hash))
(define matrix-size (make-hash))
(define cuda-vars (make-hash))
(define env-consts (hash 'struct-size 3 'warpSize 32))

(define (cuda-var? v)
  (hash-has-key? cuda-vars v))
(define (load-store? f)
  (member f '(global-to-local local-to-global global-to-shared shared-to-global)))
(define (load? f)
  (member f '(global-to-local global-to-shared)))
(define (store? f)
  (member f '(local-to-global shared-to-global)))
(define (let? x)
  (member x '(let let*)))

(define (eval-const v)
  (if (hash-has-key? env-consts v)
      (hash-ref env-consts v)
      v))

(define (sanitize x)
  (cond
    [(number? x) x]
    [(string? x) (string-replace x "-" "_")]
    [else (string-replace (symbol->string x) "-" "_")]))

(define (print-cuda l)
  (for ([s (flatten l)])
    (pretty-display s)))

(define (racket2cuda expr d #:const-map [const-map #f])
  (set! dims d)
  (match expr
    [(list 'define (list func-name tid bid bsize args ...) body ...)
     (set! thread-id tid) 
     (set! block-id bid) 
     (set! block-size bsize)
     (hash-set! env tid d)
     (hash-set! env block-id d)
     (hash-set! env block-size d)
     (hash-set! cuda-vars tid "threadIdx")
     (hash-set! cuda-vars block-id "blockIdx")
     (hash-set! cuda-vars block-size "blockDim")
     (when const-map (set! env-consts const-map))

     (define args-str
       (for/list ([arg args])
         (format "const ~a *~a" data-type (sanitize arg))))
     (define declare
       (format-indent "__global__ void ~a(~a) {" (sanitize func-name) (string-join args-str ", ")))
     (inc-indent)
     (define body-ret (for/list ([st body]) (convert-statement st)))
     (dec-indent)

     (list declare
           body-ret
           (format-indent "}"))
     ]))


(define indent-str "")
(define (inc-indent) (set! indent-str (string-append indent-str "  ")) (list))
(define (dec-indent) (set! indent-str (substring indent-str 2)) (list))
(define-syntax-rule (format-indent s args ...)
  (string-append indent-str (format s args ...)))

(define (convert-statement st)
  (match st
    [(list 'define matrix (list 'create-matrix-local (list 'x-y-z sizes ...) '#:type type))
     ;(hash-set! env matrix (length sizes))
     (hash-set! matrix-size matrix sizes)
     (format-indent "~a ~a~a;" type (sanitize matrix) (dims-str sizes))]

    [(list 'define matrix (list 'create-matrix-local (list 'x-y-z sizes ...)))
     ;(hash-set! env matrix (length sizes))
     (hash-set! matrix-size matrix sizes)
     (format-indent "~a ~a~a;" data-type (sanitize matrix) (dims-str sizes))]

    [(list 'define-shared matrix (list 'create-matrix (list 'x-y-z sizes ...) '#:type type))
     ;(hash-set! env matrix (length sizes))
     (hash-set! matrix-size matrix sizes)
     (format-indent "__shared__ ~a ~a~a;" type (sanitize matrix) (dims-str sizes))]

    [(list 'define-shared matrix (list 'create-matrix (list 'x-y-z sizes ...)))
     ;(hash-set! env matrix (length sizes))
     (hash-set! matrix-size matrix sizes)
     (format-indent "__shared__ ~a ~a~a;" data-type (sanitize matrix) (dims-str sizes))]

    ;; log M rotations
    [(list 'define y (list 'permute-vector x size
                           (list 'lambda (list i) (list 'fan fan-args ...))))
     ;(hash-set! env matrix (length sizes))
     (define statements (list))
     (define (add-st x) (set! statements (cons x statements)))

     (define n (eval-const size))
     (define (def-loop skip)
       (when (< skip n)
         (add-st (convert-statement
                  `(define ,(format "_~a~a" x (sub1 skip))
                     (make-vector ,size))))
         (def-loop (* 2 skip))))
     (def-loop 2)
     
     (add-st (convert-statement `(define ,y (make-vector ,size))))
     (hash-set! env i 1)
     (define-values (i-expr j-expr) (apply convert-fan2 fan-args))

     (add-st (format-indent "{"))
     (inc-indent)
     (add-st (format-indent "int rot = (~a) % ~a;" j-expr n))

     (define (loop skip)
       (when (< skip n)
         (add-st (rotate-log-step x y n skip 'rot i i-expr (>= (* 2 skip) n)))
         (loop (* 2 skip))))
     
     (loop 1)
     (dec-indent)
     (add-st (format-indent "}"))
     (reverse statements)]
     

    [(list 'define matrix (list 'make-vector size))
     ;(hash-set! env matrix 1)
     (hash-set! matrix-size matrix (list size))
     (format-indent "~a ~a~a;" data-type (sanitize matrix) (dims-str (list size)))]

    [(list 'define var e)
     (define-values (n f) (convert-expr e))
     (hash-set! env var n)
     (if (= n 1)
         (format-indent "~a ~a = ~a;" index-type (sanitize var) (f 0))
         (for/list ([i n])
           (format-indent "~a ~a~a = ~a;" index-type (sanitize var) i (f i))))]

    [(list 'global-to-reg global reg idx)
     (format-indent "~a = ~a~a;" (sanitize reg) (sanitize global) (dims-str (list idx)))]

    [(list 'reg-to-global reg global idx)
     (format-indent "~a~a = ~a;" (sanitize global) (dims-str (list idx)) (sanitize reg))]

    [(list (? load-store? f) A B stride offset size transpose)
     (define warp-shape
       (cond [(= dims 1) 32]
             [(= dims 2) '(x-y-z 32 1)]
             [(= dims 3) '(x-y-z 32 1 1)]))
     (define cuda-f (string-replace (symbol->string f) "-" "_"))
     (define global (if (load? f) A B))
     (define local (if (load? f) B A))
     (convert-global-to-local cuda-f
                              A B 1 stride offset size transpose warp-shape
                              (hash-ref matrix-size local))
     ]
    
    

    [(list (? load-store? f) A B stride offset size transpose '#:warp-shape warp-shape)
     (define cuda-f (string-replace (symbol->string f) "-" "_"))
     (define global (if (load? f) A B))
     (define local (if (load? f) B A))
     (convert-global-to-local cuda-f
                              A B 1 stride offset size transpose warp-shape
                              (hash-ref matrix-size local))
     ]

    [(list (? load-store? f) A B stride offset size transpose '#:round round)
     (define warp-shape
       (cond [(= dims 1) 32]
             [(= dims 2) '(x-y-z 32 1)]
             [(= dims 3) '(x-y-z 32 1 1)]))
     (define cuda-f (string-replace (symbol->string f) "-" "_"))
     (define global (if (load? f) A B))
     (define local (if (load? f) B A))
     (convert-global-to-local cuda-f
                              A B round stride offset size transpose warp-shape
                              (hash-ref matrix-size local))
     ]

    [(list (? load-store? f) A B stride offset size transpose '#:warp-shape warp-shape '#:round round)
     (define cuda-f (string-replace (symbol->string f) "-" "_"))
     (define global (if (load? f) A B))
     (define local (if (load? f) B A))
     (convert-global-to-local cuda-f
                              A B round stride offset size transpose warp-shape
                              (hash-ref matrix-size local))
     ]

    [(list 'for (list (list vs ls) ...) body)
     (for ([v vs])
       (hash-set! env v 1))
     (append
      (for/list ([v vs] [l ls])
        (let* ([x (sanitize v)]
               [b (sanitize l)]
               [temp (format-indent "for(int ~a = 0; ~a < ~a; ~a++) {" x x b x)])
          (inc-indent)
          temp))
      (list (convert-statement body))
      (for/list ([v vs])
        (dec-indent)
        (format-indent "}")))
     ]

    [(list let? (list (list vs es) ...) body ...)
     (append
      (for/list ([e es] [v vs])
        (let-values ([(n f) (convert-expr e)])
          (hash-set! env v n)
          (cond
            [(= n 1)
             (format-indent "int ~a = ~a;" (sanitize v) (f 0))]
           [else
            (for/list ([i n]) (format-indent "int ~a~a = ~a;" (sanitize v) i (f i)))])
          ))
      (for/list ([st body])
        (convert-statement st)))
     ]

    [(list 'set matrix idxs ... v)
     (define v-str
       (let-values ([(n v-f) (convert-expr v)])
         (v-f 0)))

     (format-indent "~a~a = ~a;" (sanitize matrix) (dims-str idxs) v-str)
     ]
    
    ))

(define (convert-global-to-local name A B round stride offset size transpose warp-shape local-size)
  (define-values (stride-n stride-f) (convert-expr stride))
  (define-values (offset-n offset-f) (convert-expr offset))
  (define-values (size-n size-f) (convert-expr size))
  (define-values (shape-n shape-f) (convert-expr warp-shape))
  (define-values (round-n round-f) (convert-expr round))
  (define d (max stride-n offset-n size-n))
  (define str-list
    (list (format "~a~a(~a, ~a" name d (sanitize A) (sanitize B))
          "," (string-join (for/list ([i d]) (round-f i)) ", ")
          "," (string-join (for/list ([i d]) (offset-f i)) ", ")
          "," (string-join (for/list ([i d]) (stride-f i)) ", ")
          "," (string-join (for/list ([i d]) (shape-f i)) ", ")
          ");"))
  (format-indent "~a" (string-join (flatten str-list) ""))
  )

(define (convert-expr expr)
  (match expr
    [(list 'get-warpId tid)
     (values
      1
      (lambda (i)
        (cond
          [(= dims 1) "(threadIdx.x/32)"]
          [(= dims 2) "(threadIdx.y*blockDim.x + threadIdx.x)/32)"]
        [(= dims 3) "(threadIdx.z*blockDim.y*blockDim.x + threadIdx.y*blockDim.x + threadIdx.x)/32)"]
        )))]

    [(list 'get-idInWarp tid)
     (values 1 (lambda (i) "(threadIdx.x&31)"))]

    [(list 'get-global-threadId tid bid)
     (values
      dims
      (lambda (i)
        (cond
          [(= i 0) "(blockIdx.x * blockDim.x + threadIdx.x)"]
          [(= i 1) "(blockIdx.y * blockDim.y + threadIdx.y)"]
          [(= i 2) "(blockIdx.z * blockDim.z + threadIdx.z)"])))]

    [(list 'shfl e lane)
     (define-values (e-n e-f) (convert-expr e))
     (define-values (lane-n lane-f) (convert-expr lane))
     (values 1 (lambda (i) (format "__shfl_sync(FULL_MASK, ~a, ~a)" (e-f 0) (lane-f 0))))
     ]

    [(list 'get matrix idxs ...)
     (define str-list
       (for/list ([idx idxs])
         (let-values ([(n idx-f) (convert-expr idx)])
           (idx-f 0))))

     (values 1 (lambda (i) (format "~a~a" (sanitize matrix) (dims-str str-list))))
     ]

    [(list 'fan j n* cj* dj* group* conf-fw
           k m* ck* dk*)
     (convert-fan j n* cj* dj* group* conf-fw
                  k m* ck* dk* 0)]

    [(list 'fan j n* cj* dj* group* conf-fw
           k m* ck* dk* #:offset offset)
     (convert-fan j n* cj* dj* group* conf-fw
                  k m* ck* dk* offset)]

    [(list '@dup x)
     (convert-expr x)
     ]

    [(list op args ...)
     (define max-d 1)
     (define rets 
       (for/list ([arg args])
         (let-values ([(n f) (convert-expr arg)])
           (cons n f))))

     (values max-d
             (lambda (i)
               (format "(~a)"
                       (string-join
                        (for/list ([ret rets]) ((cdr ret) i))
                        (convert-op op)))))
     ]

    [(list 'x-y-z xs ...)
     (values (length xs) (lambda (i)
                           (define-values (n f) (convert-expr (list-ref xs i)))
                           (f 0)))
     ]

    [(? cuda-var? v)
     (define name (hash-ref cuda-vars v))
     (values dims
             (lambda (i)
               (cond [(= i 0) (format "~a.x" name)]
                     [(= i 1) (format "~a.y" name)]
                     [(= i 2) (format "~a.z" name)])))
     ]

    [v
     (define d (if (hash-has-key? env v) (hash-ref env v) 1))
     (cond
       [(= d 1) (values d (lambda (i) (format "~a" (sanitize v))))]
       [else (values d (lambda (i) (format "~a~a" (sanitize v) i)))])
     ]
    ))

(define (convert-fan j n* cj* dj* group* conf-fw
                     k m* ck* dk* offset)
     (define n (eval-const n*))
     (define cj (eval-const cj*))
     (define dj (eval-const dj*))
     (define group (eval-const group*))
     (define m (eval-const m*))
     (define ck (eval-const ck*))
     (define dk (eval-const dk*))
     
     (define offset1-a
       (cond
         [(equal? dj group) 0]
         [(equal? group n) (@quotient j dj)]
         [else (@quotient (@modulo j group) dj)]))

     (define offset1-b (@* k ck))

     (define offset1-c
       (cond
         [(equal? dk group) 0] [else (@quotient k dk)]))

     (define offset1 (@+ offset1-a offset1-b offset1-c offset))

     (define common
       (if (and (number? group) (number? dj))
           (quotient group dj) (@quotient group dj)))
     (define offset2
       (cond
         [(or (= conf-fw 1) (equal? common group)) offset1]
         [else (@modulo offset1 common)]))

     (define group-offset
       (cond
         [(equal? group n) 0] [else (@* (@quotient j group) group)]))
     
     (define all (@+ group-offset
                     (@modulo (@+ (@* j cj) offset2) group)))

     (convert-expr all))

(define (convert-fan2 j n* cj* dj* group* conf-fw
                     k m* ck* dk* [offset 0])
  (define n (eval-const n*))
  (define cj (eval-const cj*))
  (define dj (eval-const dj*))
  (define group (eval-const group*))
  (define m (eval-const m*))
  (define ck (eval-const ck*))
  (define dk (eval-const dk*))
  
  (unless (equal? group n)
    (raise (format "fan function for permute-vector must have (~a) n = (~a) group." n group)))
  
  (define offset-j ;j
    (cond
      [(equal? dj group) 0]
      [(equal? group n) (@quotient j dj)]
      [else (@quotient (@modulo j group) dj)]))
  
  (define offset1-b (@* k ck)) ;k
  
  (define offset1-c
    (cond
      [(equal? dk group) 0] [else (@quotient k dk)])) ;k
  
  (define offset-k (@+ offset1-b offset1-c offset))

  (define common
    (if (and (number? group) (number? dj))
        (quotient group dj) (@quotient group dj)))

  (unless (or (= conf-fw 1) (equal? common group))
    (unless (equal? common 1)
      (raise (exn "fan function for permute-vector: invalid conf-fw, group, dj."))
      (set! offset-j 0)
      (set! offset-k 0)))

  (define-values (j-n j-f) (convert-expr (@+ (@* j cj) offset-j)))
  (define-values (k-n k-f) (convert-expr offset-k))
  (values (j-f 0) (k-f 0)))

(define (rotate-log-step x* y* n skip rot i i-expr last-iter)
  (define x (if (= skip 1) x* (format "_~a~a" x* (sub1 skip))))
  (define y (if last-iter y* (format "_~a~a" x* skip)))
  (list
   (format-indent "for(int ~a=0; ~a<~a; ~a++) {" i i n i)
   (inc-indent)
   (format-indent "if(~a & ~a) ~a[~a] = ~a[~a];" rot skip y i x i)
   (format-indent "else ~a[~a] = ~a[(~a+~a)%~a];" y i x (if last-iter i-expr i) skip n)
   (dec-indent)
   (format-indent "}")
   ))

(define (convert-op op)
  (match op
    ['quotient "/"]
    ['modulo "%"]
    [x (symbol->string x)]))

(define (dims-str idxs)
  (define str-list
       (for/list ([idx idxs])
         (let-values ([(n idx-f) (convert-expr idx)])
           (idx-f 0))))
  
  (define dims (map (lambda (s) (format "[~a]" s)) str-list))
  (string-join dims ""))

(define (@++ xs)
  (cond
    [(= (length xs) 1) (car xs)]
    [else
     (define y (@++ (cdr xs)))
     (define x (car xs))
     (cond
       [(equal? x 0) y]
       [(equal? y 0) x]
       [else `(+ ,x ,y)])
     ]))

(define (@** xs)
  (define ret
  (cond
    [(= (length xs) 1) (car xs)]
    [else
     (define y (@++ (cdr xs)))
     (define x (car xs))
     (cond
       [(equal? x 0) 0]
       [(equal? y 0) 0]
       [(equal? x 1) y]
       [(equal? y 1) x]
       [else `(* ,x ,y)])
     ]))
  ret
  )

(define-syntax-rule (@+ x ...) (@++ (list x ...)))
(define-syntax-rule (@* x ...) (@** (list x ...)))

(define (@quotient x y)
  (cond
    [(equal? y 1) x]
    [(equal? x 0) 0]
    [else `(quotient ,x ,y)]))

(define (@modulo x y)
  (cond
    [(equal? y 1) 0]
    [else `(modulo ,x ,y)]))

(define func
  '(define (AOS-load3 threadId blockID blockDim I O a b c)
     (define I-cached (create-matrix-local (x-y-z struct-size)))
     (define O-cached (create-matrix-local (x-y-z struct-size)))
     (define warpID (get-warpId threadId))
     (define offset
       (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))
     (define gid (get-global-threadId threadId blockID))
     (global-to-local
      I
      I-cached
      (x-y-z 1)
      offset
      (x-y-z (* warpSize struct-size))
      #f #:round struct-size)
     (define localId (get-idInWarp threadId))
     (define I-cached2 (permute-vector I-cached struct-size
                                       (lambda (i)
                                         (fan i struct-size 2 3 3 1 localId warpSize 0 1))))
     
     (for
         ((i struct-size))
       (let* ((lane (fan localId warpSize 3 32 32 1 i struct-size 0 1))
              (x (shfl (get I-cached2 (@dup i)) lane))
              (index-o (fan i struct-size 1 3 3 1 localId warpSize 0 warpSize)))
         (set O-cached index-o x)))
     (local-to-global
      O-cached
      O
      (x-y-z 1)
      offset
      (x-y-z (* warpSize struct-size))
      #f #:round struct-size)))
  
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

(print-cuda (racket2cuda func 1))
;(print-cuda (convert-statement loop))
;(print-cuda (convert-statement fan))