#lang racket

(provide racket2cuda print-cuda load-store?) 

(define data-type "int")
(define index-type "int")

(define dims 1)
(define thread-id #f)
(define block-id #f)
(define block-size #f)

(define env (make-hash))
(define matrix-size (make-hash))
(define cuda-vars (make-hash))
(define env-consts (hash 'struct-size 3 'warpSize 32 'n 64))
(define accumulators (make-hash))
(define acc-replace (make-hash))
(define temps (make-hash))

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
    [(equal? x #t) 1]
    [(equal? x #f) 0]
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
     
     (add-st (convert-statement `(define ,y (make-vector ,size))))
     (hash-set! env i 1)
     (define-values (i-expr j-expr) (apply convert-fan2 fan-args))

     (cond
       [(string->number j-expr)
        (add-st (roate-one-step (sanitize x) (sanitize y) n i i-expr j-expr))
        ]

       [else
        (add-st (format-indent "{"))
        (inc-indent)
        
        (define (def-loop skip)
          (when (< skip n)
            (add-st (convert-statement
                     `(define ,(format "_~a~a" x (sub1 skip))
                        (make-vector ,size))))
            (def-loop (* 2 skip))))
        (def-loop 2)
        (add-st (format-indent "int rot = (~a) % ~a;" j-expr n))
        
        (define (loop skip)
          (when (< skip n)
            (add-st (rotate-log-step (sanitize x) (sanitize y) n skip 'rot i i-expr (>= (* 2 skip) n)))
            (loop (* 2 skip))))
        
        (loop 1)
        (dec-indent)
        (add-st (format-indent "}"))]
       )
     (reverse statements)]

    [(list 'define acc (list 'create-accumulator (list 'list op-list ...) final-op block-dim))
     (hash-set! accumulators acc (cons (map convert-op op-list) final-op))
     (format-indent "~a ~a = 0;" data-type (sanitize acc))] ;; TODO: where to insert final-op?


    [(list 'accumulate acc l)
     (convert-statement (list 'accumulate acc l '#:pred #t))
     ]

    [(list 'accumulate acc l* '#:pred pred*)
     (define pred (simplify pred*))
     (cond
       [(equal? pred #f)
        (list)]

       [else
        (define l
          (match l*
            [(list 'list x ...) x]
            [_ l*]))
        (define op-list (car (hash-ref accumulators acc)))
        (define res (accumulate l (cdr (reverse op-list))))
        (define st (format "~a ~a= ~a;" acc (last op-list) res))

        (cond
          [(equal? pred #t)
           (format-indent st)]

          [else
           (define-values (pred-n pred-f) (convert-expr pred))
           (define pred-ret (pred-f 0))
           (format-indent "if(~a) ~a" pred-ret st)
           ])
        ])
     ]

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

    [(list 'global-to-reg global reg idx '#:size size)
     (define-values (idx-n idx-f) (convert-expr idx))
     (define-values (size-n size-f) (convert-expr size))
     (define ans (idx-f (sub1 idx-n)))
     (for ([t (sub1 idx-n)])
       (let ([i (- idx-n t 2)])
         (set! ans (format "(~a * ~a) + ~a" ans (size-f i) (idx-f i)))))
     (format-indent "~a = ~a[~a];" (sanitize reg) (sanitize global) ans)]

    [(list 'reg-to-global reg global idx)
     (define-values (reg-n reg-f) (convert-expr reg))
     (format-indent "~a~a = ~a;" (sanitize global) (dims-str (list idx)) (reg-f 0))]

    [(list 'reg-to-global reg global idx '#:size size)
     (define-values (reg-n reg-f) (convert-expr reg))
     (define-values (idx-n idx-f) (convert-expr idx))
     (define-values (size-n size-f) (convert-expr size))
     (define ans (idx-f (sub1 idx-n)))
     (for ([t (sub1 idx-n)])
       (let ([i (- idx-n t 2)])
         (set! ans (format "(~a * ~a) + ~a" ans (size-f i) (idx-f i)))))
     (format-indent "~a[~a] = ~a;" (sanitize global) ans (reg-f 0))]

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
     (convert-statement (list f A B stride offset size transpose '#:round round '#:size 1))
     ]
    
    [(list (? load-store? f) A B stride offset size transpose '#:size gsize)
     (convert-statement (list f A B stride offset size transpose '#:round 1 '#:size gsize))
     ]

    [(list (? load-store? f) A B stride offset size transpose '#:round round '#:size gsize)
     (define warp-shape
       (cond [(= dims 1) 32]
             [(= dims 2) '(x-y-z 32 1)]
             [(= dims 3) '(x-y-z 32 1 1)]))
     (convert-statement (list f A B stride offset size transpose '#:warp-shape warp-shape  '#:round round '#:size gsize))
     ]

    [(list (? load-store? f) A B stride offset size transpose '#:warp-shape warp-shape '#:round round)
     (convert-statement (list f A B stride offset size transpose '#:warp-shape warp-shape  '#:round round '#:size 1))
     ]

    [(list (? load-store? f) A B stride offset size transpose '#:warp-shape warp-shape '#:size gsize)
     (convert-statement (list f A B stride offset size transpose '#:warp-shape warp-shape  '#:round 1 '#:size gsize))
     ]

    [(list (? load-store? f) A B stride offset size transpose '#:warp-shape warp-shape '#:round round '#:size gsize)
     (define cuda-f (string-replace (symbol->string f) "-" "_"))
     (define global (if (load? f) A B))
     (define local (if (load? f) B A))
     (convert-global-to-local cuda-f
                              A B round stride offset size transpose warp-shape
                              (hash-ref matrix-size local) #:size gsize)
     ]

    [(list (? load-store? f) A B stride offset size transpose '#:round round
           '#:shfl (list 'lambda (list tid i) (list 'fan fan-args ...)))
     (define-values (fan-n fan-f) (apply convert-fan fan-args))
     
     (define warp-shape
       (cond [(= dims 1) 32]
             [(= dims 2) '(x-y-z 32 1)]
             [(= dims 3) '(x-y-z 32 1 1)]))
     (define cuda-f (string-replace (symbol->string f) "-" "_"))
     (define global (if (load? f) A B))
     (define local (if (load? f) B A))
     (list
      (format-indent "auto perm_~a = [=] (int ~a, int ~a) -> int{ return ~a; };" (sanitize A) tid i (fan-f 0))
      (convert-global-to-local cuda-f
                               A B round stride offset size transpose warp-shape
                              (hash-ref matrix-size local) #:shfl (format "perm_~a" (sanitize A))))
     ]

    [(list 'for (list (list vs ls) ...) body)
     (for ([v vs])
       (hash-set! env v 1))
     (define inits (list))
     (define conds (list))
     (define incs (list))

     (for ([v vs] [l ls])
       (let* ([x (sanitize v)]
              [b (sanitize l)])
         (set! inits (cons (format "~a = 0" x) inits))
         (set! conds (cons (format "(~a < ~a)" x b) conds))
         (set! incs (cons (format "~a++" x) incs))))
     (define start
       (format-indent "for(int ~a; ~a; ~a) {" (string-join inits ",") (string-join conds "&&") (string-join incs ",")))
     (inc-indent)
     (define body-ret (convert-statement body))
     (dec-indent)
     (define end (format-indent "}"))
     (list start body-ret end)
     ]

    [(list 'for* (list (list vs ls) ...) body)
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
     (define ret
     (append
      (for/list ([e es] [v vs])
        (let ([e (simplify e)])
        (let-values ([(n f) (convert-expr e)])
          (hash-set! env v n)
          (hash-set! temps v e)
          (cond
            [(= n 1)
             (format-indent "int ~a = ~a;" (sanitize v) (f 0))]
           [else
            (for/list ([i n]) (format-indent "int ~a~a = ~a;" (sanitize v) i (f i)))])
          )))
      (for/list ([st body])
        (convert-statement st))))
     (for ([v vs]) (hash-remove! temps v))
     ret
     ]

    [(list 'set matrix idxs ... v)
     (define v-str
       (let-values ([(n v-f) (convert-expr v)])
         (v-f 0)))

     (format-indent "~a~a = ~a;" (sanitize matrix) (dims-str idxs) v-str)
     ]
    
    ))

(define (convert-global-to-local name A B round stride offset load-size transpose warp-shape local-size
                                 #:shfl [shfl #f] #:size [gsize 1])
  (define-values (stride-n stride-f) (convert-expr stride))
  (define-values (offset-n offset-f) (convert-expr offset))
  ;(define-values (lsize-n lsize-f) (convert-expr local-size))
  (define-values (shape-n shape-f) (convert-expr warp-shape))
  (define-values (round-n round-f) (convert-expr round))
  (define d (max stride-n offset-n))
  (define size-str "")
  (when (> d 1)
    (define-values (gsize-n gsize-f) (convert-expr gsize))
    (define l
      (append
       (for/list ([i (sub1 d)]) (gsize-f i))
       (for/list ([i (sub1 d)]) (format "~a" (list-ref local-size i)))))
    (set! size-str (format ",~a" (string-join l ",")))
    )
  
  (define str-list
    (list (format "~a~a~a<~a>((~a*) ~a, (~a*) ~a" name (if shfl "_shlf" "") d data-type
                  data-type (sanitize A) data-type (sanitize B))
          "," (string-join (for/list ([i d]) (round-f i)) ", ")
          "," (string-join (for/list ([i d]) (offset-f i)) ", ")
          "," (string-join (for/list ([i d]) (stride-f i)) ", ")
          "," (string-join (for/list ([i d]) (shape-f i)) ", ")
          (if shfl (format ",~a" shfl) "")
          size-str ");"))
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
          [(= dims 2) "((threadIdx.y*blockDim.x + threadIdx.x)/32)"]
        [(= dims 3) "((threadIdx.z*blockDim.y*blockDim.x + threadIdx.y*blockDim.x + threadIdx.x)/32)"]
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

    #;[(list 'get matrix idxs1 ... (list 'ite c1 e1 (list 'ite c2 e2 e3)) idxs2 ...)
     (define-values (c-n c-f) (convert-expr c1))
     (define-values (geta-n geta-f) (convert-expr (append `(get ,matrix) idxs1 `(,e1) idxs2)))
     (define-values (getb-n getb-f) (convert-expr (append `(get ,matrix) idxs1 `(ite ,c2 ,e2 ,e3) idxs2)))
     (values 1 (lambda (i) (format "~a? ~a: ~a" (c-f 0) (geta-f 0) (getb-f 0))))
     ]

    [(list 'get matrix idxs1 ... (list 'ite c a b) idxs2 ...)
     (define-values (c-n c-f) (convert-expr c))
     (define-values (geta-n geta-f) (convert-expr (append `(get ,matrix) idxs1 `(,a) idxs2)))
     (define-values (getb-n getb-f) (convert-expr (append `(get ,matrix) idxs1 `(,b) idxs2)))
     (values 1 (lambda (i) (format "~a? ~a: ~a" (c-f 0) (geta-f 0) (getb-f 0))))
     ]

    [(list 'get matrix idxs ...)
     (define ites (map ite-const? idxs))
     (define ite-n (count identity ites))
     ;(pretty-display `(get ,temps ,ite-n))

     (cond
       [(> ite-n 0)
        (define new-idxs
          (for/list ([idx idxs])
            (if (ite-const? idx)
                (hash-ref temps idx)
                idx)))
        (convert-expr (append `(get ,matrix) new-idxs))
        ]

       [else
        (define str-list
          (for/list ([idx idxs])
            (let-values ([(n idx-f) (convert-expr idx)])
              (idx-f 0))))
        
        (values 1 (lambda (i) (format "~a~a" (sanitize matrix) (dims-str str-list))))]
       )
     ]

    [(list 'fan j n* cj* dj* group* conf-fw
           k m* ck* dk*)
     (convert-fan j n* cj* dj* group* conf-fw
                  k m* ck* dk* 0)]

    [(list 'fan j n* cj* dj* group* conf-fw
           k m* ck* dk* #:offset offset)
     (convert-fan j n* cj* dj* group* conf-fw
                  k m* ck* dk* offset)]

    [(list 'accumulate-final acc)
     (define final-op (cdr (hash-ref accumulators acc)))

     (match final-op
       ['identity (convert-expr acc)]
       [(list 'lambda (list arg) body)
        (hash-set! acc-replace arg acc)
        (define-values (n f) (convert-expr body))
        (hash-remove! acc-replace arg)
        (values n f)
        ])
     ]

    [(list 'get-x v)
     (define-values (n f) (convert-expr v))
     (values 1 (lambda (i) (f 0)))
     ]

    [(list 'get-y v)
     (define-values (n f) (convert-expr v))
     (values 1 (lambda (i) (f 1)))
     ]

    [(list 'get-z v)
     (define-values (n f) (convert-expr v))
     (values 1 (lambda (i) (f 2)))
     ]

    [(list '@dup x)
     (convert-expr x)
     ]
    
    [(list 'x-y-z xs ...)
     (values (length xs) (lambda (i)
                           (define-values (n f) (convert-expr (list-ref xs i)))
                           (f 0)))
     ]

    [(list 'ite c a b)
     (define new-ite (simplify expr))

     (match new-ite
       [(list 'ite _ _ _)
        (define-values (c-n c-f) (convert-expr c))
        (define-values (a-n a-f) (convert-expr a))
        (define-values (b-n b-f) (convert-expr b))
        (define max-d (max a-n b-n c-n))
        
        (values max-d
                (lambda (i)
                  (format "~a? ~a: ~a" (c-f i) (a-f i) (b-f i))))]
       [else (convert-expr new-ite)])
     ]

    [(list op args ...)
     (define new-expr (simplify expr))

     (match new-expr
       [(list op args ...)
        (define max-d 1)
        (define fs 
          (for/list ([arg args])
            (let-values ([(n f) (convert-expr arg)])
              (when (> n max-d) (set! max-d n))
              f)))
        
        (values max-d
                (lambda (i)
                  (format "(~a)"
                          (string-join
                           (for/list ([f fs])
                             (f i))
                           (convert-op op)))))]

       [v (convert-expr v)])
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
       [(hash-has-key? acc-replace v)
        (define name (hash-ref acc-replace v))
        (values d (lambda (i) (format "~a" (sanitize name))))]
       [(= d 1) (values d (lambda (i) (format "~a" (sanitize v))))]
       [else (values d (lambda (i) (format "~a~a" (sanitize v) i)))])
     ]
    ))

(define (ite-const? e)
  (if (hash-has-key? temps e)
      (let ([v (hash-ref temps e)])
        (match v
          [(list 'ite _ _ _) #t]
          ;[(list 'ite c (? number?) (? number?)) #t]
          [_ #f]))
      #f))

(define (accumulate vals ops)
  (cond
    [(= (length ops) 0)
     (define-values (v-n v-f) (convert-expr vals))
     (v-f 0)
     ]

    [else
     (define vals-ret (for/list ([v vals]) (accumulate v (cdr ops))))
     (string-join vals-ret (car ops))
     ]))

(define (convert-fan j n* cj* dj* group* conf-fw
                     k m* ck* dk* [offset 0])
     (define n (eval-const n*))
     (define cj (eval-const cj*))
     (define dj (eval-const dj*))
     (define group (eval-const group*))
     (define m (eval-const m*))
     (define ck (eval-const ck*))
     (define dk (eval-const dk*))
     
     (define offset1-a
       (cond
         [(equal? dj n) 0]
         [(equal? group n) (@quotient j dj)]
         [else (@quotient (@modulo j group) dj)]))

     (define offset1-b (@* k ck))

     (define offset1-c
       (cond
         [(equal? dk m) 0] [else (@quotient k dk)]))

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
      [(equal? dj n) 0]
      [(equal? group n) (@quotient j dj)]
      [else (@quotient (@modulo j group) dj)]))
  
  (define offset1-b (@* k ck)) ;k
  
  (define offset1-c
    (cond
      [(equal? dk m) 0] [else (@quotient k dk)])) ;k
  
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

(define (roate-one-step x y n i i-expr j-expr)
  (list
   (format-indent "for(int ~a=0; ~a<~a; ~a++) {" i i n i)
   (inc-indent)
   (format-indent "~a[~a] = ~a[(~a+~a)%~a];" y i x i-expr j-expr n)
   (dec-indent)
   (format-indent "}")
   ))

(define (rotate-log-step x* y* n skip rot i i-expr last-iter)
  (define x (if (= skip 1) x* (format "_~a~a" x* (sub1 skip))))
  (define y (if last-iter y* (format "_~a~a" x* skip)))
  (list
   (format-indent "for(int ~a=0; ~a<~a; ~a++) {" i i n i)
   (inc-indent)
   (format-indent "if((~a & ~a)==0) ~a[~a] = ~a[(~a)%~a];" rot skip y i x (if last-iter i-expr i) n)
   (format-indent "else ~a[~a] = ~a[(~a+~a)%~a];" y i x (if last-iter i-expr i) skip n)
   (dec-indent)
   (format-indent "}")
   ))

(define (convert-op op)
  (match op
    ['quotient "/"]
    ['modulo "%"]
    ['bvand "&"]
    ['bvxor "^"]
    [(? string?) op]
    [x (symbol->string x)]))

(define (dims-str idxs)
  (define str-list
       (for/list ([idx (reverse idxs)])
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

(define (@-- xs)
  (cond
    [(= (length xs) 1) (car xs)]
    [else
     (define y (@++ (cdr xs)))
     (define x (car xs))
     (cond
       [(and (equal? x 0) (number? y)) (- 0 y)]
       [(equal? y 0) x]
       [else `(- ,x ,y)])
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
(define-syntax-rule (@- x ...) (@-- (list x ...)))
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

(define (simplify e)
  (match e
    [(list 'ite c a b)
     (define new-c (simplify c))
     (cond
       [(equal? new-c #t) (simplify a)]
       [(equal? new-c #f) (simplify b)]
       [else `(ite ,new-c ,(simplify a) ,(simplify b))])]
    [(list '+ args ...) (@++ (for/list ([x args]) (simplify x)))]
    [(list '- args ...) (@-- (for/list ([x args]) (simplify x)))]
    [(list '* args ...) (@** (for/list ([x args]) (simplify x)))]
    [(list 'quotient x y) (@quotient (simplify x) (simplify y))]
    [(list 'modulo x y) (@modulo (simplify x) (simplify y))]
    [(list op a b)
     (match `(,op ,(simplify a) ,(simplify b))
       [(list '= x x) #t]
       [(list '>= x x) #t]
       [(list '<= x x) #t]
       [(list '> x x) #f]
       [(list '< x x) #f]
       [(list '= x (list '+ (? number?) x)) #f]
       [(list '<= x (list '+ (? positive?) x)) #t]
       [(list '< x (list '+ (? positive?) x)) #t]
       [(list '<= x (list '+ (? negative?) x)) #f]
       [(list '< x (list '+ (? negative?) x)) #f]
       [new-expr new-expr])]
    [(list '@dup x) (simplify x)]
    [_
     (if (hash-has-key? env-consts e)
         (hash-ref env-consts e)
         e)
     ]))

