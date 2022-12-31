program     → declaration* EOF ;
declaration → varDecl
            | funDecl
            | classDecl
            | statement ; 

block       → "{" declaration* "}" ;

classDecl → "class" IDENTIFIER "{" function* "}" ;
funDecl → "fun" function ;
function → IDENTIFIER "(" parameters? ")" block ;
parameters → IDENTIFIER ( "," IDENTIFIER )* ;

statement   → exprStmt 
            | ifStmt
            | whileStmt 
            | printStmt 
            | block ;

exprStmt  → expression ";";
printStmt → "print" expression ";" ;

ifStmt      → "if" "(" expression ")" statement ( "else" statement )? ;
whileStmt   → "while" "(" expression ")" statement ;

forStmt     → "for" "(" ( varDecl | exprStmt | ";" ) 
                expression? ";"
                expression? ")" statement ;

expression  → assignment ;
assignment  → IDENTIFIER "=" assignment
            | logic_or ;
logic_or    → logic_and ( "or" logic_and )* ;
logic_and   → equality ( "and" equality )* ;

 
equality        comparison ( ( "!=" | "==" ) comparison )* ;
→               → term ( ( ">" | ">=" | "<" | "<=" ) term )* ; 
                → factor ( ( "-" | "+" ) factor )* ;
                → unary ( ( "/" | "*" ) unary )* ; 
unary           → ( "!" | "-" ) unary | call ; 
call            → primary ( "(" arguments? ")" )* ;
arguments       → expression ( "," expression )* ;
primary         → NUMBER | STRING | "true" | "false" | "nil" | "(" expression ")" ;


private Expr expression() {
    return equalty();
}
private Expr equality() {
    Expr expr = comparison();
    while (match(BANG_EQUAL, EQUAL_EQUAL)) {
      Token operator = previous();
      Expr right = comparison();
      expr = new Expr.Binary(expr, operator, right);
    }
    return expr;
}
private boolean match(TokenType... types) {
    for (TokenType type : types) {
      if (check(type)) {
          advance();
          return true;
        } 
    }
    retur false;
}
private Token advance() {
    if (!isAtEnd()) current++;
        return previous();
    }
}
private Expr comparison() {
      Expr expr = term();
      while (match(GREATER, GREATER_EQUAL, LESS, LESS_EQUAL)) {
        Token operator = previous();
        Expr right = term();
        expr = new Expr.Binary(expr, operator, right);
    }
    return expr;
}

// match each token, parse into stmt tree... more than expression. Expression is one liner.
private Stmt statement() {
    if (match(FOR)) return forStatement();
    if (match(IF)) return ifStatement();
    if (match(WHILE)) return whileStatement();
    if (match(FUN)) return function("function");
    if (match(CLASS)) return classDeclaration();
    return expressionStatement();  // expression is a statement that not for/if/while/function/class
}

private void execute(Stmt stmt) {
      stmt.accept(this);   // calls the visitor of the type.
}

private Stmt forStatement() {
    consume(LEFT_PAREN, "Expect '(' after 'for'.");
    Stmt initializer;
    if (match(SEMICOLON)) {
        initializer = null;
    } else if (match(VAR)) {
        initializer = varDeclaration();
    } else {
        initializer = expressionStatement();
    }
}

private Stmt expressionStatement() {
      Expr expr = expression();
      consume(SEMICOLON, "Expect ';' after expression.");
      return new Stmt.Expression(expr);
}

private Expr call() {
    Expr expr = primary();
    while (true) {
    if (match(LEFT_PAREN)) {
        expr = finishCall(expr);
    } else {
        break;
    }
    return expr;
}

private Expr finishCall(Expr callee) {
    List<Expr> arguments = new ArrayList<>();
    if (!check(RIGHT_PAREN)) {
        do {
            arguments.add(expression());
        } while (match(COMMA));
    }
    Token paren = consume(RIGHT_PAREN,
                        "Expect ')' after arguments.");
    return new Expr.Call(callee, paren, arguments);
}
// Stmt/Expr Syntax call. recursively visit each children.
public Object visitCallExpr(Expr.Call expr) {
      Object callee = evaluate(expr.callee);
      List<Object> arguments = new ArrayList<>();
      for (Expr argument : expr.arguments) {
        arguments.add(evaluate(argument));
      }
      LoxCallable function = (LoxCallable)callee;
      return function.call(this, arguments);
    }


private Stmt.Function function(String kind) {
      Token name = consume(IDENTIFIER, "Expect " + kind + " name.");
}
