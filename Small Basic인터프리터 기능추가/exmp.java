package interpreter;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Point;
// A Small BASIC Interpreter.
import java.io.*;
import java.util.*;

import javax.swing.JFrame;
import javax.swing.JPanel;

// Exception class for interpreter errors.
class InterpreterException extends Exception {
	String errStr; // describes the error
	public InterpreterException(String str) {
		errStr = str;
		}
	public String toString() {
		return errStr;
		}
}
class BFrame extends JFrame{
	DrawPanel Draw;
	BFrame(){
		 Draw = new DrawPanel();
		 setContentPane(Draw);
		
	}
	class DrawPanel extends JPanel {
		Vector<Point> S_point;
		Vector<Point> L_point;
		DrawPanel(){
			S_point=new Vector<Point>();
			L_point=new Vector<Point>();
    	}
        public void paintComponent(Graphics g){
            super.paintComponent(g);
            g.setColor(Color.black);
            for(int i= 0; i<S_point.size();i++ ) {
    			g.drawLine(S_point.get(i).x,S_point.get(i).y,
    					L_point.get(i).x,L_point.get(i).y);
    		}
    	}
    	public void DrawLine(int sx,int sy,int lx,int ly) {
    		S_point.add(new Point(sx,sy));
    		L_point.add(new Point(lx,ly));
    		}   
    	public void DrawRect(int x,int y,int width,int heght) {
    		S_point.add(new Point(x,y));
    		L_point.add(new Point(x+width,y));
    		
    		S_point.add(new Point(x+width,y));
    		L_point.add(new Point(x+width,y+heght));
    		
    		S_point.add(new Point(x+width,y+heght));
    		L_point.add(new Point(x,y+heght));
    		
    		S_point.add(new Point(x,y));
    		L_point.add(new Point(x,y+heght));
    		}
        }
	public void paint(Graphics g){
		super.paintComponents(g);
}
}

// The Small BASIC interpreter.
class interpreter {
	final int PROG_SIZE = 10000; // maximum program size
	// These are the token types.
	final int NONE = 0;
	final int DELIMITER = 1;
	final int VARIABLE = 2;
	final int NUMBER = 3;
	final int COMMAND = 4;
	final int QUOTEDSTR = 5;	//almost string
	// These are the types of errors.
	final int SYNTAX = 0;
	final int UNBALPARENS = 1;
	final int NOEXP = 2;
	final int DIVBYZERO = 3;
	final int EQUALEXPECTED = 4;
	final int NOTVAR = 5;
	final int LABELTABLEFULL = 6;
	final int DUPLABEL = 7;
	final int UNDEFLABEL = 8;
	final int THENEXPECTED = 9;
	final int TOEXPECTED = 10;
	final int NEXTWITHOUTFOR = 11;
	final int RETURNWITHOUTGOSUB = 12;
	final int MISSINGQUOTE = 13;
	final int FILENOTFOUND = 14;
	final int FILEIOERROR = 15;
	final int INPUTIOERROR = 16;
	// Internal representation of the Small BASIC keywords.
	final int UNKNCOM = 0;
	final int PRINT = 1;
	final int INPUT = 2;
	final int IF = 3;
	final int THEN = 4;
	final int FOR = 5;
	final int NEXT = 6;
	final int TO = 7;
	final int GOTO = 8;
	final int GOSUB = 9;
	final int RETURN = 10;
	final int END = 11;
	final int EOL = 12;
	final int OPEN = 13;
	final int ADDRECT = 14;
	final int ADDLINE = 15;
	// This token indicates end-of-program.
	final String EOP = "\0";
	// Codes for double-operators, such as <=.
	final char LE = 1;
	final char GE = 2;
	final char NE = 3;
	// Array for variables.
	private double vars[];
	private BFrame window;
	private Graphics g;
	// This class links keywords with their keyword tokens.
	class Keyword {
		String keyword; // string form
		int keywordTok; // internal representation
		Keyword(String str, int t) {
			keyword = str;
			keywordTok = t;
		}
	}
	/* Table of keywords with their internal representation.
	All keywords must be entered lowercase. */
	Keyword kwTable[] = {
		new Keyword("print", PRINT), // in this table.
		new Keyword("input", INPUT),
		new Keyword("if", IF),
		new Keyword("then", THEN),
		new Keyword("goto", GOTO),
		new Keyword("for", FOR),
		new Keyword("next", NEXT),
		new Keyword("to", TO),
		new Keyword("gosub", GOSUB),
		new Keyword("return", RETURN),
		new Keyword("end", END),
		new Keyword("open", OPEN),
		new Keyword("addrect", ADDRECT),
		new Keyword("addline", ADDLINE)
	};
	private char[] prog; // refers to program array
	private int progIdx; // current index into program
	private String token; // holds current token
	private int tokType; // holds token's type
	private int kwToken; // internal representation of a keyword
	// Support for FOR loops.
	class ForInfo {
		int var; // counter variable
		double target; // target value
		int loc; // index in source code to loop to
	}
	// Stack for FOR loops.
	private Stack fStack;
	// Defines label table entries.
	class Label {
		String name; // label
		int loc; // index of label's location in source file
		public Label(String n, int i) {
		name = n;
		loc = i;
		}
	}
	// A map for labels.
	private TreeMap labelTable;
	// Stack for gosubs.
	private Stack gStack;
	// Relational operators.
	char rops[] = {
	GE, NE, LE, '<', '>', '=', 0
	};
	/* Create a string containing the relational
	operators in order to make checking for
	them more convenient. */
	String relops = new String(rops);
	// Constructor for SBasic.
	public interpreter(String progName) throws InterpreterException {
		char tempbuf[] = new char[PROG_SIZE];
		int size;
		// Load the program to execute.
		size = loadProgram(tempbuf, progName);
		if(size != -1) {
		// Create a properly sized array to hold the program.
		prog = new char[size];
		// Copy the program into program array.
		System.arraycopy(tempbuf, 0, prog, 0, size);
		}
	}
	// Load a program.
	private int loadProgram(char[] p, String fname) throws InterpreterException
	{
		int size = 0;
		try {
			FileReader fr = new FileReader(fname);
			BufferedReader br = new BufferedReader(fr);
			size = br.read(p, 0, PROG_SIZE);
			fr.close();
		} catch(FileNotFoundException exc) {
			handleErr(FILENOTFOUND);
		} catch(IOException exc) {
			handleErr(FILEIOERROR);
		}
		// If file ends with an EOF mark, back up.
		if(p[size-1] == (char) 26) size--;
		return size; // return size of program
	}
	// Execute the program.
	public void run() throws InterpreterException {
		// Initialize for new program run.
		vars = new double[26];
		fStack = new Stack();
		labelTable = new TreeMap();
		gStack = new Stack();
		progIdx = 0;
		scanLabels(); // find the labels in the program
		sbInterp(); // execute
	}
	// Entry point for the Small BASIC interpreter.
	private void sbInterp() throws InterpreterException
	{
		// This is the interpreter's main loop.
		do {
			getToken();
		// Check for assignment statement.
		if(tokType==VARIABLE) {
			putBack(); // return the var to the input stream
			assignment(); // handle assignment statement
		}
		else // is keyword
		switch(kwToken) {
			case PRINT:
			print();
			break;
			case GOTO:
				execGoto();
				break;
			case IF:
				execIf();
				break;
			case FOR:
				execFor();
				break;
			case NEXT:
				next();
				break;	
			case INPUT:
				input();
				break;
			case GOSUB:
				gosub();
				break;
			case RETURN:
				greturn();
				break;
			case OPEN:
				open();
				break;
			case ADDRECT:
				addrect();
				break;
			case ADDLINE:
				addline();
				break;
			case END:
				return;
			
			}
		} while (!token.equals(EOP));
	}


	// Find all labels.
	private void scanLabels() throws InterpreterException
	{
		int i;
		Object result;
		// See if the first token in the file is a label.
		getToken();
		if(tokType==NUMBER)
		labelTable.put(token, new Integer(progIdx));
		findEOL();
		do {
			getToken();
			if(tokType==NUMBER) {// must be a line number
				result = labelTable.put(token,
				new Integer(progIdx));
				if(result != null)
				handleErr(DUPLABEL);
			}
			// If not on a blank line, find next line.
			if(kwToken != EOL) findEOL();
		} while(!token.equals(EOP));
		progIdx = 0; // reset index to start of program
	}
	// Find the start of the next line.
	private void findEOL()
	{
		while(progIdx < prog.length &&
		prog[progIdx] != '\n') ++progIdx;
		if(progIdx < prog.length) progIdx++;
	}
	// Assign a variable a value.
	private void assignment() throws InterpreterException
	{
		int var;
		double value;
		char vname;
		// Get the variable name.
		getToken();
		vname = token.charAt(0);
		if(!Character.isLetter(vname)) {
			handleErr(NOTVAR);
			return;
		}
		// Convert to index into variable table.
		var = (int) Character.toUpperCase(vname) - 'A';
		// Get the equal sign.
		getToken();
		if(!token.equals("=")) {
			handleErr(EQUALEXPECTED);
			return;
		}
		// Get the value to assign.
		value = evaluate();
		// Assign the value.
		vars[var] = value;
	}
	// Execute a simple version of the PRINT statement.
	private void print() throws InterpreterException
	{
		double result;
		int len=0, spaces;
		String lastDelim = "";
		do {
			getToken(); // get next list item
			if(kwToken==EOL || token.equals(EOP)) break;
			if(tokType==QUOTEDSTR) { // is string
				System.out.print(token);
				len += token.length();
				getToken();
			}
			else { // is expression
				putBack();
				result = evaluate();
				getToken();
				System.out.print(result);
				// Add length of output to running total.
				Double t = new Double(result);
				len += t.toString().length(); // save length
			}
			lastDelim = token;
			// If comma, move to next tab stop.
			if(lastDelim.equals(",")) {
				// compute number of spaces to move to next tab
				spaces = 8 - (len % 8);
				len += spaces; // add in the tabbing position
				while(spaces != 0) {
				System.out.print(" ");
				spaces--;
			}
		}
		else if(token.equals(";")) {
			System.out.print(" ");
			len++;
		}
		else if(kwToken != EOL && !token.equals(EOP))
			handleErr(SYNTAX);
		} while (lastDelim.equals(";") || lastDelim.equals(","));
		
		if(kwToken==EOL || token.equals(EOP)) {
			if(!lastDelim.equals(";") && !lastDelim.equals(","))
				System.out.println();
		}
		else handleErr(SYNTAX);
	}
	
	// Execute a GOTO statement.
	private void execGoto() throws InterpreterException
	{
		Integer loc;
		getToken(); // get label to go to
		// Find the location of the label.
		loc = (Integer) labelTable.get(token);
		if(loc == null)
			handleErr(UNDEFLABEL); // label not defined
		else // start program running at that loc
			progIdx = loc.intValue();
	}
	// Execute an IF statement.
	private void execIf() throws InterpreterException
	{
		double result;
		result = evaluate(); // get value of expression
		/* If the result is true (non-zero),
		process target of IF. Otherwise move on
		to next line in the program. */
		if(result != 0.0) {
			getToken();
			if(kwToken != THEN) {
				handleErr(THENEXPECTED);
				return;
			} // else, target statement will be executed
		}
		else findEOL(); // find start of next line
	}
	// Execute a FOR loop.
	private void execFor() throws InterpreterException
	{
		ForInfo stckvar = new ForInfo();
		double value;
		char vname;
		getToken(); // read the control variable
		vname = token.charAt(0);
		if(!Character.isLetter(vname)) {
			handleErr(NOTVAR);
			return;
		}
		// Save index of control var.
		stckvar.var = Character.toUpperCase(vname) - 'A';
		getToken(); // read the equal sign
		if(token.charAt(0) != '=') {
			handleErr(EQUALEXPECTED);
			return;
		}
		value = evaluate(); // get initial value
		vars[stckvar.var] = value;
		getToken(); // read and discard the TO
		if(kwToken != TO) handleErr(TOEXPECTED);
		stckvar.target = evaluate(); // get target value
		/* If loop can execute at least once,
		push info on stack. */
		if(value >= vars[stckvar.var]) {
			stckvar.loc = progIdx;
			fStack.push(stckvar);
		}
		else // otherwise, skip loop code altogether
		while(kwToken != NEXT) getToken();
	}
	// Execute a NEXT statement.
	private void next() throws InterpreterException
	{
		ForInfo stckvar;
		try {
			// Retrieve info for this For loop.
			stckvar = (ForInfo) fStack.pop();
			vars[stckvar.var]++; // increment control var
			// If done, return.
			if(vars[stckvar.var] > stckvar.target) return;
			// Otherwise, restore the info.
			fStack.push(stckvar);
			progIdx = stckvar.loc; // loop
		}catch(EmptyStackException exc) {
			handleErr(NEXTWITHOUTFOR);
		}
	}
	// Execute a simple form of INPUT.
	private void input() throws InterpreterException
	{
		int var;
		double val = 0.0;
		String str;
		BufferedReader br = new
		BufferedReader(new InputStreamReader(System.in));
		getToken(); // see if prompt string is present
		if(tokType == QUOTEDSTR) {
			// if so, print it and check for comma
			System.out.print(token);
			getToken();
			if(!token.equals(",")) handleErr(SYNTAX);
				getToken();
		}
		else System.out.print("? "); // otherwise, prompt with ?
		// get the input var
		var = Character.toUpperCase(token.charAt(0)) - 'A';
		try {
			str = br.readLine();
			val = Double.parseDouble(str); // read the value
		} catch (IOException exc) {
			handleErr(INPUTIOERROR);
		} catch (NumberFormatException exc) {
			/* You might want to handle this error
			differently than the other interpreter
			errors. */
			System.out.println("Invalid input.");
		}
		vars[var] = val; // store it
	}
	// Execute a GOSUB.
	private void gosub() throws InterpreterException
	{
		Integer loc;
		getToken();
		// Find the label to call.
		loc = (Integer) labelTable.get(token);
		if(loc == null)
			handleErr(UNDEFLABEL); // label not defined
		else {
			// Save place to return to.
			gStack.push(new Integer(progIdx));
			// Start program running at that loc.
			progIdx = loc.intValue();
		}
	}
	// Return from GOSUB.
	private void greturn() throws InterpreterException
		{
		Integer t;
		try {
			// Restore program index.
			t = (Integer) gStack.pop();
			progIdx = t.intValue();
		} catch(EmptyStackException exc) {
			handleErr(RETURNWITHOUTGOSUB);
		}
	}
	private void open() throws InterpreterException {
		int size[] = new int[2];//매개변수 저장
		int length=0;//길이
		do {
		getToken();
		length++;
		if(tokType == NUMBER) {
			size[length-1] = Integer.parseInt(token);//매개변수를 2개 받아 저장
		}
		}while(length<2);
		if(window == null) {//이미 생성 되어 있지 않을경우
			window = new BFrame();//새로 생성밑 초기화
			window.setVisible(true);
			g = window.getGraphics();
			g.setColor(Color.BLACK);
		}
		window.setSize(size[0], size[1]);//크기 설정
	}
	//토큰을 각자가 요구하는 만큼 받아 저장한 뒤 작업
	//창에 그림을 그리는 부분 세부요소만 다를 뿐 거의 동일
	private void addrect() throws InterpreterException {
		int point[] = new int[4];//매개변수를 저장할 배열
		int length=0;//매개변수 길이 저장
		do {
		getToken();// 토큰을 가져옴
		length++;//길이 증가
		if(tokType == NUMBER) {//숫자일떄만 작동
			point[length-1] = Integer.parseInt(token);
		}
		}while(length<4);
		if(window != null) {
			window.Draw.DrawRect(point[0], point[1], point[2], point[3]);//그리기
			window.repaint();
		}
	}
	private void addline() throws InterpreterException {
		int point[] = new int[4];
		int length=0;
		do {
		getToken();
		length++;
		if(tokType == NUMBER&&length!=5) {
			point[length-1] = Integer.parseInt(token);
		}
		}while(length<4);
		if(window != null) {
			window.Draw.DrawLine(point[0], point[1], point[2], point[3]);
			window.repaint();
		}
	}
	// **************** Expression Parser ****************
	// Parser entry point.
	private double evaluate() throws InterpreterException
	{
		double result = 0.0;
		getToken();
		if(token.equals(EOP))
			handleErr(NOEXP); // no expression present
		// Parse and evaluate the expression.
		result = evalExp1();
		putBack();
		return result;
	}
	// Process relational operators.
	private double evalExp1() throws InterpreterException
	{
		double l_temp, r_temp, result;
		char op;
		result = evalExp2();
		// If at end of program, return.
		if(token.equals(EOP)) return result;
		op = token.charAt(0);
		if(isRelop(op)) {
			l_temp = result;
			getToken();
			r_temp = evalExp1();
			switch(op) { // perform the relational operation
				case '<':
				if(l_temp < r_temp) result = 1.0;
				else result = 0.0;
				break;
				case LE:
				if(l_temp <= r_temp) result = 1.0;
				else result = 0.0;
				break;
				case '>':
				if(l_temp > r_temp) result = 1.0;
				else result = 0.0;
				break;
				case GE:
				if(l_temp >= r_temp) result = 1.0;
				else result = 0.0;
				break;
				case '=':
				if(l_temp == r_temp) result = 1.0;
				else result = 0.0;
				break;
				case NE:
				if(l_temp != r_temp) result = 1.0;
				else result = 0.0;
				break;
			}
		}
		return result;
	}
	// Add or subtract two terms.
	private double evalExp2() throws InterpreterException
	{
		char op;
		double result;
		double partialResult;
		result = evalExp3();
		while((op = token.charAt(0)) == '+' || op == '-') {
			getToken();
			partialResult = evalExp3();
			switch(op) {
				case '-':
					result = result - partialResult;
					break;
				case '+':
					result = result + partialResult;
					break;
				}
		}
		return result;
	}
	// Multiply or divide two factors.
	private double evalExp3() throws InterpreterException
	{
		char op;
		double result;
		double partialResult;
		result = evalExp4();
		while((op = token.charAt(0)) == '*' ||
			op == '/' || op == '%') {
			getToken();
			partialResult = evalExp4();
			switch(op) {
				case '*':
					result = result * partialResult;
					break;
				case '/':
					if(partialResult == 0.0)
					handleErr(DIVBYZERO);
					result = result / partialResult;
					break;
				case '%':
					if(partialResult == 0.0)
					handleErr(DIVBYZERO);
					result = result % partialResult;
					break;
			}
		}
		return result;
	}
	// Process an exponent.
	private double evalExp4() throws InterpreterException
	{
		double result;
		double partialResult;
		double ex;
		int t;
		result = evalExp5();
		if(token.equals("^")) {
			getToken();
			partialResult = evalExp4();
			ex = result;
			if(partialResult == 0.0) {
				result = 1.0;
			} else
			for(t=(int)partialResult-1; t > 0; t--)
			result = result * ex;
		}
		return result;
	}
	// Evaluate a unary + or -.
	private double evalExp5() throws InterpreterException
	{
		double result;
		String op;
		op = "";
		if((tokType == DELIMITER) && token.equals("+") || token.equals("-")) {
			op = token;
			getToken();
		}
		result = evalExp6();
		if(op.equals("-"))
			result = -result;
		return result;
	}
	// Process a parenthesized expression.
	private double evalExp6() throws InterpreterException
	{
		double result;
		if(token.equals("(")) {
			getToken();
			result = evalExp2();
			if(!token.equals(")"))
				handleErr(UNBALPARENS);
			getToken();
		}
		else result = atom();
		return result;
	}
	// Get the value of a number or variable.
	private double atom() throws InterpreterException
	{
		double result = 0.0;
		switch(tokType) {
			case NUMBER:
			try {
				result = Double.parseDouble(token);
			} catch (NumberFormatException exc) {
				handleErr(SYNTAX);
				}
				getToken();
				break;
			case VARIABLE:
				result = findVar(token);
				getToken();
				break;
			default:
				handleErr(SYNTAX);
				break;
		}
		return result;
	}
	// Return the value of a variable.
	private double findVar(String vname) throws InterpreterException
	{
		if(!Character.isLetter(vname.charAt(0))){
			handleErr(SYNTAX);
			return 0.0;
		}
		return vars[Character.toUpperCase(vname.charAt(0))-'A'];
	}
	// Return a token to the input stream.
	private void putBack()
	{
		if(token == EOP) return;
		for(int i=0; i < token.length(); i++) progIdx--;
	}
	// Handle an error.
	private void handleErr(int error) throws InterpreterException
	{
		String[] err = {
		"Syntax Error",
		"Unbalanced Parentheses",
		"No Expression Present",
		"Division by Zero",
		"Equal sign expected",
		"Not a variable",
		"Label table full",
		"Duplicate label",
		"Undefined label",
		"THEN expected",
		"TO expected",
		"NEXT without FOR",
		"RETURN without GOSUB",
		"Closing quotes needed",
		"File not found",
		"I/O error while loading file",
		"I/O error on INPUT statement"
		};
		throw new InterpreterException(err[error]);
	}
	// Obtain the next token.
	private void getToken() throws InterpreterException
	{
		char ch;
		tokType = NONE;
		token = "";
		kwToken = UNKNCOM;
		// Check for end of program.
		if(progIdx == prog.length) {
			token = EOP;
			return;
		}
		// Skip over white space.
		while(progIdx < prog.length && isSpaceOrTab(prog[progIdx])) 
			progIdx++;
		// Trailing whitespace ends program.
		if(progIdx == prog.length) {
			token = EOP;
			tokType = DELIMITER;
			return;
		}
		if(prog[progIdx] == '\r') { // handle crlf
			progIdx += 2;
			kwToken = EOL;
			token = "\r\n";
			return;
		}
		// Check for relational operator.
		ch = prog[progIdx];
		if(ch == '<' || ch == '>') {
			if(progIdx+1 == prog.length) handleErr(SYNTAX);
			switch(ch) {
				case '<':
					if(prog[progIdx+1] == '>') {
						progIdx += 2;;
						token = String.valueOf(NE);
					}
					else if(prog[progIdx+1] == '=') {
						progIdx += 2;
						token = String.valueOf(LE);
					}
					else {
						progIdx++;
						token = "<";
					}
					break;
				case '>':
					if(prog[progIdx+1] == '=') {
						progIdx += 2;;
						token = String.valueOf(GE);
					}
				else {
					progIdx++;
					token = ">";
				}
				break;
			}
			tokType = DELIMITER;
			return;
		}
		if(isDelim(prog[progIdx])) {
			// Is an operator.
			token += prog[progIdx];
			progIdx++;
			tokType = DELIMITER;
		}
		else if(Character.isLetter(prog[progIdx])) {
			// Is a variable or keyword.
			while(!isDelim(prog[progIdx])) {
				token += prog[progIdx];
				progIdx++;
				if(progIdx >= prog.length) break;
			}
			kwToken = lookUp(token);
			if(kwToken==UNKNCOM) tokType = VARIABLE;
			else tokType = COMMAND;
		}
		else if(Character.isDigit(prog[progIdx])) {
			// Is a number.
			while(!isDelim(prog[progIdx])) {
				token += prog[progIdx];
				progIdx++;
				if(progIdx >= prog.length)
					break;
			}
			tokType = NUMBER;
		}
		else if(prog[progIdx] == '"') {
			// Is a quoted string.
			progIdx++;
			ch = prog[progIdx];
			while(ch !='"' && ch != '\r') {
				token += ch;
				progIdx++;
				ch = prog[progIdx];
			}
			if(ch == '\r') handleErr(MISSINGQUOTE);
			progIdx++;
			tokType = QUOTEDSTR;
		}
		else { // unknown character terminates program
			token = EOP;
			return;
		}
	}
	// Return true if c is a delimiter.
	private boolean isDelim(char c)
	{
		if((" \r,;<>+-/*%^=()".indexOf(c) != -1))
		return true;
		return false;
	}
	// Return true if c is a space or a tab.
	boolean isSpaceOrTab(char c)
	{
		if(c == ' ' || c =='\t') return true;
		return false;
	}
	// Return true if c is a relational operator.
	boolean isRelop(char c) {
		if(relops.indexOf(c) != -1) return true;
		return false;
	}
	/* Look up a token's internal representation in the
	token table. */
	private int lookUp(String s)
	{
		int i;
		// Convert to lowercase.
		s = s.toLowerCase();
		// See if token is in table.
		for(i=0; i < kwTable.length; i++)
			if(kwTable[i].keyword.equals(s))
				return kwTable[i].keywordTok;
		return UNKNCOM; // unknown keyword
	}
}


public class exmp{
	public static void main(String args[]) {
		try {
			interpreter i = new interpreter("SBTEST.txt");
			i.run();
		}catch(InterpreterException exc) {
			System.out.println(exc);
		}
	}
}