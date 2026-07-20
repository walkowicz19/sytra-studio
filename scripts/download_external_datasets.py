import os
import json
import pandas as pd
from huggingface_hub import hf_hub_download

def slice_parquet(repo_id, filename, output_name, prompt_col, completion_col, count=100):
    print(f"Downloading {repo_id}/{filename}...")
    local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    print(f"Loading and slicing {filename}...")
    df = pd.read_parquet(local_path)
    df_slice = df.head(count)
    
    output_path = f"fixtures/{output_name}.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df_slice.iterrows():
            item = {
                "instruction": str(row.get(prompt_col, "")),
                "output": str(row.get(completion_col, "")),
                "domain": "programming",
                "source_dataset": repo_id
            }
            f.write(json.dumps(item) + "\n")
    print(f"Saved {count} rows to {output_path}")

def slice_jsonl(repo_id, filename, output_name, prompt_col, completion_col, count=100):
    print(f"Downloading {repo_id}/{filename}...")
    local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    print(f"Loading and slicing {filename}...")
    
    output_path = f"fixtures/{output_name}.jsonl"
    rows_written = 0
    
    with open(local_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            if rows_written >= count:
                break
            try:
                row = json.loads(line)
                # Handle nested or list types if needed
                prompt_val = row.get(prompt_col, "")
                completion_val = row.get(completion_col, "")
                
                # Special parsing for datasets like APPS where solutions are stored as a JSON string list
                if output_name == "apps_train" and isinstance(completion_val, str):
                    try:
                        sols = json.loads(completion_val)
                        if isinstance(sols, list) and len(sols) > 0:
                            completion_val = sols[0]
                    except Exception:
                        pass
                
                item = {
                    "instruction": str(prompt_val),
                    "output": str(completion_val),
                    "domain": "programming",
                    "source_dataset": repo_id
                }
                outfile.write(json.dumps(item) + "\n")
                rows_written += 1
            except Exception as e:
                print(f"Skipping malformed row: {e}")
                
    print(f"Saved {rows_written} rows to {output_path}")

def main():
    os.makedirs("fixtures", exist_ok=True)
    
    # 1. SWE-bench Lite (Parquet)
    # prompt: problem_description, completion: patch
    try:
        slice_parquet(
            repo_id="SWE-bench/SWE-bench_Lite",
            filename="data/dev-00000-of-00001.parquet",
            output_name="swe_bench_lite",
            prompt_col="problem_statement",
            completion_col="patch"
        )
    except Exception as e:
        print(f"Error downloading SWE-bench Lite: {e}")
        
    # 2. APPS (JSONL)
    # prompt: question, completion: solutions
    try:
        slice_jsonl(
            repo_id="codeparrot/apps",
            filename="train.jsonl",
            output_name="apps_train",
            prompt_col="question",
            completion_col="solutions"
        )
    except Exception as e:
        print(f"Error downloading APPS: {e}")
        
    # 3. MBPP (Parquet) - Replacing LiveCodeBench due to empty test targets
    try:
        slice_parquet(
            repo_id="mbpp",
            filename="sanitized/train-00000-of-00001.parquet",
            output_name="live_code_bench",
            prompt_col="prompt",
            completion_col="code"
        )
    except Exception as e:
        print(f"Error downloading MBPP: {e}")
        
        
    # 4. CodeFeedback-Filtered-Instruction (JSONL)
    # prompt: query, completion: answer
    try:
        slice_jsonl(
            repo_id="m-a-p/CodeFeedback-Filtered-Instruction",
            filename="CodeFeedback-Filtered-Instruction.jsonl",
            output_name="code_feedback",
            prompt_col="query",
            completion_col="answer"
        )
    except Exception as e:
        print(f"Error downloading CodeFeedback: {e}")

    # 5. NousResearch/hermes-function-calling-v1 (JSON/Glaive) - 6th dataset
    try:
        download_hermes_agentic_tooling(count=100)
    except Exception as e:
        print(f"Error downloading Hermes Agentic Tooling: {e}")

def parse_glaive_item(item):
    tools_str = item.get("tools", "")
    conversations = item.get("conversations", [])
    
    human_query = ""
    gpt_response = ""
    
    for i, turn in enumerate(conversations):
        if turn["from"] == "human" and not human_query:
            human_query = turn["value"]
            if i + 1 < len(conversations) and conversations[i+1]["from"] == "gpt":
                gpt_response = conversations[i+1]["value"]
            break
            
    if not human_query or not gpt_response:
        return None
        
    instruction = (
        f"You are a function calling AI model. Here are the available tools:\n"
        f"<tools>\n{tools_str}\n</tools>\n\n"
        f"Query: {human_query}"
    )
    
    return {
        "instruction": instruction,
        "output": gpt_response,
        "domain": "agentic-tooling",
        "source_dataset": "NousResearch/hermes-function-calling-v1"
    }

def generate_direct_response_samples():
    languages = ["Python", "Java", "Rust", "Go", "JavaScript", "TypeScript"]
    topics = [
        {
            "query_template": "Write a function to check if a string is a palindrome in {lang}.",
            "output_templates": {
                "Python": "Here is a Python function to check for palindromes:\n\n```python\ndef is_palindrome(s: str) -> bool:\n    clean_s = ''.join(c.lower() for c in s if c.isalnum())\n    return clean_s == clean_s[::-1]\n```",
                "Java": "Here is a Java method to check for palindromes:\n\n```java\npublic class Palindrome {\n    public static boolean isPalindrome(String s) {\n        String clean = s.replaceAll(\"[^a-zA-Z0-9]\", \"\").toLowerCase();\n        return clean.equals(new StringBuilder(clean).reverse().toString());\n    }\n}\n```",
                "Rust": "Here is a Rust function to check for palindromes:\n\n```rust\nfn is_palindrome(s: &str) -> bool {\n    let clean: String = s.chars().filter(|c| c.is_alphanumeric()).map(|c| c.to_ascii_lowercase()).collect();\n    clean == clean.chars().rev().collect::<String>()\n}\n```",
                "Go": "Here is a Go function to check for palindromes:\n\n```go\npackage main\n\nimport (\n\t\"regexp\"\n\t\"strings\"\n)\n\nfunc isPalindrome(s string) bool {\n\tclean := strings.ToLower(regexp.MustCompile(`[^a-zA-Z0-9]`).ReplaceAllString(s, \"\"))\n\tfor i := 0; i < len(clean)/2; i++ {\n\t\tif clean[i] != clean[len(clean)-1-i] {\n\t\t\treturn false\n\t\t}\n\t}\n\treturn true\n}\n```",
                "JavaScript": "Here is a JavaScript function to check for palindromes:\n\n```javascript\nfunction isPalindrome(s) {\n  const clean = s.replace(/[^a-zA-Z0-9]/g, '').toLowerCase();\n  return clean === clean.split('').reverse().join('');\n}\n```",
                "TypeScript": "Here is a TypeScript function to check for palindromes:\n\n```typescript\nfunction isPalindrome(s: string): boolean {\n  const clean = s.replace(/[^a-zA-Z0-9]/g, '').toLowerCase();\n  return clean === clean.split('').reverse().join('');\n}\n```"
            }
        },
        {
            "query_template": "How do you read a text file line by line in {lang}?",
            "output_templates": {
                "Python": "You can read a file line by line in Python using a context manager:\n\n```python\nwith open('file.txt', 'r', encoding='utf-8') as f:\n    for line in f:\n        print(line.strip())\n```",
                "Java": "You can read a file line by line in Java using `Files.lines`:\n\n```java\nimport java.io.IOException;\nimport java.nio.file.Files;\nimport java.nio.file.Paths;\n\npublic class ReadFile {\n    public static void main(String[] args) {\n        try {\n            Files.lines(Paths.get(\"file.txt\")).forEach(System.out::println);\n        } catch (IOException e) {\n            e.printStackTrace();\n        }\n    }\n}\n```",
                "Rust": "You can read a file line by line in Rust using `BufReader`:\n\n```rust\nuse std::fs::File;\nuse std::io::{self, BufRead, BufReader};\n\nfn read_lines() -> io::Result<()> {\n    let file = File::open(\"file.txt\")?;\n    let reader = BufReader::new(file);\n    for line in reader.lines() {\n        println!(\"{}\", line?);\n    }\n    Ok(())\n}\n```",
                "Go": "You can read a file line by line in Go using `bufio.Scanner`:\n\n```go\npackage main\n\nimport (\n\t\"bufio\"\n\t\"fmt\"\n\t\"os\"\n)\n\nfunc main() {\n\tfile, _ := os.Open(\"file.txt\")\n\tdefer file.Close()\n\tscanner := bufio.NewScanner(file)\n\tfor scanner.Scan() {\n\t\tfmt.Println(scanner.Text())\n\t}\n}\n```",
                "JavaScript": "You can read a file line by line in Node.js using the `readline` module:\n\n```javascript\nconst fs = require('fs');\nconst readline = require('readline');\n\nconst rl = readline.createInterface({\n  input: fs.createReadStream('file.txt'),\n  output: process.stdout,\n  terminal: false\n});\n\nrl.on('line', (line) => {\n  console.log(line);\n});\n```",
                "TypeScript": "You can read a file line by line in TypeScript/Node.js using the `readline` module:\n\n```typescript\nimport * as fs from 'fs';\nimport * as readline from 'readline';\n\nconst rl = readline.createInterface({\n  input: fs.createReadStream('file.txt'),\n  output: process.stdout,\n  terminal: false\n});\n\nrl.on('line', (line: string) => {\n  console.log(line);\n});\n```"
            }
        },
        {
            "query_template": "Generate a unit test for a simple calculator add function in {lang}.",
            "output_templates": {
                "Python": "Here is a Python unit test using `unittest`:\n\n```python\nimport unittest\n\ndef add(a, b):\n    return a + b\n\nclass TestCalculator(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n        self.assertEqual(add(-1, 1), 0)\n```",
                "Java": "Here is a Java unit test using JUnit 5:\n\n```java\nimport org.junit.jupiter.api.Test;\nimport static org.junit.jupiter.api.Assertions.assertEquals;\n\npublic class CalculatorTest {\n    @Test\n    public void testAdd() {\n        Calculator calc = new Calculator();\n        assertEquals(5, calc.add(2, 3));\n    }\n}\n```",
                "Rust": "Here is a Rust unit test using the built-in test framework:\n\n```rust\nfn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n\n#[cfg(test)]\nmod tests {\n    use super::*;\n\n    #[test]\n    fn test_add() {\n        assert_eq!(add(2, 3), 5);\n    }\n}\n```",
                "Go": "Here is a Go unit test using the standard `testing` package:\n\n```go\npackage main\n\nimport \"testing\"\n\nfunc Add(a, b int) int {\n\treturn a + b\n}\n\nfunc TestAdd(t *testing.T) {\n\tif Add(2, 3) != 5 {\n\t\tt.Errorf(\"Expected 5, got %d\", Add(2, 3))\n\t}\n}\n```",
                "JavaScript": "Here is a JavaScript unit test using Jest:\n\n```javascript\nconst add = (a, b) => a + b;\n\ntest('adds 2 + 3 to equal 5', () => {\n  expect(add(2, 3)).toBe(5);\n});\n```",
                "TypeScript": "Here is a TypeScript unit test using Jest:\n\n```typescript\nconst add = (a: number, b: number): number => a + b;\n\ntest('adds 2 + 3 to equal 5', () => {\n  expect(add(2, 3)).toBe(5);\n});\n```"
            }
        },
        {
            "query_template": "Write a basic HTTP server in {lang}.",
            "output_templates": {
                "Python": "Here is a basic HTTP server in Python using `http.server`:\n\n```python\nfrom http.server import SimpleHTTPRequestHandler, HTTPServer\n\nclass MyHandler(SimpleHTTPRequestHandler):\n    def do_GET(self):\n        self.send_response(200)\n        self.send_header('Content-type', 'text/html')\n        self.end_headers()\n        self.wfile.write(b\"Hello, world!\")\n\nserver = HTTPServer(('localhost', 8000), MyHandler)\nserver.serve_forever()\n```",
                "Java": "Here is a basic HTTP server in Java using standard JDK components:\n\n```java\nimport com.sun.net.httpserver.HttpServer;\nimport java.io.OutputStream;\nimport java.net.InetSocketAddress;\n\npublic class SimpleServer {\n    public static void main(String[] args) throws Exception {\n        HttpServer server = HttpServer.create(new InetSocketAddress(8000), 0);\n        server.createContext(\"/\", exchange -> {\n            String response = \"Hello, world!\";\n            exchange.sendResponseHeaders(200, response.length());\n            OutputStream os = exchange.getResponseBody();\n            os.write(response.getBytes());\n            os.close();\n        });\n        server.start();\n    }\n}\n```",
                "Rust": "Here is a basic HTTP server in Rust using the standard `TcpListener`:\n\n```rust\nuse std::io::prelude::*;\nuse std::net::{TcpListener, TcpStream};\n\nfn handle_connection(mut stream: TcpStream) {\n    let response = \"HTTP/1.1 200 OK\\r\\n\\r\\nHello, world!\";\n    stream.write(response.as_bytes()).unwrap();\n    stream.flush().unwrap();\n}\n\nfn main() {\n    let listener = TcpListener::bind(\"127.0.0.1:8000\").unwrap();\n    for stream in listener.incoming() {\n        handle_connection(stream.unwrap());\n    }\n}\n```",
                "Go": "Here is a basic HTTP server in Go using `net/http`:\n\n```go\npackage main\n\nimport (\n\t\"fmt\"\n\t\"net/http\"\n)\n\nfunc main() {\n\thttp.HandleFunc(\"/\", func(w http.ResponseWriter, r *http.Request) {\n\t\tfmt.Fprintf(w, \"Hello, world!\")\n\t})\n\thttp.ListenAndServe(\":8000\", nil)\n}\n```",
                "JavaScript": "Here is a basic HTTP server in Node.js:\n\n```javascript\nconst http = require('http');\n\nconst server = http.createServer((req, res) => {\n  res.statusCode = 200;\n  res.setHeader('Content-Type', 'text/plain');\n  res.end('Hello, world!\\n');\n});\n\nserver.listen(8000, '127.0.0.1', () => {\n  console.log('Server running...');\n});\n```",
                "TypeScript": "Here is a basic HTTP server in TypeScript/Node.js:\n\n```typescript\nimport * as http from 'http';\n\nconst server = http.createServer((req: http.IncomingMessage, res: http.ServerResponse) => {\n  res.statusCode = 200;\n  res.setHeader('Content-Type', 'text/plain');\n  res.end('Hello, world!\\n');\n});\n\nserver.listen(8000, '127.0.0.1', () => {\n  console.log('Server running...');\n});\n```"
            }
        },
        {
            "query_template": "Explain the difference between stack and heap memory in {lang}.",
            "output_templates": {
                "Python": "In Python, memory management is handled automatically by the private heap space:\n\n1. **Stack Memory**: Python uses the call stack to keep track of function execution frames, primitive values, and references to objects.\n2. **Heap Memory**: All Python objects (like lists, dictionaries, custom class instances) are allocated on the private heap. The reference count and garbage collector manage heap memory dynamically.",
                "Java": "In Java, memory is clearly separated between Stack and Heap:\n\n1. **Stack Memory**: Used for execution of a thread. It contains short-lived method-specific variables, local primitives, and references to heap objects. Allocation is LIFO.\n2. **Heap Memory**: Used for dynamic allocation of Java objects and JRE classes. All objects created via `new` live on the heap. Managed by the Garbage Collector.",
                "Rust": "In Rust, memory is strictly managed according to ownership rules:\n\n1. **Stack Memory**: Used for values with a fixed, known size at compile time (like local variables, primitives, arrays, and function execution contexts). Access is very fast.\n2. **Heap Memory**: Used for data whose size can change dynamically or is unknown at compile time (e.g., `Vec`, `String`, `Box`). Deallocation is deterministic and happens when the owner goes out of scope.",
                "Go": "In Go, the compiler uses escape analysis to decide where to store variables:\n\n1. **Stack Memory**: Used for variables whose lifetime does not extend past the function execution. Very fast allocation/deallocation.\n2. **Heap Memory**: Variables that 'escape' the function boundary (e.g., returned pointers, shared variables) are allocated on the heap and managed by the Go Garbage Collector.",
                "JavaScript": "In JavaScript engines (like V8), memory is divided between Stack and Heap:\n\n1. **Stack Memory**: Used for primitive values (numbers, strings, booleans, null, undefined) and references to objects. Manages the execution context stack.\n2. **Heap Memory**: Used for reference types (objects, arrays, functions) whose size is dynamic. Cleaned up automatically by the garbage collector.",
                "TypeScript": "In TypeScript (which compiles to JavaScript), memory management follows the host JS engine:\n\n1. **Stack Memory**: Stores primitives and reference pointers during function execution.\n2. **Heap Memory**: Stores objects, arrays, and class instances. The JS garbage collector handles deallocation automatically."
            }
        }
    ]
    
    tools_str = '[{"type": "function", "function": {"name": "search_code", "description": "Search codebase for a pattern"}}, {"type": "function", "function": {"name": "execute_command", "description": "Execute a terminal command"}}, {"type": "function", "function": {"name": "write_to_file", "description": "Write code content to a file"}}, {"type": "function", "function": {"name": "read_resource", "description": "Read local documentation resource"}}]'
    
    samples = []
    for topic in topics:
        for lang in languages:
            query = topic["query_template"].format(lang=lang)
            output_content = topic["output_templates"][lang]
            
            instruction = (
                f"You are a function calling AI model. Here are the available tools:\n"
                f"<tools>\n{tools_str}\n</tools>\n\n"
                f"Query: {query}"
            )
            
            samples.append({
                "instruction": instruction,
                "output": output_content,
                "domain": "agentic-tooling",
                "source_dataset": "custom-direct-response"
            })
            
    return samples

def download_hermes_agentic_tooling(count=100):
    print("Downloading NousResearch/hermes-function-calling-v1/glaive-function-calling-5k.json...")
    local_path = hf_hub_download(
        repo_id="NousResearch/hermes-function-calling-v1",
        filename="glaive-function-calling-5k.json",
        repo_type="dataset"
    )
    print("Parsing and formatting hermes agentic tooling dataset...")
    with open(local_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    output_path = "fixtures/hermes_agentic_tooling.jsonl"
    rows_written = 0
    with open(output_path, "w", encoding="utf-8") as out:
        # 1. Write 70 Glaive function-calling examples
        for item in data:
            if rows_written >= 70:
                break
            parsed = parse_glaive_item(item)
            if parsed:
                out.write(json.dumps(parsed) + "\n")
                rows_written += 1
                
        # 2. Write 30 custom direct response examples
        direct_samples = generate_direct_response_samples()
        for parsed in direct_samples:
            out.write(json.dumps(parsed) + "\n")
            rows_written += 1
            
    print(f"Saved {rows_written} rows to {output_path}")

if __name__ == "__main__":
    main()
