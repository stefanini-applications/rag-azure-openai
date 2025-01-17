import sys

def reencode_file(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='latin1') as infile:
            content = infile.read()
        
        with open(output_file, 'w', encoding='utf-8') as outfile:
            outfile.write(content)
        
        print(f"File re-encoded successfully and saved to: {output_file}")
    except Exception as e:
        print(f"Error re-encoding file: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python reencode_to_utf8.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    reencode_file(input_file, output_file)
