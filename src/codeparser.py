import tree_sitter_python as tspython

from tree_sitter import Language, Parser


class CodeParser:

    def __init__(self):
        self.language = Language(tspython.language(), 'python')
        self.parser = Parser()
        self.parser.set_language(self.language)

    def parse_file(self, file_content: str):
        code = bytes(file_content, 'utf8')
        tree = self.parser.parse(code)
        return tree, code

    def extract_items(self, node, code):
        functions, classes, global_contexts = [], [], []

        def traverse(node, is_top_level=True):
            if not is_top_level:
                return

            current_type = node.type
            if current_type == 'decorated_definition':
                if node.children[1].type == 'function_definition':
                    content, start_line, end_line = self.extract_content_with_lines(node, code)
                    functions.append({'nodetype': 'decorated_definition', 'content': content})
                    is_top_level = False
                elif node.children[1].type == 'class_definition':
                    content, start_line, end_line = self.extract_content_with_lines(node, code)
                    classes.append({'nodetype': 'decorated_definition', 'content': content})
                    is_top_level = False

            if current_type == 'function_definition' and is_top_level:
                content, start_line, end_line = self.extract_content_with_lines(node, code)
                functions.append({'nodetype': 'function_definition', 'content': content})
                is_top_level = False

            elif current_type == 'class_definition' and is_top_level:
                content, start_line, end_line = self.extract_content_with_lines(node, code)
                classes.append({'nodetype': 'class_definition', 'content': content})
                is_top_level = False

            elif current_type != 'module' and is_top_level:
                content, start_line, end_line = self.extract_content_with_lines(node, code)
                global_contexts.append({'nodetype': 'global_context', 'content': content})
                is_top_level = False

            for child in node.children:
                traverse(child, is_top_level=is_top_level)

        traverse(node)
        return {
            'functions': functions,
            'classes': classes,
            'global_contexts': global_contexts,
        }

    def extract_content_with_lines(self, node, code):
        content = code[node.start_byte:node.end_byte].decode('utf8')
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        return content, start_line, end_line
