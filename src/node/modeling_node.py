import os

from typing import List, Dict
from src.codeparser import CodeParser

codeparser = CodeParser()


class ContextNode():
    parsed: bool = False
    embedding: List[float] = None
    score: float = 0.0
    temp_score: float = 0.0
    related: str = None
    revelance: float = 0.0

    def __init__(
        self,
        nodetype: str,
        content: str,
    ):

        self.content = content

    def __repr__(self) -> str:
        info = {
            'content': self.content,
        }
        return str(info)


class FunctionNode(ContextNode):

    def __init__(
        self,
        nodetype: str,
        content: str,
        is_class_method: bool = False,
    ):
        super().__init__(nodetype, content)

        tree, _ = codeparser.parse_file(self.content)
        try:
            assert tree.root_node.children[0].type in ['function_definition', 'decorated_definition'], 'No function definition found in the provided content.'
            if tree.root_node.children[0].type == 'function_definition':
                function_node = tree.root_node.children[0]
            else:
                function_node = tree.root_node.children[0].children[1]

            body = function_node.child_by_field_name('body').text.decode('utf8')
            func_head = self.content.split(function_node.child_by_field_name('body').text.decode('utf8'))[0]
            self.func_name = function_node.child_by_field_name('name').text.decode('utf8')
            if is_class_method:
                self.func_head = '    ' + func_head
                self.func_body = '        ' + body
            else:
                self.func_head = func_head
                self.func_body = '    ' + body

            self.parsed = True
        except:
            self.parsed = False
            print(f"Could not parse function:\n {self.content}")

    def __repr__(self) -> str:
        info = {
            'content': self.content,
        }
        return str(info)

    def get_concise_context(self):
        concise_context = self.func_head + '# Omitting the implementation details' + '\n    ' + 'pass'

        return concise_context

    def get_context(self, is_class_method=False):
        self.temp_score = 0.0
        if self.related == 'high':
            context = self.content
            self.revelance = 1.0
            self.temp_score = self.score * self.revelance
        elif self.related == 'medium':
            self.revelance = 0.5
            self.temp_score = self.score * self.revelance
            if is_class_method:
                context = self.func_head + '# Omitting the implementation details\n' + '        ' + 'pass'
            else:
                context = self.func_head + '# Omitting the implementation details\n' + '    ' + 'pass'
        else:
            self.revelance = 0.0
            self.temp_score = self.score * self.revelance
            context = ""
        return context


class ClassNode(ContextNode):

    def __init__(
        self,
        nodetype: str,
        content: str,
    ):
        super().__init__(nodetype, content)

        tree, _ = codeparser.parse_file(self.content)
        decorated_node = None
        self.class_method_nodes = []
        try:
            assert tree.root_node.children[0].type in ['class_definition', 'decorated_definition'], 'No class definition found in the provided content.'
            if tree.root_node.children[0].type == 'class_definition':
                class_node = tree.root_node.children[0]
            else:
                decorated_node = tree.root_node.children[0].children[0]
                class_node = tree.root_node.children[0].children[1]

            self.class_name = class_node.child_by_field_name('name').text.decode('utf8')
            if decorated_node is None:
                self.class_head = 'class ' + self.class_name + ':'
            else:
                self.class_head = decorated_node.text.decode('utf8') + '\n' + 'class ' + self.class_name + ':'
            self.body = class_node.child_by_field_name('body').text.decode('utf8')

            # parse class methods
            for child in class_node.child_by_field_name('body').children:
                if child.type == 'function_definition':
                    self.class_method_nodes.append(FunctionNode(
                        nodetype='function_definition',
                        content=child.text.decode('utf8'),
                        is_class_method=True,
                    ))
                if child.type == 'decorated_definition' and child.children[1].type == 'function_definition':
                    self.class_method_nodes.append(FunctionNode(
                        nodetype='decorated_definition',
                        content=child.text.decode('utf8'),
                        is_class_method=True,
                    ))
            self.parsed = True
        except:
            self.parsed = False
            print(f"Could not parse class:\n {self.content}")

    def __repr__(self) -> str:
        info = {
            'content': self.content,
        }
        return str(info)

    def get_concise_context(self):
        context = [self.class_head]
        for method in self.class_method_nodes:
            if method.parsed:
                context.append(method.func_head + '# Omitting the implementation details\n' + '        ' + 'pass')
        concise_context = '\n\n'.join(context)
        return concise_context

    def get_context(self):
        context = [self.class_head]
        self.temp_score = 0.0
        for method in self.class_method_nodes:
            if method.parsed:
                content = method.get_context(is_class_method=True)
                if content != "":
                    context.append(content)
                self.temp_score += method.temp_score
        context = '\n\n'.join(context)
        return context


class FileNode():

    score: float = 0.0

    def __init__(
        self,
        repo_path: str,
        file_path: str,
    ):

        self.repo_path = repo_path
        self.file_path = file_path
        self.file_name = file_path.split('/')[-1]
        self.file_type = file_path.split('.')[-1]
        with open(file_path, 'r') as file:
            self.content = file.read()

        # parse the file content
        tree, code = codeparser.parse_file(self.content)
        parsed_file = codeparser.extract_items(tree.root_node, code)

        self.dependencies = self.get_dependencies(tree.root_node)

        self.functions = self.build_function_nodes(parsed_file["functions"])
        self.classes = self.build_class_nodes(parsed_file["classes"])
        self.global_contexts = self.build_context_nodes(parsed_file["global_contexts"])

    def get_dependencies(self, root_node) -> List[str]:

        def find_imports(node):
            imports = []
            for child in node.children:
                if child.type == 'import_statement':
                    module_name = ''
                    imported_items = []
                    for sub_child in child.children:
                        if sub_child.type in ['dotted_name', 'relative_import']:
                            module_name = ''.join([n.text.decode('utf8') for n in sub_child.children])
                            imports.append({'module_name': module_name, 'imported_items': []})
                        if sub_child.type == 'aliased_import':
                            imported_items.extend([n.text.decode('utf8') for n in sub_child.children if n.type == 'dotted_name'])

                if child.type == 'import_from_statement':
                    module_name = ''
                    imported_items = []
                    after_from, after_import = False, False
                    for sub_child in child.children:
                        if sub_child.type == 'from':
                            after_from = True
                        if sub_child.type == 'import':
                            after_import = True

                        if sub_child.type in ['dotted_name', 'relative_import'] and after_from and not after_import:
                            module_name = ''.join([n.text.decode('utf8') for n in sub_child.children])
                        if sub_child.type in ['dotted_name', 'wildcard_import'] and after_from and after_import:
                            imported_items.append('.'.join([n.text.decode('utf8') for n in sub_child.children]))
                        if sub_child.type == 'aliased_import':
                            imported_items.extend([n.text.decode('utf8') for n in sub_child.children if n.type == 'dotted_name'])
                    imports.append({'module_name': module_name, 'imported_items': imported_items})
            return imports

        from_imports = find_imports(root_node)
        dependencies = {}
        for from_import in from_imports:
            module_name, imported_items = from_import['module_name'], from_import['imported_items']
            for import_item in imported_items:
                module_path, item = self.resolve_import(module_name, import_item)
                if module_path is not None and module_path != self.file_path:
                    if module_path not in dependencies:
                        dependencies[module_path] = []
                    dependencies[module_path].append(item)

        return dependencies

    def __repr__(self) -> str:
        info = {
            'repo_path': self.repo_path,
            'file_path': self.file_path,
            'file_name': self.file_name,
            'file_type': self.file_type,
            'content': self.content,
            'dependencies': self.dependencies,
            'functions': self.functions,
            'classes': self.classes,
            'global_contexts': self.global_contexts,
        }
        return str(info)

    def resolve_import(self, module_name: str, imported_item: str) -> str:
        # resolve relative imports
        if module_name.startswith('.'):
            parent_path = self.file_path
            _module_name = module_name
            while _module_name.startswith('.'):
                parent_path = os.path.dirname(parent_path)
                _module_name = _module_name[1:]
            module_path = _module_name.replace('.', os.sep)
            module_file_path = os.path.join(parent_path, module_path + '.py')
            if os.path.isfile(module_file_path):
                return module_file_path, imported_item
            module_file_path = os.path.join(parent_path, module_path + '.pyi')
            if os.path.isfile(module_file_path):
                return module_file_path, imported_item
            if os.path.isdir(os.path.join(parent_path, module_path)):
                init_path = os.path.join(parent_path, module_path, '__init__.py')
                if os.path.isfile(init_path):
                    return init_path, imported_item
                item_path = os.path.join(parent_path, module_path, imported_item + '.py')
                if os.path.isfile(item_path):
                    return item_path, '*'
                item_path = os.path.join(parent_path, module_path, imported_item + '.pyi')
                if os.path.isfile(item_path):
                    return item_path, '*'
            print(f"Could not resolve import: {module_name} in {self.file_path}")
            return None, None

        module_path = module_name.replace('.', os.sep)

        py_path = os.path.join(self.repo_path, module_path + '.py')
        if os.path.isfile(py_path):
            return py_path, imported_item
        py_path = os.path.join(self.repo_path, module_path + '.pyi')
        if os.path.isfile(py_path):
            return py_path, imported_item

        init_path = os.path.join(self.repo_path, module_path, '__init__.py')
        if os.path.isfile(init_path):
            return init_path, imported_item

        return None, None

    def build_context_nodes(self, contexts: List[Dict]) -> List[ContextNode]:
        return [ContextNode(**context) for context in contexts]

    def build_function_nodes(self, functions: List[Dict]) -> List[FunctionNode]:
        return [FunctionNode(**function) for function in functions]

    def build_class_nodes(self, classes: List[Dict]) -> List[ClassNode]:
        return [ClassNode(**class_) for class_ in classes]

    def get_concise_context(self):
        """
        This function only used in preliminary study when P-Level=1
        """
        context = []
        for cls in self.classes:
            context.append(cls.content)

        for func in self.functions:
            context.append(func.content)

        concise_context = '\n\n'.join(context)
        return concise_context

    def get_sparse_context(self):
        """
        This function only used in preliminary study when P-Level=2
        """
        context = []
        for cls in self.classes:
            context.append(cls.get_concise_context())

        for func in self.functions:
            context.append(func.get_concise_context())

        concise_context = '\n\n'.join(context)
        return concise_context

    def get_context(self):
        """
        This function is used by HCP strategy
        """
        context = []
        self.score = 0.0
        for cls in self.classes:
            context.append(cls.get_context())
            self.score += cls.temp_score

        for func in self.functions:
            context.append(func.get_context())
            self.score += func.temp_score

        context = '\n\n'.join(context)
        return context

    def get_query_text(self, row: int, col: int, span: int = 10):
        # take the last 10 lines and later 10 lines before the target line
        lines = self.content.split('\n')
        start = max(0, row - span)
        end = min(len(lines), row + span)
        query_text = '\n'.join(lines[start:end])

        return query_text
