"""Extract executable registrations and implementation anchors without importing upstream code."""
import ast
import json
from collections import defaultdict
from pathlib import PurePosixPath
from zipfile import ZipFile


PINS = {
    "transformers":["huggingface/transformers","d8f10a82213a3a9db48385aba3a4b57ab535de50"],
    "timm":["huggingface/pytorch-image-models","b4f423c19f778672f875e0948f296322c775f24b"],
    "diffusers":["huggingface/diffusers","2e09bbcf1c4930381a3f043e8fe7bb9f8bdafb43"],
    "sklearn":["scikit-learn/scikit-learn","f159b78dc59f250cdde8fe391a21f0bc871960ad"],
}


class Source:
    def __init__(self,z,provider):
        """Bind archive paths to the expected upstream tree at revision R."""
        repo,rev = PINS[provider]
        self.root = repo.split("/")[1]+"-"+rev+"/"
        self.url = f"https://github.com/{repo}/blob/{rev}/"
        self.paths = sorted(p[len(self.root):] for p in z.namelist() if p.startswith(self.root))
        if not self.paths:
            raise ValueError(f"Unexpected archive root for {provider}")
        self.z = z

    def tree(self,path):
        """Parse AST(source_R[path]); no model code is executed."""
        return ast.parse(self.z.read(self.root+path),filename=path)


def mappings(tree):
    """Resolve literal OrderedDict declarations and earlier mapping expansions in source order."""
    found = {}
    for n in tree.body:
        if not isinstance(n,ast.Assign) or not isinstance(n.targets[0],ast.Name):
            continue
        key = n.targets[0].id
        if not isinstance(n.value,ast.Call) or ast.unparse(n.value.func)!="OrderedDict":
            continue
        data = {}
        for item in n.value.args[0].elts:
            if isinstance(item,ast.Starred):
                expr = item.value
                if ast.unparse(expr.func)!="list" or expr.args[0].func.attr!="items":
                    raise ValueError("Unknown registry expansion")
                data.update(found[expr.args[0].func.value.id])
            else:
                k,v = ast.literal_eval(item)
                data[k] = v
        found[key] = data
    return found


def signature(node,name):
    """Project method m onto its declared input port names excluding self."""
    for func in node.body:
        if isinstance(func,ast.FunctionDef) and func.name==name:
            return [arg.arg for arg in [*func.args.posonlyargs,*func.args.args,*func.args.kwonlyargs]
                    if arg.arg!="self"]
    return []


def class_evidence(source,path,node):
    """Record direct constructor symbols and method signatures at the original class declaration."""
    init = next((n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=="__init__"),None)
    calls = set()
    for call in ast.walk(init) if init else ():
        if isinstance(call,ast.Call):
            sym = ast.unparse(call.func)
            if sym.split(".")[-1][:1].isupper() or sym.startswith("nn."):
                calls.add(sym)
    return dict(source=source.url+path+f"#L{node.lineno}",bases=[ast.unparse(b) for b in node.bases],
                forward_inputs=signature(node,"forward"),direct_constructor_symbols=sorted(calls))


def operator_surface(root,definitions):
    """Compute same-file call closure C*=C∪callee(C), then project onto declared tensor primitives."""
    seen, kernels = set(),set()
    queue = [root.name]
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        definition = definitions[name]
        if isinstance(definition,ast.ClassDef):
            queue.extend(b.id for b in definition.bases if isinstance(b,ast.Name) and b.id in definitions)
        for node in ast.walk(definition):
            if not isinstance(node,ast.Call):
                continue
            if isinstance(node.func,ast.Name) and node.func.id in definitions:
                queue.append(node.func.id)
            elif isinstance(node.func,ast.Attribute):
                fun = node.func
                while isinstance(fun,ast.Attribute):
                    fun = fun.value
                if isinstance(fun,ast.Name) and fun.id in ("F","nn","torch"):
                    kernels.add(ast.unparse(node.func))
    return sorted(kernels)


def transformers(source):
    """Take the union of config keys, static overrides and all PyTorch AutoModel (key,class) pairs."""
    base = "src/transformers/models/auto/"
    auto = mappings(source.tree(base+"auto_mappings.py"))
    configs = auto["CONFIG_MAPPING_NAMES"]
    for n in source.tree(base+"configuration_auto.py").body:
        if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=="CONFIG_MAPPING_NAMES.update":
            configs.update(ast.literal_eval(n.value.args[0]))
        if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=="CONFIG_MAPPING_NAMES":
            for keyword in n.value.keywords:
                if isinstance(keyword.value,ast.Dict):
                    configs.update(ast.literal_eval(keyword.value))
    pairs = defaultdict(set)
    for task, table in mappings(source.tree(base+"modeling_auto.py")).items():
        for family, classes in table.items():
            for cls in (classes if isinstance(classes,(tuple,list)) else [classes]):
                pairs[family,cls].add(task.removesuffix("_MAPPING_NAMES"))
    have = {family for family,_ in pairs}
    for family in configs.keys()-have:
        pairs[family,""] = set()
    wanted = {cls for _,cls in pairs} | set(configs.values())
    evidence = defaultdict(list)
    for path in source.paths:
        file = PurePosixPath(path).name
        if (not path.startswith("src/transformers/models/") or not file.endswith(".py")
                or not file.startswith(("modeling_","configuration_"))):
            continue
        nodes = source.tree(path).body
        definitions = {n.name:n for n in nodes if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
        for node in nodes:
            if isinstance(node,ast.ClassDef) and node.name in wanted:
                info = class_evidence(source,path,node)
                if file.startswith("modeling_"):
                    info["same_file_primitive_candidates"] = operator_surface(node,definitions)
                evidence[node.name].append(info)
    for (family,cls),tasks in sorted(pairs.items()):
        config = configs.get(family)
        anchors = evidence[cls] if cls else evidence[config]
        yield dict(catalogue="transformers",family=family,entry=cls,config_class=config,
                   tasks=sorted(tasks),declarations=anchors,
                   development=("qualify_variant" if (family,cls) in {
                       ("bert","BertModel"),("gpt2","GPT2LMHeadModel"),("resnet","ResNetForImageClassification")}
                       else "resolve_component" if not cls else "bind_task_head" if family in ("bert","gpt2","resnet")
                       else "lower_model_and_head"))


def timm(source):
    """Enumerate each statically @register_model-decorated factory under timm/models."""
    for path in source.paths:
        if not path.startswith("timm/models/") or not path.endswith(".py"):
            continue
        for node in source.tree(path).body:
            if isinstance(node,ast.FunctionDef) and "register_model" in [ast.unparse(d) for d in node.decorator_list]:
                returns = [ast.unparse(r.value) for r in ast.walk(node) if isinstance(r,ast.Return) and r.value]
                args = [ast.unparse(n.value) for n in node.body if isinstance(n,ast.Assign) and
                        any(isinstance(t,ast.Name) and t.id=="model_args" for t in n.targets)]
                factories = sorted({ast.unparse(call.func) for call in ast.walk(node)
                                    if isinstance(call,ast.Call) and isinstance(call.func,ast.Name)
                                    and call.func.id!="dict"})
                yield dict(catalogue="timm",family=PurePosixPath(path).stem,entry=node.name,
                           source=source.url+path+f"#L{node.lineno}",factory_returns=returns,
                           factory_calls=factories,model_args=args,development="translate_vision_factory")


def diffusers(source):
    """Enumerate top-level *Pipeline classes in the upstream Python pipeline implementation files."""
    for path in source.paths:
        file = PurePosixPath(path).name
        if not path.startswith("src/diffusers/pipelines/") or not file.startswith("pipeline_") or not file.endswith(".py"):
            continue
        for node in source.tree(path).body:
            if isinstance(node,ast.ClassDef) and node.name.endswith("Pipeline"):
                calls = {ast.unparse(c.func) for f in node.body if isinstance(f,ast.FunctionDef) and f.name=="__call__"
                         for c in ast.walk(f) if isinstance(c,ast.Call) and isinstance(c.func,ast.Attribute)}
                yield dict(catalogue="diffusers",family=PurePosixPath(path).parent.name,entry=node.name,
                           source=source.url+path+f"#L{node.lineno}",components=signature(node,"__init__"),
                           call_inputs=signature(node,"__call__"),bases=[ast.unparse(b) for b in node.bases],
                           component_call_candidates=sorted(s for s in calls if s.startswith("self.") and "(" not in s),
                           development="bind_pipeline_components")


def inventory(directory):
    """I=I_transformers ⊔ I_timm ⊔ I_diffusers, retaining source identity for each member."""
    for name,extract in (("transformers",transformers),("timm",timm),("diffusers",diffusers)):
        with ZipFile(directory/f"dry-{name}.zip") as archive:
            yield from extract(Source(archive,name))
    yield from json.loads((directory/"dry-sklearn.json").read_bytes())
