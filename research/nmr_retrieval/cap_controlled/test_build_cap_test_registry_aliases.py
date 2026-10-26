"""功能概述：source-only registry/声明aliases助手的纯人工小文件核验，无科学数据。"""
from pathlib import Path
import copy,tempfile,unittest
from unittest import mock
import build_cap_test_registry_aliases as m
from test_prepare_cap_test_metadata import Fixture

def spec_for(root):
    f=Fixture(str(root/"remote"));reg=copy.deepcopy(f.registry)
    old=reg["tool_paths"]["builder"];new=f.reader.add(f.root/"prepare_cap_test_metadata.py",b"# synthetic builder source\n")
    reg["tool_paths"]["builder"]=new
    sources={}
    for d in (reg["leaf_source_sha256"],reg["downstream_source_sha256"],reg["operator_source_sha256"],{new["path"]:new["sha256"]}):sources.update(d)
    aliases={}
    for path in sources:
        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(f.reader.data[path]);aliases[path]=str(p)
    return {"schema":"cap10_source_registry_alias_spec_20261025","remote_root":str(f.root),**reg,"source_aliases":aliases,"metadata_aliases":{"files":{},"sha256":{}}}

class RegistryTests(unittest.TestCase):
    def test_source_registry_exact_roles_and_self_registry_alias_no_cycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();spec=spec_for(root)
            registry,aliases,seen=m.construct(spec,root/"out")
            self.assertEqual(len(registry["leaf_source_sha256"]),12)
            self.assertEqual(len(seen),19)
            path=str(Path(spec["remote_root"])/"test_source_registry.json")
            self.assertEqual(aliases["sha256"][path],m.digest(m.encoded(registry)))
    def test_declared_metadata_alias_is_never_opened(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();spec=spec_for(root);path=str(root/"old/sealed/features/summary.json");local=str(root/"uncreated_summary.json")
            spec["metadata_aliases"]={"files":{path:local},"sha256":{path:"a"*64}}
            original=Path.read_bytes
            def guarded(p):
                if p.suffix!=".py":raise AssertionError("metadata content was read")
                return original(p)
            with mock.patch.object(Path,"read_bytes",guarded):r,a,_=m.construct(spec,root/"out")
            self.assertEqual(a["files"][path],local)
    def test_tensor_alias_rejected_without_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();spec=spec_for(root);path=str(root/"features.pt")
            spec["metadata_aliases"]={"files":{path:path},"sha256":{path:"a"*64}}
            with self.assertRaisesRegex(ValueError,"alias_metadata_or_source_only"):m.construct(spec,root/"out")
    def test_old_operator_role_cannot_be_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();spec=spec_for(root);ops=spec["operator_source_sha256"];k=next(k for k in ops if k.endswith("supervise_resource_ready_queue.py"));ops[str(Path(k).with_name("supervise_controlled_queue.py"))]=ops.pop(k)
            with self.assertRaisesRegex(ValueError,"exact_new_operators"):m.construct(spec,root/"out")
    def test_wrong_source_SHA_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();spec=spec_for(root);path=next(iter(spec["leaf_source_sha256"]));Path(spec["source_aliases"][path]).write_text("# changed")
            with self.assertRaisesRegex(ValueError,"actual_source_SHA"):m.construct(spec,root/"out")
    def test_source_alias_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();spec=spec_for(root);path=next(iter(spec["source_aliases"]));original=Path(spec["source_aliases"][path]);alias=root/"alias.py";alias.symlink_to(original);spec["source_aliases"][path]=str(alias)
            with self.assertRaisesRegex(ValueError,"only_real_py_source"):m.construct(spec,root/"out")
if __name__=="__main__":unittest.main()
