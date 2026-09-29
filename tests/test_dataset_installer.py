import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest

path = Path(__file__).resolve().parents[1] / 'scripts/download_syaudio.py'
spec = importlib.util.spec_from_file_location('download_syaudio', path)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)

class InstallerTests(unittest.TestCase):
    def build(self, root, name='benchmark/a.txt', payload=b'original'):
        archive = root / 'a.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            m = tarfile.TarInfo(name); m.size = len(payload)
            tar.addfile(m, io.BytesIO(payload))
        records = [{'path': name, 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}]
        return archive, records

    def test_roundtrip_and_idempotence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); a, records=self.build(root); dest=root/'out'
            installer.extract_verified(a,dest,records)
            installer.extract_verified(a,dest,records)
            self.assertEqual((dest/'benchmark/a.txt').read_bytes(),b'original')

    def test_preserves_collaborator_edits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a,records=self.build(root);dest=root/'out'
            (dest/'benchmark').mkdir(parents=True);(dest/'benchmark/a.txt').write_bytes(b'collaborator edit')
            with self.assertRaises(FileExistsError):installer.extract_verified(a,dest,records)
            self.assertEqual((dest/'benchmark/a.txt').read_bytes(),b'collaborator edit')

    def test_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a,records=self.build(root,'../outside')
            with self.assertRaises(ValueError):installer.extract_verified(a,root/'out',records)
            self.assertFalse((root/'outside').exists())

    def test_rejects_checksum_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a,records=self.build(root);records[0]['sha256']='0'*64
            with self.assertRaises(ValueError):installer.extract_verified(a,root/'out',records)
            self.assertFalse((root/'out/benchmark/a.txt').exists())

if __name__=='__main__':unittest.main()
