import os
import sys
from unittest import mock

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
PROJ_PATH = os.path.join(DIR_PATH, os.path.pardir)
sys.path.append(PROJ_PATH)

from pipeline import parse_pascal


def test_is_xml_filename_matches_extension_only():
    assert parse_pascal.is_xml_filename("annotation.xml") is True
    assert parse_pascal.is_xml_filename("annotation.XML") is True
    assert parse_pascal.is_xml_filename("annotation.Xml") is True
    assert parse_pascal.is_xml_filename("xml_notes.txt") is False
    assert parse_pascal.is_xml_filename("notes.xml.txt") is False
    assert parse_pascal.is_xml_filename("archive.tar.xml") is True
    assert parse_pascal.is_xml_filename("no_extension") is False


def test_main_parses_only_xml_files(tmp_path):
    json_dir = tmp_path / "json"
    json_dir.mkdir()
    annotation_dir = tmp_path / "_annotations"
    annotation_dir.mkdir()

    for name in ["a.xml", "b.XML", "xml_notes.txt", "notes.xml.txt"]:
        (annotation_dir / name).write_text("<root/>")

    args = mock.Mock(annotation_path=str(annotation_dir))
    with mock.patch.object(parse_pascal, "parse_argument", return_value=args), \
            mock.patch.object(parse_pascal.const, "JSON_ANNOTATION_PATH", str(json_dir)), \
            mock.patch.object(parse_pascal, "pars_xml_file", return_value=[{"img_path": "x"}]) as mock_parse:
        parse_pascal.main()

    parsed = {os.path.basename(call.args[0]) for call in mock_parse.call_args_list}
    assert parsed == {"a.xml", "b.XML"}
