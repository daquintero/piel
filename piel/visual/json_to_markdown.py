from piel.types import PathTypes


def dictionary_to_markdown_str(
    dictionary: PathTypes,
) -> str:
    def parse_dictionary(d, level=0):
        markdown = ""
        indent = "  " * level
        for key, value in d.items():
            if isinstance(value, dict):
                markdown += f"{indent}- **{key}**:\n"
                markdown += parse_dictionary(value, level + 1)
            elif isinstance(value, list):
                markdown += f"{indent}- **{key}**:\n"
                markdown += parse_list(value, level + 1)
            else:
                markdown += f"{indent}- **{key}**: {value}\n"
        return markdown

    def parse_list(lst, level=0):
        markdown = ""
        indent = "  " * level
        for item in lst:
            if isinstance(item, dict):
                markdown += f"{indent}-\n"
                markdown += parse_dictionary(item, level + 1)
            elif isinstance(item, list):
                markdown += parse_list(item, level + 1)
            else:
                markdown += f"{indent}- {item}\n"
        return markdown

    markdown_content = parse_dictionary(dictionary)

    return markdown_content
