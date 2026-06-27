import sys
content = open('run_tests.py', encoding='utf-8').read()
replacements = [
    ('\u2265', '>='),
    ('\u2264', '<='),
    ('\u2014', '--'),
    ('\u2019', "'"),
    ('\u00b1', '+/-'),
]
for old, new in replacements:
    content = content.replace(old, new)
open('run_tests.py', 'w', encoding='utf-8').write(content)
print('Done. Replaced all special chars.')
