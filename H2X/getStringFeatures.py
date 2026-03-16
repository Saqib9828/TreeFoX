import sys
import json
import pefile
import re
from collections import namedtuple

def buf_filled_with(buf, character):
    dupe_chunk = character * SLICE_SIZE
    for offset in range(0, len(buf), SLICE_SIZE):
        new_chunk = buf[offset : offset + SLICE_SIZE]
        if dupe_chunk[: len(new_chunk)] != new_chunk:
            return False
    return True


def extract_ascii_strings(buf, n=4):
    """
    Extract ASCII strings from the given binary data.
    :param buf: A bytestring.
    :type buf: str
    :param n: The minimum length of strings to extract.
    :type n: int
    :rtype: Sequence[String]
    """

    if not buf:
        return
    if (buf[0] in REPEATS) and buf_filled_with(buf, buf[0]):
        return
    r = None
    if n == 4:
        r = ASCII_RE_4
    else:
        reg = b"([%s]{%d,})" % (ASCII_BYTE, n)
        r = re.compile(reg)
    for match in r.finditer(buf):
        yield String(match.group().decode("ascii"), match.start())


def extract_unicode_strings(buf, n=4):
    """
    Extract naive UTF-16 strings from the given binary data.
    :param buf: A bytestring.
    :type buf: str
    :param n: The minimum length of strings to extract.
    :type n: int
    :rtype: Sequence[String]
    """

    if not buf:
        return
    if (buf[0] in REPEATS) and buf_filled_with(buf, buf[0]):
        return
    if n == 4:
        r = UNICODE_RE_4
    else:
        reg = b"((?:[%s]\x00){%d,})" % (ASCII_BYTE, n)
        r = re.compile(reg)
    for match in r.finditer(buf):
        try:
            yield String(match.group().decode("utf-16"), match.start())
        except UnicodeDecodeError:
            pass

ASCII_BYTE = r" !\"#\$%&\'\(\)\*\+,-\./0123456789:;<=>\?@ABCDEFGHIJKLMNOPQRSTUVWXYZ\[\]\^_`abcdefghijklmnopqrstuvwxyz\{\|\}\\\~\t".encode(
    "ascii"
)
ASCII_RE_4 = re.compile(b"([%s]{%d,})" % (ASCII_BYTE, 4))
UNICODE_RE_4 = re.compile(b"((?:[%s]\x00){%d,})" % (ASCII_BYTE, 4))
REPEATS = [b"A", b"\x00", b"\xfe", b"\xff"]
SLICE_SIZE = 4096

String = namedtuple("String", ["s", "offset"])

def isRemoveAble(s):
    flag = False
    remove_list = ['.text', '.rdata', '.data', '.debug']
    for word in remove_list:
        if word in s:
            flag = True
            break
    return flag

def isAcceptable(s):
    flag = False
    accept_list = ['HTTP:', 'HTTPS:', 'FTP']
    for word in accept_list:
        if word in s.upper():
            flag = True
            break
    return flag

def isSmallLenth(s):
    if len(s) <= smallStringLenth:
        return True
    else:
        return False

word_len = 7
minValidWordLen = 4
smallStringLenth = 0
tooSmallforValidating = 5
continueVowel = 3
continueConsonent = 4
continueSpecialCharecter = 3
continueDigit = 3
continueVowelDigit = 3
continueConsonentDigit = 4
inWord_continueConsonent = 4
inWord_continueSpecialChareter = 3
allowSPinMinWord = 2

vowel = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']
special_char = re.compile('[@_!#$%^&*()<>?/\|}{~: ]')

def isMinValidWord(s):
    if len(s)<= minValidWordLen:
        count_spchar = 0
        for i in range(len(s)):
            if special_char.search(s[i]) or s[i] == ';' or s[i] == "\\" or s[i].isdigit():
                count_spchar = count_spchar + 1
        if count_spchar >= allowSPinMinWord:
            return False
    return True


def isRoughWord(s):
    if len(s) > word_len:
        return False
    if s[0] == " ":
        return False
    if isAcceptable(s):
        return False
    inWord_CountContinueConsonent = 0
    inWord_CountContinueSpecialChareter = 0
    '''
    for i in range(min(word_len, len(s))):
        if special_char.search(s[i]):
            inWord_CountContinueSpecialChareter = inWord_CountContinueSpecialChareter + 1
            if inWord_CountContinueSpecialChareter == inWord_continueSpecialChareter:
                break
        else:
            inWord_CountContinueSpecialChareter = 0
            '''

    for i in range(min(word_len, len(s))):
        if not (s[i].isdigit() or special_char.search(s[i]) or (s[i] in vowel)):
            inWord_CountContinueConsonent = inWord_CountContinueConsonent + 1
            if inWord_CountContinueConsonent == inWord_continueConsonent:
                break
        else:
            inWord_CountContinueConsonent = 0
    if (inWord_CountContinueConsonent >= inWord_continueConsonent):
        return True
    else:
        return False


def isRoughStart(s):
    if len(s) > word_len:
        return False
    if s.isdigit():
        return False
    if s[0] == " " or s[0] == ":":
        return False
    if isAcceptable(s):
        return False
    countVowel = 0
    countSpecialChar = 0
    countDigit = 0
    countConsonent = 0
    countVowelDigit = 0
    countConsonentDigit = 0

    for i in range(min(continueVowelDigit, len(s))):
        if (s[i] in vowel) or s[i].isdigit():
            countVowelDigit = countVowelDigit + 1
    for i in range(min(continueConsonentDigit, len(s))):
        if not (special_char.search(s[i]) or (s[i] in vowel)):
            countConsonentDigit = countConsonentDigit + 1
    for i in range(min(continueVowel, len(s))):
        if s[i] in vowel:
            countVowel = countVowel + 1
    for i in range(min(continueSpecialCharecter, len(s))):
        if special_char.search(s[i]):
            countSpecialChar = countSpecialChar + 1
    for i in range(min(continueDigit, len(s))):
        if s[i].isdigit():
            countDigit = countDigit + 1
    for i in range(min(continueConsonent, len(s))):
        if not (s[i].isdigit() or special_char.search(s[i]) or (s[i] in vowel)):
            countConsonent = countConsonent + 1

    if (countConsonentDigit >= continueConsonentDigit) or (countVowelDigit >= continueVowelDigit) or (
            countVowel >= continueVowel) or (countSpecialChar >= continueSpecialCharecter) or (
            countDigit >= continueDigit) or (countConsonent >= continueConsonent):
        return True
    else:
        return False


word_len = 7
minSentLen = 8
minDirLen = 15
minValidWordLen = 4
smallStringLenth = 0
tooSmallforValidating = 5
continueVowel = 3
continueConsonent = 4
continueSpecialCharecter = 3
continueDigit = 3
continueVowelDigit = 3
continueConsonentDigit = 4
inWord_continueConsonent = 4
inWord_continueSpecialChareter = 3
allowSPinMinWord = 2

vowel = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']
special_char = re.compile('[@_!#$%^&*()<>?/\|}{~: ]')


def isMinValidWord(s):
    if len(s) <= minValidWordLen:
        count_spchar = 0
        for i in range(len(s)):
            if special_char.search(s[i]) or s[i] == ';' or s[i] == "\\" or s[i].isdigit():
                count_spchar = count_spchar + 1
        if count_spchar >= allowSPinMinWord:
            return False
    return True


def isRoughWord(s):
    if len(s) > word_len:
        return False
    if s[0] == " ":
        return False
    if isAcceptable(s):
        return False
    inWord_CountContinueConsonent = 0
    inWord_CountContinueSpecialChareter = 0
    '''
    for i in range(min(word_len, len(s))):
        if special_char.search(s[i]):
            inWord_CountContinueSpecialChareter = inWord_CountContinueSpecialChareter + 1
            if inWord_CountContinueSpecialChareter == inWord_continueSpecialChareter:
                break
        else:
            inWord_CountContinueSpecialChareter = 0
            '''

    for i in range(min(word_len, len(s))):
        if not (s[i].isdigit() or special_char.search(s[i]) or (s[i] in vowel)):
            inWord_CountContinueConsonent = inWord_CountContinueConsonent + 1
            if inWord_CountContinueConsonent == inWord_continueConsonent:
                break
        else:
            inWord_CountContinueConsonent = 0
    if (inWord_CountContinueConsonent >= inWord_continueConsonent):
        return True
    else:
        return False


def isRoughStart(s):
    if len(s) > word_len:
        return False
    if s.isdigit():
        return False
    if s[0] == " " or s[0] == ":":
        return False
    if isAcceptable(s):
        return False
    countVowel = 0
    countSpecialChar = 0
    countDigit = 0
    countConsonent = 0
    countVowelDigit = 0
    countConsonentDigit = 0

    for i in range(min(continueVowelDigit, len(s))):
        if (s[i] in vowel) or s[i].isdigit():
            countVowelDigit = countVowelDigit + 1
    for i in range(min(continueConsonentDigit, len(s))):
        if not (special_char.search(s[i]) or (s[i] in vowel)):
            countConsonentDigit = countConsonentDigit + 1
    for i in range(min(continueVowel, len(s))):
        if s[i] in vowel:
            countVowel = countVowel + 1
    for i in range(min(continueSpecialCharecter, len(s))):
        if special_char.search(s[i]):
            countSpecialChar = countSpecialChar + 1
    for i in range(min(continueDigit, len(s))):
        if s[i].isdigit():
            countDigit = countDigit + 1
    for i in range(min(continueConsonent, len(s))):
        if not (s[i].isdigit() or special_char.search(s[i]) or (s[i] in vowel)):
            countConsonent = countConsonent + 1

    if (countConsonentDigit >= continueConsonentDigit) or (countVowelDigit >= continueVowelDigit) or (
            countVowel >= continueVowel) or (countSpecialChar >= continueSpecialCharecter) or (
            countDigit >= continueDigit) or (countConsonent >= continueConsonent):
        return True
    else:
        return False


def includeString(s):
    if isSmallLenth(s):
        return False
    elif isRemoveAble(s):
        return False
    elif isRoughStart(s):
        return False
    elif isRoughWord(s):
        return False
    elif not (isMinValidWord(s)):
        return False
    else:
        return True

sentenceLen = 4
email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
def isGarbage(s):
    if includeString(s) and len(s) < sentenceLen:
        return True
    return False
def isFile(s):
    if includeString(s) and len(s)>sentenceLen:
        if s[-4] == '.' and not('.com' in s or "." in s[-3:]) and not(special_char.search(s[-4:]) ):
            return True
        elif s[-5] == '.' and not('.com' in s or "." in s[-4:]) and not(special_char.search(s[-5:]) ):
            return True
    return False
def isURL(s):
    if includeString(s) and len(s)>sentenceLen:
        if (('://' in s) and ('.' in s)) or ('www.' in s):
            return True
    return False
def isDIR(s):
    if len(s) < minDirLen :
        return False
    if includeString(s) and len(s)>sentenceLen:
        if ((s.count('/') >= 2) or (s.count('\\') >= 2)) and not (('<' in s) or ('>' in s) or ('://' in s)):
            return True
    return False
def isEmail(s):
    if includeString(s) and len(s)>sentenceLen:
        if (re.fullmatch(email_regex, s)):
            return True
    return False
def isInValEmail(s):
    if includeString(s) and len(s)>sentenceLen:
        if ('@' in s) and ('.' in s) and not(isEmail(s) or ('?' in s) or ('(' in s)):
            return True
    return False
def isLongWord(s):
    if s.isupper() or special_char.search(s):
        return  False # if all the letter are upper
    if not(s.isalpha()):
        return False # if anything except letter
    if includeString(s) and len(s)>sentenceLen:
        if not (isGarbage(s) or isFile(s) or isURL(s) or isDIR(s) or isEmail(s) or isInValEmail(s)):
            if not((" " in s) or ("=" in s) or ("`" in s) or special_char.search(s)):
                if any(ele.isupper() for ele in s[1:]) and s[0].isupper():
                    return True
    return False
def isSpecialKeyword(s):
    keywords = ['windows', 'xboxnetapi', 'system', 'xbox', 'microsoft', 'ipv4', 'ipv6']
    if any(word in s.lower() for word in keywords):
        return True
    return  False
def isIP(s):
    if re.search(r'\d.\d.\d.\d', s):
        return True
    return False
def isSentence(s):
    if len(s) < minSentLen :
        return False
    str_without_spcl = re.sub('[^a-zA-Z0-9 \n\.]', '', s)
    if len(str_without_spcl) < minSentLen or str_without_spcl.isupper():
        return False
    if len(str_without_spcl.split(' ')) == 2 and (str_without_spcl.split(' ')[0].isupper() or str_without_spcl.split(' ')[1].isupper()):
        return False
    if not(isIP(s) or isSpecialKeyword(s) or isLongWord(s) or isGarbage(s) or isFile(s) or isURL(s) or isDIR(s) or isEmail(s) or isInValEmail(s)):
        return True
    return False

garbage = []
fileName = []
URLs = []
DIRs = []
emails = []
inValEmails = []
longWord = []
specialKeyword = []
ipaddresses = []
sentences = []
def cluster(s):
    if isGarbage(s):
        garbage.append(s)
    if isFile(s):
        fileName.append(s)
    if isURL(s):
        URLs.append(s)
    if isDIR(s):
        DIRs.append(s)
    if isEmail(s):
        emails.append(s)
    if isInValEmail(s):
        inValEmails.append(s)
    if isLongWord(s):
        longWord.append(s)
    if isSpecialKeyword(s):
        specialKeyword.append(s)
    if isIP(s):
        ipaddresses.append(s)
    if isSentence(s):
        sentences.append(s)

def getStrings(file):
    string_ = {}
    importExport = {}
    pe = pefile.PE(file)
    with open(file, "rb") as f:
        b = f.read()
    count_s = 0
    for s in extract_ascii_strings(b):
        if includeString(s.s):
            cluster(s.s)

    for s in extract_unicode_strings(b):
        if includeString(s.s):
            cluster(s.s)
    # get imports
    try:
        importExport['ImportsList'] = []
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            for imp in entry.imports:
                importExport['ImportsList'].append(imp.name.decode("utf-8"))
    except AttributeError:
        importExport['ImportsList'] = []
    # get exports
    try:
        importExport['ExportsList'] = []
        for exp in pe.DIRECTORY_ENTRY_EXPORT.symbols:
            importExport['ExportsList'].append(exp.name.decode("utf-8"))
    except AttributeError:
        # No export
        importExport['ExportsList'] = []

    string_['garbage'] = garbage
    string_['fileName'] = fileName
    string_['URLs'] = URLs
    string_['DIRs'] = DIRs
    string_['emails'] = emails
    string_['inValEmails'] = inValEmails
    string_['longWord'] = longWord
    string_['specialKeyword'] = specialKeyword
    string_['ipaddresses'] = ipaddresses
    string_['sentences'] = sentences
    pe.close()
    return string_ , importExport
'''
file = "pe2"
string_ = getStrings(file)
print(json.dumps(string_, indent=4))

with open('Strings.json', 'w') as json_file:
    json.dump(string_, json_file)
    
'''
