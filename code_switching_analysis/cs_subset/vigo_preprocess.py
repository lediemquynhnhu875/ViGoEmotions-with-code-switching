"""
vigo_preprocess.py
==================
Tiền xử lý ViGoEmotions theo đúng ba scenario của bài báo.

  S1 — giữ nguyên emoji, chuẩn hoá teencode bằng từ điển thủ công
  S2 — chuyển emoji/emoticon thành mô tả tiếng Việt + cùng chuẩn hoá teencode
  S3 — giữ nguyên emoji, chuẩn hoá bằng ViSoLex (nếu không chạy được thì
       dùng lại từ điển thủ công và ghi rõ trong báo cáo)

Ba từ điển dưới đây được trích nguyên văn từ output notebook
`model/ViSoBERT.ipynb` trong repo, nên khớp với dữ liệu đã dùng để train.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# 1. Chuẩn hoá ký tự lặp / HTML entity  (áp dụng cho CẢ BA scenario)
# ---------------------------------------------------------------------------
REPLACE_LIST = {
    r'\)\)+': '))', r'\(\(+': '((', r'\]\]+': ']]', r'\>+': '>', r'\=+': '=',
    r'\:+': ':', r'\?+': '?', r'\!+': '!', r'kk+': 'haha', r'\.\.\.+': '...',
    '&gt;': '>', '&lt;': '<', '&amp;': '&', '&quot;': '"', '&apos;': "'",
    '&#39;': "'", '&nbsp;': ' ', '<br>': '\n',
}

# ---------------------------------------------------------------------------
# 2. Emoticon + emoji -> mô tả tiếng Việt  (chỉ dùng cho S2)
# ---------------------------------------------------------------------------
EMOJI_MAP = {
    ':))': 'cười lớn', '=))': 'cười rũ rượi', '=((': 'khóc', ':3': 'cười mặt mèo',
    ':v': 'há mồm', '^^': 'cười híp mắt', ':<': 'mặt méo', '=.=': 'bất lực',
    '-_-': 'bó tay', ':>': 'vui vẻ', '=]]': 'nhe răng', ':((': 'khóc', ':)': 'cười',
    ':-(': 'buồn', ':-@': 'sốc', ':#': 'im lặng', '@@': 'lăn mắt',
    '<(-_-)>': 'robot', ';-)': 'nháy mắt', ':-)': 'cười', ':-<': 'buồn',
    ':@': 'sốc', ':X': 'im lặng', ':-!': 'bối rối', 'd[-_-]b': 'dj',
    'O:-)': 'thiên thần', ';d': 'nháy mắt', ':P': 'le lưỡi', ':-$': 'bối rối',
    ':^)': 'cười', ':-D': 'cười', ":'-)": 'cười buồn', 'O*-)': 'thiên thần',
    ':-E': 'ma cà rồng', ':O': 'ngạc nhiên', ':\\': 'khó chịu', ':-&': 'bối rối',
    ':-0': 'hét', ';)': 'nháy mắt', '(:-D': 'nói xấu', ':(': 'buồn',
    '$_$': 'tham lam', 'O.o': 'bối rối', '=^.^=': 'mèo',
    '❤️': 'tim', '💗': 'yêu', '💛': 'tim vàng', '👏': 'vỗ tay', '🤗': 'ôm',
    '😂': 'cười ra nước mắt', '😡': 'tức giận', '🤩': 'háo hức', '💔': 'tim tan vỡ',
    '💓': 'đập nhanh', '💜': 'tim tím', '🙌': 'hoan hô', '😍': 'mê mẩn',
    '🤣': 'cười lớn', '😱': 'hoảng hốt', '😋': 'thèm ăn', '💕': 'tim yêu thương',
    '💙': 'tim xanh', '👍': 'thích', '💪': 'mạnh mẽ', '😤': 'tức giận', '😎': 'ngầu',
    '😴': 'buồn ngủ', '😌': 'hài lòng', '💖': 'lấp lánh', '💚': 'tim xanh lá',
    '👎': 'phản đối', '🤝': 'bắt tay', '🤯': 'kinh ngạc', '😇': 'hiền lành',
    '🙄': 'ngao ngán', '😔': 'buồn nhẹ', '🤔': 'suy nghĩ', '😢': 'khóc',
    '🥱': 'ngáp', '😭': 'khóc lớn', '💡': 'phát giác', '🤞': 'tự tin',
    '🤮': 'mắc ói', '💫': 'choáng', '😵‍💫': 'khiếp sợ', '🐛': 'sầu',
    '😶‍🌫': 'cạn lời', '😒': 'khó chịu', '❤': 'yêu', '💐': 'hoa',
    '😉': 'nháy mắt', '💀🙉': 'ma đầu', '😗': 'hôn', '😊': 'cười', '🌸': 'hoa đào',
    '💃': 'nhảy múa', '😦': 'ngạc nhiên', '👊': 'đấm', '🙎': 'bĩu môi',
    '🖕': 'ngón giữa trỏ lên', '🥥': 'dừa', '🌴': 'dừa', '🫶🏼': 'tim',
    '🫣': 'ngại ngùng', '👌': 'đồng ý', '🌚': 'mặt mỉa mai', '🐕': 'chó',
    '🍇': 'nho', '🙏': 'chắp tay lạy', '💩': 'ỉa', '😕': 'bối rối', '♥': 'tim',
    '🙁': 'buồn', '😠': 'tức giận', '😀': 'cười ngoác mồm', '😖': 'bối rối',
    '😶': 'im lặng', '😷': 'im lặng', '😹': 'cười chảy nước mắt', '😥': 'buồn bã',
    '👻': 'ma', '🖤': 'trái tim đen', '🌹': 'hoa hồng', '😜': 'lè lưỡi, trêu chọc',
    '🙇': 'cúi đầu', '👋': 'vẫy tay', '😚': 'hôn', '🙂': 'mỉm cười',
    '😆': 'cười tít mắt', '😪': 'buồn ngủ', '🙈': 'xấu hổ', '🌧': 'mây mưa',
    '😏': 'cười nhép', '😫': 'mệt mỏi', '😵': 'hôn', '😘': 'hôn gió',
    '🗿': 'lạnh lùng', '🌶': 'cay', '👿': 'tức giận', '❎': 'dấu chéo',
    '🙃': 'ngược đời', '😩': 'kiệt sức', '👨': 'người đàn ông', '🐠': 'cá',
    '😻': 'yêu', '😰': 'lo lắng toát mô hôi', '😃': 'vui vẻ', '😁': 'cười nhe răng',
    '☺': 'cười', '😓': 'buồn', '😞': 'gương mặt thất vọng hoặc lo lắng',
    '🙉': 'không nghe', '⚠': 'cảnh báo', '👮': 'cảnh sát', '📸': 'máy ảnh',
    '😬': 'lo lắng', '👀': 'mắt quan sát', '😝': 'cười lè lưỡi', '💝': 'tim',
    '😑': 'không cảm xúc', '🎓': 'tốt nghiệp', '🙀 ': 'kinh ngạc', '💋': 'hôn',
    '😅': 'cười toát mồ hôi', '☕': 'cà phê', '✨ ': 'lấp lánh', '☹': 'mặt buồn',
    '☘': 'may mắn', '😛': 'mặt lè lưỡi', '🐂': 'bò', '🔪': 'chém',
    '😼': 'cười nhếch mép', '🎶': 'âm nhạc', '😐': 'không cảm xúc', '😿': 'khóc',
    '🙆': 'đồng ý', '💯': '100 điểm', '🔥': 'cháy', '🐸': 'ếch', '✅': 'dấu tích',
    '👉👉': 'theo đường dẫn', '👉': 'theo đường dẫn', '👉👈': 'ngại ngùng, xấu hổ',
    '🌝': 'vui vẻ', '😄': 'cười ngoắc miệng', '✌': 'chào hỏi',
    '🎉': 'pháo giấy chúc mừng', '🙅': 'không đồng ý', '😟': 'lo lắng',
    '😨': 'sợ hãi', '😳': 'ngượng ngùng', '😣': 'khó chịu', '☻': 'cười',
    '🍀': 'cỏ bốn lá', '😮‍💨': 'thở phào', '🐧': 'chim cánh cụt',
    '😧': 'lo lắng', '😮': 'ngạc nhiên', '🙀': 'la hét', '🤨': 'nhướng lông mày ',
    '🐣': 'trứng nở', '🤬': 'khuôn mặt chửi rủa', '🌵': 'xương rồng', '🪑': 'ghế',
    '🫠': 'mặt tan chảy', '🤕': 'đầu quấn băng', '☠': 'xương sọ', '🤤': 'thèm khát',
    '👈': 'ngon trỏ trái', '🤫': 'mặt im lặng', '🔔': 'cái chuông', '🩳': 'quần',
    '🤪': 'mặt ngốc nghếch', '🗣': 'phát biểu', '🤍': 'trái tim trắng',
    '☝': 'ngón trỏ lên', '🥰': 'mặt hạnh phúc', '🤥': 'mặt nói dối',
    '\U0001fa77': 'trái tim hồng', '🤦‍♂️': 'che mặt', '🤐': 'khóa miệng',
    '🥵': 'nóng bức', '🤦': 'che mặt', '🥴': 'mặt choáng váng', '🫵': 'ngón trỏ tới',
    '🦽': 'xe lăn', '🚀': 'bắn pháo', '🧐': 'mắt quan sát',
    '🫶': 'tay tạo hình trái tim', '🫰': 'tay tạo hình trái tim', '🤒': 'mặt ốm',
    '🤡': 'mặt hề', '🦾': 'tay khỏe', '🥺': 'mặt buồn', '🐔': 'con gà',
    '🩹': 'băng dính', '🕸': 'mạng nhện', '🤧': 'mặt hỉ mũi', '🤢': 'mặt buồn nôn',
    '🌈': 'cầu vòng', '🥲': 'mặt chảy nước mắt', '😈': 'mặt cười có sừng',
    '🤭': 'mặt cười che miệng', '🤓': 'mặt ngố', '🥶': 'mặt lạnh như băng',
    '🤦🏻‍♀️': 'che mặt', '🫤': 'mặt miệng chéo', '🥹': 'mặt rưng rưng nước mắt',
    '🙏🏿': 'chắp tay', '💅': 'sơn móng tay', '🗾': 'bản đồ Nhật Bản',
    '📈': 'biểu đồ tăng trưởng', '🎱': 'bóng số 8', '🍦': 'kem mềm', '🦊': 'con cáo',
    '💀': 'đầu lâu', '😯': 'mặt ngạc nhiên', '\U0001fabd': 'cánh',
    '🖐': 'bàn tay xòe ra', '🐢': 'rùa', '👆': 'ngón tay trỏ lên', '👣': 'dấu chân',
    '🎬': 'phim', '🐶': 'mặt chó', '🌤': 'mặt trời sau mây', '💸': 'tiền',
    '🎀': 'ruy băng', '✊': 'nắm tay giơ lên', '🕊': 'bồ câu', '🚨': 'cảnh báo',
    '🤏': 'véo ngón tay', '❄': 'hoa tuyết', '👇': 'ngón trỏ xuống',
    '🫢': 'mặt che miệng', '🏐': 'bóng chuyền', '🍊': 'quả quyết', '🍃': 'lá bay',
    '🍜': 'bát mì nóng', '🗻': 'núi Phú Sĩ', '🔴': 'hình tròn đỏ',
    '🤑': 'mặt dấu tiền', '🥇': 'huy chương vàng', '♀': 'ký hiệu nữ',
    '🧀': 'phô mai', '🌟': 'sao', '🦔': 'nhím', '🌬': 'thở ra hơi',
    '🥸': 'mặt cải trang', '💭': 'suy nghĩ', '🍄': 'nấm', '🧏': 'khiếm thính',
    '🧬': 'DNA', '🎈': 'bóng bay', '\U0001fa76': 'tim xám', '💢': 'giận dữ',
    '💰': ':túi tiền', '🦕': 'khủng long', '🦭': 'hải cẩu', '👑': 'vương miệng',
    '❣': 'dấu trái tim', '\U0001fae8': 'mặt rung lắc', '🦖': 'khủng long',
    '😽': 'mèo hôn gió', '♐': 'cung nhân mã', '🐺': 'sói', '💤': 'ngủ',
    '👦': 'bé trai', '🪷': 'hoa sen', '💘': 'bắn xuyên tim', '🧁': 'bánh cupcake',
    '🙊': 'khỉ che miệng', '✨': 'lấp lánh', '🐰': 'mặt thỏ', '🍓': 'dâu tây',
    '🥃': 'ly rượu', '🌻': 'hướng dương', '🧚': 'tiên', '🤟': 'i love you',
    '🐳': 'cá voi', '🦫': 'hải ly', '🐱': 'mặt mèo', '🤙': 'gọi cho tôi',
    '🏪': 'cửa hàng', '🌷': 'tulip', '🥀': 'hoa héo', '🚚': 'xe giao hàng',
    '🧶': 'cuộn len', '🧡': 'tim màu cam', '🍭': 'kẹo mút', '💥': 'bùng nổ',
    '🫂': 'ôm nhau', '👩': 'phụ nữ', '🫡': 'chào nghiêm trang', '❓': 'dấu hỏi',
    '🤌': 'bóp nhẹ', '❌': 'dấu chéo', '🐻': 'gấu', '🧠': 'não', '🍸': 'ly cocktail',
    '♉': 'kim ngưu', '🖇': 'kẹp', '💨': 'gió thổi', '🌽': 'bắp',
    '👽': 'người hành tinh', '💞': 'tim quấn quanh', '🫦': 'cắn môi', '🥑': 'bơ',
    '🤷': 'người nhún vai', '👄': 'miệng', '🌫': 'sương mù', '🐟': 'cá',
    '★': 'sao', '✵': 'sao', '☆': 'sao', '♡': 'trái tim', '▽': 'tam giác',
    '𝝰': 'alpha',
}

# ---------------------------------------------------------------------------
# 3. Teencode -> tiếng Việt chuẩn  (dùng cho S1 và S2; S3 dùng ViSoLex thay thế)
# ---------------------------------------------------------------------------
TEENCODE_MAP = {
    'ctrai': 'con trai', 'khôg': 'không', 'bme': 'bố mẹ', 'cta': 'chúng ta',
    'mih': 'mình', 'cmnl': 'con mẹ nó luôn', 'mqh': 'mối quan hệ', 'ùi': 'rồi',
    'sem': 'xem', 'pải': 'phải', 'đel': 'đéo', 'cgai': 'con gái', 'nhữg': 'những',
    'mng': 'mọi người', 'r': 'rồi', 'qtam': 'quan tâm', 'thươg': 'thương',
    'lun': 'luôn', 'cute': 'dễ thương', 'kute': 'dễ thương', 'ưi': 'ơi',
    'kao': 'tao', 'tau': 'tao', 'gê': 'ghê', 'ge': 'ghê', 'kau': 'tao',
    'rùi': 'rồi', 'qtâm': 'quan tâm', 'share': 'chia sẻ', 'chug': 'chung',
    'vại': 'vậy', 'trườg': 'trường', 'hjx': 'hic', 'hix': 'hic', 'thoy': 'thôi',
    'oy': 'rồi', 'đki': 'đăng ký', 'àk': 'à', 'qcao': 'quảng cáo',
    'pr': 'quảng cáo', 'cv': 'công việc', 'tr': 'trời', 'vch': 'vãi chưởng',
    'cùg': 'cùng', 'thjck': 'thích', 'thjk': 'thích', 'ktra': 'kiểm tra',
    'cgái': 'con gái', 'nthe': 'như thế', 'chúg': 'chúng', 'tìh': 'tình',
    'phòg': 'phòng', 'lòg': 'lòng', 'từg': 'từng', 'rằg': 'rằng', 'sốg': 'sống',
    'thuj': 'thôi', 'càg': 'càng', 'đky': 'đăng ký', 'dể': 'dễ', 'ukm': 'ừ',
    'bằg': 'bằng', 'sviên': 'sinh viên', 'đág': 'đáng', 'nvay': 'như vậy',
    'vv': 'vui vẻ', 'v': 'vậy', 'xg': 'xong', 'zồi': 'rồi', 'trag': 'trang',
    'zữ': 'dữ', 'atrai': 'anh trai', 'kte': 'kinh tế', 'độg': 'động',
    'gắg': 'gắng', 'đzai': 'đẹp trai', 'thgian': 'thời gian', 'đồg': 'đồng',
    'btrai': 'bạn trai', 'vler': 'vãi lồn', 'pccc': 'phòng cháy chữa cháy',
    'nthê': 'như thế', 'vọg': 'vọng', 'đôg': 'đông', 'răg': 'răng',
    'thườg': 'thường', 'tcảm': 'tình cảm', 'đứg': 'đứng', 'ksao': 'không sao',
    'đĩ': 'đỉ', 'đũy': 'đỉ', 's': 'sao', 'dz': 'đẹp trai', 'cmày': 'chúng mày',
    'xuốg': 'xuống', 'nkư': 'như', 'lquan': 'liên quan', 'tiếg': 'tiếng',
    'xih': 'xinh', 'lày': 'này', 'dug': 'đúng', 'đug': 'đúng', 'hìh': 'hình',
    'hắt': 'hất', 'thàh': 'thành', 'ngke': 'nghe', 'dzậy': 'dậy', 'nghen': 'nha',
    'ngheng': 'nha', 'tnào': 'thế nào', 'bede': 'bê đê', 'buê đuê': 'bê đê',
    'tưởg': 'tưởng', 'đx': 'được', 'ctrinh': 'chương trình', 'phog': 'phong',
    'hog': 'không', 'hôg': 'không', 'zìa': 'về', 'kũg': 'cũng',
    'ntnao': 'như thế nào', 'tnao': 'thế nào', 'trọg': 'trọng', 'nthế': 'như thế',
    'dzớ dzẩn': 'vớ vẩn', 'năg': 'năng', 'éo': 'đéo', 'ngđó': 'người đó',
    'lquen': 'làm quen', 'dell': 'đéo', 'del': 'đéo', 'riêg': 'riêng',
    'ngag': 'ngang', 'chộm': 'trộm', 'bnhiu': 'bao nhiêu', 'ngốk': 'ngốc',
    'kậu': 'cậu', 'kqua': 'kết quả', 'kq': 'kết quả', 'htrc': 'hôm trước',
    'địh': 'định', 'gđình': 'gia đinh', 'giốg': 'giống', 'csống': 'cuộc sống',
    'zùi': 'rồi', 'bnhiêu': 'bao nhiêu', 'cbị': 'chuẩn bị', 'kòn': 'còn',
    'buôg': 'buông', 'csong': 'cuộc sống', 'chàg': 'chàng', 'sad': 'buồn',
    'đma': 'đụ má', 'nà': 'là', 'chăg': 'chăng', 'ngàh': 'ngành',
    'llac': 'liên lạc', 'nkưng': 'nhưng', 'xink': 'xinh', 'nắg': 'nắng',
    'tíh': 'tính', 'khoảg': 'khoảng', 'thík': 'thích', 'ngđo': 'người đó',
    'ngkhác': 'người khác', 'thẳg': 'thẳng', 'kảm': 'cảm', 'dàh': 'dành',
    'júp': 'giúp', 'lặg': 'lặng', 'vđê': 'vấn đề', 'bbè': 'bạn bè', 'bóg': 'bóng',
    'dky': 'đăng ký', 'dòg': 'dòng', 'uốg': 'uống', 'nvien': 'nhân viên',
    'tyêu': 'tình yêu', 'snvv': 'sinh nhật vui vẻ', 'đthoại': 'điện thoại',
    'qhe': 'quan hệ', 'cviec': 'công việc', 'tượg': 'tượng', 'qà': 'quà',
    'thjc': 'thích', 'nhưq': 'nhưng', 'cđời': 'cuộc đời', 'bthường': 'bình thường',
    'zà': 'già', 'đáh': 'đánh', 'xloi': 'xin lỗi', 'xlũi': 'xin lỗi',
    'lũi': 'lỗi', 'zám': 'dám', 'qtrọng': 'quan trọng', 'bìh': 'bình',
    'lzi': 'làm gì', 'qhệ': 'quan hệ', 'kủa': 'của', 'đóg': 'đóng',
    'ngulon': 'ngu lồn', 'cka': 'cha', 'lgi': 'làm gì', 'nvậy': 'như vậy',
    'qả': 'quả', 'đkiện': 'điều kiện', 'mi': 'mày', 'dụg': 'dụng', 'mj': 'mày',
    'nèk': 'nè', 'tlai': 'tương lai', 'bsĩ': 'bác sĩ', 'vde': 'vấn đề',
    'chta': 'chúng ta', 'òy': 'rồi', 'ltinh': 'linh tinh', 'bome': 'bỏ mẹ',
    'ngyeu': 'người yêu', 'thank': 'cảm ơn', 'tks': 'cảm ơn', 'thanks': 'cảm ơn',
    'đthoai': 'điện thoại', 'snghĩ': 'suy nghĩ', 'nặg': 'nặng', 'họk': 'học',
    'móa': 'má', 'zl': 'vãi lồn', 'dừg': 'dừng', 'hphúc': 'hạnh phúc',
    'nge': 'nghe', 'wtâm': 'quan tâm', 'fa': 'cô đơn', 'thíck': 'thích',
    'chuện': 'chuyện', 'lạh': 'lạnh', 'ntnày': 'như thế này', 'dek': 'đéo',
    'thèn': 'thằng', 'cức': 'cứt', 'mé': 'má', 'ẻm': 'em', 'lúk': 'lúc',
    'haj': 'hai', 'kím': 'kiếm', 'ngía': 'nghía', 'mớj': 'mới', 'hsơ': 'hồ sơ',
    'ctraj': 'con trai', 'nyêu': 'người yêu', 'dma': 'đụ má', 'c': 'chị',
    'kih': 'kinh', 'kb': 'kết bạn', 'dthương': 'dễ thương', 'dth': 'dễ thương',
    'nhìu': 'nhiều', 'ctrình': 'chương trình', 'mìnk': 'mình', 'mjh': 'mình',
    'ng': 'người', 'vc': 'vãi cặc', 'ck': 'chồng', 'thỳ': 'thì',
    'nyc': 'người yêu cũ', 'ex': 'người yêu cũ', 'nàg': 'nàng', 'thui': 'thôi',
    'đjên': 'điên', 'bgái': 'bạn gái', 'zới': 'với', 'hđộng': 'hành động',
    'đhọc': 'đại học', 'yêuc': 'yêu', 'đóa': 'đó', 'đuỵt': 'địt', 'mk': 'mình',
    'thik': 'thích', 'ci': 'chị', 'sx': 'sản xuất', 'vailon': 'vãi lồn',
    'sxuat': 'sản xuất', 'mn': 'mọi người', 'nógn': 'nóng', 'free': 'miễn phí',
    'hok': 'không', 'vlon': 'vãi lồn', 'vlol': 'vãi lồn', 'ditme': 'địt mẹ',
    'list': 'danh sách', 'mm': 'mẹ mày', 'm': 'mày', 'ko': 'không', 'bik': 'biết',
    'dume': 'đụ mẹ', 't': 'tao', 'đume': 'đụ mẹ', 'vs': 'với', 'cx': 'cũng',
    'mik': 'mình', 'đc': 'được', 'hag': 'hàng', 'hk': 'không', 'n': 'nó',
    'hàg': 'hàng', 'ngta': 'người ta', 'gđ': 'gia đình', 'ah': 'à',
    'vlone': 'vãi lồn', 'đcđ': 'đéo chịu được', 'dcd': 'đéo chịu được',
    'vk': 'vợ', 'cc': ' con cặc', 'ctác': 'công tác', 'sg': 'sài gòn',
    'dmm': 'địt mẹ mày', 'dm': 'đụ mẹ', 'đm': 'đụ mẹ', 'dcm': 'địt con mẹ',
    'đcm': 'địt con mẹ', 'đmm': 'địt mẹ mày', 'dkm': 'địt con mẹ', 'ae': 'anh em',
    'rì': 'gì', 'vl': 'vãi lồn', 'ms': 'mới', 'vn': 'việt nam', 'cũg': 'cũng',
    'bit': 'biết', 'đag': 'đang', 'cmm': 'con mẹ mày', 'cmnr': 'con mẹ nó rồi',
    'hnay': 'hôm nay', 'tq': 'trung quốc', 'ctr': 'chương trình',
    'nch': 'nói chuyện', 'block': 'chặn', 'nta': 'người ta', 'ngèo': 'nghèo',
    'kêh': 'kênh', 'ak': 'à', 'j': 'gì', 'ny': 'người yêu', 'qc': 'quảng cáo',
    'baoh': 'bao giờ', 'zui': 'vui', 'zẻ': 'vẻ', 'tym': 'tim',
    'aye': 'anh yêu em', 'eya': 'em yêu anh', 'z': 'vậy', 'zậy': 'vậy',
    'thich': 'thích', 'vcl': 'vãi cả lồn', 'đt': 'điện thoại', 'cl': 'cái lồn',
    'lol': 'lồn', 'loz': 'lồn', 'đuma': 'đụ má', 'lz': 'lồn', 'trc': 'trước',
    'chs': 'chẳng hiểu sao', 'đhs': 'đéo hiểu sao', 'dhs': 'đéo hiểu sao',
    'qá': 'quá', 'ntn': 'như thế nào', 'wá': 'quá', 'duma': 'đụ má', 'zô': 'dô',
    'vđ': 'vãi đái', 'vchg': 'vãi chưởng', 'ml': 'mặt lồn', 'sml': 'sấp mặt lồn',
    'xl': 'xin lỗi', 'cmn': 'con mẹ nó', 'cmt': 'bình luận', 'ns': 'nói',
    'iu': 'yêu', 'ctay': 'chia tay', 'ju': 'yêu', 'vcđ': 'vãi cả đái',
    'qq': 'quằn què', 'kh': 'không', 'zạ': 'dạ', 'mis': 'nhớ', 'h': 'giờ',
    'jo': 'giờ', 'clmm': 'cái lồn mẹ mày', 'troai': 'trai', 'wa': 'quá',
    'e': 'em', 'ji': 'gì', 'ce': 'chị em', 'lm': 'làm', 'đz': 'đẹp giai',
    'hỏg': 'hỏng', 'hoy': 'thôi', 'đbh': 'đéo bao giờ', 'dbh': 'đéo bao giờ',
    'k': 'không', 'vd': 'ví dụ', 'a': 'anh', 'cty': 'công ty', 'lở': 'lỡ',
    'kô': 'không', 'hqua': 'hôm qua', 'xog': 'xong', 'nhoé': 'nhé',
    'biet': 'biết', 'bjk': 'biết', 'bjt': 'biết', 'quí': 'quý',
    'kbh': 'không bao giờ', 'stk': 'số tài khoản', 'nghành': 'ngành',
    'trog': 'trong', 'tgian': 'thời gian', 'cf': 'cà phê', 'cafe': 'cà phê',
    'biêt': 'biết', 'tđn': 'thế đéo nào', 'bth': 'bình thường',
    'tgd': 'thế giới di động', 'khg': 'không', 'nhưg': 'nhưng', 'thằg': 'thằng',
    'đuợc': 'được', 'hj': 'hi', 'dc': 'được', 'ku': 'cu', 'thým': 'thím',
    'zú': 'vú', 'sđt': 'số điện thoại', 'klq': 'không liên quan',
    'vkl': 'vãi cả lồn',
}

# Emoticon phải thay trước khi tách từ vì chúng chứa dấu câu.
_EMOJI_KEYS = sorted(EMOJI_MAP, key=len, reverse=True)
_REPLACE_COMPILED = [(re.compile(p), r) for p, r in REPLACE_LIST.items()]
_WORD_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)


def normalize_common(text: str) -> str:
    """Bước chung cho cả ba scenario: gọn dấu câu lặp, gỡ HTML entity."""
    s = str(text)
    for pat, rep in _REPLACE_COMPILED:
        s = pat.sub(rep, s)
    return s


def emoji_to_text(text: str) -> str:
    """Thay emoticon và emoji bằng mô tả tiếng Việt (chỉ dùng ở S2)."""
    s = text
    for k in _EMOJI_KEYS:
        if k in s:
            s = s.replace(k, f" {EMOJI_MAP[k]} ")
    return s


def normalize_teencode(text: str) -> str:
    """Thay teencode bằng dạng chuẩn, so khớp theo từng từ, không phân biệt hoa thường."""
    def _sub(mo):
        w = mo.group(0)
        return TEENCODE_MAP.get(w.lower(), w)
    return _WORD_RE.sub(_sub, text)


def clean_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# ViSoLex (cho S3). Nếu không nạp được, hàm trả về None và pipeline sẽ báo rõ.
# ---------------------------------------------------------------------------
_VISOLEX = None


def load_visolex(model_name: str = "uitnlp/visolex"):
    """Nạp ViSoLex nếu có. Trả về callable(text)->text hoặc None."""
    global _VISOLEX
    if _VISOLEX is not None:
        return _VISOLEX
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        mdl = mdl.to(dev).eval()

        @torch.no_grad()
        def _run(texts, batch_size=32, max_length=128):
            out = []
            for i in range(0, len(texts), batch_size):
                enc = tok(texts[i:i + batch_size], return_tensors="pt",
                          padding=True, truncation=True, max_length=max_length).to(dev)
                gen = mdl.generate(**enc, max_length=max_length)
                out += tok.batch_decode(gen, skip_special_tokens=True)
            return out

        _VISOLEX = _run
        print(f"[OK] đã nạp ViSoLex: {model_name}")
    except Exception as e:
        print(f"[!] không nạp được ViSoLex ({type(e).__name__}: {str(e)[:80]})")
        _VISOLEX = None
    return _VISOLEX


# ---------------------------------------------------------------------------
# API chính
# ---------------------------------------------------------------------------
def preprocess(text: str, scenario: str = "s1") -> str:
    """Tiền xử lý một câu theo scenario."""
    s = normalize_common(text)
    if scenario == "s1":
        s = normalize_teencode(s)                     # giữ nguyên emoji
    elif scenario == "s2":
        s = emoji_to_text(s)                          # emoji -> chữ
        s = normalize_teencode(s)
    elif scenario == "s3":
        pass                                          # ViSoLex xử lý ở mức batch
    else:
        raise ValueError(f"scenario không hợp lệ: {scenario}")
    return clean_spaces(s)


def preprocess_series(texts, scenario: str = "s1", visolex_fallback: bool = True):
    """Tiền xử lý cả một cột. Với S3 sẽ gọi ViSoLex theo batch."""
    texts = [str(t) for t in texts]
    if scenario != "s3":
        return [preprocess(t, scenario) for t in texts]

    base = [normalize_common(t) for t in texts]
    fn = load_visolex()
    if fn is None:
        if not visolex_fallback:
            raise RuntimeError("Không nạp được ViSoLex và visolex_fallback=False")
        print("    [!] S3 dùng TẠM từ điển thủ công thay ViSoLex — "
              "phải ghi rõ điều này trong báo cáo")
        return [clean_spaces(normalize_teencode(t)) for t in base]
    print("    chạy ViSoLex...")
    return [clean_spaces(t) for t in fn(base)]


def build_all_scenarios(df: pd.DataFrame, text_col: str = "text",
                        scenarios=("s1", "s2", "s3")) -> pd.DataFrame:
    """Thêm cột text_s1 / text_s2 / text_s3 vào DataFrame."""
    out = df.copy()
    for sc in scenarios:
        print(f"  [{sc}] ...")
        out[f"text_{sc}"] = preprocess_series(out[text_col].tolist(), sc)
        n_changed = (out[f"text_{sc}"] != out[text_col].astype(str)).sum()
        print(f"       {n_changed}/{len(out)} dòng thay đổi "
              f"({100*n_changed/len(out):.1f}%)")
    return out


def demo(n=6):
    samples = [
        "ko hiểu j luôn z ad ơi 😭😭😭",
        "bức ảnh xuất sắc ❤️ =))))",
        "mik thấy bth thoi, cx dc mà mn 🙂",
        "Bài học: <br>1: 5 nhìn, 4 chạm &amp; 3 nghe!!!",
        "cute quá đi mất 😍 vcl",
        "trời ơi cười rụng rốn kkkk :))",
    ][:n]
    rows = []
    for s in samples:
        rows.append({"gốc": s,
                     "S1": preprocess(s, "s1"),
                     "S2": preprocess(s, "s2")})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.max_colwidth", 70)
    print(demo().to_string(index=False))