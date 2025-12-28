"""
VisionGuide Pro - 500 Kelimelik Nesne Sözlüğü
İngilizce-Türkçe eşleşmeleri ve mod tanımları
"""

# EV MODU (~170 kelime)
EV_MODU = {
    # Mutfak Gereçleri
    'spoon': 'Kaşık', 'fork': 'Çatal', 'knife': 'Bıçak', 'plate': 'Tabak',
    'bowl': 'Kase', 'cup': 'Fincan', 'mug': 'Bardak', 'glass': 'Cam',
    'kettle': 'Su ısıtıcısı', 'pot': 'Tencere', 'pan': 'Tava', 'ladle': 'Kepçe',
    'spatula': 'Spatula', 'whisk': 'Çırpıcı', 'grater': 'Rende', 'cutting board': 'Kesme tahtası',
    'can opener': 'Konserve açacağı', 'bottle opener': 'Şişe açacağı', 'corkscrew': 'Tıpa açacağı',
    'dishwasher': 'Bulaşık makinesi', 'refrigerator': 'Buzdolabı', 'oven': 'Fırın',
    'microwave': 'Mikrodalga', 'stove': 'Ocak', 'toaster': 'Tost makinesi',
    'coffee maker': 'Kahve makinesi', 'blender': 'Blender', 'mixer': 'Mikser',
    'sink': 'Lavabo', 'faucet': 'Musluk', 'tap': 'Musluk', 'drain': 'Gider',
    
    # Mobilya ve Ev Detayları
    'chair': 'Sandalye', 'table': 'Masa', 'sofa': 'Kanepe', 'couch': 'Kanepe',
    'bed': 'Yatak', 'pillow': 'Yastık', 'blanket': 'Battaniye', 'sheet': 'Çarşaf',
    'wardrobe': 'Gardırop', 'cabinet': 'Dolap', 'drawer': 'Çekmece', 'shelf': 'Raf',
    'desk': 'Yazı masası', 'bookshelf': 'Kitaplık', 'dresser': 'Kommode',
    'door': 'Kapı', 'door handle': 'Kapı kolu', 'door knob': 'Kapı kolu',
    'window': 'Pencere', 'curtain': 'Perde', 'blind': 'Jaluzi',
    'light switch': 'Elektrik anahtarı', 'switch': 'Anahtar', 'outlet': 'Priz',
    'socket': 'Priz', 'plug': 'Fiş', 'lamp': 'Lamba', 'light': 'Işık',
    'ceiling light': 'Tavan lambası', 'floor lamp': 'Ayaklı lamba',
    'staircase': 'Merdiven', 'stairs': 'Merdiven', 'stair railing': 'Merdiven korkuluğu',
    'handrail': 'Korkuluk', 'banister': 'Korkuluk',
    
    # Banyo Eşyaları
    'toilet': 'Tuvalet', 'toilet paper': 'Tuvalet kağıdı', 'toilet brush': 'Tuvalet fırçası',
    'shower': 'Duş', 'bathtub': 'Küvet', 'bath': 'Küvet',
    'towel': 'Havlu', 'bath towel': 'Banyo havlusu', 'hand towel': 'El havlusu',
    'soap': 'Sabun', 'shampoo': 'Şampuan', 'toothbrush': 'Diş fırçası',
    'toothpaste': 'Diş macunu', 'mirror': 'Ayna', 'bathroom mirror': 'Banyo aynası',
    'sink': 'Lavabo', 'bathroom sink': 'Banyo lavabosu', 'medicine cabinet': 'İlaç dolabı',
    
    # Ev Aletleri ve Elektronik
    'television': 'Televizyon', 'TV': 'Televizyon', 'remote control': 'Kumanda',
    'remote': 'Kumanda', 'speaker': 'Hoparlör', 'headphones': 'Kulaklık',
    'computer': 'Bilgisayar', 'laptop': 'Dizüstü bilgisayar', 'keyboard': 'Klavye',
    'mouse': 'Fare', 'monitor': 'Monitör', 'printer': 'Yazıcı',
    'vacuum cleaner': 'Elektrik süpürgesi', 'vacuum': 'Elektrik süpürgesi',
    'iron': 'Ütü', 'hair dryer': 'Saç kurutma makinesi', 'fan': 'Vantilatör',
    'air conditioner': 'Klima', 'heater': 'Isıtıcı', 'radiator': 'Radyatör',
    
    # Diğer Ev Eşyaları
    'clock': 'Saat', 'wall clock': 'Duvar saati', 'alarm clock': 'Çalar saat',
    'picture': 'Resim', 'photo': 'Fotoğraf', 'frame': 'Çerçeve',
    'plant': 'Bitki', 'flower': 'Çiçek', 'vase': 'Vazo',
    'trash can': 'Çöp kutusu', 'wastebasket': 'Çöp sepeti', 'bin': 'Çöp kutusu',
    'broom': 'Süpürge', 'mop': 'Paspas', 'bucket': 'Kova',
    'umbrella': 'Şemsiye', 'coat hanger': 'Askı', 'hanger': 'Askı',
    'key': 'Anahtar', 'keys': 'Anahtarlar', 'wallet': 'Cüzdan',
    'phone': 'Telefon', 'mobile phone': 'Cep telefonu', 'cell phone': 'Cep telefonu',
    'newspaper': 'Gazete', 'magazine': 'Dergi', 'book': 'Kitap',
    'pen': 'Kalem', 'pencil': 'Kurşun kalem', 'notebook': 'Defter',
    
    # Ek Ev Detayları
    'dining table': 'Yemek masası', 'coffee table': 'Sehpa', 'side table': 'Yan sehpa',
    'armchair': 'Koltuk', 'recliner': 'Kanepe koltuğu', 'ottoman': 'Puf',
    'mattress': 'Yatak', 'bed frame': 'Yatak çerçevesi', 'headboard': 'Başlık',
    'nightstand': 'Komodin', 'bedside table': 'Komodin',
    'closet': 'Dolap', 'wardrobe door': 'Gardırop kapağı',
    'window sill': 'Pencere eşiği', 'window frame': 'Pencere çerçevesi',
    'door frame': 'Kapı çerçevesi', 'doorbell': 'Kapı zili',
    'thermostat': 'Termostat', 'smoke detector': 'Duman dedektörü',
    'power outlet': 'Elektrik prizi', 'extension cord': 'Uzatma kablosu',
    'broom closet': 'Temizlik dolabı', 'laundry basket': 'Çamaşır sepeti',
    'washing machine': 'Çamaşır makinesi', 'dryer': 'Kurutma makinesi',
    'dish rack': 'Bulaşık kurutma rafı', 'sponge': 'Sünger',
    'dish soap': 'Bulaşık deterjanı', 'detergent': 'Deterjan',
    'trash bag': 'Çöp poşeti', 'recycling bin': 'Geri dönüşüm kutusu',
    'door mat': 'Kapı paspası', 'rug': 'Halı', 'carpet': 'Halı',
    'curtain rod': 'Perde askısı', 'blinds': 'Jaluzi',
    'picture hook': 'Resim askısı', 'nail': 'Çivi', 'screw': 'Vida',
}

# SOKAK MODU (~170 kelime)
SOKAK_MODU = {
    # Trafik İşaretleri ve Yol Elemanları
    'traffic light': 'Trafik ışığı', 'stop sign': 'Dur işareti', 'yield sign': 'Yol ver işareti',
    'speed limit sign': 'Hız sınırı işareti', 'road sign': 'Yol işareti', 'sign': 'İşaret',
    'traffic sign': 'Trafik işareti', 'warning sign': 'Uyarı işareti',
    'crosswalk': 'Yaya geçidi', 'zebra crossing': 'Yaya geçidi', 'pedestrian crossing': 'Yaya geçidi',
    'sidewalk': 'Kaldırım', 'pavement': 'Kaldırım', 'curb': 'Kaldırım kenarı',
    'road': 'Yol', 'street': 'Cadde', 'avenue': 'Bulvar', 'highway': 'Otoyol',
    'traffic cone': 'Trafik konisi', 'cone': 'Koni', 'barrier': 'Bariyer',
    'barricade': 'Bariyer', 'roadblock': 'Yol barikatı',
    'speed bump': 'Hız kesici', 'speed hump': 'Hız kesici',
    'parking meter': 'Parkmetre', 'parking sign': 'Park işareti',
    'stoplight': 'Trafik ışığı', 'red light': 'Kırmızı ışık', 'green light': 'Yeşil ışık',
    
    # Araçlar
    'car': 'Araba', 'automobile': 'Araba', 'vehicle': 'Araç',
    'truck': 'Kamyon', 'bus': 'Otobüs', 'van': 'Minibüs',
    'motorcycle': 'Motosiklet', 'bike': 'Bisiklet', 'bicycle': 'Bisiklet',
    'scooter': 'Scooter', 'electric scooter': 'Elektrikli scooter',
    'taxi': 'Taksi', 'ambulance': 'Ambulans', 'police car': 'Polis aracı',
    'fire truck': 'İtfaiye aracı', 'delivery truck': 'Kargo aracı',
    
    # Yaya Altyapısı
    'bench': 'Bank', 'park bench': 'Park bankı', 'bus stop': 'Otobüs durağı',
    'bus shelter': 'Otobüs durağı', 'stop': 'Durak',
    'streetlight': 'Sokak lambası', 'street lamp': 'Sokak lambası', 'lamp post': 'Lamba direği',
    'pole': 'Direk', 'utility pole': 'Elektrik direği', 'traffic pole': 'Trafik direği',
    
    # Doğal Öğeler
    'tree': 'Ağaç', 'tree branch': 'Ağaç dalı', 'branch': 'Dal',
    'trunk': 'Gövde', 'tree trunk': 'Ağaç gövdesi', 'leaves': 'Yapraklar',
    'grass': 'Çim', 'flower': 'Çiçek', 'bush': 'Çalı', 'shrub': 'Çalı',
    'fence': 'Çit', 'gate': 'Kapı', 'hedge': 'Çit',
    
    # Binalar ve Yapılar
    'building': 'Bina', 'house': 'Ev', 'apartment': 'Apartman',
    'store': 'Mağaza', 'shop': 'Dükkan', 'restaurant': 'Restoran',
    'cafe': 'Kafe', 'bank': 'Banka', 'ATM': 'ATM', 'atm machine': 'ATM',
    'post office': 'Postane', 'hospital': 'Hastane', 'school': 'Okul',
    'church': 'Kilise', 'mosque': 'Cami', 'temple': 'Tapınak',
    'entrance': 'Giriş', 'exit': 'Çıkış', 'door': 'Kapı',
    'window': 'Pencere', 'stairs': 'Merdiven', 'step': 'Basamak',
    'ramp': 'Rampa', 'handrail': 'Korkuluk', 'railing': 'Korkuluk',
    
    # Sokak Mobilyaları
    'mailbox': 'Posta kutusu', 'post box': 'Posta kutusu',
    'trash can': 'Çöp kutusu', 'garbage can': 'Çöp kutusu', 'waste bin': 'Çöp kutusu',
    'recycling bin': 'Geri dönüşüm kutusu', 'bin': 'Çöp kutusu',
    'newspaper stand': 'Gazete standı', 'kiosk': 'Kiosk',
    'vending machine': 'Otomat', 'drink machine': 'İçecek otomatı',
    'fire hydrant': 'Yangın musluğu', 'hydrant': 'Yangın musluğu',
    'manhole cover': 'Rögar kapağı', 'drain': 'Gider',
    
    # Güvenlik ve Uyarı
    'security camera': 'Güvenlik kamerası', 'camera': 'Kamera', 'CCTV': 'Güvenlik kamerası',
    'alarm': 'Alarm', 'warning': 'Uyarı', 'caution sign': 'Dikkat işareti',
    'fence': 'Çit', 'barrier': 'Bariyer', 'gate': 'Kapı',
    
    # Diğer
    'puddle': 'Su birikintisi', 'water': 'Su', 'pothole': 'Çukur',
    'crack': 'Çatlak', 'graffiti': 'Grafiti', 'poster': 'Poster',
    'flag': 'Bayrak', 'banner': 'Afiş', 'advertisement': 'Reklam',
    'crosswalk button': 'Yaya geçidi butonu', 'button': 'Buton',
    'pedestrian': 'Yaya', 'person': 'İnsan', 'people': 'İnsanlar',
    
    # Ek Sokak Detayları
    'parking lot': 'Otopark', 'parking space': 'Park yeri', 'parking spot': 'Park yeri',
    'crossing guard': 'Yaya geçidi görevlisi', 'traffic officer': 'Trafik polisi',
    'bike lane': 'Bisiklet yolu', 'bicycle lane': 'Bisiklet yolu',
    'median': 'Yol orta refüjü', 'center divider': 'Orta refüj',
    'roundabout': 'Döner kavşak', 'traffic circle': 'Döner kavşak',
    'bridge': 'Köprü', 'overpass': 'Üst geçit', 'underpass': 'Alt geçit',
    'tunnel': 'Tünel', 'subway entrance': 'Metro girişi',
    'newsstand': 'Gazete standı', 'phone booth': 'Telefon kulübesi',
    'public restroom': 'Umumi tuvalet', 'restroom sign': 'Tuvalet işareti',
    'elevator': 'Asansör', 'escalator': 'Yürüyen merdiven',
    'bench seat': 'Bank', 'picnic table': 'Piknik masası',
    'fountain': 'Çeşme', 'statue': 'Heykel', 'monument': 'Anıt',
    'playground': 'Oyun alanı', 'playground equipment': 'Oyun parkı ekipmanı',
    'swing': 'Salıncak', 'slide': 'Kaydırak', 'seesaw': 'Tahterevalli',
    'dog': 'Köpek', 'cat': 'Kedi', 'bird': 'Kuş',
    'pigeon': 'Güvercin', 'crow': 'Karga',
    'bicycle rack': 'Bisiklet parkı', 'bike stand': 'Bisiklet standı',
    'construction sign': 'İnşaat işareti', 'construction barrier': 'İnşaat bariyeri',
    'work zone': 'Çalışma alanı', 'detour sign': 'Yön değiştirme işareti',
    'street vendor': 'Sokak satıcısı', 'food cart': 'Yiyecek arabası',
    'bench': 'Bank', 'public bench': 'Halka açık bank',
    'street art': 'Sokak sanatı', 'mural': 'Duvar resmi',
    'parking garage': 'Otopark', 'parking structure': 'Otopark yapısı',
    'loading zone': 'Yükleme alanı', 'delivery zone': 'Teslimat alanı',
}

# MARKET/OFİS MODU (~160 kelime)
MARKET_OFIS_MODU = {
    # Ofis Malzemeleri
    'stapler': 'Zımba', 'staples': 'Zımba teli', 'paper clip': 'Ataş',
    'clip': 'Ataş', 'binder clip': 'Dosya klipsi',
    'pen': 'Kalem', 'pencil': 'Kurşun kalem', 'marker': 'Kalem',
    'highlighter': 'Fosforlu kalem', 'eraser': 'Silgi', 'correction tape': 'Düzeltme bandı',
    'scissors': 'Makas', 'tape': 'Bant', 'adhesive tape': 'Yapışkan bant',
    'glue': 'Yapıştırıcı', 'glue stick': 'Yapıştırıcı çubuk',
    'ruler': 'Cetvel', 'protractor': 'Açıölçer', 'compass': 'Pergel',
    'calculator': 'Hesap makinesi', 'notebook': 'Defter', 'notepad': 'Not defteri',
    'folder': 'Klasör', 'file': 'Dosya', 'binder': 'Cilt',
    'envelope': 'Zarf', 'letter': 'Mektup', 'document': 'Belge',
    'paper': 'Kağıt', 'sheet': 'Kağıt', 'copy paper': 'Fotokopi kağıdı',
    'printer': 'Yazıcı', 'scanner': 'Tarayıcı', 'photocopier': 'Fotokopi makinesi',
    'desk': 'Yazı masası', 'office chair': 'Ofis koltuğu', 'chair': 'Sandalye',
    'computer': 'Bilgisayar', 'laptop': 'Dizüstü bilgisayar', 'monitor': 'Monitör',
    'keyboard': 'Klavye', 'mouse': 'Fare', 'mouse pad': 'Fare altlığı',
    'telephone': 'Telefon', 'desk phone': 'Masa telefonu', 'headset': 'Kulaklık',
    'whiteboard': 'Beyaz tahta', 'blackboard': 'Kara tahta', 'chalk': 'Tebeşir',
    'marker': 'Kalem', 'dry erase marker': 'Beyaz tahta kalemi',
    'bulletin board': 'İlan tahtası', 'pin board': 'Pano',
    'push pin': 'Raptiye', 'thumbtack': 'Raptiye', 'pin': 'İğne',
    'calendar': 'Takvim', 'clock': 'Saat', 'wall clock': 'Duvar saati',
    'lamp': 'Lamba', 'desk lamp': 'Masa lambası', 'light': 'Işık',
    'wastebasket': 'Çöp sepeti', 'trash can': 'Çöp kutusu', 'recycling bin': 'Geri dönüşüm kutusu',
    'cabinet': 'Dolap', 'filing cabinet': 'Dosya dolabı', 'drawer': 'Çekmece',
    'shelf': 'Raf', 'bookcase': 'Kitaplık', 'bookshelf': 'Kitaplık',
    
    # Market Ürün Kategorileri
    'bottle': 'Şişe', 'water bottle': 'Su şişesi', 'plastic bottle': 'Plastik şişe',
    'glass bottle': 'Cam şişe', 'wine bottle': 'Şarap şişesi',
    'can': 'Kutu', 'tin can': 'Konserve kutusu', 'aluminum can': 'Alüminyum kutu',
    'jar': 'Kavanoz', 'glass jar': 'Cam kavanoz', 'mason jar': 'Kavanoz',
    'container': 'Konteyner', 'box': 'Kutu', 'cardboard box': 'Karton kutu',
    'package': 'Paket', 'packaging': 'Ambalaj', 'wrapper': 'Ambalaj',
    'bag': 'Poşet', 'plastic bag': 'Plastik poşet', 'shopping bag': 'Alışveriş çantası',
    'paper bag': 'Kağıt poşet', 'tote bag': 'Bez çanta',
    'cart': 'Sepet', 'shopping cart': 'Alışveriş arabası', 'basket': 'Sepet',
    'basket': 'Sepet', 'shopping basket': 'Alışveriş sepeti',
    
    # Gıda Kategorileri
    'bread': 'Ekmek', 'loaf': 'Somun ekmek', 'baguette': 'Baget',
    'milk': 'Süt', 'milk carton': 'Süt kutusu', 'yogurt': 'Yoğurt',
    'cheese': 'Peynir', 'butter': 'Tereyağı', 'eggs': 'Yumurta',
    'fruit': 'Meyve', 'apple': 'Elma', 'banana': 'Muz', 'orange': 'Portakal',
    'vegetable': 'Sebze', 'tomato': 'Domates', 'potato': 'Patates',
    'cereal': 'Tahıl', 'cereal box': 'Tahıl kutusu', 'breakfast cereal': 'Kahvaltılık tahıl',
    'crackers': 'Kraker', 'chips': 'Cips', 'snacks': 'Atıştırmalık',
    'candy': 'Şeker', 'chocolate': 'Çikolata', 'cookies': 'Kurabiye',
    'soup': 'Çorba', 'soup can': 'Çorba kutusu', 'canned food': 'Konserve',
    'pasta': 'Makarna', 'rice': 'Pirinç', 'flour': 'Un',
    'sugar': 'Şeker', 'salt': 'Tuz', 'pepper': 'Biber',
    'oil': 'Yağ', 'cooking oil': 'Yemeklik yağ', 'olive oil': 'Zeytinyağı',
    'vinegar': 'Sirke', 'sauce': 'Sos', 'ketchup': 'Ketçap',
    'coffee': 'Kahve', 'tea': 'Çay', 'coffee bag': 'Kahve paketi',
    'tea bag': 'Çay poşeti',
    
    # Teknolojik Cihazlar
    'smartphone': 'Akıllı telefon', 'phone': 'Telefon', 'mobile phone': 'Cep telefonu',
    'tablet': 'Tablet', 'iPad': 'Tablet', 'laptop': 'Dizüstü bilgisayar',
    'computer': 'Bilgisayar', 'desktop': 'Masaüstü bilgisayar',
    'monitor': 'Monitör', 'screen': 'Ekran', 'display': 'Ekran',
    'keyboard': 'Klavye', 'mouse': 'Fare', 'webcam': 'Web kamerası',
    'speaker': 'Hoparlör', 'headphones': 'Kulaklık', 'earbuds': 'Kulaklık',
    'charger': 'Şarj aleti', 'cable': 'Kablo', 'USB cable': 'USB kablosu',
    'power bank': 'Powerbank', 'battery': 'Pil', 'AA battery': 'AA pil',
    'flash drive': 'USB bellek', 'USB drive': 'USB bellek', 'memory stick': 'USB bellek',
    'hard drive': 'Sabit disk', 'external drive': 'Harici disk',
    'router': 'Yönlendirici', 'modem': 'Modem', 'WiFi router': 'WiFi yönlendirici',
    'camera': 'Kamera', 'digital camera': 'Dijital kamera', 'webcam': 'Web kamerası',
    'watch': 'Saat', 'smartwatch': 'Akıllı saat', 'fitness tracker': 'Fitness takipçisi',
    
    # Ek Market/Ofis Detayları
    'shopping list': 'Alışveriş listesi', 'receipt': 'Fiş', 'cash register': 'Kasa',
    'cashier': 'Kasiyer', 'checkout counter': 'Kasa', 'conveyor belt': 'Bant',
    'price tag': 'Fiyat etiketi', 'barcode': 'Barkod', 'scanner': 'Barkod okuyucu',
    'shopping cart': 'Alışveriş arabası', 'basket': 'Sepet', 'hand basket': 'El sepeti',
    'produce section': 'Meyve-sebze reyonu', 'dairy section': 'Süt ürünleri reyonu',
    'meat section': 'Et reyonu', 'bakery': 'Fırın', 'deli counter': 'Şarküteri',
    'frozen food': 'Dondurulmuş gıda', 'freezer': 'Dondurucu', 'ice cream': 'Dondurma',
    'candy aisle': 'Şeker reyonu', 'snack aisle': 'Atıştırmalık reyonu',
    'beverage aisle': 'İçecek reyonu', 'soda': 'Gazoz', 'juice': 'Meyve suyu',
    'water': 'Su', 'sparkling water': 'Maden suyu',
    'meat': 'Et', 'chicken': 'Tavuk', 'beef': 'Sığır eti', 'pork': 'Domuz eti',
    'fish': 'Balık', 'seafood': 'Deniz ürünleri',
    'produce': 'Meyve-sebze', 'lettuce': 'Marul', 'carrot': 'Havuç', 'onion': 'Soğan',
    'garlic': 'Sarımsak', 'pepper': 'Biber', 'cucumber': 'Salatalık',
    'stapler remover': 'Zımba sökmek', 'hole punch': 'Delgeç', 'binder': 'Cilt',
    'folder tab': 'Klasör etiketi', 'label': 'Etiket', 'sticker': 'Etiket',
    'rubber band': 'Lastik bant', 'paperweight': 'Kağıt ağırlığı',
    'desk organizer': 'Masa düzenleyici', 'pen holder': 'Kalemlik',
    'inbox': 'Gelen kutusu', 'outbox': 'Giden kutusu',
    'projector': 'Projeksiyon', 'screen': 'Perde', 'pointer': 'Lazer pointer',
    'conference table': 'Toplantı masası', 'meeting room': 'Toplantı odası',
}

# Tüm modları birleştir
OBJ_MAP = {}
OBJ_MAP.update(EV_MODU)
OBJ_MAP.update(SOKAK_MODU)
OBJ_MAP.update(MARKET_OFIS_MODU)

# Mod tanımları
MODES = {
    'E': {
        'name': 'EV',
        'classes': list(EV_MODU.keys()),
        'turkish_name': 'Ev Modu'
    },
    'S': {
        'name': 'SOKAK',
        'classes': list(SOKAK_MODU.keys()),
        'turkish_name': 'Sokak Modu'
    },
    'M': {
        'name': 'MARKET/OFİS',
        'classes': list(MARKET_OFIS_MODU.keys()),
        'turkish_name': 'Market/Ofis Modu'
    }
}

