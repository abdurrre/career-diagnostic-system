
Isi file **knowledge_base_skills.csv** dari beberapa informasi utama:
- **job_category**	(Nama profesi atau role)
- **skill**	(Nama skill)
- **frequency**	(Jumlah kemunculan skill pada job posting role terkait)
- **rank_in_role**	(Ranking skill berdasarkan frekuensi kemunculan)
- **tier**	(Tingkat prioritas skill)

# Data Dictionary
Secara sederhana, Data Dictionary bisa dianggap sebagai “kamus” yang dipakai sistem untuk memahami data. Isinya berupa daftar istilah, aturan, atau informasi penting tentang data yang digunakan selama proses analisis.
Dalam project ini, Data Dictionary dipakai terutama untuk membantu proses NLP dan pengolahan teks, khususnya saat membaca skill, job title, dan keyword dari job posting maupun CV user.

Beberapa fungsi utamanya:
- Menyamakan Penulisan Skill
  (Banyak skill ditulis dengan format berbeda-beda. Data Dictionary membantu menyatukan variasi tersebut menjadi istilah yang konsisten.
Contohnya seperti JS, Java Script, dan Reactjs yang dipetakan menjadi format standar tertentu agar sistem tidak menganggapnya sebagai skill yang berbeda.)
- Membantu Cleaning Data
  (Dictionary juga dipakai untuk menyaring kata yang tidak relevan, typo, atau noise pada data mentah sehingga hasil preprocessing jadi lebih bersih dan konsisten)
- Mempermudah Kolaborasi Tim
  (Dengan adanya dictionary, seluruh anggota tim punya acuan yang sama mengenai penamaan skill, kategori role, maupun format data yang digunakan di project)
- Membantu Proses NLP dan Skill Extraction
  (Saat sistem melakukan ekstraksi skill dari CV atau job posting, Data Dictionary menjadi referensi utama agar model bisa mengenali skill penting dengan lebih akurat tanpa harus “menebak” arti kata dari teks mentah)
