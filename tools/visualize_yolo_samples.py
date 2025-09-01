import os, random, cv2
import matplotlib.pyplot as plt

CLASS_NAMES = ['pi','me','gl','no']

def draw_boxes(ax, img_bgr, label_path):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5: 
                    continue
                cls, xc, yc, bw, bh = int(parts[0]), *map(float, parts[1:5])
                xmin = int((xc-bw/2)*w); ymin = int((yc-bh/2)*h)
                xmax = int((xc+bw/2)*w); ymax = int((yc+bh/2)*h)
                cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (255,0,0), 2)
                cv2.putText(img, CLASS_NAMES[cls], (xmin, max(10,ymin-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    ax.imshow(img); ax.axis('off')

def show_samples(yolo_root, split='val', n=8, seed=42):
    random.seed(seed)
    img_dir = os.path.join(yolo_root, 'images', split)
    lbl_dir = os.path.join(yolo_root, 'labels', split)
    imgs = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))]
    if not imgs:
        raise FileNotFoundError(f"No images under {img_dir}")
    sel = random.sample(imgs, min(n, len(imgs)))
    rows, cols = 2, (n+1)//2
    plt.figure(figsize=(4*cols, 4*rows))
    for i, name in enumerate(sel, 1):
        ax = plt.subplot(rows, cols, i)
        img_path = os.path.join(img_dir, name)
        lbl_path = os.path.join(lbl_dir, os.path.splitext(name)[0]+'.txt')
        img = cv2.imread(img_path)
        draw_boxes(ax, img, lbl_path)
        ax.set_title(name)
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    show_samples("data/yolo", split="val", n=8)
