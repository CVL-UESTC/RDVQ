import os
from tqdm import *
from PIL import Image
from cleanfid import fid

def split_image(img, name, output_dir, start_x, start_y, tile_size=256, count=0):
    """
    将图片分割为指定起始点和尺寸的互不重叠小图并保存。
    
    参数:
        image_path (str): 原始图片路径
        output_dir (str): 输出目录
        start_x (int): 起始x坐标
        start_y (int): 起始y坐标
        tile_size (int): 小图尺寸,默认为256x256
    """
    width, height = img.size
    os.makedirs(output_dir, exist_ok=True)

    # 生成所有有效的x坐标
    x_coords = []
    x = start_x
    while x + tile_size <= width:
        if x < 0:
            x_coords.append(0)
        else:
            x_coords.append(x)
        x += tile_size

    # 生成所有有效的y坐标
    y_coords = []
    y = start_y
    while y + tile_size <= height:
        if y < 0:
            y_coords.append(0)
        else:
            y_coords.append(y)
        y += tile_size

    # 保存所有有效区域的小图
    for x in x_coords:
        for y in y_coords:
            # 定义裁剪区域 (左, 上, 右, 下)
            box = (x, y, x + tile_size, y + tile_size)
            tile = img.crop(box)
            
            # 生成文件名并保存
            filename = f"tile_{name}_{count}.png"
            tile.save(os.path.join(output_dir, filename))
            count += 1
    
    return count

def main():

    source = "/home/hanminghao/workspace/VDC/samples/div2k/rdm_sota_128bs_finetuned/scale:0.95/rho=11.0,alpha=0.06,beta=0.0,std_mul=1.0,scale_min=0.13,step=2,noise=gaussian"
    path = os.path.join(source, "x_hat")
    output_dir = os.path.join(source, "tiles")

    img_list = []
    for file in os.listdir(path):
        if file[-3:] in ["jpg", "png", "peg"]:
            img_list.append(file)

    tile_size = 256
    count = 0

    if not os.path.exists(output_dir):
        for img_name in tqdm(img_list):
            img_path = os.path.join(path, img_name)
            img = Image.open(img_path).convert('RGB')

            cur_cnt = 0

            # first division
            cur_cnt = split_image(img, img_name, output_dir, 0, 0, tile_size, cur_cnt)
            # second division
            cur_cnt = split_image(img, img_name, output_dir, tile_size // 2, tile_size // 2, tile_size, cur_cnt)

            count += cur_cnt
        
    print(f"generate {count} tiles in total")
    score_kid = fid.compute_kid(output_dir, 
                           dataset_name="div2k", mode="clean", dataset_split="custom")
    score_fid = fid.compute_fid(output_dir, 
                           dataset_name="div2k", mode="clean", dataset_split="custom")
    print(score_kid)
    print(score_fid)
    os.mkdir(os.path.join(source, f"fid:{score_fid:.4f}_kid:{score_kid}"))

if __name__ == "__main__":
    main()