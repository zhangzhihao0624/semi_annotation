import os.path
import shutil


def each_file(filepath, new_filepath):
    '''
    读取每个文件夹，将遇到的指定文件统统转移到指定目录中
    :param filepath: 想要获取的文件的目录
    :param new_filepath: 想要转移的指定目录
    :return:
    '''
    l_dir = os.listdir(filepath)  # 读取目录下的文件或文件夹

    for one_dir in l_dir:  # 进行循环
        full_path = os.path.join(filepath, one_dir)  # 构造路径
        new_full_path = os.path.join(new_filepath, one_dir)
        if os.path.isfile(full_path):  # 如果是文件类型就执行转移操作
            if one_dir.split('.')[1] == 'PNG':  # 只转移txt/PNG文件，修改相应后缀就可以转移不同的文件
                shutil.copy(full_path, new_full_path)  # 这个是转移的语句，最关键的一句话
        else:   # 不为文件类型就继续递归
            each_file(full_path, new_filepath)  # 如果是文件夹类型就有可能下面还有文件，要继续递归


if __name__ == '__main__':
    old_path = '/home/zhihao/yolo-training/assets/breakfest1'
    new_path = '/home/zhihao/yolo-training/assets/png'
    each_file(old_path, new_path)
