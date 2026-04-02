import random
from settings import CONFIG, GameConfig

# 7种经典方块形状定义 (3x3 或 4x4 矩阵)
SHAPES = {
    'I': ['....', 'XXXX', '....', '....'],
    'J': ['X..', 'XXX', '...'],
    'L': ['..X', 'XXX', '...'],
    'O': ['XX', 'XX'],
    'S': ['.XX', 'XX.', '...'],
    'T': ['.X.', 'XXX', '...'],
    'Z': ['XX.', '.XX', '...']
}

class Tetromino:
    """
    俄罗斯方块核心逻辑类
    负责方块的生成、坐标管理、旋转（矩阵变换）、碰撞检测与落点预测。
    """
    def __init__(self, shape_name=None, config: GameConfig | None = None, start_x: int | None = None):
        """
        初始化方块
        :param shape_name: 可选，指定生成哪种方块('I','J','L','O','S','T','Z')。如果不传则随机生成。
        :param config: 全局配置
        :param start_x: 可选，指定方块出生的 X 坐标。不指定则默认在屏幕正中间。
        """
        # 1. 随机生成逻辑
        if shape_name is None:
            shape_name = random.choice(list(SHAPES.keys()))
        self.shape_name = shape_name
        self.config = config if config is not None else CONFIG
        
        # 2. 将字符串数组转换为 2D 字符矩阵，方便进行矩阵旋转运算
        # 例如 'T' 变成: [['.', 'X', '.'], ['X', 'X', 'X'], ['.', '.', '.']]
        self.matrix = [list(row) for row in SHAPES[shape_name]]
        
        # 3. 初始坐标：出生在网格顶部的中间或指定位置
        if start_x is not None:
            self.x = start_x
        else:
            self.x = self.config.grid_cols // 2 - len(self.matrix[0]) // 2
        self.y = 0

    def get_rotated_matrix(self):
        """
        获取旋转后的矩阵（顺时针 90 度），但不直接修改当前方块，用于预判碰撞。
        核心数学逻辑：顺时针旋转 = 矩阵转置 (Transpose) + 每一行水平反转 (Reverse)
        :return: 旋转后的 2D 列表
        """
        # 第一步：矩阵转置 (行变列，列变行)
        # 使用 zip(*self.matrix) 解包并重新打包
        transposed = [list(row) for row in zip(*self.matrix)]
        
        # 第二步：水平反转每一行
        rotated = [row[::-1] for row in transposed]
        return rotated

    def rotate(self, grid):
        """
        执行旋转操作。如果旋转后发生碰撞，则取消旋转（保持原样）。
        :param grid: 当前游戏网格（2D列表），用于碰撞检测
        :return: 是否旋转成功 (bool)
        """
        # O型方块（正方形）不需要旋转
        if self.shape_name == 'O':
            return True
            
        new_matrix = self.get_rotated_matrix()
        
        # 旋转预判：检测旋转后的矩阵在当前位置是否会发生碰撞
        if not self.check_collision(custom_matrix=new_matrix, grid=grid):
            self.matrix = new_matrix # 无碰撞，确认旋转
            return True
        return False # 发生碰撞，放弃旋转

    def check_collision(self, dx=0, dy=0, custom_matrix=None, grid=None):
        """
        核心碰撞检测逻辑：检测方块移动或旋转后，是否触碰边界或已有方块。
        :param dx: X 轴预判偏移量 (左右移动)
        :param dy: Y 轴预判偏移量 (向下掉落)
        :param custom_matrix: 用于预判旋转后的矩阵，如果不传则使用当前矩阵
        :param grid: 当前游戏网格（必须提供，0表示空，非0表示有方块）
        :return: True 表示发生碰撞，False 表示安全
        """
        if grid is None:
            raise ValueError("必须传入 grid 才能进行碰撞检测！")

        matrix_to_check = custom_matrix if custom_matrix else self.matrix

        # 遍历方块矩阵的每一个小单元
        for row_idx, row in enumerate(matrix_to_check):
            for col_idx, cell in enumerate(row):
                if cell == 'X': # 只检测实心部分
                    # 计算该单元格在游戏网格中的绝对坐标
                    board_x = self.x + col_idx + dx
                    board_y = self.y + row_idx + dy

                    # 边界条件 1：左右撞墙
                    if board_x < 0 or board_x >= self.config.grid_cols:
                        return True
                    
                    # 边界条件 2：触底
                    if board_y >= self.config.grid_rows:
                        return True
                    
                    # 边界条件 3：与网格中已固定的方块重叠
                    # 注意：方块刚出生时 y 可能为负数（在屏幕上方），所以要确保 board_y >= 0 才检测网格
                    if board_y >= 0 and grid[board_y][board_x] != 0:
                        return True
                        
        return False

    def get_drop_position(self, grid):
        """
        落点预测（幻影方块 Ghost Piece）：计算当前方块如果直接掉落，最终会停在哪一行。
        :param grid: 当前游戏网格
        :return: 最终预测的 Y 坐标
        """
        drop_y = self.y
        # 模拟不断向下移动，直到发生碰撞
        while not self.check_collision(dx=0, dy=(drop_y - self.y + 1), grid=grid):
            drop_y += 1
            
        return drop_y
    


 
