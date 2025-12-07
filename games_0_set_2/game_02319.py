
# Generated: 2025-08-28T04:26:44.482051
# Source Brief: brief_02319.md
# Brief Index: 2319

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move and push boxes onto the green platforms."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A classic grid-based puzzle game. Push all the boxes onto the target platforms before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    GRID_WIDTH = 16
    GRID_HEIGHT = 10
    CELL_SIZE = 40
    SCREEN_WIDTH = GRID_WIDTH * CELL_SIZE
    SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE

    COLOR_BG = (40, 40, 40)
    COLOR_GRID = (60, 60, 60)
    COLOR_PLAYER = (50, 150, 255)
    COLOR_BOX = (200, 120, 50)
    COLOR_PLATFORM = (80, 180, 80)
    COLOR_BOX_ON_PLATFORM = (150, 220, 100)
    COLOR_TEXT = (240, 240, 240)

    MAX_MOVES = 25
    NUM_BOXES = 6

    LEVELS = [
        [
            "                ",
            "  ..@..         ",
            "  .$ $.$        ",
            "  .$ $.$        ",
            "  .. ..         ",
            "                ",
            "                ",
            "                ",
            "                ",
            "                ",
        ],
        [
            "                ",
            "   @            ",
            "  .$$.          ",
            " ..$$..         ",
            "  .$$.          ",
            "                ",
            "                ",
            "                ",
            "                ",
            "                ",
        ],
        [
            "                ",
            "     . . .      ",
            "     . @ .      ",
            "     .$$.$      ",
            "    $  .  $     ",
            "     . . .      ",
            "                ",
            "                ",
            "                ",
            "                ",
        ]
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 60, bold=True)

        self.player_pos = [0, 0]
        self.boxes_pos = []
        self.platforms_pos = []
        self.current_level_index = -1
        self.last_moved_object = None
        self.last_pos = [0, 0]

        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.remaining_moves = self.MAX_MOVES

        self.last_moved_object = None
        self.last_pos = [0, 0]

        self.current_level_index = (self.current_level_index + 1) % len(self.LEVELS)
        self._load_level(self.current_level_index)

        return self._get_observation(), self._get_info()

    def _load_level(self, level_index):
        self.player_pos = [0, 0]
        self.boxes_pos = []
        self.platforms_pos = []
        level_data = self.LEVELS[level_index]
        
        occupied_pos = set()

        for r, row_str in enumerate(level_data):
            for c, char in enumerate(row_str):
                pos = (c, r)
                if char == '@':
                    self.player_pos = list(pos)
                    occupied_pos.add(pos)
                elif char == '$':
                    self.boxes_pos.append(list(pos))
                    occupied_pos.add(pos)
                elif char == '.':
                    self.platforms_pos.append(list(pos))
                    occupied_pos.add(pos)
                elif char == '*':
                    self.boxes_pos.append(list(pos))
                    self.platforms_pos.append(list(pos))
                    occupied_pos.add(pos)
                elif char == '+':
                    self.player_pos = list(pos)
                    self.platforms_pos.append(list(pos))
                    occupied_pos.add(pos)

        while len(self.boxes_pos) < self.NUM_BOXES:
            pos = self._get_random_empty_pos(occupied_pos)
            self.boxes_pos.append(pos)
            occupied_pos.add(tuple(pos))
        
        while len(self.platforms_pos) < self.NUM_BOXES:
            pos = self._get_random_empty_pos(occupied_pos)
            self.platforms_pos.append(pos)
            occupied_pos.add(tuple(pos))

        self.boxes_pos = self.boxes_pos[:self.NUM_BOXES]
        self.platforms_pos = self.platforms_pos[:self.NUM_BOXES]

    def _get_random_empty_pos(self, occupied_pos):
        while True:
            pos = (
                self.np_random.integers(0, self.GRID_WIDTH),
                self.np_random.integers(0, self.GRID_HEIGHT)
            )
            if pos not in occupied_pos:
                return list(pos)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        self.steps += 1
        reward = 0.0

        self.last_moved_object = None

        if movement != 0:
            self.remaining_moves -= 1
            reward -= 0.1
            
            boxes_on_platform_before = self._count_boxes_on_platforms()

            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(movement, (0, 0))
            
            target_pos = [self.player_pos[0] + dx, self.player_pos[1] + dy]

            if not (0 <= target_pos[0] < self.GRID_WIDTH and 0 <= target_pos[1] < self.GRID_HEIGHT):
                pass # Hit a wall
            else:
                box_idx_to_push = next((i for i, pos in enumerate(self.boxes_pos) if pos == target_pos), -1)
                
                if box_idx_to_push != -1:
                    box_target_pos = [target_pos[0] + dx, target_pos[1] + dy]
                    
                    is_wall = not (0 <= box_target_pos[0] < self.GRID_WIDTH and 0 <= box_target_pos[1] < self.GRID_HEIGHT)
                    is_other_box = any(pos == box_target_pos for pos in self.boxes_pos)
                    
                    if not is_wall and not is_other_box:
                        # sfx_push_box.wav
                        self.last_moved_object = ('box', box_idx_to_push)
                        self.last_pos = self.boxes_pos[box_idx_to_push][:]
                        self.boxes_pos[box_idx_to_push] = box_target_pos
                        self.player_pos = target_pos
                else:
                    # sfx_player_step.wav
                    self.last_moved_object = ('player',)
                    self.last_pos = self.player_pos[:]
                    self.player_pos = target_pos
            
            boxes_on_platform_after = self._count_boxes_on_platforms()
            newly_placed_boxes = boxes_on_platform_after - boxes_on_platform_before
            if newly_placed_boxes > 0:
                # sfx_box_on_platform.wav
                reward += newly_placed_boxes * 1.0

        if self._count_boxes_on_platforms() == self.NUM_BOXES:
            if not self.game_over: # Grant reward only on the frame of winning
                # sfx_win_level.wav
                reward += 50
            self.game_over = True
        elif self.remaining_moves <= 0 and movement != 0:
            if not self.game_over: # Grant penalty only on the frame of losing
                # sfx_lose_level.wav
                reward -= 50
            self.game_over = True
        
        self.score += reward
        terminated = self.game_over or self.steps >= 1000

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _count_boxes_on_platforms(self):
        platform_set = {tuple(p) for p in self.platforms_pos}
        return sum(1 for box_pos in self.boxes_pos if tuple(box_pos) in platform_set)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        transposed_arr = np.transpose(arr, (1, 0, 2))
        return transposed_arr.astype(np.uint8)

    def _render_game(self):
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        for pos in self.platforms_pos:
            rect = pygame.Rect(pos[0] * self.CELL_SIZE, pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, rect)
            pygame.draw.rect(self.screen, self.COLOR_BG, rect.inflate(-8, -8))

        if self.last_moved_object is not None:
            obj_type = self.last_moved_object[0]
            color = self.COLOR_PLAYER if obj_type == 'player' else self.COLOR_BOX
            ghost_surf = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            ghost_color = (*color, 70)
            rect = pygame.Rect(4, 4, self.CELL_SIZE - 8, self.CELL_SIZE - 8)
            pygame.draw.rect(ghost_surf, ghost_color, rect, border_radius=6)
            self.screen.blit(ghost_surf, (self.last_pos[0] * self.CELL_SIZE, self.last_pos[1] * self.CELL_SIZE))

        platform_set = {tuple(p) for p in self.platforms_pos}
        for pos in self.boxes_pos:
            rect = pygame.Rect(pos[0] * self.CELL_SIZE + 4, pos[1] * self.CELL_SIZE + 4, self.CELL_SIZE - 8, self.CELL_SIZE - 8)
            color = self.COLOR_BOX_ON_PLATFORM if tuple(pos) in platform_set else self.COLOR_BOX
            pygame.draw.rect(self.screen, color, rect, border_radius=6)
            pygame.draw.rect(self.screen, tuple(c * 0.8 for c in color), rect, 3, border_radius=6)

        player_rect = pygame.Rect(self.player_pos[0] * self.CELL_SIZE + 6, self.player_pos[1] * self.CELL_SIZE + 6, self.CELL_SIZE - 12, self.CELL_SIZE - 12)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)
        pygame.draw.rect(self.screen, tuple(c * 0.7 for c in self.COLOR_PLAYER), player_rect, 2, border_radius=4)

    def _render_ui(self):
        moves_text = self.font_ui.render(f"Moves: {self.remaining_moves}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (10, 5))
        
        boxes_count = self._count_boxes_on_platforms()
        boxes_text = self.font_ui.render(f"Placed: {boxes_count}/{self.NUM_BOXES}", True, self.COLOR_TEXT)
        text_rect = boxes_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 5))
        self.screen.blit(boxes_text, text_rect)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win = self._count_boxes_on_platforms() == self.NUM_BOXES
            message = "LEVEL CLEAR!" if win else "OUT OF MOVES"
            color = (100, 255, 100) if win else (255, 100, 100)
            game_over_text = self.font_game_over.render(message, True, color)
            text_rect = game_over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(game_over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "remaining_moves": self.remaining_moves,
            "boxes_on_platforms": self._count_boxes_on_platforms(),
            "level": self.current_level_index,
        }

    def close(self):
        pygame.quit()