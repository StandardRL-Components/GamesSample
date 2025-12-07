
# Generated: 2025-08-28T05:13:29.378851
# Source Brief: brief_02503.md
# Brief Index: 2503

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
from dataclasses import dataclass
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper dataclass for blocks
@dataclass
class Block:
    id: int
    pos: tuple[int, int]
    target_pos: tuple[int, int]
    color: tuple[int, int, int]
    is_on_target: bool = False

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing strings
    user_guide = (
        "Controls: Arrow keys to push the selected block (highlighted). "
        "Use Space/Shift to select the next/previous block. "
        "Get all blocks to their colored targets!"
    )
    game_description = (
        "An isometric puzzle game. Push blocks onto their targets in as few moves as possible. "
        "Pushing one block can cause a chain reaction!"
    )

    # Game settings
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Screen dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # Colors
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_GRID = (60, 65, 80)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_SELECT_GLOW = (255, 255, 100)
        self.BLOCK_COLORS = [
            (255, 100, 100), (100, 255, 100), (100, 100, 255),
            (255, 255, 100), (100, 255, 255), (255, 100, 255)
        ]

        # Isometric grid parameters
        self.TILE_WIDTH_HALF = 32
        self.TILE_HEIGHT_HALF = 16
        self.BLOCK_HEIGHT = 25
        self.grid_origin_x = self.SCREEN_WIDTH // 2
        self.grid_origin_y = 100
        
        # Action mapping to grid vectors
        self.ACTION_TO_VEC = {
            1: (-1, 0),  # Up-Left
            2: (1, 0),   # Down-Right
            3: (0, 1),   # Down-Left
            4: (0, -1),  # Up-Right
        }
        
        # State variables will be initialized in reset()
        self.level = 1
        self.max_levels = 3
        self.grid_size = 0
        self.moves_remaining = 0
        self.blocks = []
        self.block_map = {}
        self.selected_block_idx = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None

        # This will be called once to set the initial state
        self.reset()
        
    def _setup_level(self):
        self.blocks.clear()
        
        if self.level == 1:
            self.grid_size = 6
            self.moves_remaining = 20
            block_defs = [((2, 1), (4, 3)), ((3, 4), (1, 2))]
        elif self.level == 2:
            self.grid_size = 8
            self.moves_remaining = 35
            block_defs = [
                ((2, 2), (5, 5)), ((2, 3), (5, 4)),
                ((3, 2), (4, 5)), ((3, 3), (4, 4)),
            ]
        else: # Level 3 and beyond
            self.grid_size = 10
            self.moves_remaining = 50
            block_defs = [
                ((2, 2), (7, 7)), ((2, 3), (7, 6)),
                ((3, 2), (6, 7)), ((3, 3), (6, 6)),
                ((4, 5), (2, 7)), ((5, 4), (7, 2)),
            ]
        
        for i, (start, target) in enumerate(block_defs):
            color = self.BLOCK_COLORS[i % len(self.BLOCK_COLORS)]
            self.blocks.append(Block(id=i, pos=start, target_pos=target, color=color))

        self._update_block_map()
        self.selected_block_idx = 0
        self._check_and_update_all_blocks_on_target()

    def _update_block_map(self):
        self.block_map = {b.pos: b for b in self.blocks}

    def _check_and_update_all_blocks_on_target(self):
        all_on_target = True
        for block in self.blocks:
            on_target = block.pos == block.target_pos
            block.is_on_target = on_target
            if not on_target:
                all_on_target = False
        return all_on_target

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if options and 'level' in options:
            self.level = options['level']
        else:
            self.level = 1

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self._setup_level()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1

        # Action priority: Push > Select
        if movement != 0:
            self.moves_remaining -= 1
            pre_move_positions = {b.id: b.pos for b in self.blocks}
            
            selected_block = self.blocks[self.selected_block_idx]
            if not selected_block.is_on_target:
                push_succeeded = self._push_block_chain(self.selected_block_idx, movement)
                if push_succeeded:
                    reward += self._calculate_move_reward(pre_move_positions)
            
        elif space_pressed or shift_pressed:
            movable_indices = [i for i, b in enumerate(self.blocks) if not b.is_on_target]
            if movable_indices:
                try:
                    current_selection_in_movable = movable_indices.index(self.selected_block_idx)
                except ValueError:
                    current_selection_in_movable = 0
                
                if space_pressed and not shift_pressed:
                    next_idx = (current_selection_in_movable + 1) % len(movable_indices)
                else: # shift_pressed or both
                    next_idx = (current_selection_in_movable - 1 + len(movable_indices)) % len(movable_indices)
                
                self.selected_block_idx = movable_indices[next_idx]
        
        term_reward, terminated = self._check_termination()
        reward += term_reward
        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _push_block_chain(self, start_idx, movement):
        direction_vec = self.ACTION_TO_VEC.get(movement)
        if not direction_vec:
            return False

        chain = []
        q = [self.blocks[start_idx]]
        visited_ids = {start_idx}
        
        while q:
            current_block = q.pop(0)
            chain.append(current_block)
            
            next_pos = (current_block.pos[0] + direction_vec[0], current_block.pos[1] + direction_vec[1])
            
            if not (0 <= next_pos[0] < self.grid_size and 0 <= next_pos[1] < self.grid_size):
                return False # Chain hits outer wall
            
            if next_pos in self.block_map:
                next_block = self.block_map[next_pos]
                if next_block.id in visited_ids: continue
                
                if next_block.is_on_target:
                    return False # Chain is blocked by a locked block
                
                q.append(next_block)
                visited_ids.add(next_block.id)
        
        for block in reversed(chain):
            block.pos = (block.pos[0] + direction_vec[0], block.pos[1] + direction_vec[1])
        
        self._update_block_map()
        return True

    def _calculate_move_reward(self, pre_move_positions):
        reward = 0
        for block in self.blocks:
            if block.pos != pre_move_positions[block.id]:
                dist_before = self._manhattan_distance(pre_move_positions[block.id], block.target_pos)
                dist_after = self._manhattan_distance(block.pos, block.target_pos)

                if dist_after < dist_before:
                    reward += 1
                elif dist_after > dist_before:
                    reward -= 1

                was_on_target = pre_move_positions[block.id] == block.target_pos
                is_now_on_target = block.pos == block.target_pos
                if is_now_on_target and not was_on_target:
                    reward += 5
                    block.is_on_target = True
        return reward
    
    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _check_termination(self):
        if self.game_over:
            return 0, True

        all_on_target = self._check_and_update_all_blocks_on_target()

        if all_on_target:
            self.game_over = True
            if self.level < self.max_levels:
                self.level += 1
            return 100, True
        
        if self.moves_remaining <= 0:
            self.game_over = True
            return -50, True
            
        if self.steps >= 1000:
            self.game_over = True
            return 0, True

        return 0, False

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "moves_remaining": self.moves_remaining,
        }

    def _iso_to_screen(self, x, y):
        screen_x = self.grid_origin_x + (x - y) * self.TILE_WIDTH_HALF
        screen_y = self.grid_origin_y + (x + y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _draw_iso_poly(self, surface, x, y, color, height, outline_color=None):
        top_points = [
            self._iso_to_screen(x, y), self._iso_to_screen(x + 1, y),
            self._iso_to_screen(x + 1, y + 1), self._iso_to_screen(x, y + 1)
        ]
        
        c_dark = tuple(max(0, c - 50) for c in color)
        c_darker = tuple(max(0, c - 80) for c in color)

        bottom_right_points = [top_points[1], (top_points[1][0], top_points[1][1] + height), (top_points[2][0], top_points[2][1] + height), top_points[2]]
        pygame.draw.polygon(surface, c_dark, bottom_right_points)
        bottom_left_points = [top_points[3], (top_points[3][0], top_points[3][1] + height), (top_points[2][0], top_points[2][1] + height), top_points[2]]
        pygame.draw.polygon(surface, c_darker, bottom_left_points)
        
        pygame.draw.polygon(surface, color, top_points)

        if outline_color:
            pygame.draw.aalines(surface, outline_color, True, top_points, 1)

    def _render_game(self):
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                points = [
                    self._iso_to_screen(x, y), self._iso_to_screen(x + 1, y),
                    self._iso_to_screen(x + 1, y + 1), self._iso_to_screen(x, y + 1)
                ]
                pygame.draw.aalines(self.screen, self.COLOR_GRID, True, points, 1)

        for block in self.blocks:
            points = [
                self._iso_to_screen(block.target_pos[0], block.target_pos[1]),
                self._iso_to_screen(block.target_pos[0] + 1, block.target_pos[1]),
                self._iso_to_screen(block.target_pos[0] + 1, block.target_pos[1] + 1),
                self._iso_to_screen(block.target_pos[0], block.target_pos[1] + 1)
            ]
            pygame.draw.polygon(self.screen, block.color, points, 4)

        sorted_blocks = sorted(self.blocks, key=lambda b: b.pos[0] + b.pos[1])
        for block in sorted_blocks:
            outline = self.COLOR_SELECT_GLOW if block.id == self.selected_block_idx and not self.game_over else None
            self._draw_iso_poly(self.screen, block.pos[0], block.pos[1], block.color, self.BLOCK_HEIGHT, outline)
            
            if block.is_on_target:
                check_center = self._iso_to_screen(block.pos[0] + 0.5, block.pos[1] + 0.5)
                check_center = (check_center[0], check_center[1] - self.BLOCK_HEIGHT // 2)
                p1 = (check_center[0] - 5, check_center[1])
                p2 = (check_center[0] - 1, check_center[1] + 4)
                p3 = (check_center[0] + 5, check_center[1] - 4)
                pygame.draw.line(self.screen, (255,255,255), p1, p2, 3)
                pygame.draw.line(self.screen, (255,255,255), p2, p3, 3)

    def _render_ui(self):
        level_text = self.font_small.render(f"Level: {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (10, 10))

        moves_text = self.font_large.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.SCREEN_WIDTH - moves_text.get_width() - 10, 10))

        if self.game_over:
            all_on_target = all(b.is_on_target for b in self.blocks)
            end_text_str = "LEVEL COMPLETE!" if all_on_target else "OUT OF MOVES"
            end_color = (100, 255, 100) if all_on_target else (255, 100, 100)
            
            end_text = self.font_large.render(end_text_str, True, end_color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 50))
            
            s = pygame.Surface((text_rect.width + 20, text_rect.height + 20), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (text_rect.x-10, text_rect.y-10))
            self.screen.blit(end_text, text_rect)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def validate_implementation(self):
        print("Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv()
    env.validate_implementation()
    
    obs, info = env.reset()
    done = False
    
    real_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Isometric Block Pusher")
    
    running = True
    while running:
        action = [0, 0, 0]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_SPACE: action[1] = 1
                elif event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]: action[2] = 1
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    done = False
                    continue
        
        if any(a != 0 for a in action) and not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Done: {done}")

        draw_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(draw_surface, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30)
        
    pygame.quit()