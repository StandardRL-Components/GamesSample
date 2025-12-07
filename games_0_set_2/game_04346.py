
# Generated: 2025-08-28T02:07:28.991008
# Source Brief: brief_04346.md
# Brief Index: 4346

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A grid-based puzzle game where the player rearranges colored blocks to match a target pattern.

    The player selects a block using the 'shift' and 'space' actions and moves it with the
    directional actions. The goal is to match the target pattern within a limited number of moves.
    The game provides rewards for placing blocks correctly and a large bonus for solving the puzzle,
    while penalizing each move and running out of moves.

    The visual style is clean and minimalist, with bright, high-contrast elements for clarity
    and satisfying particle effects for player feedback.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the selected block. Press Space to cycle selection forward, "
        "and Shift to cycle backward. Match the target pattern on the right."
    )

    game_description = (
        "A block-sliding puzzle game. Rearrange the blocks on the main grid to perfectly match "
        "the target pattern shown on the right side. Plan your moves carefully, as you only have a "
        "limited number to solve each puzzle!"
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_size = (640, 400)
        self.screen = pygame.Surface(self.screen_size)
        self.clock = pygame.time.Clock()
        
        # --- Visuals & Layout ---
        self.GRID_AREA_WIDTH = 400
        self.UI_AREA_X = self.GRID_AREA_WIDTH + 1
        
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GRID = (50, 60, 70)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_TEXT_ACCENT = (100, 200, 255)
        self.COLOR_WIN = (100, 255, 150)
        self.COLOR_LOSE = (255, 100, 100)
        self.BLOCK_COLORS = [
            (255, 80, 80),   # Red
            (80, 150, 255),  # Blue
            (100, 255, 100), # Green
            (255, 200, 80),  # Yellow
            (200, 100, 255), # Purple
            (80, 220, 220),  # Cyan
            (255, 150, 80),  # Orange
        ]
        
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_title = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_msg = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- Game Constants ---
        self.MAX_MOVES = 50
        self.MAX_EPISODE_STEPS = 500 # Safety break
        
        # --- Game State (initialized in reset) ---
        self.level = 0
        self.win_streak = 0
        self.grid_size = 0
        self.cell_size = 0
        self.grid_offset_x = 0
        self.grid_offset_y = 0
        self.blocks = []
        self.target_map = {}
        self.target_blocks = []
        self.selected_block_idx = 0
        self.moves_left = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win_state = False
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if hasattr(self, 'win_state') and self.win_state:
            self.win_streak += 1
        else:
            self.win_streak = 0
        
        self.level = self.win_streak
        
        self.grid_size = 4 if self.level >= 3 else 3
        num_blocks = min(self.grid_size * self.grid_size - 1, 3 + self.level)
        
        self.cell_size = self.GRID_AREA_WIDTH // (self.grid_size + 1)
        grid_pixel_size = self.cell_size * self.grid_size
        self.grid_offset_x = (self.GRID_AREA_WIDTH - grid_pixel_size) // 2
        self.grid_offset_y = (self.screen_size[1] - grid_pixel_size) // 2

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.moves_left = self.MAX_MOVES
        self.selected_block_idx = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []

        self._generate_puzzle(num_blocks)
        
        return self._get_observation(), self._get_info()

    def _generate_puzzle(self, num_blocks):
        all_positions = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
        
        # Generate target pattern
        target_indices = self.np_random.choice(len(all_positions), size=num_blocks, replace=False)
        target_positions = [all_positions[i] for i in target_indices]
        
        color_indices = self.np_random.choice(len(self.BLOCK_COLORS), size=num_blocks, replace=False)
        chosen_colors = [self.BLOCK_COLORS[i] for i in color_indices]

        self.target_blocks = [{'pos': pos, 'color': color} for pos, color in zip(target_positions, chosen_colors)]
        self.target_map = {b['pos']: b['color'] for b in self.target_blocks}
        
        # Create initial state by shuffling from solved state
        self.blocks = [b.copy() for b in self.target_blocks]
        shuffle_moves = 5 + self.level * 2
        for _ in range(shuffle_moves):
            block_idx = self.np_random.integers(0, len(self.blocks))
            move = self.np_random.integers(1, 5) # 1-4 for up/down/left/right
            
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][move - 1]
            old_pos = self.blocks[block_idx]['pos']
            new_pos = (old_pos[0] + dx, old_pos[1] + dy)
            
            if self._is_valid_move(new_pos):
                self.blocks[block_idx]['pos'] = new_pos

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = False
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        
        # 1. Handle selection change (on press)
        if space_held and not self.last_space_held:
            self.selected_block_idx = (self.selected_block_idx + 1) % len(self.blocks)
        if shift_held and not self.last_shift_held:
            self.selected_block_idx = (self.selected_block_idx - 1 + len(self.blocks)) % len(self.blocks)
        
        self.last_space_held, self.last_shift_held = space_held, shift_held

        # 2. Handle movement
        if movement > 0:
            block = self.blocks[self.selected_block_idx]
            old_pos = block['pos']
            
            was_correct = self.target_map.get(old_pos) == block['color']
            
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][movement - 1]
            new_pos = (old_pos[0] + dx, old_pos[1] + dy)

            if self._is_valid_move(new_pos):
                # Valid move
                block['pos'] = new_pos
                self.moves_left -= 1
                reward -= 0.1 # Cost per move
                
                is_correct = self.target_map.get(new_pos) == block['color']
                if not was_correct and is_correct:
                    reward += 1.0 # Reward for placing a block correctly
                elif was_correct and not is_correct:
                    reward -= 1.0 # Penalty for moving a correct block
                
                # sfx: move_block.wav
                self._create_particles(old_pos, block['color'])

        # 3. Update game logic and check for termination
        self._update_particles()
        
        if self._check_win_condition():
            reward += 100.0
            terminated = True
            self.game_over = True
            self.win_state = True
            # sfx: puzzle_solved.wav
        elif self.moves_left <= 0 or self.steps >= self.MAX_EPISODE_STEPS:
            reward -= 10.0
            terminated = True
            self.game_over = True
            self.win_state = False
            # sfx: puzzle_failed.wav

        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _is_valid_move(self, pos):
        x, y = pos
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return False # Out of bounds
        
        occupied_positions = {b['pos'] for b in self.blocks}
        if pos in occupied_positions:
            return False # Another block is there
        
        return True

    def _check_win_condition(self):
        if len(self.blocks) != len(self.target_map):
            return False
        for block in self.blocks:
            if self.target_map.get(block['pos']) != block['color']:
                return False
        return True

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw main grid
        self._draw_grid(self.grid_offset_x, self.grid_offset_y, self.grid_size, self.cell_size)
        
        # Draw target pattern grid
        target_cell_size = (self.screen_size[0] - self.UI_AREA_X - 40) // self.grid_size
        target_grid_size = target_cell_size * self.grid_size
        target_offset_x = self.UI_AREA_X + (self.screen_size[0] - self.UI_AREA_X - target_grid_size) // 2
        target_offset_y = 120
        self._draw_grid(target_offset_x, target_offset_y, self.grid_size, target_cell_size)

        # Draw target blocks
        for block in self.target_blocks:
            self._draw_block(block, target_offset_x, target_offset_y, target_cell_size)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['size']), color)
            
        # Draw main blocks
        for i, block in enumerate(self.blocks):
            is_selected = (i == self.selected_block_idx) and not self.game_over
            self._draw_block(block, self.grid_offset_x, self.grid_offset_y, self.cell_size, is_selected)

    def _draw_grid(self, ox, oy, grid_size, cell_size):
        grid_pixel_size = grid_size * cell_size
        for i in range(grid_size + 1):
            # Vertical lines
            start_pos = (ox + i * cell_size, oy)
            end_pos = (ox + i * cell_size, oy + grid_pixel_size)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
            # Horizontal lines
            start_pos = (ox, oy + i * cell_size)
            end_pos = (ox + grid_pixel_size, oy + i * cell_size)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

    def _draw_block(self, block, ox, oy, cell_size, is_selected=False):
        grid_x, grid_y = block['pos']
        color = block['color']
        
        px = ox + grid_x * cell_size
        py = oy + grid_y * cell_size
        
        inset = cell_size // 10
        rect = pygame.Rect(px + inset, py + inset, cell_size - 2 * inset, cell_size - 2 * inset)
        
        pygame.draw.rect(self.screen, color, rect, border_radius=cell_size // 8)
        
        if is_selected:
            # Pulsating glow effect
            pulse = abs(math.sin(self.steps * 0.2))
            glow_size = int(cell_size * (0.6 + pulse * 0.15))
            glow_alpha = int(80 + pulse * 40)
            
            glow_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (255, 255, 255, glow_alpha), (glow_size, glow_size), glow_size)
            
            center_x = px + cell_size // 2
            center_y = py + cell_size // 2
            self.screen.blit(glow_surf, (center_x - glow_size, center_y - glow_size), special_flags=pygame.BLEND_RGBA_ADD)
            
            # Bright border
            pygame.draw.rect(self.screen, (255, 255, 255), rect, 2, border_radius=cell_size // 8)

    def _render_ui(self):
        # UI Panel background
        ui_rect = pygame.Rect(self.UI_AREA_X, 0, self.screen_size[0] - self.UI_AREA_X, self.screen_size[1])
        pygame.draw.rect(self.screen, (30, 35, 40), ui_rect)
        pygame.draw.line(self.screen, self.COLOR_GRID, (self.UI_AREA_X, 0), (self.UI_AREA_X, self.screen_size[1]), 2)
        
        # Moves Left
        moves_text = self.font_main.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.UI_AREA_X + 20, 20))
        
        # Score
        score_text = self.font_main.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT_ACCENT)
        self.screen.blit(score_text, (self.UI_AREA_X + 20, 50))

        # Target Pattern Title
        target_title = self.font_title.render("TARGET PATTERN", True, self.COLOR_TEXT)
        title_rect = target_title.get_rect(centerx=(self.UI_AREA_X + (self.screen_size[0] - self.UI_AREA_X)/2), centery=100)
        self.screen.blit(target_title, title_rect)
        
        # Win/Loss Message
        if self.game_over:
            msg_text, msg_color = ("SUCCESS!", self.COLOR_WIN) if self.win_state else ("FAILED!", self.COLOR_LOSE)
            msg_surf = self.font_msg.render(msg_text, True, msg_color)
            msg_rect = msg_surf.get_rect(center=(self.GRID_AREA_WIDTH / 2, self.screen_size[1] / 2))
            
            # Draw a semi-transparent background for the message
            overlay = pygame.Surface(self.screen_size, pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            self.screen.blit(msg_surf, msg_rect)

    def _create_particles(self, grid_pos, color):
        center_x = self.grid_offset_x + grid_pos[0] * self.cell_size + self.cell_size // 2
        center_y = self.grid_offset_y + grid_pos[1] * self.cell_size + self.cell_size // 2
        
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'pos': [center_x, center_y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': random.uniform(15, 30),
                'max_life': 30,
                'color': color,
                'size': random.uniform(2, 5)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95 # friction
            p['vel'][1] *= 0.95
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "level": self.level,
            "win": self.win_state,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
    # To run and play the game
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode(env.screen_size)
    pygame.display.set_caption("Block Puzzle Environment")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    print("\n" + "="*30)
    print(env.game_description)
    print("="*30)
    print(env.user_guide)
    print("="*30 + "\n")

    while not done:
        # --- Action mapping from keyboard ---
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Event handling ---
        action_taken = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_SPACE, pygame.K_LSHIFT, pygame.K_RSHIFT]:
                    action_taken = True
                if event.key == pygame.K_r:
                    print("Resetting environment...")
                    obs, info = env.reset()
                    action_taken = False # Don't step on reset frame

        # Since auto_advance is False, we only step when an action is taken
        if action_taken:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step: {info['steps']}, Moves Left: {info['moves_left']}, Reward: {reward:.2f}, Score: {info['score']:.2f}")
            if terminated:
                print(f"Game Over! Final Score: {info['score']:.2f}. You {'WON' if info['win'] else 'LOST'}.")
                print("Press 'R' to play again or close the window.")
        
        # --- Rendering ---
        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit FPS

    env.close()