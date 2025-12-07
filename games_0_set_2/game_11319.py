import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:00:20.666318
# Source Brief: brief_01319.md
# Brief Index: 1319
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A match-3 puzzle game Gymnasium environment.

    The player controls a cursor on a 3x3 grid of colored blocks.
    The goal is to swap adjacent blocks to create horizontal or vertical
    lines of three or more matching colors. Successful matches award points,
    and cleared blocks are replaced by new ones falling from above,
    potentially creating cascading combos.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right) for the cursor.
    - actions[1]: Space button (0=released, 1=held) to select/swap blocks.
    - actions[2]: Shift button (0=released, 1=held) to deselect a block.

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game state.

    Reward Structure:
    - +1 for a 3-block match.
    - +5 for each additional block in a match beyond 3.
    - +100 for reaching the win score of 500.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A classic match-3 puzzle game. Swap adjacent blocks to create lines of three or more matching colors to score points."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to select or swap a block, and press shift to deselect."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 3
    CELL_SIZE = 100
    GRID_LINE_WIDTH = 4
    BLOCK_PADDING = 10
    BLOCK_BORDER_RADIUS = 15
    CURSOR_WIDTH = 6

    # Animation Speeds
    SWAP_SPEED = 0.25  # seconds
    FALL_SPEED = 0.2  # seconds
    FLASH_SPEED = 0.3  # seconds
    FPS = 30

    # Colors
    COLOR_BG = (25, 28, 36)
    COLOR_GRID = (60, 65, 80)
    COLOR_TEXT = (230, 230, 230)
    COLOR_SCORE_PANEL = (40, 45, 60, 180)
    CURSOR_COLOR = (255, 200, 0)
    BLOCK_COLORS = {
        1: (224, 85, 85),   # Red
        2: (85, 190, 120),  # Green
        3: (85, 130, 224),  # Blue
    }
    BLOCK_SHADOW_COLORS = {
        1: (180, 60, 60),
        2: (60, 150, 90),
        3: (60, 95, 180),
    }

    # Game Rules
    WIN_SCORE = 500
    MAX_STEPS = 1000
    NUM_COLORS = 3

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
        self.font_main = pygame.font.Font(None, 36)
        self.font_game_over = pygame.font.Font(None, 72)

        self.grid_origin_x = (self.SCREEN_WIDTH - self.GRID_SIZE * self.CELL_SIZE) // 2
        self.grid_origin_y = (self.SCREEN_HEIGHT - self.GRID_SIZE * self.CELL_SIZE) // 2

        self.grid = None
        self.cursor_pos = None
        self.selected_pos = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.animations = []
        self.game_phase = 'player_turn'
        self.space_was_held = False
        self.shift_was_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.animations = []
        self.game_phase = 'player_turn'
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_pos = None
        self.space_was_held = True # Prevent action on first frame
        self.shift_was_held = True

        self._fill_board()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.space_was_held
        shift_press = shift_held and not self.shift_was_held
        self.space_was_held = space_held
        self.shift_was_held = shift_held

        reward = 0
        
        self._update_animations()

        if self.game_phase == 'player_turn':
            if not self.game_over:
                swap_initiated = self._handle_player_input(movement, space_press, shift_press)
                if swap_initiated:
                    self.game_phase = 'swapping'
                    # sfx: block_swap_start.wav

        elif self.game_phase == 'swapping':
            if not self._is_animation_running('swap'):
                self.game_phase = 'checking_matches'

        elif self.game_phase == 'checking_matches':
            lines = self._find_match_lines()
            if lines:
                matched_coords = set()
                wave_reward = 0
                for line in lines:
                    wave_reward += 1 + max(0, len(line) - 3) * 5
                    for coord in line:
                        matched_coords.add(coord)
                
                reward += wave_reward
                self.score += wave_reward
                # sfx: match_success.wav
                self._initiate_clear_animation(matched_coords)
                self._clear_blocks_from_grid(matched_coords)
                self.game_phase = 'clearing'
            else:
                # If a swap resulted in no match, the turn is over.
                if self._was_a_swap_just_made():
                    self.game_phase = 'player_turn'
                else: # This branch handles cascades
                    self._initiate_fall_animation()
                    if self.animations: # Only fall if there are gaps
                        self.game_phase = 'falling'
                    else:
                        self.game_phase = 'player_turn'
                
        elif self.game_phase == 'clearing':
            if not self._is_animation_running('clear'):
                self._initiate_fall_animation()
                # sfx: blocks_fall.wav
                self.game_phase = 'falling'
                
        elif self.game_phase == 'falling':
            if not self._is_animation_running('fall'):
                self.game_phase = 'checking_matches' # Cascade

        self.steps += 1
        terminated = (self.score >= self.WIN_SCORE) or (self.steps >= self.MAX_STEPS)
        truncated = False
        
        if terminated and not self.game_over:
            if self.score >= self.WIN_SCORE:
                reward += 100
                # sfx: game_win.wav
            else:
                # sfx: game_over.wav
                pass
            self.game_over = True
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    # --- Game Logic Helpers ---

    def _fill_board(self):
        while True:
            self.grid = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_SIZE, self.GRID_SIZE))
            if not self._find_match_lines():
                break
    
    def _handle_player_input(self, movement, space_press, shift_press):
        if movement == 1 and self.cursor_pos[0] > 0: self.cursor_pos[0] -= 1
        elif movement == 2 and self.cursor_pos[0] < self.GRID_SIZE - 1: self.cursor_pos[0] += 1
        elif movement == 3 and self.cursor_pos[1] > 0: self.cursor_pos[1] -= 1
        elif movement == 4 and self.cursor_pos[1] < self.GRID_SIZE - 1: self.cursor_pos[1] += 1
        
        if shift_press and self.selected_pos is not None:
            self.selected_pos = None
            # sfx: deselect.wav
            return False

        if space_press:
            if self.selected_pos is None:
                self.selected_pos = list(self.cursor_pos)
                # sfx: select.wav
            else:
                dist = abs(self.selected_pos[0] - self.cursor_pos[0]) + abs(self.selected_pos[1] - self.cursor_pos[1])
                if dist == 1:
                    r1, c1 = self.selected_pos
                    r2, c2 = self.cursor_pos
                    self._initiate_swap_animation((r1, c1), (r2, c2))
                    self.grid[r1, c1], self.grid[r2, c2] = self.grid[r2, c2], self.grid[r1, c1]
                    self.selected_pos = None
                    return True
                elif self.selected_pos == list(self.cursor_pos):
                    self.selected_pos = None
                    # sfx: deselect.wav
                else:
                    self.selected_pos = list(self.cursor_pos)
                    # sfx: select_fail.wav
        return False

    def _find_match_lines(self):
        lines = []
        matched_coords = set()

        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE - 2):
                if self.grid[r, c] != 0 and self.grid[r, c] == self.grid[r, c+1] == self.grid[r, c+2]:
                    line = [(r, c), (r, c+1), (r, c+2)]
                    if not any(coord in matched_coords for coord in line):
                        lines.append(line)
                        matched_coords.update(line)
        
        for c in range(self.GRID_SIZE):
            for r in range(self.GRID_SIZE - 2):
                if self.grid[r, c] != 0 and self.grid[r, c] == self.grid[r+1, c] == self.grid[r+2, c]:
                    line = [(r, c), (r+1, c), (r+2, c)]
                    if not any(coord in matched_coords for coord in line):
                        lines.append(line)
                        matched_coords.update(line)
        return lines

    def _clear_blocks_from_grid(self, coords):
        for r, c in coords:
            self.grid[r, c] = 0

    def _initiate_fall_animation(self):
        for c in range(self.GRID_SIZE):
            empty_row = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] != 0:
                    if r != empty_row:
                        self.animations.append({
                            'type': 'fall', 'progress': 0, 'duration': self.FALL_SPEED * self.FPS,
                            'start_pos': (r, c), 'end_pos': (empty_row, c), 'color_idx': self.grid[r, c]
                        })
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = 0
                    empty_row -= 1
        
        for c in range(self.GRID_SIZE):
            new_blocks_in_col = 0
            for r in range(self.GRID_SIZE):
                if self.grid[r, c] == 0:
                    new_blocks_in_col += 1
            
            for i in range(new_blocks_in_col):
                r = new_blocks_in_col - 1 - i
                new_color = self.np_random.integers(1, self.NUM_COLORS + 1)
                self.grid[r, c] = new_color
                self.animations.append({
                    'type': 'fall', 'progress': 0, 'duration': self.FALL_SPEED * self.FPS,
                    'start_pos': (-1 - i, c), 'end_pos': (r, c), 'color_idx': new_color
                })

    def _was_a_swap_just_made(self):
        # A bit of a hack: if there are no other animations, assume a swap just happened.
        # A better way would be a flag, but this works for this state machine.
        return len(self.animations) == 0

    # --- Animation ---

    def _lerp(self, a, b, t):
        return a + (b - a) * t

    def _initiate_swap_animation(self, pos1, pos2):
        r1, c1 = pos1
        r2, c2 = pos2
        self.animations.append({
            'type': 'swap', 'progress': 0, 'duration': self.SWAP_SPEED * self.FPS,
            'pos1': (r1, c1), 'pos2': (r2, c2),
            'color1': self.grid[r2, c2], 'color2': self.grid[r1, c1] # Note: colors are from swapped grid
        })

    def _initiate_clear_animation(self, coords):
        for r, c in coords:
            self.animations.append({
                'type': 'clear', 'progress': 0, 'duration': self.FLASH_SPEED * self.FPS,
                'pos': (r, c), 'color_idx': self.grid[r, c]
            })

    def _update_animations(self):
        if not self.animations: return
        self.animations = [anim for anim in self.animations if anim['progress'] < anim['duration']]
        for anim in self.animations:
            anim['progress'] += 1
    
    def _is_animation_running(self, anim_type):
        return any(anim['type'] == anim_type for anim in self.animations)

    # --- Rendering ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_blocks()
        self._render_animations()
        self._render_cursor()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _grid_to_pixel(self, r, c):
        x = self.grid_origin_x + c * self.CELL_SIZE
        y = self.grid_origin_y + r * self.CELL_SIZE
        return x, y

    def _render_grid(self):
        for i in range(self.GRID_SIZE + 1):
            start_x = self.grid_origin_x + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (start_x, self.grid_origin_y), (start_x, self.grid_origin_y + self.GRID_SIZE * self.CELL_SIZE), self.GRID_LINE_WIDTH)
            start_y = self.grid_origin_y + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.grid_origin_x, start_y), (self.grid_origin_x + self.GRID_SIZE * self.CELL_SIZE, start_y), self.GRID_LINE_WIDTH)

    def _draw_block(self, pos_x, pos_y, color_idx, scale=1.0):
        if color_idx == 0: return
        
        size = (self.CELL_SIZE - self.BLOCK_PADDING * 2) * scale
        shadow_offset = size * 0.08
        
        block_rect = pygame.Rect(
            pos_x + self.BLOCK_PADDING + (self.CELL_SIZE - self.BLOCK_PADDING*2 - size)/2,
            pos_y + self.BLOCK_PADDING + (self.CELL_SIZE - self.BLOCK_PADDING*2 - size)/2,
            size, size
        )
        shadow_rect = block_rect.copy()
        shadow_rect.y += shadow_offset
        
        pygame.draw.rect(self.screen, self.BLOCK_SHADOW_COLORS[color_idx], shadow_rect, border_radius=int(self.BLOCK_BORDER_RADIUS * scale))
        pygame.draw.rect(self.screen, self.BLOCK_COLORS[color_idx], block_rect, border_radius=int(self.BLOCK_BORDER_RADIUS * scale))

    def _render_blocks(self):
        animating_coords = set()
        for anim in self.animations:
            if anim['type'] == 'swap':
                animating_coords.add(anim['pos1'])
                animating_coords.add(anim['pos2'])
            elif anim['type'] in ['clear', 'fall']:
                animating_coords.add(anim['start_pos'] if 'start_pos' in anim else anim['pos'])
        
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if (r, c) not in animating_coords and self.grid[r, c] != 0:
                    px, py = self._grid_to_pixel(r, c)
                    self._draw_block(px, py, self.grid[r, c])

    def _render_animations(self):
        for anim in self.animations:
            t = anim['progress'] / anim['duration']
            
            if anim['type'] == 'swap':
                px1, py1 = self._grid_to_pixel(*anim['pos1'])
                px2, py2 = self._grid_to_pixel(*anim['pos2'])
                self._draw_block(self._lerp(px1, px2, t), self._lerp(py1, py2, t), anim['color1'])
                self._draw_block(self._lerp(px2, px1, t), self._lerp(py2, py1, t), anim['color2'])

            elif anim['type'] == 'clear':
                px, py = self._grid_to_pixel(*anim['pos'])
                scale = self._lerp(1.0, 1.5, t)
                alpha = self._lerp(1.0, 0.0, t)
                
                size = (self.CELL_SIZE - self.BLOCK_PADDING * 2) * scale
                block_rect = pygame.Rect(
                    px + self.BLOCK_PADDING + (self.CELL_SIZE - self.BLOCK_PADDING*2 - size)/2,
                    py + self.BLOCK_PADDING + (self.CELL_SIZE - self.BLOCK_PADDING*2 - size)/2,
                    size, size
                )
                flash_surface = pygame.Surface(block_rect.size, pygame.SRCALPHA)
                flash_surface.fill((255, 255, 255, int(255 * alpha)))
                self.screen.blit(flash_surface, block_rect.topleft)

            elif anim['type'] == 'fall':
                px_start, py_start = self._grid_to_pixel(*anim['start_pos'])
                px_end, py_end = self._grid_to_pixel(*anim['end_pos'])
                self._draw_block(px_end, self._lerp(py_start, py_end, t), anim['color_idx'])

    def _render_cursor(self):
        r, c = self.cursor_pos
        px, py = self._grid_to_pixel(r, c)
        cursor_rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.CURSOR_COLOR, cursor_rect, self.CURSOR_WIDTH, border_radius=5)
        
        if self.selected_pos is not None:
            r_sel, c_sel = self.selected_pos
            px_sel, py_sel = self._grid_to_pixel(r_sel, c_sel)
            
            glow_alpha = 100 + 50 * math.sin(pygame.time.get_ticks() * 0.01)
            glow_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(glow_surface, (*self.CURSOR_COLOR, glow_alpha), (0,0, self.CELL_SIZE, self.CELL_SIZE), 0, border_radius=5)
            self.screen.blit(glow_surface, (px_sel, py_sel))

    def _render_ui(self):
        panel_rect = pygame.Rect(10, 10, 200, 50)
        panel_surf = pygame.Surface(panel_rect.size, pygame.SRCALPHA)
        panel_surf.fill(self.COLOR_SCORE_PANEL)
        self.screen.blit(panel_surf, panel_rect.topleft)
        
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (25, 22))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            text_surface = self.font_game_over.render(msg, True, self.COLOR_TEXT)
            text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cursor_pos": self.cursor_pos,
            "selected_pos": self.selected_pos,
            "game_phase": self.game_phase,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # The original code had a validation function that is not part of the Gym API
    # and a main loop for human play. These are useful for testing but are
    # removed here to keep the environment class clean, as they are not
    # required by the standard Gym interface.
    # To run this environment, you would typically do:
    # env = GameEnv()
    # obs, info = env.reset()
    # ... and then interact with env.step(action) in a loop.
    
    # For demonstration purposes, we can re-add a simple execution block.
    # This part is not evaluated by the tests but helps in running the file directly.
    try:
        env = GameEnv()
        obs, info = env.reset()
        
        # To make it runnable for a human, we need a display
        os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc.
        pygame.display.init()
        human_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Match-3 Gym Environment")

        running = True
        game_clock = pygame.time.Clock()
        
        action = [0, 0, 0] 
        
        print(GameEnv.game_description)
        print(GameEnv.user_guide)

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP: action[0] = 1
                    elif event.key == pygame.K_DOWN: action[0] = 2
                    elif event.key == pygame.K_LEFT: action[0] = 3
                    elif event.key == pygame.K_RIGHT: action[0] = 4
                    elif event.key == pygame.K_SPACE: action[1] = 1
                    elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 1
                    elif event.key == pygame.K_r: 
                        obs, info = env.reset()
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if reward > 0:
                print(f"Step: {info['steps']}, Reward: {reward}, Score: {info['score']}")

            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            human_screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            # Reset action after step
            action = [0, 0, 0] 
            
            if terminated or truncated:
                print(f"Episode finished. Final Score: {info['score']}, Steps: {info['steps']}")
                # Wait for a moment then reset
                pygame.time.wait(2000)
                obs, info = env.reset()
                
            game_clock.tick(env.FPS)
            
        env.close()
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        print("This might be because you are running in a headless environment.")
        print("The __main__ block is for human play and requires a display.")