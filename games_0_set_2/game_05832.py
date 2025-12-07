
# Generated: 2025-08-28T06:14:43.508550
# Source Brief: brief_05832.md
# Brief Index: 5832

        
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
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move the falling crystal, Space to drop it."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a crystal through a grid, matching colors to fill the board and conquer the Crystal Caverns."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 10
    CELL_SIZE = 32
    GRID_WIDTH = GRID_SIZE * CELL_SIZE
    GRID_HEIGHT = GRID_SIZE * CELL_SIZE
    GRID_X = (SCREEN_WIDTH - GRID_WIDTH) // 2
    GRID_Y = (SCREEN_HEIGHT - GRID_HEIGHT) // 2 + 20
    
    MAX_MOVES = 50
    MAX_STEPS = 500
    
    # --- Colors ---
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 80)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_VALUE = (255, 255, 255)
    
    CRYSTAL_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 120, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
    ]
    
    # --- Game States ---
    STATE_PLAYER_CONTROL = 0
    STATE_ANIM_DROP = 1
    STATE_ANIM_MATCH = 2
    STATE_ANIM_GRAVITY = 3

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)
        
        # Initialize state variables
        self.grid = None
        self.falling_crystal = None
        self.score = 0
        self.steps = 0
        self.moves_left = 0
        self.game_over = False
        self.game_won = False
        self.game_state = self.STATE_PLAYER_CONTROL
        self.animation_timer = 0
        self.animation_data = {}
        self.last_space_state = 0
        self.combo_multiplier = 1
        
        self.reset()
        
        # Self-validation check
        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.score = 0
        self.steps = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.game_won = False
        self.game_state = self.STATE_PLAYER_CONTROL
        self.animation_timer = 0
        self.animation_data = {}
        self.last_space_state = 0
        self.combo_multiplier = 1
        
        self._spawn_new_crystal()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.steps += 1
        reward = 0
        terminated = False

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_state
        self.last_space_state = space_held

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- State Machine for Animations and Gameplay ---
        if self.animation_timer > 0:
            self.animation_timer -= 1
            if self.animation_timer == 0:
                reward += self._end_animation_phase()
        else:
            if self.game_state == self.STATE_PLAYER_CONTROL:
                # Handle player input
                if movement == 3: # Left
                    self.falling_crystal['x'] = max(0, self.falling_crystal['x'] - 1)
                elif movement == 4: # Right
                    self.falling_crystal['x'] = min(self.GRID_SIZE - 1, self.falling_crystal['x'] + 1)
                
                if space_pressed:
                    # Sound: Crystal Drop
                    reward += 1 # Base reward for placing a crystal
                    self.moves_left -= 1
                    self._start_drop_animation()
        
        # Check for termination conditions
        if not self.game_over:
            if self.moves_left <= 0:
                self.game_over = True
                reward -= 100
            elif self.steps >= self.MAX_STEPS:
                self.game_over = True
            
            if self.game_won:
                self.game_over = True
                reward += 100

        terminated = self.game_over

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _start_drop_animation(self):
        col = self.falling_crystal['x']
        target_y = self.GRID_SIZE - 1
        while target_y >= 0 and self.grid[target_y][col] != 0:
            target_y -= 1

        if target_y < 0: # Column is full, game over
            self.game_over = True
            return

        self.game_state = self.STATE_ANIM_DROP
        drop_duration = int(max(3, (target_y + 1) * 1.5))
        self.animation_timer = drop_duration
        self.animation_data = {
            'crystal': self.falling_crystal.copy(),
            'start_y_px': self.GRID_Y - self.CELL_SIZE,
            'end_y_px': self.GRID_Y + target_y * self.CELL_SIZE,
            'target_row': target_y,
            'duration': drop_duration
        }
        self.falling_crystal = None

    def _end_animation_phase(self):
        reward = 0
        if self.game_state == self.STATE_ANIM_DROP:
            # Crystal has landed, place it in the grid
            crystal = self.animation_data['crystal']
            row, col = self.animation_data['target_row'], crystal['x']
            self.grid[row][col] = crystal['color_idx']
            self.animation_data = {}
            # Sound: Crystal Land
            
            # Immediately check for matches
            reward += self._check_for_matches()

        elif self.game_state == self.STATE_ANIM_MATCH:
            # Flashing is over, clear the crystals
            for r, c in self.animation_data['matches']:
                self.grid[r][c] = 0 # 0 is empty
            self.animation_data = {}
            # Sound: Match Clear
            
            # Immediately apply gravity
            self._apply_gravity()

        elif self.game_state == self.STATE_ANIM_GRAVITY:
            # Gravity animation is over, update grid state
            new_grid = np.zeros_like(self.grid)
            for fall_info in self.animation_data['falls']:
                new_grid[fall_info['end_r']][fall_info['c']] = fall_info['color_idx']
            
            # Copy non-falling crystals
            for r in range(self.GRID_SIZE):
                for c in range(self.GRID_SIZE):
                    if self.grid[r][c] != 0 and new_grid[r][c] == 0:
                        is_falling = any(d['c'] == c and d['start_r'] == r for d in self.animation_data.get('falls', []))
                        if not is_falling:
                             new_grid[r][c] = self.grid[r][c]

            self.grid = new_grid
            self.animation_data = {}
            # Sound: Crystals Settle
            
            # Check for combo matches
            self.combo_multiplier += 1
            reward += self._check_for_matches()

        return reward

    def _check_for_matches(self):
        matches = self._find_matches()
        if not matches:
            # No matches found, end the turn
            self.combo_multiplier = 1
            if np.all(self.grid != 0): # Win condition
                self.game_won = True
            else:
                self._spawn_new_crystal()
                if self.grid[0][self.falling_crystal['x']] != 0: # Block out
                    self.game_over = True
            return 0
        
        # Matches found, start match animation
        # Sound: Match Found!
        reward = 0
        match_count = len(matches)
        if match_count == 3: reward = 10
        elif match_count == 4: reward = 20
        else: reward = 30
        reward *= self.combo_multiplier
        self.score += reward

        self.game_state = self.STATE_ANIM_MATCH
        self.animation_timer = 15 # frames
        self.animation_data = {'matches': matches, 'duration': 15}
        return reward

    def _find_matches(self):
        to_remove = set()
        # Horizontal
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE - 2):
                color = self.grid[r][c]
                if color != 0 and color == self.grid[r][c+1] == self.grid[r][c+2]:
                    to_remove.add((r, c))
                    to_remove.add((r, c+1))
                    to_remove.add((r, c+2))
                    # Check for longer matches
                    for k in range(c + 3, self.GRID_SIZE):
                        if self.grid[r][k] == color:
                            to_remove.add((r,k))
                        else:
                            break
        # Vertical
        for c in range(self.GRID_SIZE):
            for r in range(self.GRID_SIZE - 2):
                color = self.grid[r][c]
                if color != 0 and color == self.grid[r+1][c] == self.grid[r+2][c]:
                    to_remove.add((r, c))
                    to_remove.add((r+1, c))
                    to_remove.add((r+2, c))
                    # Check for longer matches
                    for k in range(r + 3, self.GRID_SIZE):
                        if self.grid[k][c] == color:
                            to_remove.add((k, c))
                        else:
                            break
        return list(to_remove)

    def _apply_gravity(self):
        falls = []
        grid_copy = self.grid.copy()

        for c in range(self.GRID_SIZE):
            write_row = self.GRID_SIZE - 1
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if grid_copy[r][c] != 0:
                    if r != write_row:
                        falls.append({
                            'start_r': r, 'c': c, 'end_r': write_row, 
                            'color_idx': grid_copy[r][c]
                        })
                    write_row -= 1

        if not falls:
            # No gravity needed, check for combos (which will fail and spawn new crystal)
            self.combo_multiplier += 1
            self._check_for_matches()
            return
        
        # Start gravity animation
        self.game_state = self.STATE_ANIM_GRAVITY
        duration = 10
        self.animation_timer = duration
        self.animation_data = {'falls': falls, 'duration': duration}

    def _spawn_new_crystal(self):
        self.game_state = self.STATE_PLAYER_CONTROL
        self.falling_crystal = {
            'x': self.GRID_SIZE // 2,
            'y': -1,
            'color_idx': self.np_random.integers(1, len(self.CRYSTAL_COLORS) + 1),
            'pulse': 0
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
        }

    def _render_game(self):
        self._render_grid()
        self._render_grid_crystals()
        
        if self.falling_crystal:
            self._render_falling_crystal(self.falling_crystal)
            
        self._render_animations()

        if self.game_over:
            self._render_game_over()

    def _render_grid(self):
        for i in range(self.GRID_SIZE + 1):
            # Vertical lines
            start_pos = (self.GRID_X + i * self.CELL_SIZE, self.GRID_Y)
            end_pos = (self.GRID_X + i * self.CELL_SIZE, self.GRID_Y + self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
            # Horizontal lines
            start_pos = (self.GRID_X, self.GRID_Y + i * self.CELL_SIZE)
            end_pos = (self.GRID_X + self.GRID_WIDTH, self.GRID_Y + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

    def _render_grid_crystals(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color_idx = self.grid[r][c]
                if color_idx != 0:
                    is_matching = self.game_state == self.STATE_ANIM_MATCH and (r, c) in self.animation_data.get('matches', [])
                    is_falling = self.game_state == self.STATE_ANIM_GRAVITY and any(d['c'] == c and d['start_r'] == r for d in self.animation_data.get('falls', []))
                    
                    if not is_matching and not is_falling:
                        self._draw_crystal(c, r, color_idx)

    def _render_falling_crystal(self, crystal):
        crystal['pulse'] = (crystal['pulse'] + 0.1) % (2 * math.pi)
        glow_alpha = int(100 + 60 * math.sin(crystal['pulse']))
        
        cx = int(self.GRID_X + crystal['x'] * self.CELL_SIZE + self.CELL_SIZE / 2)
        cy = int(self.GRID_Y - self.CELL_SIZE / 2)
        color = self.CRYSTAL_COLORS[crystal['color_idx'] - 1]
        
        # Glow
        pygame.gfxdraw.filled_circle(self.screen, cx, cy, self.CELL_SIZE // 2, (*color, glow_alpha))
        # Crystal
        self._draw_crystal_at_pos(cx, cy, crystal['color_idx'])


    def _render_animations(self):
        if self.game_state == self.STATE_ANIM_DROP:
            data = self.animation_data
            progress = 1 - (self.animation_timer / data['duration'])
            
            y_pos = data['start_y_px'] + (data['end_y_px'] - data['start_y_px']) * progress
            cx = int(self.GRID_X + data['crystal']['x'] * self.CELL_SIZE + self.CELL_SIZE / 2)
            cy = int(y_pos + self.CELL_SIZE / 2)
            self._draw_crystal_at_pos(cx, cy, data['crystal']['color_idx'])

        elif self.game_state == self.STATE_ANIM_MATCH:
            data = self.animation_data
            progress = self.animation_timer / data['duration']
            scale = 1.0 + 0.5 * math.sin(progress * math.pi * 4) # Pulsing scale
            alpha = int(255 * progress)
            
            for r, c in data['matches']:
                color_idx = self.grid[r][c]
                self._draw_crystal(c, r, color_idx, scale, alpha)

        elif self.game_state == self.STATE_ANIM_GRAVITY:
            data = self.animation_data
            progress = 1 - (self.animation_timer / data['duration'])
            
            for fall_info in data['falls']:
                start_y = self.GRID_Y + fall_info['start_r'] * self.CELL_SIZE
                end_y = self.GRID_Y + fall_info['end_r'] * self.CELL_SIZE
                
                y_pos = start_y + (end_y - start_y) * progress
                cx = int(self.GRID_X + fall_info['c'] * self.CELL_SIZE + self.CELL_SIZE / 2)
                cy = int(y_pos + self.CELL_SIZE / 2)
                self._draw_crystal_at_pos(cx, cy, fall_info['color_idx'])

    def _draw_crystal(self, c, r, color_idx, scale=1.0, alpha=255):
        cx = int(self.GRID_X + c * self.CELL_SIZE + self.CELL_SIZE / 2)
        cy = int(self.GRID_Y + r * self.CELL_SIZE + self.CELL_SIZE / 2)
        self._draw_crystal_at_pos(cx, cy, color_idx, scale, alpha)

    def _draw_crystal_at_pos(self, cx, cy, color_idx, scale=1.0, alpha=255):
        color = self.CRYSTAL_COLORS[color_idx - 1]
        radius = int(self.CELL_SIZE * 0.4 * scale)
        
        # Main body
        pygame.gfxdraw.filled_circle(self.screen, cx, cy, radius, (*color, alpha))
        pygame.gfxdraw.aacircle(self.screen, cx, cy, radius, (*color, alpha))
        
        # Highlight
        highlight_color = tuple(min(255, val + 60) for val in color)
        pygame.gfxdraw.filled_circle(self.screen, cx - radius // 3, cy - radius // 3, radius // 4, (*highlight_color, alpha))
        pygame.gfxdraw.aacircle(self.screen, cx - radius // 3, cy - radius // 3, radius // 4, (*highlight_color, alpha))

    def _render_ui(self):
        score_text = self.font_ui.render("SCORE", True, self.COLOR_UI_TEXT)
        score_val = self.font_ui.render(f"{self.score}", True, self.COLOR_UI_VALUE)
        moves_text = self.font_ui.render("MOVES", True, self.COLOR_UI_TEXT)
        moves_val = self.font_ui.render(f"{self.moves_left}", True, self.COLOR_UI_VALUE)
        
        self.screen.blit(score_text, (20, 20))
        self.screen.blit(score_val, (20, 45))
        self.screen.blit(moves_text, (self.SCREEN_WIDTH - moves_text.get_width() - 20, 20))
        self.screen.blit(moves_val, (self.SCREEN_WIDTH - moves_val.get_width() - 20, 45))

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        msg = "VICTORY!" if self.game_won else "GAME OVER"
        color = (100, 255, 100) if self.game_won else (255, 100, 100)
        
        text = self.font_game_over.render(msg, True, color)
        text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text, text_rect)

    def validate_implementation(self):
        print("✓ Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Crystal Caverns")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # Game loop
    while running:
        # --- Event Handling ---
        movement = 0 # No-op
        space = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1
            
        action = [movement, space, 0] # Movement, Space, Shift
        
        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is the rendered screen, so we just need to show it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}")
            # Wait for a moment before allowing reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Limit to 30 FPS

    pygame.quit()