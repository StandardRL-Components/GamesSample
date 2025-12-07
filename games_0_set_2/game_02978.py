
# Generated: 2025-08-27T22:00:11.421258
# Source Brief: brief_02978.md
# Brief Index: 2978

        
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

    user_guide = (
        "Controls: ←→ to move the dropper. Press space to drop the next crystal. Align 5 to win."
    )

    game_description = (
        "An isometric puzzle game. Drop crystals into the cavern to form lines of 5 of the same color."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 64)

        # --- Game Constants ---
        self.GRID_WIDTH, self.GRID_HEIGHT = 8, 8
        self.TILE_WIDTH, self.TILE_HEIGHT = 48, 24
        self.ORIGIN_X = 640 // 2
        self.ORIGIN_Y = 120
        self.MAX_MOVES = 25
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (50, 60, 80)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_DROPPER = (255, 255, 255)
        
        self.CRYSTAL_COLORS = {
            1: (255, 50, 50),   # Red
            2: (50, 255, 50),   # Green
            3: (80, 80, 255),   # Blue
            4: (255, 255, 80),  # Yellow
            5: (200, 80, 255),  # Purple
        }
        self.GLOW_COLORS = {k: tuple(min(255, c + 50) for c in v) for k, v in self.CRYSTAL_COLORS.items()}

        # --- State Variables ---
        self.grid = None
        self.dropper_pos = 0
        self.next_crystal = 0
        self.moves_left = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win_state = False
        
        # Visual Effects
        self.particles = []
        self.last_drop_info = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        
        # Procedurally generate a stable starting board
        num_initial_crystals = self.np_random.integers(15, 25)
        for _ in range(num_initial_crystals):
            col = self.np_random.integers(0, self.GRID_WIDTH)
            color = self.np_random.integers(1, len(self.CRYSTAL_COLORS) + 1)
            self._drop_crystal_into_col(col, color, apply_gravity=False)
        self._apply_gravity() # Settle all crystals once

        self.dropper_pos = self.GRID_WIDTH // 2
        self.next_crystal = self.np_random.integers(1, len(self.CRYSTAL_COLORS) + 1)
        self.moves_left = self.MAX_MOVES
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win_state = False
        
        self.particles = []
        self.last_drop_info = None

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Reset transient visual effects
        self.particles.clear()
        self.last_drop_info = None

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        reward = 0
        terminated = False
        
        if shift_held:
            # For RL: allow agent to quit, with a penalty
            self.game_over = True
            terminated = True
            reward = -10.0
            return self._get_observation(), reward, terminated, False, self._get_info()

        # --- Handle Actions ---
        # Movement
        if movement == 3:  # Left
            self.dropper_pos = max(0, self.dropper_pos - 1)
        elif movement == 4: # Right
            self.dropper_pos = min(self.GRID_WIDTH - 1, self.dropper_pos + 1)

        # Drop Action
        if space_held:
            landing_y = self._get_landing_y(self.dropper_pos)
            if landing_y is not None:
                self.moves_left -= 1
                
                # Store drop info for rendering
                start_screen_pos = self._grid_to_iso(self.dropper_pos, -1)
                end_screen_pos = self._grid_to_iso(self.dropper_pos, landing_y)
                self.last_drop_info = {
                    "start": start_screen_pos, 
                    "end": end_screen_pos,
                    "color": self.CRYSTAL_COLORS[self.next_crystal]
                }
                
                self.grid[self.dropper_pos, landing_y] = self.next_crystal
                # sfx: crystal_land.wav
                
                # --- Match and Gravity Loop ---
                chain_bonus = 1.0
                while True:
                    matches = self._find_matches()
                    if not matches:
                        break
                    
                    match_reward, match_score = self._process_matches(matches, chain_bonus)
                    reward += match_reward
                    self.score += match_score
                    
                    self._apply_gravity()
                    # sfx: gravity_shift.wav
                    chain_bonus *= 1.5 # Increase bonus for chain reactions
                
                self.next_crystal = self.np_random.integers(1, len(self.CRYSTAL_COLORS) + 1)
            else:
                # Penalty for trying to drop in a full column
                reward -= 0.1
                # sfx: error.wav
        
        # --- Termination Check ---
        if self.win_state:
            self.game_over = True
            terminated = True
            reward += 100 # Large win bonus
            # sfx: win_jingle.wav
        elif self.moves_left <= 0 or self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True
            # sfx: lose_sound.wav

        return self._get_observation(), float(reward), terminated, False, self._get_info()

    def _grid_to_iso(self, x, y):
        screen_x = self.ORIGIN_X + (x - y) * self.TILE_WIDTH / 2
        screen_y = self.ORIGIN_Y + (x + y) * self.TILE_HEIGHT / 2
        return int(screen_x), int(screen_y)

    def _get_landing_y(self, x):
        for y in range(self.GRID_HEIGHT - 1, -1, -1):
            if self.grid[x, y] == 0:
                return y
        return None

    def _drop_crystal_into_col(self, x, color, apply_gravity=True):
        if apply_gravity:
            y = self._get_landing_y(x)
            if y is not None:
                self.grid[x, y] = color
        else: # Place at bottom
            self.grid[x, self.GRID_HEIGHT - 1] = color

    def _apply_gravity(self):
        for x in range(self.GRID_WIDTH):
            empty_slots = 0
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[x, y] == 0:
                    empty_slots += 1
                elif empty_slots > 0:
                    self.grid[x, y + empty_slots] = self.grid[x, y]
                    self.grid[x, y] = 0

    def _find_matches(self):
        to_clear = set()
        
        # Check for lines of 5 first (win condition)
        for length in [5, 4, 3]:
            # Horizontal
            for y in range(self.GRID_HEIGHT):
                for x in range(self.GRID_WIDTH - length + 1):
                    if self.grid[x, y] != 0 and all(self.grid[x, y] == self.grid[x + i, y] for i in range(1, length)):
                        for i in range(length): to_clear.add((x + i, y))
            # Vertical
            for x in range(self.GRID_WIDTH):
                for y in range(self.GRID_HEIGHT - length + 1):
                    if self.grid[x, y] != 0 and all(self.grid[x, y] == self.grid[x, y + i] for i in range(1, length)):
                        for i in range(length): to_clear.add((x, y + i))
            # Diagonal (down-right)
            for x in range(self.GRID_WIDTH - length + 1):
                for y in range(self.GRID_HEIGHT - length + 1):
                    if self.grid[x, y] != 0 and all(self.grid[x, y] == self.grid[x + i, y + i] for i in range(1, length)):
                        for i in range(length): to_clear.add((x + i, y + i))
            # Diagonal (up-right)
            for x in range(self.GRID_WIDTH - length + 1):
                for y in range(length - 1, self.GRID_HEIGHT):
                    if self.grid[x, y] != 0 and all(self.grid[x, y] == self.grid[x + i, y - i] for i in range(1, length)):
                        for i in range(length): to_clear.add((x + i, y - i))
            
            if to_clear and length == 5:
                self.win_state = True
                return { "length": 5, "coords": to_clear }
            elif to_clear:
                return { "length": length, "coords": to_clear }
                
        return None

    def _process_matches(self, matches, chain_bonus):
        length = matches["length"]
        coords = matches["coords"]
        
        if length == 5:
            reward, score = 10.0 * chain_bonus, 500 * chain_bonus
        elif length == 4:
            reward, score = 5.0 * chain_bonus, 100 * chain_bonus
        else: # length == 3
            reward, score = 1.0 * chain_bonus, 20 * chain_bonus
            
        for x, y in coords:
            color_id = self.grid[x, y]
            if color_id != 0:
                self._create_particles(self._grid_to_iso(x, y), self.CRYSTAL_COLORS[color_id])
                self.grid[x, y] = 0
        # sfx: match_clear.wav
        return reward, score

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({"pos": list(pos), "vel": vel, "color": color, "size": self.np_random.uniform(2, 5)})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid base
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                points = [
                    self._grid_to_iso(x, y),
                    self._grid_to_iso(x + 1, y),
                    self._grid_to_iso(x + 1, y + 1),
                    self._grid_to_iso(x, y + 1)
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)

        # Draw dropper guide line
        col_x = self.dropper_pos
        top_pos = self._grid_to_iso(col_x, 0)
        bottom_pos = self._grid_to_iso(col_x, self.GRID_HEIGHT)
        pygame.draw.line(self.screen, self.COLOR_GRID, (top_pos[0] + self.TILE_WIDTH/2, top_pos[1]), (bottom_pos[0]+self.TILE_WIDTH/2, bottom_pos[1]), 1)

        # Draw drop trail effect
        if self.last_drop_info:
            start = self.last_drop_info["start"]
            end = self.last_drop_info["end"]
            color = self.last_drop_info["color"]
            pygame.draw.line(self.screen, color, (start[0] + self.TILE_WIDTH/2, start[1] + self.TILE_HEIGHT/2), (end[0] + self.TILE_WIDTH/2, end[1] + self.TILE_HEIGHT/2), 3)

        # Draw crystals
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                color_id = self.grid[x, y]
                if color_id != 0:
                    self._draw_crystal(x, y, color_id)
        
        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p["color"], p["pos"], p["size"])

        # Draw dropper and next crystal
        dropper_iso_x, dropper_iso_y = self._grid_to_iso(self.dropper_pos, -0.5)
        self._draw_crystal_at(dropper_iso_x, dropper_iso_y, self.next_crystal, is_dropper=True)

    def _draw_crystal(self, x, y, color_id):
        iso_x, iso_y = self._grid_to_iso(x, y)
        self._draw_crystal_at(iso_x, iso_y, color_id)

    def _draw_crystal_at(self, iso_x, iso_y, color_id, is_dropper=False):
        color = self.CRYSTAL_COLORS[color_id]
        glow_color = self.GLOW_COLORS[color_id]
        
        top = (iso_x + self.TILE_WIDTH / 2, iso_y)
        right = (iso_x + self.TILE_WIDTH, iso_y + self.TILE_HEIGHT / 2)
        bottom = (iso_x + self.TILE_WIDTH / 2, iso_y + self.TILE_HEIGHT)
        left = (iso_x, iso_y + self.TILE_HEIGHT / 2)
        
        points = [top, right, bottom, left]
        
        # Glow effect
        glow_points = [
            (top[0], top[1] - 2),
            (right[0] + 2, right[1]),
            (bottom[0], bottom[1] + 2),
            (left[0] - 2, left[1])
        ]
        pygame.gfxdraw.filled_polygon(self.screen, glow_points, (*glow_color, 80))
        
        # Main crystal body
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, glow_color)
        
        # Highlight
        highlight_points = [
            (top[0], top[1] + 2),
            (top[0] + (self.TILE_WIDTH / 4), top[1] + (self.TILE_HEIGHT / 4) + 2),
            (top[0], top[1] + (self.TILE_HEIGHT / 2) + 2),
            (top[0] - (self.TILE_WIDTH / 4), top[1] + (self.TILE_HEIGHT / 4) + 2)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, highlight_points, (255, 255, 255, 120))
        
        if is_dropper:
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_DROPPER)


    def _render_ui(self):
        # Score and Moves
        score_text = self.font_small.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (630 - score_text.get_width(), 10))
        
        moves_text = self.font_small.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (10, 10))
        
        # Game Over / Win Message
        if self.game_over:
            overlay = pygame.Surface((640, 400), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.win_state else "GAME OVER"
            color = (180, 255, 180) if self.win_state else (255, 180, 180)
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(640 // 2, 400 // 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "win": self.win_state,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Crystal Cavern")
    clock = pygame.time.Clock()
    
    print(GameEnv.user_guide)

    while not done:
        # --- Action mapping for human play ---
        movement, space, shift = 0, 0, 0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]

        # --- Event handling ---
        should_step = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                # In turn-based, we only step on a key press
                should_step = True
                if event.key == pygame.K_r: # Add a manual reset key
                    obs, info = env.reset()

        if should_step:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(f"Action: {action}, Reward: {reward:.2f}, Info: {info}")

        # --- Rendering ---
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit FPS for human play

    env.close()
    pygame.quit()