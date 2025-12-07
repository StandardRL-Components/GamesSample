import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move selector. Space to rotate clockwise, Shift to rotate counter-clockwise."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Rotate tiles to match the hidden color pattern. You have a limited number of moves. Good luck!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_DIM = (10, 10)
        self.MAX_MOVES = 20
        self.NUM_PALETTE_COLORS = 6

        # --- Colors ---
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (50, 50, 70)
        self.COLOR_TILE_BG = (35, 35, 50)
        self.COLOR_SELECTOR = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_CORRECT_GLOW = (100, 255, 100)
        self.PALETTE = [
            (240, 163, 255), (0, 117, 220), (43, 206, 72),
            (255, 164, 5), (255, 0, 16), (224, 255, 102)
        ]
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.Font(pygame.font.get_default_font(), 18)
            self.font_big = pygame.font.Font(pygame.font.get_default_font(), 48)
            self.font_reward = pygame.font.Font(pygame.font.get_default_font(), 16)
        except IOError:
            self.font_ui = pygame.font.SysFont("sans-serif", 18)
            self.font_big = pygame.font.SysFont("sans-serif", 48)
            self.font_reward = pygame.font.SysFont("sans-serif", 16)

        # --- Grid Layout ---
        self.grid_area_height = self.SCREEN_HEIGHT - 80
        self.tile_size = min(
            (self.SCREEN_WIDTH - 40) // self.GRID_DIM[0],
            (self.grid_area_height - 40) // self.GRID_DIM[1]
        )
        self.grid_width = self.tile_size * self.GRID_DIM[0]
        self.grid_height = self.tile_size * self.GRID_DIM[1]
        self.grid_start_x = (self.SCREEN_WIDTH - self.grid_width) // 2
        self.grid_start_y = (self.SCREEN_HEIGHT - self.grid_height) // 2 + 20

        # --- Initialize state variables ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self.moves_left = 0
        self.selector_pos = [0, 0]
        self.tiles = []
        self.target_pattern = np.zeros(self.GRID_DIM)
        self.particles = []
        self.last_reward_text = ""
        self.last_reward_alpha = 0
        self.np_random = None

        # self.validate_implementation() # This is removed as it's for internal testing
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Initialize Game State ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self.moves_left = self.MAX_MOVES
        self.selector_pos = [self.GRID_DIM[0] // 2, self.GRID_DIM[1] // 2]
        self.particles = []
        self.last_reward_text = ""
        self.last_reward_alpha = 0

        # --- Generate Puzzle ---
        self.target_pattern = self.np_random.integers(0, self.NUM_PALETTE_COLORS, size=self.GRID_DIM)
        self.tiles = []
        palette_indices = list(range(self.NUM_PALETTE_COLORS))
        
        all_tiles_correct = True
        for y in range(self.GRID_DIM[1]):
            row = []
            for x in range(self.GRID_DIM[0]):
                target_color_idx = self.target_pattern[y, x]
                other_colors = [c for c in palette_indices if c != target_color_idx]
                tile_colors = [target_color_idx] + list(self.np_random.choice(other_colors, size=3, replace=False))
                self.np_random.shuffle(tile_colors)
                
                initial_rotation = self.np_random.integers(0, 4)
                
                if tile_colors[initial_rotation] == target_color_idx:
                    # If randomly correct, rotate it once to make it incorrect.
                    initial_rotation = (initial_rotation + 1) % 4
                
                row.append({'colors': tile_colors, 'rotation': initial_rotation})
                
                if tile_colors[initial_rotation] != target_color_idx:
                    all_tiles_correct = False

            self.tiles.append(row)

        if all_tiles_correct:
            # Extremely rare case, just re-run reset.
            return self.reset(seed=seed, options=options)

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # --- Unpack Action ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self.steps += 1
        reward = 0.0
        terminated = False
        
        # Fade out reward text from previous step
        self.last_reward_alpha = max(0, self.last_reward_alpha - 25)

        # --- Handle Cursor Movement ---
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        if dx != 0 or dy != 0:
            self.selector_pos[0] = (self.selector_pos[0] + dx) % self.GRID_DIM[0]
            self.selector_pos[1] = (self.selector_pos[1] + dy) % self.GRID_DIM[1]

        # --- Handle Rotation ---
        rotation_dir = 0
        if space_held and not shift_held: rotation_dir = 1  # Clockwise
        if shift_held and not space_held: rotation_dir = -1 # Counter-clockwise
        
        if rotation_dir != 0 and not self.game_over:
            num_correct_before = self._count_correct_tiles()
            
            # Perform rotation
            sel_x, sel_y = self.selector_pos
            tile = self.tiles[sel_y][sel_x]
            tile['rotation'] = (tile['rotation'] + rotation_dir) % 4
            
            self.moves_left -= 1
            
            # Calculate immediate reward
            num_correct_after = self._count_correct_tiles()
            delta_correct = num_correct_after - num_correct_before
            reward = delta_correct * 0.5
            self.score += reward
            
            # Visual feedback
            self._create_particles(sel_x, sel_y)
            if reward != 0:
                self.last_reward_text = f"{reward:+.1f}"
                self.last_reward_alpha = 255

        # --- Check for Termination ---
        is_solved = self._count_correct_tiles() == self.GRID_DIM[0] * self.GRID_DIM[1]
        
        if is_solved and not self.game_over:
            terminated = True
            self.game_over = True
            self.win = True
            completion_reward = 10.0
            bonus_reward = 10.0 * (self.moves_left / self.MAX_MOVES)
            reward += completion_reward + bonus_reward
            self.score += completion_reward + bonus_reward
            self.last_reward_text = f"SOLVED! +{completion_reward + bonus_reward:.1f}"
            self.last_reward_alpha = 255

        elif self.moves_left <= 0 and not self.game_over:
            terminated = True
            self.game_over = True
            self.win = False
            reward = -10.0
            self.score += reward # Apply penalty
            self.last_reward_text = "OUT OF MOVES"
            self.last_reward_alpha = 255
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._update_and_draw_particles()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw tiles
        for y in range(self.GRID_DIM[1]):
            for x in range(self.GRID_DIM[0]):
                tile = self.tiles[y][x]
                rect = pygame.Rect(
                    self.grid_start_x + x * self.tile_size,
                    self.grid_start_y + y * self.tile_size,
                    self.tile_size, self.tile_size
                )
                
                # Draw tile background and border
                pygame.draw.rect(self.screen, self.COLOR_TILE_BG, rect)
                
                # Draw main color
                color_idx = tile['colors'][tile['rotation']]
                color = self.PALETTE[color_idx]
                inner_rect = rect.inflate(-4, -4)
                pygame.draw.rect(self.screen, color, inner_rect, border_radius=4)

                # Draw correctness indicator
                if self._is_tile_correct(x, y):
                    center = inner_rect.center
                    pygame.gfxdraw.aacircle(self.screen, center[0], center[1], self.tile_size // 8, self.COLOR_CORRECT_GLOW)
                    pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], self.tile_size // 8, self.COLOR_CORRECT_GLOW)

        # Draw grid lines
        for i in range(self.GRID_DIM[0] + 1):
            start_pos = (self.grid_start_x + i * self.tile_size, self.grid_start_y)
            end_pos = (self.grid_start_x + i * self.tile_size, self.grid_start_y + self.grid_height)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
        for i in range(self.GRID_DIM[1] + 1):
            start_pos = (self.grid_start_x, self.grid_start_y + i * self.tile_size)
            end_pos = (self.grid_start_x + self.grid_width, self.grid_start_y + i * self.tile_size)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

        # Draw selector
        if not self.game_over:
            sel_x, sel_y = self.selector_pos
            selector_rect = pygame.Rect(
                self.grid_start_x + sel_x * self.tile_size,
                self.grid_start_y + sel_y * self.tile_size,
                self.tile_size, self.tile_size
            )
            pulse = (math.sin(self.steps * 0.3) + 1) / 2
            line_width = int(2 + pulse * 2)
            pygame.draw.rect(self.screen, self.COLOR_SELECTOR, selector_rect, width=line_width, border_radius=2)

    def _render_ui(self):
        # Draw Moves Left
        moves_text = f"Moves: {self.moves_left}/{self.MAX_MOVES}"
        self._draw_text(moves_text, self.font_ui, (10, 10))

        # Draw Score
        score_text = f"Score: {self.score:.1f}"
        self._draw_text(score_text, self.font_ui, (self.SCREEN_WIDTH - 10, 10), align="right")

        # Draw Progress Bar
        bar_y = 40
        bar_height = 8
        bar_width = self.SCREEN_WIDTH - 20
        progress = self.moves_left / self.MAX_MOVES
        
        bg_rect = pygame.Rect(10, bar_y, bar_width, bar_height)
        pygame.draw.rect(self.screen, self.COLOR_TILE_BG, bg_rect, border_radius=4)
        
        if progress > 0:
            # FIX: pygame.Color.lerp(color, amount) interpolates from the instance color to the argument color.
            # The original code had the wrong syntax and arguments.
            start_color = pygame.Color(255, 0, 0)
            end_color = pygame.Color(0, 255, 0)
            bar_color = start_color.lerp(end_color, progress)
            fill_rect = pygame.Rect(10, bar_y, int(bar_width * progress), bar_height)
            pygame.draw.rect(self.screen, bar_color, fill_rect, border_radius=4)

        # Draw floating reward text
        if self.last_reward_alpha > 0:
            sel_x, sel_y = self.selector_pos
            pos_x = self.grid_start_x + sel_x * self.tile_size + self.tile_size // 2
            pos_y = self.grid_start_y + sel_y * self.tile_size - 10
            self._draw_text(
                self.last_reward_text, self.font_reward, (pos_x, pos_y),
                alpha=self.last_reward_alpha, align="center"
            )

        # Draw Game Over/Win screen
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 200))
            self.screen.blit(overlay, (0, 0))
            
            text = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_CORRECT_GLOW if self.win else (255, 80, 80)
            self._draw_text(text, self.font_big, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), color=color, align="center")

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "correct_tiles": self._count_correct_tiles()
        }
        
    def _is_tile_correct(self, x, y):
        tile = self.tiles[y][x]
        current_color_idx = tile['colors'][tile['rotation']]
        target_color_idx = self.target_pattern[y, x]
        return current_color_idx == target_color_idx

    def _count_correct_tiles(self):
        count = 0
        for y in range(self.GRID_DIM[1]):
            for x in range(self.GRID_DIM[0]):
                if self._is_tile_correct(x, y):
                    count += 1
        return count
        
    def _create_particles(self, tile_x, tile_y):
        tile = self.tiles[tile_y][tile_x]
        color_idx = tile['colors'][tile['rotation']]
        color = self.PALETTE[color_idx]
        
        center_x = self.grid_start_x + tile_x * self.tile_size + self.tile_size / 2
        center_y = self.grid_start_y + tile_y * self.tile_size + self.tile_size / 2
        
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': [center_x, center_y], 'vel': vel, 'life': life, 'color': color})

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['life'] / 30))
                color = (*p['color'], alpha)
                radius = int(p['life'] / 10) + 1
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                
                temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (radius, radius), radius)
                self.screen.blit(temp_surf, (pos[0]-radius, pos[1]-radius))
                
    def _draw_text(self, text, font, pos, color=None, alpha=255, align="left"):
        if color is None:
            color = self.COLOR_TEXT
        
        text_surface = font.render(text, True, color)
        text_surface.set_alpha(alpha)
        
        text_rect = text_surface.get_rect()
        if align == "right":
            text_rect.topright = pos
        elif align == "center":
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for human play and is not part of the Gymnasium interface
    # It will not be run by the evaluation server.
    # To use, you will need to `pip install pygame`
    # and remove the `os.environ.setdefault("SDL_VIDEODRIVER", "dummy")` line.
    
    # Re-enable the display for human play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv()
    obs, info = env.reset(seed=42)
    terminated = False
    
    print(env.user_guide)
    print(f"Action space: {env.action_space}")

    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Tile Rotator")

    running = True
    while running:
        action_taken_this_frame = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_SPACE, pygame.K_LSHIFT, pygame.K_RSHIFT]:
                    action_taken_this_frame = True
                if event.key == pygame.K_r:
                    obs, info = env.reset(seed=random.randint(0, 10000))
                    terminated = False
                    action_taken_this_frame = True
                if event.key == pygame.K_q:
                    running = False

        if terminated and not action_taken_this_frame:
            pass
        else:
            keys = pygame.key.get_pressed()
            move_action = 0
            if keys[pygame.K_UP]: move_action = 1
            elif keys[pygame.K_DOWN]: move_action = 2
            elif keys[pygame.K_LEFT]: move_action = 3
            elif keys[pygame.K_RIGHT]: move_action = 4
            
            space_action = 1 if keys[pygame.K_SPACE] else 0
            shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [move_action, space_action, shift_action]
            
            if any(action) or action_taken_this_frame:
                obs, reward, terminated, truncated, info = env.step(action)
                if any(action):
                    print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Info: {info}")

        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30)

    env.close()