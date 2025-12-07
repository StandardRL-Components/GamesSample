import os
import os
import pygame

os.environ["SDL_VIDEODRIVER"] = "dummy"

import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    """
    A puzzle game where the player combines numbers on a grid to reach a target value.

    The player controls a cursor to select and combine adjacent numbers. Each combination
    creates a new number equal to the sum of the two combined numbers. The goal is to
    create a number that matches the target value before the time runs out.

    The game is designed with a focus on visual feedback and satisfying interactions,
    featuring smooth animations, particle effects, and a clean, minimalist aesthetic.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor. Space to select a number. "
        "Select a second adjacent number to combine them. Shift to deselect."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Combine numbers on a grid to reach the target value before time runs out. "
        "Strategically merge adjacent numbers to create larger sums and hit the goal."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_W, SCREEN_H = 640, 400
    GRID_DIM = (8, 5)  # width, height
    TIME_LIMIT_SECONDS = 30
    FPS = 30
    
    # Colors
    COLOR_BG = (25, 30, 35)
    COLOR_GRID = (50, 60, 70)
    COLOR_UI_TEXT = (220, 220, 230)
    COLOR_UI_BG = (40, 45, 50)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_SELECT = (100, 200, 255)
    COLOR_LOW_VAL = (60, 120, 220)  # Cool Blue
    COLOR_HIGH_VAL = (255, 100, 80) # Warm Orange/Red
    COLOR_TARGET_MATCH = (100, 255, 150) # Bright Green

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_H, self.SCREEN_W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_W, self.SCREEN_H))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_huge = pygame.font.SysFont("Arial", 48, bold=True)

        # Grid and cell dimensions
        self.top_margin = 60
        self.grid_h = self.SCREEN_H - self.top_margin - 20
        self.grid_w = self.grid_h * (self.GRID_DIM[0] / self.GRID_DIM[1])
        self.grid_x = (self.SCREEN_W - self.grid_w) / 2
        self.grid_y = self.top_margin
        self.cell_w = self.grid_w / self.GRID_DIM[0]
        self.cell_h = self.grid_h / self.GRID_DIM[1]

        # State variables initialized in reset()
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.time_limit_steps = self.TIME_LIMIT_SECONDS * self.FPS
        self.np_random = None
        self.grid = []
        self.target_number = 0
        self.cursor_pos = [0, 0]
        self.visual_cursor_pos = [0.0, 0.0]
        self.selected_pos = None
        self.move_cooldown = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.particles = []
        self.number_visuals = {} # For smooth size animations
        self.win_message = ""

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win_message = ""
        
        self.target_number = self.np_random.integers(low=50, high=150)
        self._generate_grid()
        
        self.cursor_pos = [self.GRID_DIM[0] // 2, self.GRID_DIM[1] // 2]
        self.visual_cursor_pos = self._grid_to_pixel(self.cursor_pos[0], self.cursor_pos[1])
        self.selected_pos = None
        
        self.move_cooldown = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.particles.clear()
        self.number_visuals.clear()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0.0
        self.steps += 1
        
        if not self.game_over:
            self._handle_input(movement, space_held, shift_held)
            reward = self._process_combinations(space_held)
            self.score += reward

        self._update_animations()
        
        terminated = self.game_over
        
        # Check for time-out loss condition
        if not self.game_over and self.steps >= self.time_limit_steps:
            self.game_over = True
            terminated = True
            reward = -100.0
            self.score += reward
            self.win_message = "TIME UP!"
            # sfx: game_over_lose

        # Update previous button states for next frame
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _generate_grid(self):
        while True:
            self.grid = [[0] * self.GRID_DIM[0] for _ in range(self.GRID_DIM[1])]
            total_sum = 0
            num_cells_to_fill = self.np_random.integers(low=12, high=20)
            
            for _ in range(num_cells_to_fill):
                x, y = self.np_random.integers(0, self.GRID_DIM[0]), self.np_random.integers(0, self.GRID_DIM[1])
                if self.grid[y][x] == 0:
                    val = self.np_random.integers(1, 10)
                    self.grid[y][x] = val
                    total_sum += val
            
            if total_sum >= self.target_number:
                break

    def _handle_input(self, movement, space_held, shift_held):
        # --- Movement ---
        self.move_cooldown = max(0, self.move_cooldown - 1)
        if self.move_cooldown == 0 and movement != 0:
            dx, dy = 0, 0
            if movement == 1: dy = -1  # Up
            elif movement == 2: dy = 1  # Down
            elif movement == 3: dx = -1  # Left
            elif movement == 4: dx = 1  # Right
            
            self.cursor_pos[0] = (self.cursor_pos[0] + dx) % self.GRID_DIM[0]
            self.cursor_pos[1] = (self.cursor_pos[1] + dy) % self.GRID_DIM[1]
            self.move_cooldown = 4 # Cooldown to prevent hyperspeed movement

        # --- Deselect ---
        is_shift_press = shift_held and not self.prev_shift_held
        if is_shift_press and self.selected_pos is not None:
            # sfx: deselect
            self.selected_pos = None

    def _process_combinations(self, space_held):
        is_space_press = space_held and not self.prev_space_held
        if not is_space_press:
            return 0.0

        cx, cy = self.cursor_pos
        
        # Case 1: Nothing selected -> try to select
        if self.selected_pos is None:
            if self.grid[cy][cx] > 0:
                self.selected_pos = [cx, cy]
                # sfx: select
            return 0.0
        
        # Case 2: Something is selected -> try to combine or deselect
        else:
            sx, sy = self.selected_pos
            
            # Deselect if clicking the same cell
            if sx == cx and sy == cy:
                self.selected_pos = None
                # sfx: deselect
                return 0.0

            # Check for adjacency and valid target cell
            is_adjacent = abs(sx - cx) + abs(sy - cy) == 1
            if self.grid[cy][cx] > 0 and is_adjacent:
                val1 = self.grid[sy][sx]
                val2 = self.grid[cy][cx]
                new_val = val1 + val2

                # Update grid state
                self.grid[cy][cx] = new_val
                self.grid[sy][sx] = 0
                
                # Animate new number
                self.number_visuals[(cx, cy)] = {'size_mult': 1.5, 'value': new_val}
                
                # Create particles
                center_px = self._grid_to_pixel(cx, cy)
                color = self._get_color_for_value(new_val)
                self._create_particles(center_px, color)
                # sfx: combine

                # Clear selection
                self.selected_pos = None
                
                # Check for win condition
                if new_val == self.target_number:
                    self.game_over = True
                    self.win_message = "TARGET REACHED!"
                    self.score += 50.0
                    # sfx: win_fanfare
                
                # Calculate reward based on progress
                dist_before = abs(self.target_number - max(val1, val2))
                dist_after = abs(self.target_number - new_val)
                
                return 1.0 if dist_after < dist_before else -0.1
            else:
                # Invalid combine attempt
                # sfx: error_buzz
                return 0.0
        return 0.0

    def _update_animations(self):
        # Smooth cursor movement
        target_pixel_pos = self._grid_to_pixel(self.cursor_pos[0], self.cursor_pos[1])
        self.visual_cursor_pos[0] += (target_pixel_pos[0] - self.visual_cursor_pos[0]) * 0.4
        self.visual_cursor_pos[1] += (target_pixel_pos[1] - self.visual_cursor_pos[1]) * 0.4

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['radius'] = max(0, p['radius'] * 0.95)

        # Update number size animations
        for key in list(self.number_visuals.keys()):
            vis = self.number_visuals[key]
            vis['size_mult'] += (1.0 - vis['size_mult']) * 0.15
            if abs(vis['size_mult'] - 1.0) < 0.01:
                del self.number_visuals[key]

    def _get_observation(self):
        if self.np_random is None: self.reset()
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_DIM[0] + 1):
            x = self.grid_x + i * self.cell_w
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.grid_y), (x, self.grid_y + self.grid_h))
        for i in range(self.GRID_DIM[1] + 1):
            y = self.grid_y + i * self.cell_h
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.grid_x, y), (self.grid_x + self.grid_w, y))

        # Draw selection highlight
        if self.selected_pos is not None:
            sx, sy = self.selected_pos
            px, py = self._grid_to_pixel(sx, sy)
            rect = pygame.Rect(px - self.cell_w/2, py - self.cell_h/2, self.cell_w, self.cell_h)
            pygame.draw.rect(self.screen, self.COLOR_SELECT, rect.inflate(4,4), 0, 10)

        # Draw cursor
        cx_px, cy_px = self.visual_cursor_pos
        cursor_rect = pygame.Rect(cx_px - self.cell_w/2, cy_px - self.cell_h/2, self.cell_w, self.cell_h)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, 10)

        # Draw numbers
        for y in range(self.GRID_DIM[1]):
            for x in range(self.GRID_DIM[0]):
                val = self.grid[y][x]
                if val > 0:
                    center_px = self._grid_to_pixel(x, y)
                    color = self._get_color_for_value(val)
                    
                    size_mult = self.number_visuals.get((x, y), {}).get('size_mult', 1.0)
                    base_radius = min(self.cell_w, self.cell_h) / 2.5
                    radius = int(base_radius * size_mult)

                    pygame.gfxdraw.filled_circle(self.screen, int(center_px[0]), int(center_px[1]), radius, color)
                    pygame.gfxdraw.aacircle(self.screen, int(center_px[0]), int(center_px[1]), radius, color)

                    text_surf = self.font_large.render(str(val), True, self.COLOR_UI_BG)
                    text_rect = text_surf.get_rect(center=center_px)
                    self.screen.blit(text_surf, text_rect)
        
        # Draw particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), p['color'])

    def _render_ui(self):
        # Top UI bar background
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (0, 0, self.SCREEN_W, self.top_margin - 10))

        # Score
        score_text = self.font_large.render(f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 15))

        # Target Number
        target_text = self.font_large.render("Target", True, self.COLOR_UI_TEXT)
        target_rect = target_text.get_rect(center=(self.SCREEN_W / 2, 15))
        self.screen.blit(target_text, target_rect)
        
        target_val_text = self.font_large.render(str(self.target_number), True, self.COLOR_TARGET_MATCH)
        target_val_rect = target_val_text.get_rect(center=(self.SCREEN_W / 2, 40))
        self.screen.blit(target_val_text, target_val_rect)

        # Timer
        time_left = max(0, (self.time_limit_steps - self.steps) / self.FPS)
        time_color = self.COLOR_HIGH_VAL if time_left < 5 else self.COLOR_UI_TEXT
        time_text = self.font_large.render(f"Time: {time_left:.1f}", True, time_color)
        time_rect = time_text.get_rect(topright=(self.SCREEN_W - 20, 15))
        self.screen.blit(time_text, time_rect)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_W, self.SCREEN_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_huge.render(self.win_message, True, self.COLOR_UI_TEXT)
            end_rect = end_text.get_rect(center=(self.SCREEN_W / 2, self.SCREEN_H / 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "target": self.target_number}

    def _grid_to_pixel(self, x, y):
        px = self.grid_x + (x + 0.5) * self.cell_w
        py = self.grid_y + (y + 0.5) * self.cell_h
        return [px, py]
    
    def _lerp_color(self, c1, c2, t):
        t = max(0, min(1, t))
        return (c1[0] + (c2[0] - c1[0]) * t, c1[1] + (c2[1] - c1[1]) * t, c1[2] + (c2[2] - c1[2]) * t)

    def _get_color_for_value(self, value):
        if value == self.target_number:
            return self.COLOR_TARGET_MATCH
        # Normalize value for color interpolation. Let's cap at target_number for mapping.
        t = min(1.0, value / (self.target_number * 0.8)) 
        return self._lerp_color(self.COLOR_LOW_VAL, self.COLOR_HIGH_VAL, t)

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(15, 31),
                'radius': self.np_random.uniform(3, 8),
                'color': color
            })

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # --- Human Controls ---
    # Map Pygame keys to the MultiDiscrete action space
    key_to_action = {
        pygame.K_UP:    [1, 0, 0],
        pygame.K_DOWN:  [2, 0, 0],
        pygame.K_LEFT:  [3, 0, 0],
        pygame.K_RIGHT: [4, 0, 0],
        pygame.K_SPACE: [0, 1, 0],
        pygame.K_LSHIFT:[0, 0, 1],
        pygame.K_RSHIFT:[0, 0, 1],
    }

    # Pygame setup for human play
    render_screen = pygame.display.set_mode((GameEnv.SCREEN_W, GameEnv.SCREEN_H))
    pygame.display.set_caption(GameEnv.game_description)
    clock = pygame.time.Clock()

    print("--- Playing Game ---")
    print(GameEnv.user_guide)
    
    total_reward = 0
    while not terminated:
        # Construct action from keyboard state
        action = [0, 0, 0] # Default action: no-op
        keys = pygame.key.get_pressed()
        
        # Combine actions (e.g., move while holding space)
        # Movement is prioritized
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        
        # Process Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(GameEnv.FPS)

    print(f"Game Over! Final Score: {info['score']:.2f}")
    env.close()