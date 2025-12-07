import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:22:20.269056
# Source Brief: brief_01684.md
# Brief Index: 1684
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Match three or more blocks of the same color to clear them from the board. "
        "Rotate and move the falling triplets to create chains and score points before time runs out."
    )
    user_guide = (
        "Controls: ←→ to move, ↑↓ to rotate. Press space to hard drop the piece and shift to swap with the next piece."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 10, 15
    BLOCK_SIZE = 24
    BOARD_WIDTH = GRID_WIDTH * BLOCK_SIZE
    BOARD_HEIGHT = GRID_HEIGHT * BLOCK_SIZE
    BOARD_X_OFFSET = (WIDTH - BOARD_WIDTH) // 2
    BOARD_Y_OFFSET = (HEIGHT - BOARD_HEIGHT)

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 70)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (10, 10, 15)
    
    BLOCK_COLORS = {
        1: ((255, 80, 80), (255, 120, 120)),  # Red
        2: ((80, 255, 80), (120, 255, 120)),  # Green
        3: ((80, 80, 255), (120, 120, 255)),  # Blue
    }
    
    # Game parameters
    MAX_TIMER = 300 * 30  # 300 seconds at 30 FPS
    MAX_STEPS = 5000
    WIN_SCORE = 1000
    FPS = 30

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_popup = pygame.font.SysFont("Verdana", 16, bold=True)

        # self.reset() # reset is called by the wrapper/runner
        # self.validate_implementation() # This should not be called in init

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.int8)
        self.steps = 0
        self.score = 0
        self.timer = self.MAX_TIMER
        self.game_over = False
        
        self.fall_speed = 0.1  # Grids per step
        self.score_milestone = 200
        
        self.next_triplet_colors = self._generate_triplet_colors()
        self._spawn_new_triplet()

        self.particles = []
        self.floating_texts = []
        
        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.clock.tick(self.FPS)
        self.steps += 1
        self.timer -= 1
        
        reward_this_step = 0

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Input (Edge-triggered for drop/swap) ---
        hard_drop = space_held and not self.prev_space_held
        swap_action = shift_held and not self.prev_shift_held
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        self._handle_input(movement, hard_drop, swap_action)

        # --- Game Logic Update ---
        landed = False
        if hard_drop:
            while self._is_valid_position(self.current_triplet['x'], self.current_triplet['y'] + 1, self.current_triplet['rotation']):
                self.current_triplet['y'] += 1
            landed = True
        else:
            self.current_triplet['y'] += self.fall_speed
            if not self._is_valid_position(self.current_triplet['x'], self.current_triplet['y'], self.current_triplet['rotation']):
                self.current_triplet['y'] = math.floor(self.current_triplet['y'])
                landed = True

        if landed:
            self._place_triplet_on_grid()
            reward_this_step += 0.1  # Reward for placing a block
            
            # Chain reaction loop
            chain_multiplier = 1.0
            while True:
                matches, cleared_blocks = self._find_and_process_matches()
                if not matches:
                    break
                
                # Score and Reward
                num_cleared = len(cleared_blocks)
                score_gain = int(10 * (num_cleared // 3) * chain_multiplier)
                self.score += score_gain
                
                reward_this_step += (num_cleared // 3) * 1.0 # Base reward for matches
                if chain_multiplier > 1.0:
                    reward_this_step += 5.0 # Chain reaction bonus
                
                if score_gain > 0:
                    # Find center of cleared blocks for text popup
                    avg_x = sum(c for r, c in cleared_blocks) / num_cleared
                    avg_y = sum(r for r, c in cleared_blocks) / num_cleared
                    self._create_floating_text(f"+{score_gain}", avg_x, avg_y)

                self._apply_grid_gravity()
                chain_multiplier += 0.5
            
            # Speed up based on score milestones
            if self.score >= self.score_milestone:
                self.fall_speed *= 1.05
                self.score_milestone += 200

            self._spawn_new_triplet()
            if not self._is_valid_position(self.current_triplet['x'], self.current_triplet['y'], self.current_triplet['rotation']):
                self.game_over = True # Top-out condition

        self._update_effects()
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        final_reward = reward_this_step
        if terminated:
            if self.score >= self.WIN_SCORE:
                final_reward += 100 # Win bonus
            elif self.timer <= 0:
                final_reward -= 100 # Timeout penalty
        
        return self._get_observation(), final_reward, terminated, truncated, self._get_info()

    # --- Private Helper Methods ---

    def _handle_input(self, movement, hard_drop, swap_action):
        # Movement: 0=none, 1=rot+, 2=rot-, 3=left, 4=right
        if movement == 1: # Rotate Clockwise
            new_rot = (self.current_triplet['rotation'] + 1) % 2
            if self._is_valid_position(self.current_triplet['x'], self.current_triplet['y'], new_rot):
                self.current_triplet['rotation'] = new_rot
        elif movement == 2: # Rotate Counter-Clockwise
            new_rot = (self.current_triplet['rotation'] - 1 + 2) % 2
            if self._is_valid_position(self.current_triplet['x'], self.current_triplet['y'], new_rot):
                self.current_triplet['rotation'] = new_rot
        elif movement == 3: # Left
            if self._is_valid_position(self.current_triplet['x'] - 1, self.current_triplet['y'], self.current_triplet['rotation']):
                self.current_triplet['x'] -= 1
        elif movement == 4: # Right
            if self._is_valid_position(self.current_triplet['x'] + 1, self.current_triplet['y'], self.current_triplet['rotation']):
                self.current_triplet['x'] += 1

        if swap_action and not self.swap_used:
            self.current_triplet['colors'], self.next_triplet_colors = self.next_triplet_colors, self.current_triplet['colors']
            self.swap_used = True
            # Check if new piece is valid, if not, swap back
            if not self._is_valid_position(self.current_triplet['x'], self.current_triplet['y'], self.current_triplet['rotation']):
                self.current_triplet['colors'], self.next_triplet_colors = self.next_triplet_colors, self.current_triplet['colors']
                self.swap_used = False


    def _get_triplet_coords(self, grid_x, grid_y, rotation):
        coords = []
        iy = math.floor(grid_y)
        if rotation == 0:  # Vertical
            coords = [(grid_x, iy - 1), (grid_x, iy), (grid_x, iy + 1)]
        elif rotation == 1:  # Horizontal
            coords = [(grid_x - 1, iy), (grid_x, iy), (grid_x + 1, iy)]
        return coords

    def _is_valid_position(self, grid_x, grid_y, rotation):
        coords = self._get_triplet_coords(grid_x, grid_y, rotation)
        for x, y in coords:
            if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT):
                return False
            if y >= 0 and self.grid[y, x] != 0:
                return False
        return True

    def _place_triplet_on_grid(self):
        coords = self._get_triplet_coords(self.current_triplet['x'], self.current_triplet['y'], self.current_triplet['rotation'])
        colors = self.current_triplet['colors']
        for i, (x, y) in enumerate(coords):
            if 0 <= y < self.GRID_HEIGHT and 0 <= x < self.GRID_WIDTH:
                self.grid[y, x] = colors[i]
        # sfx: place_block.wav

    def _find_and_process_matches(self):
        to_clear = set()
        visited = np.zeros_like(self.grid, dtype=bool)
        all_matches = []

        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] != 0 and not visited[r, c]:
                    color = self.grid[r, c]
                    component = []
                    q = [(r, c)]
                    visited[r, c] = True
                    head = 0
                    while head < len(q):
                        row, col = q[head]
                        head += 1
                        component.append((row, col))
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = row + dr, col + dc
                            if 0 <= nr < self.GRID_HEIGHT and 0 <= nc < self.GRID_WIDTH and \
                               not visited[nr, nc] and self.grid[nr, nc] == color:
                                visited[nr, nc] = True
                                q.append((nr, nc))
                    
                    if len(component) >= 3:
                        all_matches.append(component)
                        for pos in component:
                            to_clear.add(pos)
        
        if to_clear:
            for r, c in to_clear:
                self._create_particles(c, r, self.grid[r, c])
                self.grid[r, c] = 0
            # sfx: match_clear.wav
        
        return all_matches, to_clear

    def _apply_grid_gravity(self):
        for c in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] != 0:
                    if r != empty_row:
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.grid[r, c] = 0
                    empty_row -= 1

    def _spawn_new_triplet(self):
        self.current_triplet = {
            'x': self.GRID_WIDTH // 2,
            'y': 1.0,
            'rotation': 0,
            'colors': self.next_triplet_colors
        }
        self.next_triplet_colors = self._generate_triplet_colors()
        self.swap_used = False

    def _generate_triplet_colors(self):
        return [self.np_random.integers(1, 4) for _ in range(3)]

    def _check_termination(self):
        return self.score >= self.WIN_SCORE or self.timer <= 0 or self.game_over

    def _update_effects(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1 # gravity on particles
            p['life'] -= 1
            p['size'] = max(0, p['size'] * 0.95)

        self.floating_texts = [t for t in self.floating_texts if t['life'] > 0]
        for t in self.floating_texts:
            t['y'] -= t['vy']
            t['life'] -= 1

    def _create_particles(self, grid_x, grid_y, color_idx):
        px = self.BOARD_X_OFFSET + grid_x * self.BLOCK_SIZE + self.BLOCK_SIZE // 2
        py = self.BOARD_Y_OFFSET + grid_y * self.BLOCK_SIZE + self.BLOCK_SIZE // 2
        color = self.BLOCK_COLORS[color_idx][0]
        for _ in range(10):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'x': px, 'y': py,
                'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'life': self.np_random.integers(15, 30),
                'size': self.np_random.uniform(2, 5),
                'color': color
            })
    
    def _create_floating_text(self, text, grid_x, grid_y):
        px = self.BOARD_X_OFFSET + grid_x * self.BLOCK_SIZE
        py = self.BOARD_Y_OFFSET + grid_y * self.BLOCK_SIZE
        self.floating_texts.append({
            'text': text, 'x': px, 'y': py,
            'vy': 1.0, 'life': 45, 'color': (255, 255, 100)
        })

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "timer": self.timer}

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    # --- Rendering Methods ---

    def _render_all(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background_grid()
        self._render_grid_blocks()
        if not self.game_over:
            self._render_falling_triplet()
        self._render_effects()
        self._render_ui()

    def _render_background_grid(self):
        for x in range(self.GRID_WIDTH + 1):
            px = self.BOARD_X_OFFSET + x * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.BOARD_Y_OFFSET), (px, self.HEIGHT), 1)
        for y in range(self.GRID_HEIGHT + 1):
            py = self.BOARD_Y_OFFSET + y * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.BOARD_X_OFFSET, py), (self.BOARD_X_OFFSET + self.BOARD_WIDTH, py), 1)
        
        # Draw border
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (self.BOARD_X_OFFSET, self.BOARD_Y_OFFSET, self.BOARD_WIDTH, self.BOARD_HEIGHT), 2)

    def _render_block(self, surface, x, y, color_idx):
        color_main, color_light = self.BLOCK_COLORS[color_idx]
        rect = pygame.Rect(x, y, self.BLOCK_SIZE, self.BLOCK_SIZE)
        
        pygame.draw.rect(surface, color_main, rect, border_radius=4)
        inner_rect = pygame.Rect(x + 3, y + 3, self.BLOCK_SIZE - 6, self.BLOCK_SIZE - 6)
        pygame.draw.rect(surface, color_light, inner_rect, border_radius=2)
        pygame.draw.rect(surface, (0,0,0,50), rect, 1, border_radius=4)

    def _render_grid_blocks(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] != 0:
                    px = self.BOARD_X_OFFSET + c * self.BLOCK_SIZE
                    py = self.BOARD_Y_OFFSET + r * self.BLOCK_SIZE
                    self._render_block(self.screen, px, py, self.grid[r, c])

    def _render_falling_triplet(self):
        coords = self._get_triplet_coords(self.current_triplet['x'], self.current_triplet['y'], self.current_triplet['rotation'])
        colors = self.current_triplet['colors']

        # Ghost piece
        ghost_y = self.current_triplet['y']
        while self._is_valid_position(self.current_triplet['x'], ghost_y + 1, self.current_triplet['rotation']):
            ghost_y += 1
        ghost_y = math.floor(ghost_y)
        ghost_coords = self._get_triplet_coords(self.current_triplet['x'], ghost_y, self.current_triplet['rotation'])

        for gx, gy in ghost_coords:
            if gy >= 0:
                px = self.BOARD_X_OFFSET + gx * self.BLOCK_SIZE
                py = self.BOARD_Y_OFFSET + gy * self.BLOCK_SIZE
                rect = pygame.Rect(px, py, self.BLOCK_SIZE, self.BLOCK_SIZE)
                pygame.draw.rect(self.screen, (255, 255, 255, 50), rect, 2, border_radius=4)

        # Actual piece
        for i, (cx, cy) in enumerate(coords):
            # This logic is complex to handle smooth sub-pixel rendering of the falling piece
            px_float = self.BOARD_X_OFFSET + cx * self.BLOCK_SIZE
            py_float = self.BOARD_Y_OFFSET + (cy - math.floor(self.current_triplet['y'])) * self.BLOCK_SIZE + self.current_triplet['y'] * self.BLOCK_SIZE
            self._render_block(self.screen, int(px_float), int(py_float), colors[i])


    def _render_effects(self):
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), int(p['size']), (*p['color'], int(255 * p['life']/30)))
        
        for t in self.floating_texts:
            alpha = min(255, int(255 * (t['life'] / 30)))
            text_surf = self.font_popup.render(t['text'], True, t['color'])
            text_surf.set_alpha(alpha)
            self.screen.blit(text_surf, (int(t['x']), int(t['y'])))

    def _render_ui(self):
        # Score
        self._draw_text(f"SCORE: {self.score}", (20, 10), self.font_main)
        
        # Timer
        time_str = f"TIME: {self.timer // self.FPS:03d}"
        self._draw_text(time_str, (self.WIDTH - 150, 10), self.font_main)
        
        # Next Piece Preview
        self._draw_text("NEXT", (self.BOARD_X_OFFSET + self.BOARD_WIDTH + 20, self.HEIGHT - 140), self.font_small)
        for i, color_idx in enumerate(self.next_triplet_colors):
            px = self.BOARD_X_OFFSET + self.BOARD_WIDTH + 40
            py = self.HEIGHT - 110 + (i * (self.BLOCK_SIZE + 2))
            self._render_block(self.screen, px, py, color_idx)

        # Game Over message
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            msg = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            self._draw_text(msg, (self.WIDTH//2, self.HEIGHT//2 - 20), self.font_main, center=True)

    def _draw_text(self, text, pos, font, color=COLOR_TEXT, shadow_color=COLOR_TEXT_SHADOW, center=False):
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, shadow_color)
        
        x, y = pos
        if center:
            text_rect = text_surf.get_rect(center=(x, y))
        else:
            text_rect = text_surf.get_rect(topleft=(x, y))

        self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()
    
    def validate_implementation(self):
        print("Validating implementation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs, _ = self.reset()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # Set the display for manual playing, which is different from headless training
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv()
    env.validate_implementation()
    
    # --- Manual Play ---
    pygame.display.set_caption("Block Drop Environment")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    # Action state
    movement = 0
    space_held = 0
    shift_held = 0
    
    running = True
    while running:
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                
                # Update actions on key down
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space_held = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_held = 1
            
            if event.type == pygame.KEYUP:
                # Reset actions on key up
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT] and movement != 0:
                    movement = 0
                elif event.key == pygame.K_SPACE:
                    space_held = 0
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_held = 0

        if not terminated:
            # Step the environment
            action = [movement, space_held, shift_held]
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait for 'r' to reset
            wait_for_reset = True
            while wait_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        wait_for_reset = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        terminated = False
                        truncated = False
                        wait_for_reset = False
    
    env.close()