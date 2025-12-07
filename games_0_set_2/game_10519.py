import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:37:24.288439
# Source Brief: brief_00519.md
# Brief Index: 519
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent repairs a virtual motherboard.
    
    The agent controls a cursor on a grid of code blocks. Some blocks are
    corrupted (red and flashing). The goal is to repair all blocks, turning
    them stable (green).

    Repairing a block consumes a resource and triggers a "rewind" effect,
    which has a chance to fix the target block and can cause cascading
    changes (fixes or corruptions) in adjacent blocks.

    The game is won by stabilizing all blocks and lost by running out of
    repair resources.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Repair a virtual motherboard by stabilizing corrupted code blocks. Each repair attempt consumes a charge and may cause unpredictable cascading effects on adjacent blocks."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move the cursor. Press space to attempt a repair on the selected block."
    )
    auto_advance = True

    # Class attributes for difficulty progression across resets
    _initial_corruption_level = 3
    _wins_at_current_level = 0
    DIFFICULTY_INCREASE_THRESHOLD = 5 # Wins needed to increase difficulty

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
        self.screen_width, self.screen_height = 640, 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # --- Visual & Thematic Constants ---
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_GRID_LINE = (20, 30, 50)
        self.COLOR_STABLE = (0, 255, 150)
        self.COLOR_CORRUPTED = (255, 50, 50)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_UI_BG = (20, 40, 70, 180)
        self.COLOR_UI_TEXT = (200, 220, 255)
        
        try:
            self.font_ui = pygame.font.SysFont("consolas", 18)
            self.font_game_over = pygame.font.SysFont("consolas", 48, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.SysFont(None, 24)
            self.font_game_over = pygame.font.SysFont(None, 60, bold=True)

        # --- Game Grid & Block Configuration ---
        self.grid_cols, self.grid_rows = 16, 10
        self.block_margin = 4
        self.grid_area_width = self.screen_width - 40
        self.grid_area_height = self.screen_height - 60
        self.block_width = (self.grid_area_width - (self.grid_cols - 1) * self.block_margin) / self.grid_cols
        self.block_height = (self.grid_area_height - (self.grid_rows - 1) * self.block_margin) / self.grid_rows
        self.grid_start_x = (self.screen_width - self.grid_area_width) / 2
        self.grid_start_y = (self.screen_height - self.grid_area_height) / 2 + 10

        # --- Game Mechanics Parameters ---
        self.MAX_STEPS = 1000
        self.INITIAL_RESOURCES = 20
        self.STABLE, self.CORRUPTED = 0, 1
        self.REPAIR_SUCCESS_CHANCE = 0.9
        self.CASCADE_FIX_CHANCE = 0.3
        self.CASCADE_CORRUPT_CHANCE = 0.1
        self.MOVE_COOLDOWN_FRAMES = 4 # Cooldown for holding down a move key

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.grid_state = []
        self.cursor_pos = [0, 0]
        self.resources = 0
        self.particles = []
        self.data_traces = []
        self.move_cooldown = 0
        
        self._initialize_data_traces()
        # self.reset() is called by the environment wrapper

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.resources = self.INITIAL_RESOURCES
        self.cursor_pos = [self.grid_cols // 2, self.grid_rows // 2]
        self.particles.clear()
        self.move_cooldown = 0

        # Set up the grid
        self.grid_state = [[self.STABLE for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]
        
        # Introduce corruption
        num_corrupted = min(self._initial_corruption_level, self.grid_cols * self.grid_rows)
        
        # Use np_random for reproducibility
        all_indices = np.arange(self.grid_cols * self.grid_rows)
        self.np_random.shuffle(all_indices)
        corrupted_indices = all_indices[:num_corrupted]

        for index in corrupted_indices:
            row = index // self.grid_cols
            col = index % self.grid_cols
            self.grid_state[row][col] = self.CORRUPTED
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.1  # Cost of taking a step

        # Unpack factorized action
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Player Actions ---
        self._handle_movement(movement)
        
        if space_pressed and self.resources > 0:
            # // Sound effect placeholder: play('repair_start')
            self.resources -= 1
            stabilized_count, corrupted_count = self._trigger_repair_cascade()
            reward += stabilized_count * 1.0  # Reward for fixing blocks
            reward -= corrupted_count * 0.5 # Penalty for causing new corruption
        
        # --- Update Game World ---
        self._update_particles()
        self._update_data_traces()
        
        # --- Check for Termination ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        # Apply terminal rewards
        if terminated:
            if self.win_state:
                reward += 100
                # // Sound effect placeholder: play('win_game')
                GameEnv._wins_at_current_level += 1
                if GameEnv._wins_at_current_level >= GameEnv.DIFFICULTY_INCREASE_THRESHOLD:
                    GameEnv._initial_corruption_level += 1
                    GameEnv._wins_at_current_level = 0
            else: # Loss
                reward -= 100
                # // Sound effect placeholder: play('lose_game')
        
        self.score += reward
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_movement(self, movement_action):
        if self.move_cooldown > 0:
            self.move_cooldown -= 1
            return

        dx, dy = 0, 0
        if movement_action == 1: dy = -1  # Up
        elif movement_action == 2: dy = 1   # Down
        elif movement_action == 3: dx = -1  # Left
        elif movement_action == 4: dx = 1   # Right
        
        if dx != 0 or dy != 0:
            self.cursor_pos[0] = (self.cursor_pos[0] + dx) % self.grid_cols
            self.cursor_pos[1] = (self.cursor_pos[1] + dy) % self.grid_rows
            self.move_cooldown = self.MOVE_COOLDOWN_FRAMES
            # // Sound effect placeholder: play('cursor_move')

    def _trigger_repair_cascade(self):
        cx, cy = self.cursor_pos
        stabilized_count = 0
        corrupted_count = 0
        
        # --- Create Visual Effect ---
        self._create_shockwave(cx, cy)

        # Queue changes to avoid read-write conflicts during cascade
        changes = []

        # 1. Attempt to fix the target block
        if self.grid_state[cy][cx] == self.CORRUPTED:
            if self.np_random.random() < self.REPAIR_SUCCESS_CHANCE:
                changes.append(((cy, cx), self.STABLE))
        
        # 2. Affect neighbors
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < self.grid_cols and 0 <= ny < self.grid_rows:
                neighbor_state = self.grid_state[ny][nx]
                if neighbor_state == self.CORRUPTED:
                    if self.np_random.random() < self.CASCADE_FIX_CHANCE:
                        changes.append(((ny, nx), self.STABLE))
                elif neighbor_state == self.STABLE:
                    if self.np_random.random() < self.CASCADE_CORRUPT_CHANCE:
                        changes.append(((ny, nx), self.CORRUPTED))
        
        # 3. Apply changes
        for (r, c), new_state in changes:
            if self.grid_state[r][c] != new_state:
                if new_state == self.STABLE:
                    stabilized_count += 1
                else:
                    corrupted_count += 1
                self.grid_state[r][c] = new_state
        
        return stabilized_count, corrupted_count

    def _check_termination(self):
        is_all_stable = all(self.grid_state[r][c] == self.STABLE for r in range(self.grid_rows) for c in range(self.grid_cols))
        
        if is_all_stable:
            self.game_over = True
            self.win_state = True
        elif self.resources <= 0:
            self.game_over = True
            
        return self.game_over

    def _get_observation(self):
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        # Draw faint grid lines
        for i in range(self.grid_cols + 1):
            x = self.grid_start_x + i * (self.block_width + self.block_margin) - self.block_margin / 2
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (x, self.grid_start_y), (x, self.grid_start_y + self.grid_area_height), 1)
        for i in range(self.grid_rows + 1):
            y = self.grid_start_y + i * (self.block_height + self.block_margin) - self.block_margin / 2
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (self.grid_start_x, y), (self.grid_start_x + self.grid_area_width, y), 1)

        # Draw data traces
        for trace in self.data_traces:
            pygame.draw.rect(self.screen, trace['color'], trace['rect'])

    def _render_game_elements(self):
        # Render blocks
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                x = self.grid_start_x + c * (self.block_width + self.block_margin)
                y = self.grid_start_y + r * (self.block_height + self.block_margin)
                rect = pygame.Rect(x, y, self.block_width, self.block_height)
                
                if self.grid_state[r][c] == self.STABLE:
                    self._draw_glow_rect(rect, self.COLOR_STABLE, 10)
                else: # CORRUPTED
                    flash_alpha = (math.sin(self.steps * 0.2) + 1) / 2
                    color = (
                        self.COLOR_CORRUPTED[0],
                        int(50 + 100 * flash_alpha),
                        int(50 + 100 * flash_alpha)
                    )
                    self._draw_glow_rect(rect, color, 15)
        
        # Render particles
        self._update_and_draw_particles()

        # Render cursor
        cursor_x_pos = self.grid_start_x + self.cursor_pos[0] * (self.block_width + self.block_margin)
        cursor_y_pos = self.grid_start_y + self.cursor_pos[1] * (self.block_height + self.block_margin)
        cursor_rect = pygame.Rect(cursor_x_pos, cursor_y_pos, self.block_width, self.block_height)
        
        pulse = (math.sin(self.steps * 0.15) + 1) / 2 * 5
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect.inflate(pulse, pulse), 2)
        
    def _render_ui(self):
        # Draw UI background panel
        ui_panel = pygame.Surface((self.screen_width, 40), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, self.screen_height - 40))
        
        # Render UI text
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (15, self.screen_height - 30))
        
        resource_text = self.font_ui.render(f"REPAIR CHARGES: {self.resources}", True, self.COLOR_UI_TEXT)
        self.screen.blit(resource_text, (self.screen_width - resource_text.get_width() - 15, self.screen_height - 30))

        # Render Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            message = "SYSTEM STABLE" if self.win_state else "SYSTEM FAILURE"
            color = self.COLOR_STABLE if self.win_state else self.COLOR_CORRUPTED
            
            text_surf = self.font_game_over.render(message, True, color)
            text_rect = text_surf.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            
            self.screen.blit(overlay, (0, 0))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "resources": self.resources,
            "cursor_pos": list(self.cursor_pos),
            "corruption_level": self._initial_corruption_level
        }

    def close(self):
        pygame.quit()

    # --- Visual Effect Helpers ---

    def _initialize_data_traces(self):
        self.data_traces.clear()
        for _ in range(20):
            is_horizontal = random.choice([True, False])
            if is_horizontal:
                rect = pygame.Rect(random.randint(0, self.screen_width), random.randint(0, self.screen_height), random.randint(20, 50), 1)
                speed = random.uniform(1, 3) * random.choice([-1, 1])
            else:
                rect = pygame.Rect(random.randint(0, self.screen_width), random.randint(0, self.screen_height), 1, random.randint(20, 50))
                speed = random.uniform(1, 3) * random.choice([-1, 1])
            self.data_traces.append({'rect': rect, 'speed': speed, 'is_horizontal': is_horizontal, 'color': (20, 50, 90, random.randint(50, 150))})

    def _update_data_traces(self):
        for trace in self.data_traces:
            if trace['is_horizontal']:
                trace['rect'].x += trace['speed']
                if trace['rect'].left > self.screen_width: trace['rect'].right = 0
                if trace['rect'].right < 0: trace['rect'].left = self.screen_width
            else:
                trace['rect'].y += trace['speed']
                if trace['rect'].top > self.screen_height: trace['rect'].bottom = 0
                if trace['rect'].bottom < 0: trace['rect'].top = self.screen_height

    def _create_shockwave(self, grid_x, grid_y):
        center_x = self.grid_start_x + grid_x * (self.block_width + self.block_margin) + self.block_width / 2
        center_y = self.grid_start_y + grid_y * (self.block_height + self.block_margin) + self.block_height / 2
        
        # Shockwave particle
        self.particles.append({
            'pos': [center_x, center_y], 'vel': [0, 0], 'type': 'shockwave',
            'radius': 10, 'max_radius': 100, 'lifespan': 20, 'max_lifespan': 20
        })
        # Debris particles
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 6)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 31)
            self.particles.append({
                'pos': [center_x, center_y], 'vel': vel, 'type': 'debris',
                'size': self.np_random.uniform(1, 3), 'lifespan': lifespan, 'max_lifespan': lifespan,
                'color': random.choice([self.COLOR_CURSOR, (180, 255, 255)])
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)
                continue
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            if p['type'] == 'shockwave':
                p['radius'] += (p['max_radius'] - p['radius']) * 0.2
            else: # Debris
                p['vel'][0] *= 0.95 # friction
                p['vel'][1] *= 0.95

    def _update_and_draw_particles(self):
        for p in self.particles:
            alpha = p['lifespan'] / p['max_lifespan']
            if p['type'] == 'shockwave':
                color = (*self.COLOR_CURSOR, int(alpha * 150))
                pygame.gfxdraw.aacircle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), color)
                pygame.gfxdraw.aacircle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius'] - 2), color)
            else: # Debris
                color = (*p['color'], int(alpha * 255))
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                size = int(p['size'])
                if size > 0:
                    surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                    pygame.draw.circle(surf, color, (size, size), size)
                    self.screen.blit(surf, (pos[0]-size, pos[1]-size), special_flags=pygame.BLEND_RGBA_ADD)

    def _draw_glow_rect(self, rect, color, glow_size):
        # Draw the solid center
        pygame.draw.rect(self.screen, color, rect, border_radius=3)
        # Draw the glow effect
        for i in range(glow_size, 0, -1):
            alpha = 1 - (i / glow_size)
            glow_color = (*color, int(alpha * 40))
            glow_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, glow_color, (0,0, *rect.size), border_radius=5)
            self.screen.blit(glow_surf, rect.topleft, special_flags=pygame.BLEND_RGBA_ADD)


if __name__ == '__main__':
    # --- Manual Play Example ---
    # This block will not run in a headless environment but is useful for local testing.
    # To run this, you may need to comment out the `os.environ.setdefault` line at the top.
    try:
        env = GameEnv()
        obs, info = env.reset()
        
        screen = pygame.display.set_mode((640, 400))
        pygame.display.set_caption("Motherboard Repair")
        clock = pygame.time.Clock()
        
        running = True
        total_reward = 0
        
        while running:
            movement, space, shift = 0, 0, 0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- ENV RESET ---")

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Render the observation from the environment
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            
            pygame.display.flip()
            
            if terminated or truncated:
                print(f"Episode Finished. Final Score: {info['score']:.2f}, Steps: {info['steps']}")
                # Wait for a moment before auto-resetting
                pygame.time.wait(2000)
                obs, info = env.reset()
                total_reward = 0
                print("--- ENV RESET ---")

            clock.tick(30) # Run at 30 FPS
            
        env.close()
    except pygame.error as e:
        print(f"Pygame error: {e}")
        print("This might be because the environment is running in headless mode.")
        print("To run the manual test, you might need to comment out the line:")
        print("os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')")