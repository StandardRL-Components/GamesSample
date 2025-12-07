import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

# Generated: 2025-08-26T12:04:18.160576
# Source Brief: brief_00860.md
# Brief Index: 860
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Use colored blasts to create chain reactions on walls and destroy all enemies before you run out of ammo."
    )
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to fire a blast at a wall. Press shift to cycle through blast colors."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array", start_level=1):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 20
        self.GRID_W, self.GRID_H = self.WIDTH // self.GRID_SIZE, self.HEIGHT // self.GRID_SIZE

        # Colors (Neon Cyberpunk)
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_WALL = (40, 50, 70)
        self.COLOR_ENEMY = (120, 20, 40)
        self.COLOR_ENEMY_DAMAGED = (255, 100, 100)
        self.COLOR_CURSOR = (255, 255, 255)
        self.BLAST_COLORS = [
            (255, 0, 80),    # Red/Pink
            (0, 255, 120),   # Green
            (0, 150, 255),   # Blue
        ]
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.MAX_EPISODE_STEPS = 2000

        # Game Mechanics
        self.CURSOR_SPEED = 10
        self.ACTION_COOLDOWN = 5 # steps
        self.ENEMY_BASE_HEALTH = 3

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_big = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- State Variables ---
        self.start_level = start_level
        self.level = 1
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level_won = False

        self.grid = None
        self.color_map = None
        self.enemies = []
        
        self.cursor_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.blast_count = 0
        self.selected_color_idx = 0
        
        self.fire_cooldown = 0
        self.shift_cooldown = 0

        self.particles = []
        self.effects = [] # For blast expansions, chain lines, etc.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.level = options.get('level', self.start_level) if options else self.start_level

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level_won = False

        self._generate_level()

        self.cursor_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.selected_color_idx = 0
        
        self.fire_cooldown = 0
        self.shift_cooldown = 0

        self.particles = []
        self.effects = []

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.grid = np.zeros((self.GRID_H, self.GRID_W), dtype=np.uint8)
        self.color_map = np.zeros((self.GRID_H, self.GRID_W), dtype=np.uint8)
        self.enemies.clear()

        # Determine level parameters
        num_enemies = min(10, 2 + self.level)
        self.blast_count = max(5, 20 - (self.level - 1) // 3)
        
        # Place wall structures (simple rectangles)
        num_walls = self.np_random.integers(4, 8)
        for _ in range(num_walls):
            w, h = self.np_random.integers(2, self.GRID_W//3, size=2)
            x, y = self.np_random.integers(0, self.GRID_W - w), self.np_random.integers(0, self.GRID_H - h)
            self.grid[y:y+h, x:x+w] = 1 # 1 for wall

        # Place enemies in empty spaces
        enemy_id_counter = 0
        attempts = 0
        while len(self.enemies) < num_enemies and attempts < 100:
            attempts += 1
            ex, ey = self.np_random.integers(1, self.GRID_W-1), self.np_random.integers(1, self.GRID_H-1)
            if self.grid[ey, ex] == 0:
                self.grid[ey, ex] = 2 # 2 for enemy
                rect = pygame.Rect(ex * self.GRID_SIZE, ey * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
                self.enemies.append({
                    'id': enemy_id_counter,
                    'rect': rect,
                    'health': self.ENEMY_BASE_HEALTH,
                    'max_health': self.ENEMY_BASE_HEALTH,
                    'grid_pos': (ex, ey),
                    'last_damage_time': -100
                })
                enemy_id_counter += 1
        
        # Ensure borders are walls
        self.grid[0, :] = self.grid[-1, :] = 1
        self.grid[:, 0] = self.grid[:, -1] = 1

    def step(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = 0
        self.game_over = False
        self.level_won = False
        truncated = False

        # --- Update Cooldowns ---
        if self.fire_cooldown > 0: self.fire_cooldown -= 1
        if self.shift_cooldown > 0: self.shift_cooldown -= 1

        # --- Handle Actions ---
        # Color Cycling (Shift)
        if shift_pressed and self.shift_cooldown == 0:
            self.selected_color_idx = (self.selected_color_idx + 1) % len(self.BLAST_COLORS)
            self.shift_cooldown = self.ACTION_COOLDOWN
            # SFX: color_swap.wav

        # Cursor Movement
        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED  # Up
        elif movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED  # Down
        elif movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED  # Left
        elif movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.HEIGHT)

        # Firing (Space)
        if space_pressed and self.fire_cooldown == 0 and self.blast_count > 0:
            self.fire_cooldown = self.ACTION_COOLDOWN
            self.blast_count -= 1
            reward -= 0.1 # Penalty for using a blast
            # SFX: fire_blast.wav

            grid_x, grid_y = int(self.cursor_pos[0] / self.GRID_SIZE), int(self.cursor_pos[1] / self.GRID_SIZE)
            
            if 0 <= grid_x < self.GRID_W and 0 <= grid_y < self.GRID_H and self.grid[grid_y, grid_x] == 1:
                blast_reward = self._handle_chain_reaction(grid_x, grid_y)
                reward += blast_reward
            else:
                # Dud shot effect
                self._create_particles(self.cursor_pos, 10, self.COLOR_WALL)

        # --- Update Game State ---
        self._update_effects_and_particles()
        self.steps += 1
        
        # --- Check Termination Conditions ---
        remaining_enemies = sum(1 for e in self.enemies if e['health'] > 0)
        
        if remaining_enemies == 0:
            self.level_won = True
            self.game_over = True
            reward += 100
        elif self.blast_count <= 0 and not any(e['type'] == 'blast' for e in self.effects):
            self.game_over = True
            reward -= 50
        elif self.steps >= self.MAX_EPISODE_STEPS:
            self.game_over = True
            truncated = True

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            self.game_over,
            truncated,
            self._get_info()
        )

    def _handle_chain_reaction(self, start_gx, start_gy):
        event_reward = 0
        color_id = self.selected_color_idx + 1 # 1, 2, 3
        
        # Only proceed if the wall is neutral or a different color
        if self.color_map[start_gy, start_gx] == color_id:
            return 0 # No change, no reward

        # Tag the initial wall
        if self.color_map[start_gy, start_gx] == 0:
            event_reward += 1 # Reward for tagging a new wall
        self.color_map[start_gy, start_gx] = color_id
        
        # Add blast expansion visual effect
        blast_center = ((start_gx + 0.5) * self.GRID_SIZE, (start_gy + 0.5) * self.GRID_SIZE)
        self.effects.append({'type': 'blast', 'pos': blast_center, 'radius': 0, 'max_radius': self.GRID_SIZE * 1.5, 'life': 20, 'color': self.BLAST_COLORS[self.selected_color_idx]})

        # --- BFS to find the connected color blob ---
        q = deque([(start_gx, start_gy)])
        visited = set([(start_gx, start_gy)])
        blob = []

        while q:
            gx, gy = q.popleft()
            blob.append((gx, gy))
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < self.GRID_W and 0 <= ny < self.GRID_H and (nx, ny) not in visited:
                    if self.grid[ny, nx] == 1 and self.color_map[ny, nx] == color_id:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        
        # --- Process the blob ---
        if len(blob) > 1:
            event_reward += 5 # Chain reaction bonus
            # SFX: chain_reaction.wav
            # Create line effects for the chain
            for i in range(len(blob) - 1):
                p1 = ((blob[i][0] + 0.5) * self.GRID_SIZE, (blob[i][1] + 0.5) * self.GRID_SIZE)
                p2 = ((blob[i+1][0] + 0.5) * self.GRID_SIZE, (blob[i+1][1] + 0.5) * self.GRID_SIZE)
                self.effects.append({'type': 'line', 'start': p1, 'end': p2, 'life': 25, 'color': self.BLAST_COLORS[self.selected_color_idx]})

        # --- Damage adjacent enemies ---
        damaged_enemy_ids = set()
        for gx, gy in blob:
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < self.GRID_W and 0 <= ny < self.GRID_H and self.grid[ny, nx] == 2:
                    for enemy in self.enemies:
                        if enemy['grid_pos'] == (nx, ny) and enemy['id'] not in damaged_enemy_ids:
                            damaged_enemy_ids.add(enemy['id'])
        
        for enemy_id in damaged_enemy_ids:
            enemy = next(e for e in self.enemies if e['id'] == enemy_id)
            if enemy['health'] > 0:
                enemy['health'] -= 1
                enemy['last_damage_time'] = self.steps
                # SFX: enemy_damage.wav
                self._create_particles(enemy['rect'].center, 20, self.COLOR_ENEMY_DAMAGED)
                if enemy['health'] <= 0:
                    event_reward += 10 # Destruction bonus
                    # SFX: enemy_explode.wav
                    self._create_particles(enemy['rect'].center, 100, self.BLAST_COLORS[self.selected_color_idx], 2)
        
        return event_reward

    def _update_effects_and_particles(self):
        # Update particles
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

        # Update effects
        for e in self.effects:
            e['life'] -= 1
            if e['type'] == 'blast':
                e['radius'] += e['max_radius'] / 20
        self.effects = [e for e in self.effects if e['life'] > 0]

    def _create_particles(self, pos, count, color, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': np.array(pos, dtype=np.float32), 'vel': vel, 'life': life, 'color': color})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw walls and colored segments
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                rect = pygame.Rect(x * self.GRID_SIZE, y * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
                if self.grid[y, x] == 1: # Wall
                    color_id = self.color_map[y, x]
                    color = self.BLAST_COLORS[color_id - 1] if color_id > 0 else self.COLOR_WALL
                    pygame.draw.rect(self.screen, color, rect)
                    if color_id > 0: # Add glow to colored walls
                        glow_rect = rect.inflate(4, 4)
                        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
                        pygame.draw.rect(glow_surf, color + (30,), (0,0, *glow_rect.size), border_radius=3)
                        self.screen.blit(glow_surf, glow_rect.topleft)


        # Draw enemies
        for enemy in self.enemies:
            if enemy['health'] > 0:
                is_damaged = (self.steps - enemy['last_damage_time']) < 10
                color = self.COLOR_ENEMY_DAMAGED if is_damaged else self.COLOR_ENEMY
                pygame.draw.rect(self.screen, color, enemy['rect'])
                # Health bar
                health_ratio = enemy['health'] / enemy['max_health']
                health_bar_rect = pygame.Rect(enemy['rect'].left, enemy['rect'].top - 6, enemy['rect'].width, 4)
                pygame.draw.rect(self.screen, (50,0,0), health_bar_rect)
                health_bar_rect.width = int(health_bar_rect.width * health_ratio)
                pygame.draw.rect(self.screen, (255,0,0), health_bar_rect)

        # Draw effects
        for e in self.effects:
            if e['type'] == 'blast':
                alpha = int(255 * (e['life'] / 20))
                if alpha > 0:
                    # Outer glow
                    pygame.gfxdraw.aacircle(self.screen, int(e['pos'][0]), int(e['pos'][1]), int(e['radius'] * 1.2), e['color'] + (int(alpha/4),))
                    # Inner circle
                    pygame.gfxdraw.aacircle(self.screen, int(e['pos'][0]), int(e['pos'][1]), int(e['radius']), e['color'] + (alpha,))
            elif e['type'] == 'line':
                alpha = int(255 * (e['life'] / 25))
                if alpha > 0:
                    pygame.draw.aaline(self.screen, e['color'] + (alpha,), e['start'], e['end'], 2)

        # Draw particles
        for p in self.particles:
            alpha = max(0, int(255 * (p['life'] / 30)))
            if len(p['color']) == 3:
                color_with_alpha = p['color'] + (alpha,)
            else:
                color_with_alpha = p['color'][:3] + (alpha,)
            size = int(p['life'] / 10) + 1
            rect = pygame.Rect(int(p['pos'][0] - size/2), int(p['pos'][1] - size/2), size, size)
            pygame.draw.rect(self.screen, color_with_alpha, rect)

        # Draw cursor
        x, y = int(self.cursor_pos[0]), int(self.cursor_pos[1])
        color = self.BLAST_COLORS[self.selected_color_idx]
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (x - 10, y), (x - 5, y), 1)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (x + 5, y), (x + 10, y), 1)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (x, y - 10), (x, y - 5), 1)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (x, y + 5), (x, y + 10), 1)
        pygame.gfxdraw.aacircle(self.screen, x, y, 5, color)

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, pos, font, color, shadow_color=(0,0,0)):
            text_surf = font.render(text, True, color)
            shadow_surf = font.render(text, True, shadow_color)
            self.screen.blit(shadow_surf, (pos[0]+2, pos[1]+2))
            self.screen.blit(text_surf, pos)
        
        # Blasts remaining
        blast_text = f"BLASTS: {self.blast_count}"
        draw_text(blast_text, (10, 10), self.font_ui, self.COLOR_UI_TEXT)

        # Score
        score_text = f"SCORE: {int(self.score)}"
        text_w = self.font_ui.size(score_text)[0]
        draw_text(score_text, (self.WIDTH - text_w - 10, 10), self.font_ui, self.COLOR_UI_TEXT)

        # Level
        level_text = f"LEVEL {self.level}"
        text_w = self.font_ui.size(level_text)[0]
        draw_text(level_text, (self.WIDTH/2 - text_w/2, 10), self.font_ui, self.COLOR_UI_TEXT)

        # Current Color Indicator
        color = self.BLAST_COLORS[self.selected_color_idx]
        center_x, center_y = self.WIDTH // 2, self.HEIGHT - 30
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, 20, color)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, 20, (255,255,255))

        # Game Over / Win Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            msg = "LEVEL CLEAR" if self.level_won else "OUT OF BLASTS"
            color = (100, 255, 150) if self.level_won else (255, 100, 100)
            text_w, text_h = self.font_big.size(msg)
            draw_text(msg, (self.WIDTH/2 - text_w/2, self.HEIGHT/2 - text_h/2), self.font_big, color)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "blast_count": self.blast_count,
            "remaining_enemies": sum(1 for e in self.enemies if e['health'] > 0),
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        """Call this after __init__ to verify implementation."""
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3), f"Obs shape is {obs.shape}"
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        # Test observation space (after reset)
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # For manual play, we don't want the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    
    # Run validation on the created instance
    try:
        env.validate_implementation()
    except AssertionError as e:
        print(f"Validation Failed: {e}")
        env.close()
        exit()

    obs, info = env.reset()
    done = False
    
    # Pygame setup for manual play
    pygame.display.set_caption("Electric Graffiti War")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    running = True
    
    while running:
        # Action defaults
        movement = 0 # none
        space = 0 # released
        shift = 0 # released

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    done = False
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift = 1
                if event.key == pygame.K_SPACE:
                    space = 1

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4

        action = [movement, space, shift]
        
        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Run at 30 FPS

    env.close()