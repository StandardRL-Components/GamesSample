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


# Set Pygame to run in a headless mode
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the cursor. Press space to place a defensive block."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your fortress by placing blocks to redirect waves of enemies in this isometric strategy game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # Game constants
        self.GRID_WIDTH = 22
        self.GRID_HEIGHT = 14
        self.ISO_TILE_WIDTH = 32
        self.ISO_TILE_HEIGHT = 16
        self.ISO_BLOCK_DEPTH = 20
        self.MAX_STEPS = 3000
        self.MAX_WAVES = 20
        
        self.screen_origin_x = self.screen_width / 2
        self.screen_origin_y = self.screen_height * 0.25

        # Colors
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_GRID = (40, 45, 50)
        self.COLOR_FORTRESS = (100, 200, 100)
        self.COLOR_BLOCK = (120, 120, 160)
        self.COLOR_ENEMY = (220, 50, 50)
        self.COLOR_PROJECTILE = (255, 200, 0)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_BAR_BG = (80, 20, 20)
        self.COLOR_HEALTH_BAR_FG = (20, 180, 20)
        self.COLOR_GAMEOVER = (200, 30, 30)
        self.COLOR_WIN = (30, 200, 30)

        # Fonts
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.fortress_health = 0
        self.max_fortress_health = 100
        self.fortress_coords = set()
        self.blocks = {}
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = [0, 0]
        self.current_wave = 0
        self.wave_cooldown = 0
        self.space_was_held = False

        self.np_random = None

        # self.reset() is called by the gym wrapper, no need to call it here.
        # self.validate_implementation() # This can be useful for debugging but should not be in the final constructor

    def _to_iso(self, grid_x, grid_y):
        screen_x = self.screen_origin_x + (grid_x - grid_y) * (self.ISO_TILE_WIDTH / 2)
        screen_y = self.screen_origin_y + (grid_x + grid_y) * (self.ISO_TILE_HEIGHT / 2)
        return int(screen_x), int(screen_y)

    def _draw_iso_cube(self, surface, grid_pos, color, depth_override=None):
        depth = depth_override if depth_override is not None else self.ISO_BLOCK_DEPTH
        base_x, base_y = self._to_iso(grid_pos[0], grid_pos[1])
        
        w_half = self.ISO_TILE_WIDTH / 2
        h_half = self.ISO_TILE_HEIGHT / 2

        top_face = [
            (base_x, base_y - depth),
            (base_x + w_half, base_y - h_half - depth),
            (base_x, base_y - self.ISO_TILE_HEIGHT - depth),
            (base_x - w_half, base_y - h_half - depth)
        ]
        
        # FIX: The generator expression was consumed on the first use, causing the second call to get an empty list.
        # Create lists directly to be reused.
        side_color_l = [int(max(0, c - 40)) for c in color]
        side_color_r = [int(max(0, c - 60)) for c in color]

        left_face = [
            (base_x - w_half, base_y - h_half - depth),
            (base_x, base_y - self.ISO_TILE_HEIGHT - depth),
            (base_x, base_y),
            (base_x - w_half, base_y - h_half)
        ]
        right_face = [
            (base_x + w_half, base_y - h_half - depth),
            (base_x, base_y - self.ISO_TILE_HEIGHT - depth),
            (base_x, base_y),
            (base_x + w_half, base_y - h_half)
        ]

        pygame.gfxdraw.filled_polygon(surface, left_face, side_color_l)
        pygame.gfxdraw.aapolygon(surface, left_face, side_color_l)
        pygame.gfxdraw.filled_polygon(surface, right_face, side_color_r)
        pygame.gfxdraw.aapolygon(surface, right_face, side_color_r)
        pygame.gfxdraw.filled_polygon(surface, top_face, color)
        pygame.gfxdraw.aapolygon(surface, top_face, color)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
            random.seed(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.fortress_health = self.max_fortress_health
        
        self.fortress_coords = set()
        fs = 1 # fortress size
        for x in range(-fs, fs + 1):
            for y in range(-fs, fs + 1):
                self.fortress_coords.add((self.GRID_WIDTH // 2 + x, self.GRID_HEIGHT // 2 + y))
        
        self.blocks = {}
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = [self.GRID_WIDTH // 2, 0]
        self.current_wave = 0
        self.wave_cooldown = 120 # Time before first wave
        self.space_was_held = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        
        if not self.game_over:
            # Unpack factorized action
            movement = action[0]
            space_held = action[1] == 1
            
            # 1. Handle Player Input
            self._update_cursor(movement)
            reward += self._place_block(space_held)
            self.space_was_held = space_held

            # 2. Update Game Logic
            enemy_reward = self._update_enemies()
            projectile_reward = self._update_projectiles()
            self._update_particles()
            reward += enemy_reward + projectile_reward

            # 3. Handle Waves
            if not self.enemies and self.wave_cooldown <= 0:
                if self.current_wave > 0: # Don't reward for wave 0
                    reward += 10 # Survived a wave
                self.current_wave += 1
                if self.current_wave > self.MAX_WAVES:
                    self.win = True
                else:
                    self._spawn_wave()
                    self.wave_cooldown = 150 # Cooldown between waves
            
            if self.wave_cooldown > 0:
                self.wave_cooldown -= 1

        # 4. Update State and Check Termination
        self.steps += 1
        self.score += reward
        
        terminated = self.fortress_health <= 0 or self.win
        truncated = self.steps >= self.MAX_STEPS
        
        if (terminated or truncated) and not self.game_over:
            self.game_over = True
            if self.win:
                reward += 100
                self.score += 100
            elif self.fortress_health <= 0:
                reward -= 100
                self.score -= 100
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_cursor(self, movement):
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

    def _place_block(self, space_held):
        cursor_tuple = tuple(self.cursor_pos)
        if space_held and not self.space_was_held:
            if cursor_tuple not in self.blocks and cursor_tuple not in self.fortress_coords:
                self.blocks[cursor_tuple] = 10 # Block health
                # sfx: block_place
                return -0.01
        return 0

    def _spawn_wave(self):
        num_enemies = self.current_wave
        speed = 0.03 + 0.005 * math.floor((self.current_wave - 1) / 2)
        
        for _ in range(num_enemies):
            side = random.randint(0, 3)
            if side == 0: # Top
                pos = [random.uniform(0, self.GRID_WIDTH-1), -1]
            elif side == 1: # Bottom
                pos = [random.uniform(0, self.GRID_WIDTH-1), self.GRID_HEIGHT]
            elif side == 2: # Left
                pos = [-1, random.uniform(0, self.GRID_HEIGHT-1)]
            else: # Right
                pos = [self.GRID_WIDTH, random.uniform(0, self.GRID_HEIGHT-1)]

            self.enemies.append({
                "pos": pos,
                "speed": speed,
                "attack_cooldown": random.randint(60, 120),
            })

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            # Find closest fortress point
            min_dist_sq = float('inf')
            target_pos = None
            for f_pos in self.fortress_coords:
                dist_sq = (enemy["pos"][0] - f_pos[0])**2 + (enemy["pos"][1] - f_pos[1])**2
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    target_pos = f_pos
            
            # Movement
            if target_pos:
                dx = target_pos[0] - enemy["pos"][0]
                dy = target_pos[1] - enemy["pos"][1]
                dist = math.hypot(dx, dy)
                
                if dist > 1: # Don't move if very close
                    next_pos_x = enemy["pos"][0] + (dx / dist) * enemy["speed"]
                    next_pos_y = enemy["pos"][1] + (dy / dist) * enemy["speed"]
                    
                    # Simple collision with blocks
                    grid_pos_next = (int(next_pos_x), int(next_pos_y))
                    if grid_pos_next not in self.blocks:
                        enemy["pos"] = [next_pos_x, next_pos_y]

            # Attack
            enemy["attack_cooldown"] -= 1
            if enemy["attack_cooldown"] <= 0 and target_pos:
                # sfx: enemy_fire
                enemy["attack_cooldown"] = random.randint(90, 150)
                proj_dx = target_pos[0] - enemy["pos"][0]
                proj_dy = target_pos[1] - enemy["pos"][1]
                proj_dist = math.hypot(proj_dx, proj_dy)
                if proj_dist > 0:
                    self.projectiles.append({
                        "pos": list(enemy["pos"]),
                        "vel": [(proj_dx/proj_dist)*2, (proj_dy/proj_dist)*2],
                        "owner": "enemy"
                    })
        return reward

    def _update_projectiles(self):
        reward = 0
        for p in self.projectiles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            
            grid_pos = (int(p["pos"][0]), int(p["pos"][1]))

            # Out of bounds check
            if not (0 <= grid_pos[0] < self.GRID_WIDTH and 0 <= grid_pos[1] < self.GRID_HEIGHT):
                self.projectiles.remove(p)
                continue

            # Collision check
            hit = False
            if grid_pos in self.blocks:
                # sfx: block_hit
                self.blocks[grid_pos] -= 5
                if self.blocks[grid_pos] <= 0:
                    del self.blocks[grid_pos]
                hit = True
                reward -= 0.1
            elif grid_pos in self.fortress_coords:
                # sfx: fortress_hit
                self.fortress_health -= 5
                hit = True
                reward -= 1.0

            if hit:
                self._create_explosion(self._to_iso(p["pos"][0], p["pos"][1]), self.COLOR_PROJECTILE)
                if p in self.projectiles:
                    self.projectiles.remove(p)
        return reward

    def _create_explosion(self, pos, color):
        for _ in range(15):
            self.particles.append({
                'pos': list(pos),
                'vel': [random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5)],
                'life': random.randint(15, 30),
                'color': color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid floor
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                sx, sy = self._to_iso(x, y)
                points = [
                    (sx, sy),
                    (sx + self.ISO_TILE_WIDTH / 2, sy + self.ISO_TILE_HEIGHT / 2),
                    (sx, sy + self.ISO_TILE_HEIGHT),
                    (sx - self.ISO_TILE_WIDTH / 2, sy + self.ISO_TILE_HEIGHT / 2)
                ]
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_GRID)

        # Create a sorted list of all objects to draw for correct isometric layering
        draw_list = []
        for pos in self.fortress_coords:
            draw_list.append({'type': 'fortress', 'pos': pos})
        for pos in self.blocks.keys():
            draw_list.append({'type': 'block', 'pos': pos})
        for enemy in self.enemies:
            draw_list.append({'type': 'enemy', 'pos': enemy['pos']})
        
        # Sort by grid y then grid x for painter's algorithm
        draw_list.sort(key=lambda item: (item['pos'][1], item['pos'][0]))

        for item in draw_list:
            if item['type'] == 'fortress':
                self._draw_iso_cube(self.screen, item['pos'], self.COLOR_FORTRESS)
            elif item['type'] == 'block':
                health_ratio = self.blocks[item['pos']] / 10.0
                color = [int(c * health_ratio) for c in self.COLOR_BLOCK]
                self._draw_iso_cube(self.screen, item['pos'], color)
            elif item['type'] == 'enemy':
                sx, sy = self._to_iso(item['pos'][0], item['pos'][1])
                sy -= self.ISO_BLOCK_DEPTH # Place on top of tiles
                pygame.gfxdraw.filled_circle(self.screen, int(sx), int(sy), 8, self.COLOR_ENEMY)
                pygame.gfxdraw.aacircle(self.screen, int(sx), int(sy), 8, self.COLOR_ENEMY)

        # Draw projectiles on top
        for p in self.projectiles:
            sx, sy = self._to_iso(p['pos'][0], p['pos'][1])
            sy -= self.ISO_BLOCK_DEPTH
            pygame.draw.circle(self.screen, self.COLOR_PROJECTILE, (int(sx), int(sy)), 4)
        
        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 30.0))))
            color = p['color'] + (alpha,)
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, (0, 0, 4, 4))
            self.screen.blit(temp_surf, (int(p['pos'][0]-2), int(p['pos'][1]-2)))

        # Draw cursor
        cursor_sx, cursor_sy = self._to_iso(self.cursor_pos[0], self.cursor_pos[1])
        w_half, h_half = self.ISO_TILE_WIDTH / 2, self.ISO_TILE_HEIGHT / 2
        cursor_points = [
            (cursor_sx - w_half, cursor_sy + h_half), (cursor_sx, cursor_sy),
            (cursor_sx + w_half, cursor_sy + h_half), (cursor_sx, cursor_sy + self.ISO_TILE_HEIGHT)
        ]
        pygame.draw.aalines(self.screen, self.COLOR_CURSOR, True, cursor_points, 2)
    
    def _render_ui(self):
        # Fortress Health Bar
        health_ratio = max(0, self.fortress_health / self.max_fortress_health)
        bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (10, 10, int(bar_width * health_ratio), 20))
        health_text = self.font_small.render(f"Fortress: {int(self.fortress_health)}/{self.max_fortress_health}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Wave Counter
        wave_text_str = f"Wave: {self.current_wave}/{self.MAX_WAVES}"
        if not self.enemies and self.wave_cooldown > 0 and not self.win:
             wave_text_str = f"Next wave in: {self.wave_cooldown // 30 + 1}"
        wave_text = self.font_small.render(wave_text_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.screen_width - wave_text.get_width() - 10, 10))

        # Score
        score_text = self.font_small.render(f"Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.screen_width - score_text.get_width() - 10, 35))

        # Game Over / Win message
        if self.game_over:
            if self.win:
                msg = "YOU WIN!"
                color = self.COLOR_WIN
            else:
                msg = "GAME OVER"
                color = self.COLOR_GAMEOVER
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "fortress_health": self.fortress_health,
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # Re-enable the display for direct play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Isometric Fortress Defense")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0 # no-op
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        
        # The environment returns a transposed array, so we fix it for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Run at 30 FPS
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Wave: {info['wave']}")
            # Wait for a moment before allowing reset
            pygame.time.wait(2000)
            obs, info = env.reset()

    env.close()