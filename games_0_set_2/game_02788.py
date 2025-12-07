
# Generated: 2025-08-27T21:25:57.956400
# Source Brief: brief_02788.md
# Brief Index: 2788

        
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
    An isometric fortress defense game.

    The player places blocks to build a fortress and defend against waves of enemies.
    The goal is to survive 10 waves. The game ends if the fortress is destroyed
    (all blocks are gone) or after a fixed number of steps.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing strings
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to place a block."
    )
    game_description = (
        "Build and defend an isometric fortress against waves of increasingly difficult enemies."
    )

    # Frame advance behavior
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 3000  # 100 seconds at 30 FPS, allows for ~10s per wave
        self.MAX_WAVES = 10

        # --- Grid and Isometric Projection ---
        self.GRID_WIDTH, self.GRID_HEIGHT = 20, 20
        self.TILE_W_HALF, self.TILE_H_HALF = 24, 12
        self.GRID_OFFSET_X = self.WIDTH // 2
        self.GRID_OFFSET_Y = 100

        # --- Colors ---
        self.COLOR_BG = (34, 32, 52)
        self.COLOR_GRID = (60, 56, 72)
        self.COLOR_CURSOR = (255, 255, 0, 150)
        self.COLOR_BLOCK = (0, 200, 120)
        self.COLOR_BLOCK_SIDE = (0, 150, 90)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_PROJECTILE = (0, 150, 255)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_UI_BG = (0, 0, 0, 128)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # --- Internal State ---
        # These are initialized in reset()
        self.steps = None
        self.score = None
        self.game_over = None
        self.win = None
        self.np_random = None
        self.cursor_pos = None
        self.fortress_blocks = None
        self.enemies = None
        self.projectiles = None
        self.particles = None
        self.wave_number = None
        self.inter_wave_timer = None
        self.blocks_remaining = None
        self.space_was_held = None
        self.last_reward = 0

        # --- Run initial reset and validation ---
        self.reset()
        # self.validate_implementation() # Uncomment for debugging

    def _iso_to_screen(self, grid_x, grid_y):
        """Converts grid coordinates to screen coordinates for isometric view."""
        screen_x = (grid_x - grid_y) * self.TILE_W_HALF + self.GRID_OFFSET_X
        screen_y = (grid_x + grid_y) * self.TILE_H_HALF + self.GRID_OFFSET_Y
        return int(screen_x), int(screen_y)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, seed = gym.utils.seeding.np_random(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.fortress_blocks = {}  # {(gx, gy): health}
        self.enemies = []
        self.projectiles = []
        self.particles = []

        self.wave_number = 0
        self.inter_wave_timer = 90  # 3 seconds at 30fps
        self.blocks_remaining = 50
        self.space_was_held = False
        self.last_reward = 0

        self._start_next_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return (
                self._get_observation(),
                0,
                True,
                False,
                self._get_info(),
            )

        self.steps += 1
        reward = 0

        # 1. Handle Input
        self._handle_input(action)

        # 2. Update Game Logic
        reward += self._update_enemies()
        reward += self._update_projectiles()
        self._update_fortress()
        self._update_particles()
        wave_end_reward = self._update_wave_logic()
        reward += wave_end_reward

        self.score += reward

        # 3. Check Termination
        terminated = False
        fortress_destroyed = len(self.fortress_blocks) == 0 and self.blocks_remaining <= 0
        
        if fortress_destroyed:
            reward -= 100
            self.game_over = True
            terminated = True
        elif self.win:
            reward += 100
            self.game_over = True
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        self.last_reward = reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, action):
        movement, space_held, _ = action
        
        # Move cursor
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # Place block
        space_pressed = space_held and not self.space_was_held
        if space_pressed:
            pos_tuple = tuple(self.cursor_pos)
            if self.blocks_remaining > 0 and pos_tuple not in self.fortress_blocks:
                self.fortress_blocks[pos_tuple] = 100  # Block health
                self.blocks_remaining -= 1
                # sfx: place_block.wav
                for _ in range(10):
                    self._create_particle(self._iso_to_screen(*pos_tuple), self.COLOR_BLOCK)

        self.space_was_held = space_held

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            enemy['health'] -= enemy['damage_taken']
            enemy['damage_taken'] = 0

            if enemy['health'] <= 0:
                self.enemies.remove(enemy)
                reward += 1 # Reward for destroying an enemy
                # sfx: enemy_explode.wav
                for _ in range(20):
                    self._create_particle(enemy['screen_pos'], self.COLOR_ENEMY)
                continue

            # Movement
            if not self.fortress_blocks:
                target_pos = self._iso_to_screen(self.GRID_WIDTH / 2, self.GRID_HEIGHT / 2)
            else:
                # Find closest block
                closest_dist = float('inf')
                target_block = None
                for block_pos in self.fortress_blocks:
                    dist = math.hypot(enemy['pos'][0] - block_pos[0], enemy['pos'][1] - block_pos[1])
                    if dist < closest_dist:
                        closest_dist = dist
                        target_block = block_pos
                target_pos = self._iso_to_screen(*target_block)

            angle = math.atan2(target_pos[1] - enemy['screen_pos'][1], target_pos[0] - enemy['screen_pos'][0])
            enemy['pos'][0] += math.cos(angle) * enemy['speed']
            enemy['pos'][1] += math.sin(angle) * enemy['speed']
            
            # Bobbing motion for visual flair
            enemy['bob_angle'] += 0.1
            bob_offset = math.sin(enemy['bob_angle']) * 3
            enemy['screen_pos'] = (int(enemy['pos'][0]), int(enemy['pos'][1] + bob_offset))

            # Firing
            enemy['fire_cooldown'] -= 1
            if enemy['fire_cooldown'] <= 0:
                enemy['fire_cooldown'] = enemy['fire_rate']
                self._create_projectile(enemy['screen_pos'], target_pos, enemy['projectile_damage'])
                # sfx: enemy_fire.wav
        return reward

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1

            if p['lifespan'] <= 0:
                self.projectiles.remove(p)
                continue

            # Collision with fortress
            for block_pos, health in self.fortress_blocks.items():
                block_screen_pos = self._iso_to_screen(*block_pos)
                if math.hypot(p['pos'][0] - block_screen_pos[0], p['pos'][1] - block_screen_pos[1]) < self.TILE_H_HALF + 5:
                    self.fortress_blocks[block_pos] -= p['damage']
                    # sfx: block_hit.wav
                    self._create_particle(p['pos'], self.COLOR_PROJECTILE, 10)
                    if p in self.projectiles: self.projectiles.remove(p)
                    
                    # Reflect damage back to closest enemy
                    closest_enemy = None
                    min_dist = float('inf')
                    for e in self.enemies:
                        dist = math.hypot(p['pos'][0] - e['screen_pos'][0], p['pos'][1] - e['screen_pos'][1])
                        if dist < min_dist:
                            min_dist = dist
                            closest_enemy = e
                    if closest_enemy:
                        closest_enemy['damage_taken'] += p['damage'] * 0.25 # Reflect 25% of damage
                    break
        return 0 # No direct reward from projectiles

    def _update_fortress(self):
        destroyed_blocks = [pos for pos, health in self.fortress_blocks.items() if health <= 0]
        for pos in destroyed_blocks:
            del self.fortress_blocks[pos]
            # sfx: block_destroy.wav
            for _ in range(20):
                self._create_particle(self._iso_to_screen(*pos), self.COLOR_BLOCK_SIDE)

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _update_wave_logic(self):
        reward = 0
        if len(self.enemies) == 0 and self.inter_wave_timer > 0:
            self.inter_wave_timer -= 1
            if self.inter_wave_timer == 0:
                if self.wave_number >= self.MAX_WAVES:
                    self.win = True
                else:
                    # End of wave reward
                    reward += len(self.fortress_blocks) * 0.1
                    self._start_next_wave()
        return reward

    def _start_next_wave(self):
        self.wave_number += 1
        if self.wave_number > self.MAX_WAVES:
            return

        num_enemies = 2 + self.wave_number
        enemy_speed = 0.5 + self.wave_number * 0.05
        enemy_health = 50 + self.wave_number * 10
        
        for _ in range(num_enemies):
            edge = self.np_random.integers(4)
            if edge == 0: x, y = -20, self.np_random.uniform(0, self.HEIGHT) # Left
            elif edge == 1: x, y = self.WIDTH + 20, self.np_random.uniform(0, self.HEIGHT) # Right
            elif edge == 2: x, y = self.np_random.uniform(0, self.WIDTH), -20 # Top
            else: x, y = self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 20 # Bottom
            
            self.enemies.append({
                'pos': [x, y],
                'screen_pos': (int(x), int(y)),
                'speed': enemy_speed,
                'health': enemy_health,
                'damage_taken': 0,
                'fire_rate': max(30, 120 - self.wave_number * 5),
                'fire_cooldown': self.np_random.integers(30, 90),
                'projectile_damage': 10 + self.wave_number * 2,
                'bob_angle': self.np_random.uniform(0, 2 * math.pi)
            })
        
        self.inter_wave_timer = -1 # Wave active

    def _create_projectile(self, start_pos, target_pos, damage):
        angle = math.atan2(target_pos[1] - start_pos[1], target_pos[0] - start_pos[0])
        speed = 4
        self.projectiles.append({
            'pos': list(start_pos),
            'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
            'damage': damage,
            'lifespan': 150 # 5 seconds
        })

    def _create_particle(self, pos, color, count=1):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'lifespan': self.np_random.integers(15, 30),
                'color': color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_grid()
        self._render_fortress()
        self._render_cursor()
        for enemy in self.enemies: self._render_enemy(enemy)
        for p in self.projectiles: self._render_projectile(p)
        for p in self.particles: self._render_particle(p)

    def _render_grid(self):
        for i in range(self.GRID_WIDTH + 1):
            start = self._iso_to_screen(i, 0)
            end = self._iso_to_screen(i, self.GRID_HEIGHT)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
        for i in range(self.GRID_HEIGHT + 1):
            start = self._iso_to_screen(0, i)
            end = self._iso_to_screen(self.GRID_WIDTH, i)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)

    def _render_fortress(self):
        sorted_blocks = sorted(self.fortress_blocks.keys(), key=lambda p: p[0] + p[1])
        for gx, gy in sorted_blocks:
            health_ratio = self.fortress_blocks[(gx, gy)] / 100.0
            color = (
                int(self.COLOR_BLOCK[0] * health_ratio),
                int(self.COLOR_BLOCK[1] * health_ratio),
                int(self.COLOR_BLOCK[2] * health_ratio)
            )
            side_color = (
                int(self.COLOR_BLOCK_SIDE[0] * health_ratio),
                int(self.COLOR_BLOCK_SIDE[1] * health_ratio),
                int(self.COLOR_BLOCK_SIDE[2] * health_ratio)
            )
            self._render_iso_cube(gx, gy, color, side_color)

    def _render_iso_cube(self, gx, gy, top_color, side_color, height=16):
        x, y = self._iso_to_screen(gx, gy)
        hw, hh = self.TILE_W_HALF, self.TILE_H_HALF
        points_top = [(x, y - height), (x + hw, y - height + hh), (x, y - height + hh * 2), (x - hw, y - height + hh)]
        points_left = [(x - hw, y - height + hh), (x, y - height + hh*2), (x, y + hh*2), (x-hw, y+hh)]
        points_right = [(x + hw, y - height + hh), (x, y - height + hh*2), (x, y + hh*2), (x+hw, y+hh)]

        pygame.gfxdraw.filled_polygon(self.screen, points_left, side_color)
        pygame.gfxdraw.filled_polygon(self.screen, points_right, side_color)
        pygame.gfxdraw.filled_polygon(self.screen, points_top, top_color)
        pygame.gfxdraw.aapolygon(self.screen, points_top, top_color)

    def _render_cursor(self):
        if not self.game_over:
            color = self.COLOR_CURSOR
            if tuple(self.cursor_pos) in self.fortress_blocks or self.blocks_remaining <= 0:
                color = (255, 0, 0, 150) # Red if invalid spot
            
            x, y = self._iso_to_screen(*self.cursor_pos)
            hw, hh = self.TILE_W_HALF, self.TILE_H_HALF
            points = [(x, y), (x + hw, y + hh), (x, y + hh * 2), (x - hw, y + hh)]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
    
    def _render_enemy(self, enemy):
        x, y = enemy['screen_pos']
        pygame.gfxdraw.filled_circle(self.screen, x, y, 8, self.COLOR_ENEMY)
        pygame.gfxdraw.aacircle(self.screen, x, y, 8, self.COLOR_ENEMY)
        # Health bar
        health_ratio = enemy['health'] / (50 + self.wave_number * 10)
        pygame.draw.rect(self.screen, (255,0,0), (x - 10, y - 15, 20, 3))
        pygame.draw.rect(self.screen, (0,255,0), (x - 10, y - 15, 20 * health_ratio, 3))

    def _render_projectile(self, p):
        x, y = int(p['pos'][0]), int(p['pos'][1])
        pygame.gfxdraw.filled_circle(self.screen, x, y, 3, self.COLOR_PROJECTILE)
        pygame.gfxdraw.aacircle(self.screen, x, y, 3, self.COLOR_PROJECTILE)

    def _render_particle(self, p):
        alpha = max(0, 255 * (p['lifespan'] / 30.0))
        color = (*p['color'], alpha)
        # This is slow, but required for per-pixel alpha on a non-alpha surface
        temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, color, (2, 2), 2)
        self.screen.blit(temp_surf, (int(p['pos'][0]) - 2, int(p['pos'][1]) - 2))

    def _render_ui(self):
        # UI Panel
        ui_panel = pygame.Surface((self.WIDTH, 40), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, 0))

        # Wave Text
        wave_text = f"WAVE: {self.wave_number}/{self.MAX_WAVES}"
        text_surf = self.font_small.render(wave_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 10))

        # Score Text
        score_text = f"SCORE: {int(self.score)}"
        text_surf = self.font_small.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (180, 10))

        # Blocks Remaining
        blocks_text = f"BLOCKS: {self.blocks_remaining}"
        text_surf = self.font_small.render(blocks_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (320, 10))

        # Fortress Health
        total_health = sum(self.fortress_blocks.values())
        max_health = len(self.fortress_blocks) * 100 if self.fortress_blocks else 1
        health_ratio = total_health / max_health
        health_text = f"FORTRESS HEALTH"
        text_surf = self.font_small.render(health_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (450, 10))
        pygame.draw.rect(self.screen, (200,0,0), (580, 10, 50, 15))
        if health_ratio > 0:
            pygame.draw.rect(self.screen, (0,200,0), (580, 10, 50 * health_ratio, 15))

        # Game Over / Win Text
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (0, 255, 0) if self.win else (255, 0, 0)
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "blocks_remaining": self.blocks_remaining,
            "fortress_blocks": len(self.fortress_blocks),
            "enemies": len(self.enemies),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Isometric Fortress Defense")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    print(env.user_guide)

    while not done:
        # --- Action mapping from keyboard to MultiDiscrete ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Environment step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to convert it back to a Pygame surface to display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        clock.tick(env.FPS)

    env.close()