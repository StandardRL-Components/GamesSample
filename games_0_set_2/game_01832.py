
# Generated: 2025-08-27T18:25:53.590216
# Source Brief: brief_01832.md
# Brief Index: 1832

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Arrow keys to move the placement cursor. "
        "Press Space to place a block. Press Shift to cycle block types."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A top-down tower defense game. Defend your base from waves of enemies "
        "by strategically placing walls and turrets."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Colors ---
    COLOR_BG = (15, 20, 30)
    COLOR_BASE = (0, 200, 100)
    COLOR_BASE_DMG = (255, 100, 100)
    COLOR_ENEMY = (220, 50, 50)
    COLOR_WALL = (60, 120, 200)
    COLOR_TURRET = (80, 180, 255)
    COLOR_PROJECTILE = (255, 220, 50)
    COLOR_TEXT = (230, 230, 230)
    COLOR_CURSOR = (255, 255, 255, 100)

    # --- Game Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_SIZE = 20
    MAX_STEPS = 30 * 120  # 120 seconds at 30fps
    WIN_WAVE = 20
    INTER_WAVE_DELAY = 30 * 4 # 4 seconds

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 32, bold=True)
        
        # Initialize state variables
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.base_pos = None
        self.base_size = None
        self.base_health = None
        self.max_base_health = None
        self.wave_number = 0
        self.enemies = []
        self.blocks = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = None
        self.block_types = ["WALL", "TURRET"]
        self.selected_block_type_idx = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.wave_timer = 0
        
        self.reset()
        # self.validate_implementation() # Optional: Uncomment to run validation on init

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            # Fallback if seed is not provided
            if self.np_random is None:
                self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.base_pos = pygame.math.Vector2(self.WIDTH // 2, self.HEIGHT // 2)
        self.base_size = 40
        self.max_base_health = 100
        self.base_health = self.max_base_health
        
        self.wave_number = 0
        self.enemies = []
        self.blocks = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_pos = pygame.math.Vector2(self.WIDTH // 2, self.HEIGHT // 4)
        self.selected_block_type_idx = 0
        self.last_space_held = False
        self.last_shift_held = False
        
        self.wave_timer = self.INTER_WAVE_DELAY - 1 # Start first wave quickly

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        
        self._handle_input(action)
        
        # --- Update Game State ---
        reward += self._update_waves()
        self._update_turrets()
        reward += self._update_projectiles()
        reward += self._update_enemies()
        self._update_particles()
        
        # --- Check for termination ---
        self.steps += 1
        terminated = False
        if self.base_health <= 0:
            terminated = True
            self._create_explosion(self.base_pos, self.COLOR_BASE, 100, 50)
        if self.steps >= self.MAX_STEPS:
            terminated = True
        if self.game_won:
            reward += 100 # Win bonus
            terminated = True
            
        if terminated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Cursor Movement ---
        cursor_speed = 6
        if movement == 1: self.cursor_pos.y -= cursor_speed
        if movement == 2: self.cursor_pos.y += cursor_speed
        if movement == 3: self.cursor_pos.x -= cursor_speed
        if movement == 4: self.cursor_pos.x += cursor_speed
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.HEIGHT)

        # --- Cycle Block Type (on key press) ---
        if shift_held and not self.last_shift_held:
            self.selected_block_type_idx = (self.selected_block_type_idx + 1) % len(self.block_types)
        self.last_shift_held = shift_held
        
        # --- Place Block (on key press) ---
        if space_held and not self.last_space_held:
            self._place_block()
        self.last_space_held = space_held

    def _place_block(self):
        grid_x = int(self.cursor_pos.x // self.GRID_SIZE) * self.GRID_SIZE
        grid_y = int(self.cursor_pos.y // self.GRID_SIZE) * self.GRID_SIZE
        new_block_rect = pygame.Rect(grid_x, grid_y, self.GRID_SIZE, self.GRID_SIZE)

        # Prevent placing on base
        base_rect = pygame.Rect(self.base_pos.x - self.base_size/2, self.base_pos.y - self.base_size/2, self.base_size, self.base_size)
        if base_rect.colliderect(new_block_rect):
            return

        # Prevent placing on other blocks
        if any(b['rect'].colliderect(new_block_rect) for b in self.blocks):
            return
            
        block_type = self.block_types[self.selected_block_type_idx]
        new_block = {'type': block_type, 'rect': new_block_rect}
        if block_type == "TURRET":
            new_block['cooldown'] = 0
            new_block['fire_rate'] = 60 # frames
            new_block['range'] = 150
        
        self.blocks.append(new_block)
        # sfx: place_block
        self._create_explosion(pygame.math.Vector2(new_block_rect.center), self.COLOR_WALL, 10, 5)

    def _update_waves(self):
        reward = 0
        if not self.enemies and not self.game_won:
            self.wave_timer += 1
            if self.wave_timer == 1 and self.wave_number > 0: # Just finished a wave
                reward += 5 # Wave survival bonus
                self.score += self.wave_number * 10
            if self.wave_timer >= self.INTER_WAVE_DELAY:
                self.wave_timer = 0
                self.wave_number += 1
                if self.wave_number > self.WIN_WAVE:
                    self.game_won = True
                else:
                    self._spawn_wave()
        return reward

    def _spawn_wave(self):
        num_enemies = 5 + self.wave_number * 2
        speed = 0.5 + (self.wave_number - 1) * 0.05
        health = 1 + (self.wave_number - 1) // 2
        
        for _ in range(num_enemies):
            edge = self.np_random.integers(0, 4)
            if edge == 0: pos = pygame.math.Vector2(self.np_random.uniform(0, self.WIDTH), -10)
            elif edge == 1: pos = pygame.math.Vector2(self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 10)
            elif edge == 2: pos = pygame.math.Vector2(-10, self.np_random.uniform(0, self.HEIGHT))
            else: pos = pygame.math.Vector2(self.WIDTH + 10, self.np_random.uniform(0, self.HEIGHT))

            self.enemies.append({
                'pos': pos,
                'health': health,
                'max_health': health,
                'speed': speed,
                'size': 8
            })

    def _update_turrets(self):
        for turret in filter(lambda b: b['type'] == 'TURRET', self.blocks):
            turret['cooldown'] = max(0, turret['cooldown'] - 1)
            if turret['cooldown'] == 0 and self.enemies:
                turret_pos = pygame.math.Vector2(turret['rect'].center)
                
                # Find closest enemy in range
                closest_enemy = None
                min_dist_sq = turret['range'] ** 2
                
                for enemy in self.enemies:
                    dist_sq = turret_pos.distance_squared_to(enemy['pos'])
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        closest_enemy = enemy
                
                if closest_enemy:
                    # sfx: turret_fire
                    direction = (closest_enemy['pos'] - turret_pos).normalize()
                    self.projectiles.append({
                        'pos': turret_pos,
                        'vel': direction * 6,
                        'damage': 1,
                        'size': 4
                    })
                    turret['cooldown'] = turret['fire_rate']

    def _update_projectiles(self):
        reward = 0
        projectiles_to_keep = []
        for p in self.projectiles:
            p['pos'] += p['vel']
            
            hit = False
            for enemy in self.enemies:
                if p['pos'].distance_to(enemy['pos']) < enemy['size'] + p['size']:
                    enemy['health'] -= p['damage']
                    reward += 0.1 # Hit bonus
                    self._create_explosion(p['pos'], self.COLOR_PROJECTILE, 15, 8)
                    hit = True
                    break
            
            if not hit and 0 < p['pos'].x < self.WIDTH and 0 < p['pos'].y < self.HEIGHT:
                projectiles_to_keep.append(p)

        self.projectiles = projectiles_to_keep
        return reward

    def _update_enemies(self):
        reward = 0
        enemies_to_keep = []
        base_rect = pygame.Rect(self.base_pos.x - self.base_size/2, self.base_pos.y - self.base_size/2, self.base_size, self.base_size)

        for enemy in self.enemies:
            if enemy['health'] <= 0:
                # sfx: enemy_die
                reward += 1 # Kill bonus
                self.score += 5
                self._create_explosion(enemy['pos'], self.COLOR_ENEMY, 30, 20)
                continue

            # --- Movement and Collision ---
            direction = (self.base_pos - enemy['pos']).normalize()
            enemy['pos'] += direction * enemy['speed']
            
            # Collision with base
            if enemy['pos'].distance_to(self.base_pos) < enemy['size'] + self.base_size / 2:
                # sfx: base_hit
                self.base_health -= 1
                self._create_explosion(enemy['pos'], self.COLOR_BASE_DMG, 20, 10)
                continue # Enemy is destroyed on impact

            # Collision with blocks
            enemy_rect = pygame.Rect(enemy['pos'].x - enemy['size'], enemy['pos'].y - enemy['size'], enemy['size']*2, enemy['size']*2)
            for block in self.blocks:
                if block['rect'].colliderect(enemy_rect):
                    # Push out logic
                    delta_x = enemy['pos'].x - block['rect'].centerx
                    delta_y = enemy['pos'].y - block['rect'].centery
                    if abs(delta_x) > abs(delta_y): # Horizontal collision
                        enemy['pos'].x = block['rect'].right + enemy['size'] if delta_x > 0 else block['rect'].left - enemy['size']
                    else: # Vertical collision
                        enemy['pos'].y = block['rect'].bottom + enemy['size'] if delta_y > 0 else block['rect'].top - enemy['size']
            
            enemies_to_keep.append(enemy)

        self.enemies = enemies_to_keep
        return reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _create_explosion(self, pos, color, count, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, max_speed)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed * 0.1
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(10, 30),
                'color': color,
                'size': self.np_random.uniform(1, 3)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render blocks
        for block in self.blocks:
            color = self.COLOR_TURRET if block['type'] == 'TURRET' else self.COLOR_WALL
            pygame.draw.rect(self.screen, color, block['rect'])
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in color), block['rect'], 2)

        # Render base
        base_rect = pygame.Rect(0, 0, self.base_size, self.base_size)
        base_rect.center = (int(self.base_pos.x), int(self.base_pos.y))
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect, border_radius=5)
        
        # Render enemies
        for enemy in self.enemies:
            pos = (int(enemy['pos'].x), int(enemy['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], enemy['size'], self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], enemy['size'], self.COLOR_ENEMY)

        # Render projectiles
        for p in self.projectiles:
            pos = (int(p['pos'].x), int(p['pos'].y))
            pygame.draw.rect(self.screen, self.COLOR_PROJECTILE, (pos[0]-p['size']//2, pos[1]-p['size']//2, p['size'], p['size']))

        # Render particles
        for p in self.particles:
            alpha = max(0, min(255, int(p['life'] * 8)))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['pos'].x - p['size']), int(p['pos'].y - p['size'])))

        # Render cursor
        grid_x = int(self.cursor_pos.x // self.GRID_SIZE) * self.GRID_SIZE
        grid_y = int(self.cursor_pos.y // self.GRID_SIZE) * self.GRID_SIZE
        cursor_rect = pygame.Rect(grid_x, grid_y, self.GRID_SIZE, self.GRID_SIZE)
        
        cursor_surf = pygame.Surface((self.GRID_SIZE, self.GRID_SIZE), pygame.SRCALPHA)
        block_type = self.block_types[self.selected_block_type_idx]
        color = self.COLOR_TURRET if block_type == 'TURRET' else self.COLOR_WALL
        cursor_surf.fill((*color, 80))
        pygame.draw.rect(cursor_surf, (*self.COLOR_TEXT, 150), (0,0,self.GRID_SIZE, self.GRID_SIZE), 1)
        self.screen.blit(cursor_surf, cursor_rect.topleft)

    def _render_ui(self):
        # Health bar for base
        health_pct = self.base_health / self.max_base_health
        bar_width = self.base_size * 1.5
        bar_height = 8
        bar_x = self.base_pos.x - bar_width / 2
        bar_y = self.base_pos.y - self.base_size / 2 - bar_height - 5
        pygame.draw.rect(self.screen, (50,50,50), (bar_x, bar_y, bar_width, bar_height), border_radius=2)
        pygame.draw.rect(self.screen, self.COLOR_BASE, (bar_x, bar_y, bar_width * health_pct, bar_height), border_radius=2)

        # Top-left info text
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        wave_text = self.font_small.render(f"WAVE: {self.wave_number}/{self.WIN_WAVE}", True, self.COLOR_TEXT)
        health_text = self.font_small.render(f"BASE HP: {max(0, self.base_health)}", True, self.COLOR_TEXT)
        block_text = self.font_small.render(f"BLOCK: {self.block_types[self.selected_block_type_idx]}", True, self.COLOR_TEXT)
        
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(wave_text, (10, 30))
        self.screen.blit(health_text, (10, 50))
        self.screen.blit(block_text, (10, 70))

        # Wave transition text
        if not self.enemies and not self.game_won and not self.game_over:
            time_left = (self.INTER_WAVE_DELAY - self.wave_timer) / 30
            msg = f"WAVE {self.wave_number} CLEARED" if self.wave_number > 0 else "GET READY"
            msg_surf = self.font_large.render(msg, True, self.COLOR_TEXT)
            self.screen.blit(msg_surf, msg_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20)))
            
            next_wave_msg = f"NEXT WAVE IN {time_left:.1f}s"
            next_wave_surf = self.font_small.render(next_wave_msg, True, self.COLOR_TEXT)
            self.screen.blit(next_wave_surf, next_wave_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 20)))
        
        if self.game_won:
            win_surf = self.font_large.render("VICTORY!", True, self.COLOR_BASE)
            self.screen.blit(win_surf, win_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2)))
        elif self.game_over:
            lose_surf = self.font_large.render("GAME OVER", True, self.COLOR_ENEMY)
            self.screen.blit(lose_surf, lose_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2)))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "base_health": self.base_health,
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Tower Defense Gym Environment")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    while not done:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        
        movement = 0 # no-op
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
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        # --- Frame rate ---
        clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Info: {info}")
    env.close()