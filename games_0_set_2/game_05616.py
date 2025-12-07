
# Generated: 2025-08-28T05:33:26.990080
# Source Brief: brief_05616.md
# Brief Index: 5616

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to place a block. "
        "Hold Shift to cycle between Wall and Turret blocks. Doing nothing will pause the game."
    )

    game_description = (
        "Defend your Core from waves of enemies by strategically placing defensive blocks. "
        "Survive 10 waves to win. If the Core is destroyed, you lose."
    )

    auto_advance = True

    # --- Constants ---
    # Game settings
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = 20
    GRID_WIDTH, GRID_HEIGHT = SCREEN_WIDTH // GRID_SIZE, SCREEN_HEIGHT // GRID_SIZE
    MAX_STEPS = 30 * 60  # 60 seconds at 30 FPS
    MAX_WAVES = 10
    WAVE_INTERVAL = 30 * 15 # 15 seconds between waves
    CORE_POS = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    CORE_SIZE = 20
    CORE_MAX_HEALTH = 1000
    
    # Colors
    COLOR_BG = (25, 28, 36)
    COLOR_GRID = (40, 44, 52)
    COLOR_CORE = (100, 110, 130)
    COLOR_CORE_DAMAGE = (255, 100, 100)
    COLOR_ENEMY = (224, 108, 117)
    COLOR_PROJECTILE = (97, 175, 239)
    COLOR_CURSOR = (255, 255, 255, 150)
    COLOR_TEXT = (200, 200, 210)
    
    BLOCK_TYPES = {
        0: {"name": "Wall", "cost": 1, "hp": 200, "color": (152, 195, 121)},
        1: {"name": "Turret", "cost": 3, "hp": 50, "color": (198, 120, 221), "range": 150, "cooldown": 20, "damage": 10},
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        self.enemies = []
        self.blocks = []
        self.projectiles = []
        self.particles = []
        
        self.reset()
        
        # Run validation check
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.core_health = self.CORE_MAX_HEALTH
        self.wave_number = 0
        self.wave_timer = 0
        self.num_blocks_available = 10
        self.selected_block_type = 0
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT - 3]
        
        self.enemies.clear()
        self.blocks.clear()
        self.projectiles.clear()
        self.particles.clear()
        
        self.space_was_pressed = False
        self.shift_was_pressed = False
        
        self._start_next_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        
        if self.game_over or self.win:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Actions ---
        reward += self._handle_input(movement, space_held, shift_held)
        
        # --- Update Game State (if not paused) ---
        is_paused = (movement == 0)
        if not is_paused:
            self.steps += 1
            step_reward = self._update_game_state()
            reward += step_reward

        # --- Check for Termination ---
        if self.core_health <= 0:
            self.game_over = True
            reward -= 100
            self._create_explosion(self.CORE_POS, 100, self.COLOR_CORE_DAMAGE, 50)
            # sfx: core_destruction_sound
        
        if self.win:
            reward += 100
        
        if self.steps >= self.MAX_STEPS:
            self.game_over = True

        terminated = self.game_over or self.win
        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        reward = 0
        # Cursor movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # Place block (on key press)
        if space_held and not self.space_was_pressed:
            if self._place_block():
                reward -= 0.01
        self.space_was_pressed = space_held

        # Cycle block type (on key press)
        if shift_held and not self.shift_was_pressed:
            self.selected_block_type = (self.selected_block_type + 1) % len(self.BLOCK_TYPES)
            # sfx: ui_cycle_sound
        self.shift_was_pressed = shift_held
        
        return reward

    def _update_game_state(self):
        reward = 0
        
        # Wave management
        self.wave_timer -= 1
        if self.wave_timer <= 0 and self.wave_number <= self.MAX_WAVES:
            wave_reward = self._start_next_wave()
            reward += wave_reward

        # Update enemies
        for enemy in self.enemies[:]:
            enemy['attack_cooldown'] -= 1
            target_pos = self.CORE_POS
            
            # Simple obstacle check
            is_blocked = False
            for block in self.blocks:
                if math.hypot(enemy['pos'][0] - block['pos'][0], enemy['pos'][1] - block['pos'][1]) < self.GRID_SIZE:
                    is_blocked = True
                    if enemy['attack_cooldown'] <= 0:
                        block['hp'] -= enemy['damage']
                        self._create_explosion(block['pos'], 3, (255, 255, 255), 3)
                        # sfx: block_hit_sound
                        enemy['attack_cooldown'] = 30
                    break
            
            if not is_blocked:
                angle = math.atan2(target_pos[1] - enemy['pos'][1], target_pos[0] - enemy['pos'][0])
                enemy['pos'][0] += math.cos(angle) * enemy['speed']
                enemy['pos'][1] += math.sin(angle) * enemy['speed']
            
            # Check collision with core
            if math.hypot(enemy['pos'][0] - self.CORE_POS[0], enemy['pos'][1] - self.CORE_POS[1]) < self.CORE_SIZE:
                self.core_health -= enemy['damage'] * 5 # Enemies do more damage to core
                self._create_explosion(self.CORE_POS, 10, self.COLOR_CORE_DAMAGE)
                # sfx: core_hit_sound
                self.enemies.remove(enemy)
                continue

        # Update blocks and turrets
        for block in self.blocks[:]:
            if block['hp'] <= 0:
                self._create_explosion(block['pos'], 20, block['color'])
                # sfx: block_destroy_sound
                self.blocks.remove(block)
                continue
            
            # Turret logic
            if block['type'] == 1:
                block['cooldown'] -= 1
                if block['cooldown'] <= 0:
                    target_enemy = None
                    min_dist = block['range']
                    for enemy in self.enemies:
                        dist = math.hypot(enemy['pos'][0] - block['pos'][0], enemy['pos'][1] - block['pos'][1])
                        if dist < min_dist:
                            min_dist = dist
                            target_enemy = enemy
                    
                    if target_enemy:
                        self._fire_projectile(block, target_enemy)
                        block['cooldown'] = self.BLOCK_TYPES[1]['cooldown']

        # Update projectiles
        for proj in self.projectiles[:]:
            proj['pos'][0] += proj['vel'][0]
            proj['pos'][1] += proj['vel'][1]
            proj['ttl'] -= 1
            
            if proj['ttl'] <= 0:
                self.projectiles.remove(proj)
                continue
            
            for enemy in self.enemies[:]:
                if math.hypot(proj['pos'][0] - enemy['pos'][0], proj['pos'][1] - enemy['pos'][1]) < 8:
                    enemy['hp'] -= proj['damage']
                    self._create_explosion(proj['pos'], 5, self.COLOR_PROJECTILE)
                    # sfx: enemy_hit_sound
                    if enemy['hp'] <= 0:
                        self.enemies.remove(enemy)
                        reward += 0.1
                        self.num_blocks_available += 1
                        self._create_explosion(enemy['pos'], 30, self.COLOR_ENEMY)
                        # sfx: enemy_destroy_sound
                    self.projectiles.remove(proj)
                    break
        
        # Update particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        return reward

    def _start_next_wave(self):
        if self.wave_number >= self.MAX_WAVES:
            self.win = True
            return 0

        reward_for_survival = 1.0 if self.wave_number > 0 else 0
        self.wave_number += 1
        self.wave_timer = self.WAVE_INTERVAL
        
        num_enemies = 2 + self.wave_number * 2
        enemy_speed = 0.8 + self.wave_number * 0.1
        enemy_health = 10 + self.wave_number * 5
        enemy_damage = 5 + self.wave_number

        for _ in range(num_enemies):
            side = self.np_random.integers(4)
            if side == 0: x, y = self.np_random.integers(self.SCREEN_WIDTH), -20
            elif side == 1: x, y = self.np_random.integers(self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 20
            elif side == 2: x, y = -20, self.np_random.integers(self.SCREEN_HEIGHT)
            else: x, y = self.SCREEN_WIDTH + 20, self.np_random.integers(self.SCREEN_HEIGHT)
            
            self.enemies.append({
                "pos": [x, y],
                "hp": enemy_health,
                "max_hp": enemy_health,
                "speed": enemy_speed,
                "damage": enemy_damage,
                "attack_cooldown": 0
            })
        
        self.num_blocks_available += 5
        return reward_for_survival
    
    def _place_block(self):
        cost = self.BLOCK_TYPES[self.selected_block_type]['cost']
        if self.num_blocks_available < cost:
            # sfx: error_sound
            return False
            
        grid_x, grid_y = self.cursor_pos
        pixel_x = grid_x * self.GRID_SIZE + self.GRID_SIZE // 2
        pixel_y = grid_y * self.GRID_SIZE + self.GRID_SIZE // 2
        
        # Check if space is occupied by another block or the core
        if math.hypot(pixel_x - self.CORE_POS[0], pixel_y - self.CORE_POS[1]) < self.CORE_SIZE + self.GRID_SIZE:
            return False
        for block in self.blocks:
            if block['grid_pos'] == (grid_x, grid_y):
                return False

        block_info = self.BLOCK_TYPES[self.selected_block_type]
        new_block = {
            "pos": (pixel_x, pixel_y),
            "grid_pos": (grid_x, grid_y),
            "type": self.selected_block_type,
            "hp": block_info['hp'],
            "max_hp": block_info['hp'],
            "color": block_info['color'],
        }
        if self.selected_block_type == 1: # Turret
            new_block.update({
                "range": block_info['range'],
                "cooldown": 0,
            })
        
        self.blocks.append(new_block)
        self.num_blocks_available -= cost
        self._create_explosion((pixel_x, pixel_y), 10, block_info['color'], 5)
        # sfx: place_block_sound
        return True

    def _fire_projectile(self, turret, target):
        start_pos = list(turret['pos'])
        angle = math.atan2(target['pos'][1] - start_pos[1], target['pos'][0] - start_pos[0])
        speed = 8
        self.projectiles.append({
            "pos": start_pos,
            "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
            "ttl": 60,
            "damage": self.BLOCK_TYPES[1]['damage']
        })
        # sfx: turret_fire_sound

    def _create_explosion(self, pos, count, color, max_radius=20):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            life = self.np_random.integers(10, 30)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": life,
                "max_life": life,
                "color": color,
                "radius": self.np_random.uniform(1, max_radius / 5)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Grid
        for x in range(0, self.SCREEN_WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Core
        core_rect = pygame.Rect(0, 0, self.CORE_SIZE * 2, self.CORE_SIZE * 2)
        core_rect.center = self.CORE_POS
        pygame.draw.rect(self.screen, self.COLOR_CORE, core_rect, border_radius=4)
        
        # Blocks
        for block in self.blocks:
            rect = pygame.Rect(0, 0, self.GRID_SIZE - 2, self.GRID_SIZE - 2)
            rect.center = block['pos']
            
            # Health tint
            health_perc = block['hp'] / block['max_hp']
            tint = tuple(int(c * health_perc) for c in block['color'])
            pygame.draw.rect(self.screen, tint, rect, border_radius=2)
            if block['type'] == 1: # Turret
                pygame.gfxdraw.aacircle(self.screen, int(block['pos'][0]), int(block['pos'][1]), 6, (255,255,255))


        # Enemies
        for enemy in self.enemies:
            pos = (int(enemy['pos'][0]), int(enemy['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 7, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 7, self.COLOR_ENEMY)

        # Projectiles
        for proj in self.projectiles:
            pos = (int(proj['pos'][0]), int(proj['pos'][1]))
            pygame.draw.rect(self.screen, self.COLOR_PROJECTILE, (pos[0]-2, pos[1]-2, 4, 4))

        # Particles
        for p in self.particles:
            life_perc = p['life'] / p['max_life']
            radius = int(p['radius'] * life_perc)
            if radius > 0:
                color = tuple(int(c * life_perc) for c in p['color'])
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, color)
        
        # Cursor
        if not self.game_over and not self.win:
            cursor_rect = pygame.Rect(self.cursor_pos[0] * self.GRID_SIZE, self.cursor_pos[1] * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
            surf = pygame.Surface(cursor_rect.size, pygame.SRCALPHA)
            cost = self.BLOCK_TYPES[self.selected_block_type]['cost']
            cursor_color = (97, 175, 239) if self.num_blocks_available >= cost else (224, 108, 117)
            pygame.draw.rect(surf, cursor_color + (100,), surf.get_rect(), border_radius=3)
            pygame.draw.rect(surf, cursor_color + (200,), surf.get_rect(), 2, border_radius=3)
            self.screen.blit(surf, cursor_rect.topleft)

    def _render_ui(self):
        # Wave info
        wave_text = self.font_small.render(f"Wave: {self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 10))

        # Block count
        block_text = self.font_small.render(f"Blocks: {self.num_blocks_available}", True, self.COLOR_TEXT)
        self.screen.blit(block_text, (self.SCREEN_WIDTH - block_text.get_width() - 10, 10))

        # Selected block type
        block_info = self.BLOCK_TYPES[self.selected_block_type]
        type_text = self.font_small.render(f"Selected: {block_info['name']} (Cost: {block_info['cost']})", True, self.COLOR_TEXT)
        self.screen.blit(type_text, (self.SCREEN_WIDTH // 2 - type_text.get_width() // 2, self.SCREEN_HEIGHT - 25))

        # Core health bar
        health_perc = max(0, self.core_health / self.CORE_MAX_HEALTH)
        bar_width = 100
        bar_height = 10
        bar_x = self.CORE_POS[0] - bar_width // 2
        bar_y = self.CORE_POS[1] - self.CORE_SIZE - bar_height - 5
        pygame.draw.rect(self.screen, (80, 80, 80), (bar_x, bar_y, bar_width, bar_height), border_radius=2)
        pygame.draw.rect(self.screen, self.BLOCK_TYPES[0]['color'], (bar_x, bar_y, bar_width * health_perc, bar_height), border_radius=2)

        # Game Over / Win Text
        if self.game_over:
            text = self.font_large.render("GAME OVER", True, self.COLOR_ENEMY)
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(text, text_rect)
        elif self.win:
            text = self.font_large.render("VICTORY!", True, self.BLOCK_TYPES[0]['color'])
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "core_health": self.core_health,
            "blocks_available": self.num_blocks_available,
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
    # Set SDL_VIDEODRIVER to "dummy" for headless execution
    import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    
    # --- Human Play ---
    # To play, you need a window.
    # The environment itself is headless, but we can display its output.
    try:
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption(env.game_description)
        clock = pygame.time.Clock()

        obs, info = env.reset()
        terminated = False
        
        while not terminated:
            # --- Action Mapping for Human ---
            keys = pygame.key.get_pressed()
            movement = 0 # No-op/Pause
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()

            obs, reward, terminated, truncated, info = env.step(action)
            
            # --- Rendering ---
            # The observation is already a rendered frame
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(30) # Limit to 30 FPS
            if reward != 0:
                print(f"Step: {info['steps']}, Score: {info['score']:.2f}, Reward: {reward:.2f}, Wave: {info['wave']}")

    finally:
        env.close()