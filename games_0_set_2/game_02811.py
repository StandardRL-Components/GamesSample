
# Generated: 2025-08-28T06:02:10.356595
# Source Brief: brief_02811.md
# Brief Index: 2811

        
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
        "Controls: Arrows to move cursor. Space to place defensive turrets. Shift to place resource generators."
    )

    game_description = (
        "Defend your base from waves of enemies. Place turrets to attack them and generators to earn resources."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 40
        self.GRID_W = self.WIDTH // self.GRID_SIZE
        self.GRID_H = self.HEIGHT // self.GRID_SIZE
        self.MAX_STEPS = 2500  # Increased for longer games
        self.MAX_WAVES = 20

        # Colors
        self.COLOR_BG = (15, 18, 28)
        self.COLOR_GRID = (30, 35, 50)
        self.COLOR_BASE = (70, 80, 100)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_DEFENSE_BLOCK = (60, 220, 180)
        self.COLOR_RESOURCE_BLOCK = (250, 200, 80)
        self.COLOR_ENEMY = (255, 80, 100)
        self.COLOR_PROJECTILE = (100, 240, 255)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_BAR = (40, 200, 120)
        self.COLOR_HEALTH_BG = (120, 40, 40)

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_small = pygame.font.SysFont("Consolas", 16)
            self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        except pygame.error:
            self.font_small = pygame.font.Font(None, 20)
            self.font_large = pygame.font.Font(None, 30)

        # Initialize state variables
        self.base_pos = None
        self.base_health = None
        self.max_base_health = None
        self.resources = None
        self.blocks = None
        self.enemies = None
        self.projectiles = None
        self.particles = None
        self.cursor_grid_pos = None
        self.current_wave = None
        self.wave_countdown = None
        self.game_over = None
        self.score = None
        self.steps = None
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            self.np_random = np.random.default_rng()


        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_wave = 0
        self.wave_countdown = 120 # Frames until first wave
        
        # Base
        self.max_base_health = 100
        self.base_health = self.max_base_health
        self.base_pos = pygame.Vector2(self.WIDTH // 2, self.HEIGHT // 2)

        # Player state
        self.resources = 50
        self.cursor_grid_pos = [self.GRID_W // 2, self.GRID_H // 2]
        
        # Game entities
        self.blocks = []
        self.enemies = []
        self.projectiles = []
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False

        if not self.game_over:
            # Unpack and handle action
            self._handle_input(action)

            # Update game logic
            reward += self._update_resources()
            reward += self._update_blocks()
            reward += self._update_projectiles()
            reward += self._update_enemies()
            self._update_particles()
            
            wave_reward, wave_cleared = self._update_waves()
            reward += wave_reward
            if wave_cleared:
                self.score += 1 # Add to score for wave clear

            self.score += reward

        # Check termination conditions
        if self.base_health <= 0 and not self.game_over:
            terminated = True
            reward -= 100
            self.score -= 100
            self.game_over = True
            self._create_explosion(self.base_pos, 100)

        if self.current_wave > self.MAX_WAVES and not self.game_over:
            terminated = True
            reward += 100
            self.score += 100
            self.game_over = True
            
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        # Move cursor
        if movement == 1: self.cursor_grid_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_grid_pos[1] += 1  # Down
        elif movement == 3: self.cursor_grid_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_grid_pos[0] += 1  # Right
        
        self.cursor_grid_pos[0] = np.clip(self.cursor_grid_pos[0], 0, self.GRID_W - 1)
        self.cursor_grid_pos[1] = np.clip(self.cursor_grid_pos[1], 0, self.GRID_H - 1)

        # Place blocks
        grid_pos_tuple = tuple(self.cursor_grid_pos)
        is_occupied = any(b['grid_pos'] == grid_pos_tuple for b in self.blocks)
        is_base = (self.WIDTH // (2 * self.GRID_SIZE) == grid_pos_tuple[0] and
                   self.HEIGHT // (2 * self.GRID_SIZE) == grid_pos_tuple[1])

        if not is_occupied and not is_base:
            if space_pressed and self.resources >= 25:
                self.resources -= 25
                self.blocks.append({
                    'type': 'defense', 'grid_pos': grid_pos_tuple,
                    'pos': pygame.Vector2(grid_pos_tuple[0] * self.GRID_SIZE + self.GRID_SIZE // 2,
                                          grid_pos_tuple[1] * self.GRID_SIZE + self.GRID_SIZE // 2),
                    'range': 120, 'cooldown': 0, 'fire_rate': 45 # frames
                })
                # sfx: place_turret.wav
            elif shift_pressed and self.resources >= 15:
                self.resources -= 15
                self.blocks.append({
                    'type': 'resource', 'grid_pos': grid_pos_tuple,
                    'pos': pygame.Vector2(grid_pos_tuple[0] * self.GRID_SIZE + self.GRID_SIZE // 2,
                                          grid_pos_tuple[1] * self.GRID_SIZE + self.GRID_SIZE // 2),
                    'timer': 0, 'gen_rate': 180 # frames
                })
                # sfx: place_generator.wav

    def _update_resources(self):
        reward = 0
        for block in self.blocks:
            if block['type'] == 'resource':
                block['timer'] += 1
                if block['timer'] >= block['gen_rate']:
                    block['timer'] = 0
                    self.resources += 5
                    reward += 0.01
        return reward

    def _update_blocks(self):
        for block in self.blocks:
            if block['type'] == 'defense':
                block['cooldown'] = max(0, block['cooldown'] - 1)
                if block['cooldown'] == 0:
                    target = None
                    min_dist = block['range']
                    for enemy in self.enemies:
                        dist = block['pos'].distance_to(enemy['pos'])
                        if dist < min_dist:
                            min_dist = dist
                            target = enemy
                    
                    if target:
                        self.projectiles.append({
                            'pos': block['pos'].copy(), 'target': target,
                            'speed': 8, 'damage': 5
                        })
                        block['cooldown'] = block['fire_rate']
                        # sfx: turret_shoot.wav
        return 0

    def _update_projectiles(self):
        reward = 0
        for p in self.projectiles[:]:
            if p['target'] not in self.enemies:
                self.projectiles.remove(p)
                continue
            
            direction = (p['target']['pos'] - p['pos']).normalize()
            p['pos'] += direction * p['speed']
            
            if p['pos'].distance_to(p['target']['pos']) < 10:
                p['target']['health'] -= p['damage']
                self._create_explosion(p['pos'], 5, self.COLOR_PROJECTILE)
                self.projectiles.remove(p)
                # sfx: projectile_hit.wav
                if p['target']['health'] <= 0:
                    if p['target'] in self.enemies:
                        self._create_explosion(p['target']['pos'], 15, self.COLOR_ENEMY)
                        self.enemies.remove(p['target'])
                        reward += 0.1
                        self.resources += 2
                        # sfx: enemy_die.wav
        return reward

    def _update_enemies(self):
        for enemy in self.enemies[:]:
            direction = (self.base_pos - enemy['pos'])
            if direction.length() > 0:
                direction.normalize_ip()
            enemy['pos'] += direction * enemy['speed']
            
            if enemy['pos'].distance_to(self.base_pos) < self.GRID_SIZE * 0.75:
                self.base_health -= enemy['damage']
                self._create_explosion(enemy['pos'], 10)
                self.enemies.remove(enemy)
                # sfx: base_hit.wav
        return 0

    def _update_waves(self):
        reward = 0
        cleared = False
        if not self.enemies and self.current_wave <= self.MAX_WAVES:
            self.wave_countdown -= 1
            if self.wave_countdown <= 0:
                if self.current_wave > 0: # Reward for clearing previous wave
                    reward += 1.0
                    cleared = True
                self.current_wave += 1
                if self.current_wave <= self.MAX_WAVES:
                    self._spawn_wave()
                    self.wave_countdown = 300 # Time between waves
        return reward, cleared

    def _spawn_wave(self):
        num_enemies = 3 + (self.current_wave - 1)
        health = 10 * (1.05 ** (self.current_wave - 1))
        speed = 1.0 * (1.05 ** (self.current_wave - 1))
        damage = 10

        for _ in range(num_enemies):
            edge = self.np_random.integers(4)
            if edge == 0: # Top
                pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), -20)
            elif edge == 1: # Bottom
                pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 20)
            elif edge == 2: # Left
                pos = pygame.Vector2(-20, self.np_random.uniform(0, self.HEIGHT))
            else: # Right
                pos = pygame.Vector2(self.WIDTH + 20, self.np_random.uniform(0, self.HEIGHT))

            self.enemies.append({
                'pos': pos, 'health': health, 'max_health': health,
                'speed': speed, 'damage': damage
            })
        # sfx: wave_start.wav

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _create_explosion(self, pos, num_particles, color=None):
        if color is None: color = self.COLOR_ENEMY
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(), 'vel': vel, 'life': self.np_random.integers(15, 30),
                'color': color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw base
        base_rect = pygame.Rect(self.base_pos.x - self.GRID_SIZE//2, self.base_pos.y - self.GRID_SIZE//2, self.GRID_SIZE, self.GRID_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect, border_radius=4)
        
        # Draw blocks
        for block in self.blocks:
            color = self.COLOR_DEFENSE_BLOCK if block['type'] == 'defense' else self.COLOR_RESOURCE_BLOCK
            pos = (int(block['pos'].x), int(block['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.GRID_SIZE // 3, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.GRID_SIZE // 3, color)

        # Draw enemies
        for enemy in self.enemies:
            pos = (int(enemy['pos'].x), int(enemy['pos'].y))
            size = int(10 + (enemy['health'] / enemy['max_health']) * 5)
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size + 3, (*self.COLOR_ENEMY, 50))
            # Main body
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, self.COLOR_ENEMY)

        # Draw projectiles
        for p in self.projectiles:
            pos = (int(p['pos'].x), int(p['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 3, self.COLOR_PROJECTILE)

        # Draw particles
        for p in self.particles:
            pos = (int(p['pos'].x), int(p['pos'].y))
            size = int(p['life'] / 10)
            if size > 0:
                alpha = int((p['life'] / 30) * 255)
                color = (*p['color'], alpha)
                pygame.draw.rect(self.screen, color, (pos[0], pos[1], size, size))

        # Draw cursor
        cursor_x = self.cursor_grid_pos[0] * self.GRID_SIZE
        cursor_y = self.cursor_grid_pos[1] * self.GRID_SIZE
        cursor_rect = pygame.Rect(cursor_x, cursor_y, self.GRID_SIZE, self.GRID_SIZE)
        s = pygame.Surface((self.GRID_SIZE, self.GRID_SIZE), pygame.SRCALPHA)
        s.fill((*self.COLOR_PLAYER, 80))
        self.screen.blit(s, (cursor_x, cursor_y))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, cursor_rect, 1)
        
    def _render_ui(self):
        # Base Health Bar
        health_ratio = max(0, self.base_health / self.max_base_health)
        bar_width = 200
        bar_height = 15
        bar_x = self.WIDTH // 2 - bar_width // 2
        bar_y = self.HEIGHT - 30
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (bar_x, bar_y, bar_width * health_ratio, bar_height), border_radius=4)
        health_text = self.font_small.render(f"BASE HP", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (bar_x + bar_width // 2 - health_text.get_width() // 2, bar_y - 18))

        # Resources
        res_text = self.font_large.render(f"${int(self.resources)}", True, self.COLOR_RESOURCE_BLOCK)
        self.screen.blit(res_text, (15, 10))

        # Wave counter
        wave_str = f"WAVE: {self.current_wave}/{self.MAX_WAVES}"
        if not self.enemies and self.current_wave < self.MAX_WAVES:
            wave_str += f" (Next in {self.wave_countdown // 30}s)"
        wave_text = self.font_large.render(wave_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 15, 10))

        # Game Over / Win message
        if self.game_over:
            message = "YOU WIN!" if self.current_wave > self.MAX_WAVES else "GAME OVER"
            color = self.COLOR_HEALTH_BAR if self.current_wave > self.MAX_WAVES else self.COLOR_ENEMY
            end_text = self.font_large.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 40))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "resources": self.resources,
            "base_health": self.base_health,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    is_human_play = True # Set to False to run an agent
    
    # Re-initialize pygame for display
    pygame.display.init()
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption(env.game_description)
    
    total_reward = 0
    
    while running:
        action = np.array([0, 0, 0]) # Default no-op
        
        if is_human_play:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    running = False

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: action[0] = 1
            elif keys[pygame.K_DOWN]: action[0] = 2
            elif keys[pygame.K_LEFT]: action[0] = 3
            elif keys[pygame.K_RIGHT]: action[0] = 4
            
            if keys[pygame.K_SPACE]: action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        else:
            # Simple agent: move cursor randomly and place blocks if possible
            action = env.action_space.sample()
            if env.resources < 15: # Can't afford anything
                action[1] = 0
                action[2] = 0
            elif env.resources < 25: # Can only afford resource blocks
                action[1] = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render to display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}, Waves Survived: {info['wave']-1}")
            obs, info = env.reset()
            total_reward = 0
            if is_human_play:
                pygame.time.wait(3000) # Pause before restarting

        env.clock.tick(30) # 30 FPS

    env.close()