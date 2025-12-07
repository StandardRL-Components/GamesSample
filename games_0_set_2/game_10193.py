import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T09:58:14.661999
# Source Brief: brief_00193.md
# Brief Index: 193
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    game_description = (
        "Defend your core from incoming enemies by teleporting around the grid and deploying defensive clones."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to teleport your cursor. Press space to deploy a defensive clone at your current location."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 16, 10
        self.CELL_SIZE = 40
        self.MAX_STEPS = 2000

        # --- Colors ---
        self.COLOR_BG = (16, 16, 32)
        self.COLOR_GRID = (32, 32, 64)
        self.COLOR_PLAYER = (0, 255, 255)
        self.COLOR_ENEMY = (255, 64, 64)
        self.COLOR_CORE = (64, 255, 64)
        self.COLOR_PROJECTILE = (255, 255, 128)
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_ENERGY = (0, 128, 255)
        
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
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.core_health = None
        self.player_energy = None
        self.enemies = None
        self.clones = None
        self.projectiles = None
        self.particles = None
        self.prev_space_held = None
        self.enemy_spawn_timer = None
        self.current_enemy_spawn_interval = None
        self.current_enemy_speed = None
        self.core_pos_pixel = pygame.Vector2(self.WIDTH // 2, self.HEIGHT // 2)

        # --- Run initial reset ---
        # self.reset() # Removed to align with standard Gym API usage
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            # Note: Pygame doesn't have a global RNG seed function.
            # Seeding Python's random is the conventional approach.
            random.seed(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = [self.GRID_W // 2, self.GRID_H // 2 + 1]
        self.core_health = 100.0
        self.player_energy = 100.0
        
        self.enemies = []
        self.clones = []
        self.projectiles = []
        self.particles = deque(maxlen=200) # Performance cap
        
        self.prev_space_held = False
        self.enemy_spawn_timer = 0
        self._update_difficulty()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        
        # --- Handle player actions ---
        reward += self._handle_player_actions(action)
        
        # --- Update game logic ---
        self._update_difficulty()
        self._update_energy()
        self._spawn_enemies()
        
        kill_reward = self._update_projectiles()
        reward += kill_reward
        
        self._update_enemies()
        # Core damage is penalized by the large negative terminal reward
        
        self._update_clones()
        # clone_reward is not used, but the method is necessary for game logic

        self._update_particles()
        
        # --- Step and survival reward ---
        self.steps += 1
        reward += 0.01 # Small reward for surviving one more step
        
        # --- Check for termination ---
        terminated = self.core_health <= 0
        truncated = self.steps >= self.MAX_STEPS

        if terminated and not self.game_over:
            self.game_over = True
            reward += -100.0 # Defeat penalty
            self.score -= 100.0
        
        if truncated and not self.game_over:
            self.game_over = True
            reward += 100.0 # Victory reward
            self.score += 100.0
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_player_actions(self, action):
        movement, space_held_bool, _ = action[0], action[1] == 1, action[2] == 1
        
        # Teleportation
        if movement != 0:
            px, py = self.player_pos
            if movement == 1: py -= 1 # Up
            if movement == 2: py += 1 # Down
            if movement == 3: px -= 1 # Left
            if movement == 4: px += 1 # Right
            
            # Boundary checks
            self.player_pos[0] = max(0, min(self.GRID_W - 1, px))
            self.player_pos[1] = max(0, min(self.GRID_H - 1, py))
            self._create_particles(self._grid_to_pixel(self.player_pos), 10, self.COLOR_PLAYER, 2, 15)

        # Clone Deployment
        is_space_press = space_held_bool and not self.prev_space_held
        if is_space_press and self.player_energy >= 10:
            self.player_energy -= 10
            clone_pos = list(self.player_pos)
            # Prevent multiple clones on the same spot
            if not any(c['pos'] == clone_pos for c in self.clones):
                self.clones.append({
                    'pos': clone_pos,
                    'kills_left': 5,
                    'attack_cooldown': 0,
                    'range_pixels': 5 * self.CELL_SIZE,
                })
                self._create_particles(self._grid_to_pixel(clone_pos), 30, self.COLOR_ENERGY, 4, 20, is_shield=True)
        
        self.prev_space_held = space_held_bool
        return 0.0

    def _update_difficulty(self):
        difficulty_tier = self.steps // 200
        self.current_enemy_spawn_interval = max(10, 50 - difficulty_tier * 5)
        self.current_enemy_speed = min(2.0, 0.5 + difficulty_tier * 0.05)

    def _update_energy(self):
        self.player_energy = min(100.0, self.player_energy + 0.1)

    def _spawn_enemies(self):
        self.enemy_spawn_timer += 1
        if self.enemy_spawn_timer >= self.current_enemy_spawn_interval:
            self.enemy_spawn_timer = 0
            edge = random.randint(0, 3)
            if edge == 0: # Top
                pos = pygame.Vector2(random.uniform(0, self.WIDTH), -10)
            elif edge == 1: # Bottom
                pos = pygame.Vector2(random.uniform(0, self.WIDTH), self.HEIGHT + 10)
            elif edge == 2: # Left
                pos = pygame.Vector2(-10, random.uniform(0, self.HEIGHT))
            else: # Right
                pos = pygame.Vector2(self.WIDTH + 10, random.uniform(0, self.HEIGHT))

            self.enemies.append({
                'pos': pos,
                'speed': self.current_enemy_speed * random.uniform(0.8, 1.2),
                'health': 1,
            })

    def _update_enemies(self):
        for enemy in reversed(self.enemies):
            direction = (self.core_pos_pixel - enemy['pos']).normalize()
            enemy['pos'] += direction * enemy['speed']
            
            if enemy['pos'].distance_to(self.core_pos_pixel) < 15: # Collision with core
                self.core_health -= 1
                if enemy in self.enemies:
                    self.enemies.remove(enemy)
                self._create_particles(self.core_pos_pixel, 20, self.COLOR_ENEMY, 3, 25)

    def _update_clones(self):
        for clone in reversed(self.clones):
            if clone['kills_left'] <= 0:
                self._create_particles(self._grid_to_pixel(clone['pos']), 15, self.COLOR_PLAYER, 1, 15, alpha_fade=True)
                if clone in self.clones:
                    self.clones.remove(clone)
                continue

            clone['attack_cooldown'] = max(0, clone['attack_cooldown'] - 1)
            if clone['attack_cooldown'] == 0:
                target = self._find_closest_enemy(clone)
                if target:
                    clone['attack_cooldown'] = 20 # 1.5 shots per second at 30fps
                    start_pos = self._grid_to_pixel(clone['pos'])
                    self.projectiles.append({
                        'pos': start_pos,
                        'target': target,
                        'speed': 8,
                    })
    
    def _find_closest_enemy(self, clone):
        closest_enemy = None
        min_dist = float('inf')
        clone_pixel_pos = self._grid_to_pixel(clone['pos'])
        
        for enemy in self.enemies:
            dist = clone_pixel_pos.distance_to(enemy['pos'])
            if dist < clone['range_pixels'] and dist < min_dist:
                min_dist = dist
                closest_enemy = enemy
        return closest_enemy

    def _update_projectiles(self):
        kill_reward = 0
        for p in reversed(self.projectiles):
            if p['target'] not in self.enemies:
                if p in self.projectiles:
                    self.projectiles.remove(p)
                continue
            
            direction = (p['target']['pos'] - p['pos']).normalize()
            p['pos'] += direction * p['speed']
            
            if p['pos'].distance_to(p['target']['pos']) < 8:
                p['target']['health'] -= 1
                if p['target']['health'] <= 0:
                    self._create_particles(p['target']['pos'], 15, self.COLOR_ENEMY, 2, 20)
                    # Find which clone fired this to attribute kill
                    for clone in self.clones:
                        if self._grid_to_pixel(clone['pos']).distance_to(p['pos']) < clone['range_pixels'] * 1.5:
                             clone['kills_left'] -= 1
                             break
                    if p['target'] in self.enemies:
                        self.enemies.remove(p['target'])
                    kill_reward += 1.0
                    self.score += 1.0
                if p in self.projectiles:
                    self.projectiles.remove(p)
        return kill_reward

    def _update_particles(self):
        for p in list(self.particles):
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                if p in self.particles:
                    self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "core_health": self.core_health, "energy": self.player_energy}

    def _render_game(self):
        # Draw Grid
        for x in range(0, self.WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw Core
        pulse = (1 + math.sin(self.steps * 0.1)) * 5
        for i in range(5):
            alpha = 150 - i * 30
            color = self.COLOR_CORE + (alpha,)
            pygame.gfxdraw.filled_circle(self.screen, int(self.core_pos_pixel.x), int(self.core_pos_pixel.y), int(15 + pulse + i*2), color)
        pygame.gfxdraw.aacircle(self.screen, int(self.core_pos_pixel.x), int(self.core_pos_pixel.y), int(15 + pulse), self.COLOR_CORE)

        # Draw Clones and their ranges
        for clone in self.clones:
            pos = self._grid_to_pixel(clone['pos'])
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), clone['range_pixels'], self.COLOR_GRID + (64,))
            self._draw_entity(pos, self.COLOR_PLAYER, 12, alpha=128)

        # Draw Enemies
        for enemy in self.enemies:
            self._draw_entity(enemy['pos'], self.COLOR_ENEMY, 8, shape='rect')

        # Draw Player
        player_pixel_pos = self._grid_to_pixel(self.player_pos)
        self._draw_entity(player_pixel_pos, self.COLOR_PLAYER, 15, glow=True)
        
        # Draw Projectiles
        for p in self.projectiles:
            if p['target'] in self.enemies:
                pygame.draw.line(self.screen, self.COLOR_PROJECTILE, p['pos'], p['pos'] + (p['target']['pos'] - p['pos']).normalize() * 5, 2)
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), 3, self.COLOR_PROJECTILE)

        # Draw Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life'])) if p['alpha_fade'] else 255
            color = p['color'] + (alpha,)
            radius = int(p['radius'] * (p['life'] / p['max_life'])) if not p['is_shield'] else int(p['radius'] * (1 - (p['life'] / p['max_life'])))
            if p['is_shield']:
                pygame.gfxdraw.aacircle(self.screen, int(p['pos'].x), int(p['pos'].y), radius, color)
            else:
                 if radius > 0:
                    pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), radius, color)

    def _render_ui(self):
        # Core Health
        health_text = self.font_large.render(f"CORE: {max(0, int(self.core_health))}%", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (10, 10))

        # Energy Bar
        energy_text = self.font_large.render("ENERGY", True, self.COLOR_UI_TEXT)
        self.screen.blit(energy_text, (self.WIDTH - 160, 10))
        bar_w = 150
        bar_h = 20
        pygame.draw.rect(self.screen, self.COLOR_GRID, (self.WIDTH - bar_w - 10, 40, bar_w, bar_h))
        fill_w = max(0, (self.player_energy / 100) * bar_w)
        pygame.draw.rect(self.screen, self.COLOR_ENERGY, (self.WIDTH - bar_w - 10, 40, fill_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (self.WIDTH - bar_w - 10, 40, bar_w, bar_h), 1)

        # Step Count
        step_text = self.font_small.render(f"TIME: {self.steps} / {self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        text_rect = step_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT - 20))
        self.screen.blit(step_text, text_rect)

    def _grid_to_pixel(self, grid_pos):
        x = grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2
        y = grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2
        return pygame.Vector2(x, y)

    def _create_particles(self, pos, count, color, speed_max, life, is_shield=False, alpha_fade=False):
        for _ in range(count):
            if is_shield:
                vel = pygame.Vector2(0,0)
                radius = 1
            else:
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(1, speed_max)
                vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
                radius = random.randint(2, 5)

            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': life,
                'max_life': life,
                'color': color,
                'radius': self.CELL_SIZE * 5 if is_shield else radius,
                'is_shield': is_shield,
                'alpha_fade': alpha_fade
            })

    def _draw_entity(self, pos, color, size, shape='circle', alpha=255, glow=False):
        x, y = int(pos.x), int(pos.y)
        if glow:
            for i in range(3):
                glow_alpha = int((alpha / 255) * (100 - i * 30))
                pygame.gfxdraw.filled_circle(self.screen, x, y, size + i * 3, color + (glow_alpha,))

        if shape == 'circle':
            pygame.gfxdraw.filled_circle(self.screen, x, y, size, color + (alpha,))
            pygame.gfxdraw.aacircle(self.screen, x, y, size, color + (alpha,))
        elif shape == 'rect':
            rect = pygame.Rect(x - size, y - size, size * 2, size * 2)
            pygame.draw.rect(self.screen, color, rect)

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # Make sure to unset the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Quantum Fortress")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0.0
    
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0 # This action is unused in the current game logic
        
        # Check for player input
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        
        action = [movement, space_held, shift_held]
        
        # Handle window closing and reset
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0.0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Transpose observation back for Pygame display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']:.2f}, Steps: {info['steps']}")
            # Wait for a moment before closing to show the final state
            pygame.time.wait(2000)
            running = False 
            
        clock.tick(env.metadata["render_fps"])
        
    env.close()