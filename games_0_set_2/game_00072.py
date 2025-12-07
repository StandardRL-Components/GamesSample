
# Generated: 2025-08-27T12:31:25.784474
# Source Brief: brief_00072.md
# Brief Index: 72

        
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

    # User-facing strings
    user_guide = (
        "Controls: Use arrow keys to move the placement reticle. "
        "Press Space to place a fast-firing tower, or Shift to place a long-range tower."
    )
    game_description = (
        "Defend your base from waves of enemies by strategically placing towers in a minimalist, top-down tower defense game."
    )

    # Frame advance setting
    auto_advance = True

    # --- Constants ---
    # Game parameters
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1800  # 60 seconds at 30 FPS logic steps
    BASE_SIZE = 40
    RETICLE_SPEED = 8
    ENEMY_SPAWN_RATE_INITIAL = 75  # Steps per enemy
    ENEMY_SPAWN_RATE_DIFFICULTY_INTERVAL = 300 # Steps between difficulty increase
    ENEMY_SPAWN_RATE_DECREASE = 5 # Amount to decrease spawn rate by

    # Colors
    COLOR_BG = (32, 32, 32)
    COLOR_BASE = (64, 224, 208)
    COLOR_ENEMY = (255, 65, 54)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_RETICLE = (255, 255, 255)
    COLOR_HEALTH_BAR_BG = (80, 80, 80)

    # Tower specifications
    TOWER_SPECS = {
        'fast': {
            'color': (0, 116, 217),
            'range': 80,
            'cooldown': 15, # steps
            'projectile_speed': 10,
            'projectile_damage': 10,
            'projectile_color': (255, 220, 0),
            'size': 12,
        },
        'slow': {
            'color': (177, 13, 201),
            'range': 150,
            'cooldown': 45, # steps
            'projectile_speed': 7,
            'projectile_damage': 35,
            'projectile_color': (127, 219, 255),
            'size': 16,
        }
    }
    TOWER_PLACEMENT_RADIUS = 20

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # State variables are initialized in reset()
        self.base_pos = None
        self.reticle_pos = None
        self.enemies = None
        self.towers = None
        self.projectiles = None
        self.particles = None
        self.steps = None
        self.score = None
        self.enemies_defeated = None
        self.enemy_spawn_timer = None
        self.current_enemy_spawn_rate = None
        self.game_over = None
        self.game_won = None

        self.reset()
        
        # self.validate_implementation() # Optional: call to test during development

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.base_pos = pygame.math.Vector2(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
        self.reticle_pos = pygame.math.Vector2(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 4)

        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.steps = 0
        self.score = 0
        self.enemies_defeated = 0
        
        self.current_enemy_spawn_rate = self.ENEMY_SPAWN_RATE_INITIAL
        self.enemy_spawn_timer = self.current_enemy_spawn_rate

        self.game_over = False
        self.game_won = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        reward = 0.0
        terminated = False

        self._handle_input(movement, space_pressed, shift_pressed)

        self._update_towers()
        self._update_projectiles()
        
        enemy_kill_reward = self._update_enemies()
        reward += enemy_kill_reward
        
        self._update_particles()
        self._spawn_enemies()
        
        self.steps += 1
        
        # Check termination conditions
        if self.game_over:
            terminated = True
            reward -= 100.0
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_won = True
            reward += 100.0
        else:
            # Survival reward
            reward += 0.01

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _handle_input(self, movement, space_pressed, shift_pressed):
        # Move reticle
        if movement == 1: self.reticle_pos.y -= self.RETICLE_SPEED
        elif movement == 2: self.reticle_pos.y += self.RETICLE_SPEED
        elif movement == 3: self.reticle_pos.x -= self.RETICLE_SPEED
        elif movement == 4: self.reticle_pos.x += self.RETICLE_SPEED

        # Clamp reticle to screen bounds
        self.reticle_pos.x = max(0, min(self.SCREEN_WIDTH, self.reticle_pos.x))
        self.reticle_pos.y = max(0, min(self.SCREEN_HEIGHT, self.reticle_pos.y))

        # Place towers (prioritize shift)
        tower_type_to_place = None
        if shift_pressed:
            tower_type_to_place = 'slow'
        elif space_pressed:
            tower_type_to_place = 'fast'

        if tower_type_to_place:
            self._place_tower(tower_type_to_place)

    def _place_tower(self, tower_type):
        # Check if placement is too close to another tower or the base
        if self.reticle_pos.distance_to(self.base_pos) < self.BASE_SIZE:
            return # Too close to base
        for tower in self.towers:
            if self.reticle_pos.distance_to(tower['pos']) < self.TOWER_PLACEMENT_RADIUS * 2:
                return # Too close to another tower

        # SFX: place_tower.wav
        specs = self.TOWER_SPECS[tower_type]
        self.towers.append({
            'pos': self.reticle_pos.copy(),
            'type': tower_type,
            'cooldown_timer': 0,
            'range': specs['range'],
            'cooldown': specs['cooldown'],
        })

    def _spawn_enemies(self):
        self.enemy_spawn_timer -= 1
        if self.enemy_spawn_timer <= 0:
            # SFX: enemy_spawn.wav
            edge = self.np_random.integers(4)
            if edge == 0: # Top
                pos = pygame.math.Vector2(self.np_random.integers(self.SCREEN_WIDTH), 0)
            elif edge == 1: # Bottom
                pos = pygame.math.Vector2(self.np_random.integers(self.SCREEN_WIDTH), self.SCREEN_HEIGHT)
            elif edge == 2: # Left
                pos = pygame.math.Vector2(0, self.np_random.integers(self.SCREEN_HEIGHT))
            else: # Right
                pos = pygame.math.Vector2(self.SCREEN_WIDTH, self.np_random.integers(self.SCREEN_HEIGHT))

            self.enemies.append({
                'pos': pos,
                'health': 100,
                'max_health': 100,
                'speed': self.np_random.uniform(1.0, 1.5),
            })
            
            # Increase difficulty over time
            if self.steps > 0 and self.steps % self.ENEMY_SPAWN_RATE_DIFFICULTY_INTERVAL == 0:
                self.current_enemy_spawn_rate = max(20, self.current_enemy_spawn_rate - self.ENEMY_SPAWN_RATE_DECREASE)

            self.enemy_spawn_timer = self.current_enemy_spawn_rate

    def _update_enemies(self):
        kill_reward = 0
        for enemy in self.enemies[:]:
            # Move towards base
            direction = (self.base_pos - enemy['pos']).normalize()
            enemy['pos'] += direction * enemy['speed']

            # Check for collision with base
            if enemy['pos'].distance_to(self.base_pos) < self.BASE_SIZE / 2:
                self.game_over = True
                self.enemies.remove(enemy)
                # SFX: base_hit.wav
                self._create_explosion(self.base_pos, self.COLOR_BASE, 50)
                continue

            if enemy['health'] <= 0:
                # SFX: enemy_explode.wav
                kill_reward += 1.0
                self.score += 1
                self.enemies_defeated += 1
                self._create_explosion(enemy['pos'], self.COLOR_ENEMY, 20)
                self.enemies.remove(enemy)
        return kill_reward

    def _update_towers(self):
        for tower in self.towers:
            if tower['cooldown_timer'] > 0:
                tower['cooldown_timer'] -= 1
                continue

            # Find target
            target = None
            min_dist = float('inf')
            for enemy in self.enemies:
                dist = tower['pos'].distance_to(enemy['pos'])
                if dist < tower['range'] and dist < min_dist:
                    min_dist = dist
                    target = enemy

            if target:
                # SFX: tower_shoot.wav
                specs = self.TOWER_SPECS[tower['type']]
                self.projectiles.append({
                    'pos': tower['pos'].copy(),
                    'target': target,
                    'speed': specs['projectile_speed'],
                    'damage': specs['projectile_damage'],
                    'color': specs['projectile_color'],
                })
                tower['cooldown_timer'] = tower['cooldown']

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            if p['target'] not in self.enemies:
                self.projectiles.remove(p)
                continue

            direction = (p['target']['pos'] - p['pos'])
            if direction.length() < p['speed']:
                # Hit target
                p['target']['health'] -= p['damage']
                self.projectiles.remove(p)
                # SFX: projectile_hit.wav
            else:
                p['pos'] += direction.normalize() * p['speed']

    def _update_particles(self):
        for particle in self.particles[:]:
            particle['pos'] += particle['vel']
            particle['lifespan'] -= 1
            if particle['lifespan'] <= 0:
                self.particles.remove(particle)

    def _create_explosion(self, pos, color, num_particles):
        for _ in range(num_particles):
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.math.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2)),
                'lifespan': self.np_random.integers(15, 30),
                'color': color,
                'size': self.np_random.integers(2, 5)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "enemies_defeated": self.enemies_defeated,
            "towers_placed": len(self.towers),
        }

    def _render_game(self):
        # Draw tower ranges
        for tower in self.towers:
            specs = self.TOWER_SPECS[tower['type']]
            pygame.gfxdraw.filled_circle(
                self.screen, int(tower['pos'].x), int(tower['pos'].y), 
                specs['range'], (*specs['color'], 20)
            )
            pygame.gfxdraw.aacircle(
                self.screen, int(tower['pos'].x), int(tower['pos'].y), 
                specs['range'], (*specs['color'], 60)
            )

        # Draw base
        base_rect = pygame.Rect(0, 0, self.BASE_SIZE, self.BASE_SIZE)
        base_rect.center = (int(self.base_pos.x), int(self.base_pos.y))
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect, border_radius=4)
        pygame.draw.rect(self.screen, (255,255,255), base_rect, width=2, border_radius=4)

        # Draw towers
        for tower in self.towers:
            specs = self.TOWER_SPECS[tower['type']]
            tower_rect = pygame.Rect(0, 0, specs['size'], specs['size'])
            tower_rect.center = (int(tower['pos'].x), int(tower['pos'].y))
            pygame.draw.rect(self.screen, specs['color'], tower_rect, border_radius=2)
            pygame.draw.rect(self.screen, (255,255,255), tower_rect, width=1, border_radius=2)

        # Draw projectiles
        for p in self.projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), 3, p['color'])

        # Draw enemies
        for enemy in self.enemies:
            # Triangle pointing to base
            direction = (self.base_pos - enemy['pos']).normalize()
            p1 = enemy['pos'] + direction * 10
            p2 = enemy['pos'] + direction.rotate(135) * 8
            p3 = enemy['pos'] + direction.rotate(-135) * 8
            pygame.gfxdraw.aapolygon(self.screen, [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)], self.COLOR_ENEMY)
            pygame.gfxdraw.filled_polygon(self.screen, [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)], self.COLOR_ENEMY)

            # Health bar
            health_pct = max(0, enemy['health'] / enemy['max_health'])
            bar_width = 20
            bar_height = 4
            bar_pos = (enemy['pos'].x - bar_width / 2, enemy['pos'].y - 20)
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (*bar_pos, bar_width, bar_height))
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (*bar_pos, bar_width * health_pct, bar_height))

        # Draw particles
        for particle in self.particles:
            alpha = int(255 * (particle['lifespan'] / 30.0))
            color = (*particle['color'], alpha)
            temp_surf = pygame.Surface((particle['size']*2, particle['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (particle['size'], particle['size']), particle['size'])
            self.screen.blit(temp_surf, (int(particle['pos'].x - particle['size']), int(particle['pos'].y - particle['size'])))

        # Draw reticle
        if not self.game_over:
            x, y = int(self.reticle_pos.x), int(self.reticle_pos.y)
            pygame.draw.line(self.screen, self.COLOR_RETICLE, (x - 10, y), (x + 10, y), 2)
            pygame.draw.line(self.screen, self.COLOR_RETICLE, (x, y - 10), (x, y + 10), 2)

    def _render_ui(self):
        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / 30)
        timer_text = self.font_small.render(f"Time: {time_left:.1f}s", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (10, 10))

        # Score
        score_text = self.font_small.render(f"Defeated: {self.enemies_defeated}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))
        
        # Game Over / Win Text
        if self.game_over or self.game_won:
            message = "YOU SURVIVED" if self.game_won else "BASE DESTROYED"
            color = self.COLOR_BASE if self.game_won else self.COLOR_ENEMY
            end_text = self.font_large.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
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
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    
    # --- Human Player Controls ---
    # To map keyboard presses to the MultiDiscrete action space
    key_to_action = {
        pygame.K_UP:    1,
        pygame.K_DOWN:  2,
        pygame.K_LEFT:  3,
        pygame.K_RIGHT: 4,
    }

    def get_human_action(keys):
        action = [0, 0, 0] # [movement, space, shift]
        for key, move_action in key_to_action.items():
            if keys[key]:
                action[0] = move_action
                break # Only one movement key at a time
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1
            
        return action

    # --- Game Loop ---
    obs, info = env.reset()
    terminated = False
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not terminated:
            keys = pygame.key.get_pressed()
            action = get_human_action(keys)
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Run at 30 FPS

    env.close()