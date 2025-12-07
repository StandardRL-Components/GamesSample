import os
import os
import pygame

os.environ["SDL_VIDEODRIVER"] = "dummy"

import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→↑↓ to select a tower spot. Shift to cycle tower type. Space to build."
    )

    game_description = (
        "A classic Tower Defense game. Place towers to defend your base from waves of enemies. "
        "Survive 10 waves to win."
    )

    auto_advance = False

    # --- Game Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 3000

    # --- Colors ---
    COLOR_BG = (20, 25, 30)
    COLOR_PATH = (40, 50, 60)
    COLOR_PATH_BORDER = (60, 75, 90)
    COLOR_BASE = (0, 100, 200)
    COLOR_BASE_BORDER = (50, 150, 255)
    COLOR_TEXT = (220, 220, 220)
    COLOR_GOLD = (255, 215, 0)
    COLOR_WIN = (0, 255, 127)
    COLOR_LOSE = (255, 69, 0)

    # --- Tower Placement & Path ---
    PATH = [
        (-20, 200), (100, 200), (100, 100), (300, 100), (300, 300),
        (540, 300), (540, 150), (660, 150)
    ]
    TOWER_SPOTS = [
        (180, 50), (180, 150), (420, 180), (420, 350), (50, 280), (220, 250)
    ]
    BASE_POS = (640, 150)
    BASE_SIZE = 40

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
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        self._define_tower_types()
        self.reset()
        

    def _define_tower_types(self):
        self.TOWER_TYPES = [
            {
                "name": "Gun", "cost": 25, "range": 80, "damage": 2,
                "fire_rate": 10, "color": (0, 255, 255), "projectile_speed": 8, "aoe": 0
            },
            {
                "name": "Cannon", "cost": 75, "range": 120, "damage": 10,
                "fire_rate": 40, "color": (255, 165, 0), "projectile_speed": 6, "aoe": 0
            },
            {
                "name": "Rocket", "cost": 100, "range": 150, "damage": 5,
                "fire_rate": 60, "color": (255, 0, 255), "projectile_speed": 4, "aoe": 30
            },
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.base_health = 100
        self.gold = 80
        self.wave_number = 0
        self.inter_wave_timer = 150 # Time until first wave starts

        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.enemies_to_spawn = []
        self.enemy_spawn_timer = 0
        
        self.selected_tower_spot_idx = 0
        self.selected_tower_type_idx = 0
        
        self.last_action_feedback = {}

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = 0
        self.steps += 1
        self.last_action_feedback.clear()

        # --- Handle Player Input ---
        if not self.game_over:
            if movement in [1, 4]:  # Up or Right
                self.selected_tower_spot_idx = (self.selected_tower_spot_idx + 1) % len(self.TOWER_SPOTS)
            elif movement in [2, 3]:  # Down or Left
                self.selected_tower_spot_idx = (self.selected_tower_spot_idx - 1 + len(self.TOWER_SPOTS)) % len(self.TOWER_SPOTS)
            
            if shift_pressed:
                self.selected_tower_type_idx = (self.selected_tower_type_idx + 1) % len(self.TOWER_TYPES)
                self.last_action_feedback['text'] = f"Selected {self.TOWER_TYPES[self.selected_tower_type_idx]['name']}"
                self.last_action_feedback['timer'] = 30
            
            if space_pressed:
                reward += self._place_tower()

        # --- Update Game State ---
        if not self.game_over:
            # Wave Management
            if not self.enemies and not self.enemies_to_spawn:
                self.inter_wave_timer -= 1
                if self.inter_wave_timer <= 0:
                    if self.wave_number > 0:
                        reward += 1 # Wave complete bonus
                    self.wave_number += 1
                    if self.wave_number > 10:
                        self.win = True
                        self.game_over = True
                        reward += 50
                    else:
                        self._start_next_wave()
            
            # Enemy Spawning
            self.enemy_spawn_timer -= 1
            if self.enemy_spawn_timer <= 0 and self.enemies_to_spawn:
                enemy_type = self.enemies_to_spawn.pop(0)
                self.enemies.append(self.Enemy(self.np_random, enemy_type, self.wave_number))
                self.enemy_spawn_timer = 20 # Time between spawns

            # Tower Logic
            for tower in self.towers:
                new_projectiles = tower.update(self.enemies)
                if new_projectiles:
                    self.projectiles.extend(new_projectiles)

            # Projectile Logic
            for proj in self.projectiles[:]:
                if proj.update(self.enemies):
                    self.projectiles.remove(proj)
                    # Create impact particles
                    if proj.aoe > 0: # Rocket explosion
                        # sfx: explosion
                        for _ in range(20):
                            self.particles.append(self.Particle(self.np_random, proj.pos, color=(255, 100, 100), duration=25, size=3, speed=4))
                        # AOE damage
                        for enemy in self.enemies:
                            if math.hypot(enemy.pos[0] - proj.pos[0], enemy.pos[1] - proj.pos[1]) <= proj.aoe:
                                enemy.health -= proj.damage
                    else: # Standard hit
                        # sfx: hit
                        for _ in range(5):
                            self.particles.append(self.Particle(self.np_random, proj.pos, color=(200, 200, 200), duration=15, size=2, speed=2))
            
            # Enemy Logic
            for enemy in self.enemies[:]:
                if enemy.update(self.PATH):
                    # Enemy reached base
                    self.base_health -= enemy.damage_to_base
                    self.enemies.remove(enemy)
                    # sfx: base_damage
                    for _ in range(30):
                        self.particles.append(self.Particle(self.np_random, (self.BASE_POS[0] - self.BASE_SIZE//2, enemy.pos[1]), color=(255, 69, 0), duration=40, size=4, speed=5, gravity=0.1))

                elif enemy.health <= 0:
                    # Enemy defeated
                    reward += 0.1
                    self.gold += enemy.gold_value
                    self.enemies.remove(enemy)
                    # sfx: enemy_die
                    for _ in range(15):
                        self.particles.append(self.Particle(self.np_random, enemy.pos, color=enemy.color, duration=20, size=3, speed=3))
                    for _ in range(5):
                         self.particles.append(self.Particle(self.np_random, enemy.pos, color=self.COLOR_GOLD, duration=30, size=2, speed=2))


        # Particle Logic
        for p in self.particles[:]:
            if p.update():
                self.particles.remove(p)

        # Check Termination Conditions
        if self.base_health <= 0:
            self.base_health = 0
            if not self.game_over: # Apply penalty only once
                reward -= 50
                self.game_over = True
        
        if self.steps >= self.MAX_STEPS and not self.game_over:
            self.game_over = True
            reward -= 50 # Penalty for timeout
        
        self.score += reward
        terminated = self.game_over

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _start_next_wave(self):
        self.inter_wave_timer = 200 # Time before next wave
        num_enemies = 5 + self.wave_number * 2
        for i in range(num_enemies):
            if self.wave_number > 6 and i % 5 == 0:
                self.enemies_to_spawn.append('fast')
            elif self.wave_number > 3 and i % 4 == 0:
                self.enemies_to_spawn.append('strong')
            else:
                self.enemies_to_spawn.append('normal')
        self.np_random.shuffle(self.enemies_to_spawn)

    def _place_tower(self):
        spot_pos = self.TOWER_SPOTS[self.selected_tower_spot_idx]
        tower_type = self.TOWER_TYPES[self.selected_tower_type_idx]
        
        # Check if spot is occupied
        is_occupied = any(t.pos == spot_pos for t in self.towers)
        if is_occupied:
            self.last_action_feedback['text'] = "Spot occupied!"
            self.last_action_feedback['timer'] = 30
            return 0

        # Check if enough gold
        if self.gold >= tower_type['cost']:
            self.gold -= tower_type['cost']
            self.towers.append(self.Tower(self.np_random, spot_pos, tower_type))
            # sfx: build_tower
            for _ in range(20):
                self.particles.append(self.Particle(self.np_random, spot_pos, color=(200,200,255), duration=25, size=3, speed=3))
            self.last_action_feedback['text'] = f"Built {tower_type['name']}!"
            self.last_action_feedback['timer'] = 30
            return 0 # No direct reward for building
        else:
            self.last_action_feedback['text'] = "Not enough gold!"
            self.last_action_feedback['timer'] = 30
            return -0.01 # Small penalty for failed build attempt
            
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw path
        for i in range(len(self.PATH) - 1):
            pygame.draw.line(self.screen, self.COLOR_PATH, self.PATH[i], self.PATH[i+1], 30)
            pygame.draw.line(self.screen, self.COLOR_PATH_BORDER, self.PATH[i], self.PATH[i+1], 32)
        
        # Draw tower spots
        for i, pos in enumerate(self.TOWER_SPOTS):
            is_selected = (i == self.selected_tower_spot_idx) and not self.game_over
            color = (0, 255, 0, 100) if is_selected else (0, 100, 0, 50)
            radius = 18 if is_selected else 15
            
            # Use gfxdraw for antialiasing
            surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(surf, radius, radius, radius, color)
            self.screen.blit(surf, (pos[0] - radius, pos[1] - radius))

        # Draw base
        base_rect = pygame.Rect(self.BASE_POS[0] - self.BASE_SIZE/2, self.BASE_POS[1] - self.BASE_SIZE/2, self.BASE_SIZE, self.BASE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_BASE_BORDER, base_rect, 2, border_radius=5)
        # Base health bar
        if self.base_health > 0:
            health_pct = self.base_health / 100
            health_bar_width = 50
            health_bar_rect = pygame.Rect(self.BASE_POS[0] - health_bar_width / 2, self.BASE_POS[1] - self.BASE_SIZE/2 - 15, health_bar_width, 8)
            fill_rect = pygame.Rect(health_bar_rect.left, health_bar_rect.top, health_bar_width * health_pct, 8)
            pygame.draw.rect(self.screen, (80,0,0), health_bar_rect, border_radius=2)
            pygame.draw.rect(self.screen, (0,200,0), fill_rect, border_radius=2)

        # Draw towers, projectiles, enemies, particles
        for tower in self.towers: tower.draw(self.screen)
        for proj in self.projectiles: proj.draw(self.screen)
        for enemy in self.enemies: enemy.draw(self.screen)
        for p in self.particles: p.draw(self.screen)

    def _render_ui(self):
        # Top-left info: Wave
        wave_text = self.font_small.render(f"Wave: {self.wave_number}/10", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 10))
        
        # Top-right info: Gold
        gold_text = self.font_small.render(f"Gold: {self.gold}", True, self.COLOR_GOLD)
        self.screen.blit(gold_text, (self.SCREEN_WIDTH - gold_text.get_width() - 10, 10))

        # Bottom info: Selected Tower
        if not self.game_over:
            tower_type = self.TOWER_TYPES[self.selected_tower_type_idx]
            tower_info = f"Build: {tower_type['name']} (Cost: {tower_type['cost']})"
            color = self.COLOR_GOLD if self.gold >= tower_type['cost'] else (150, 150, 150)
            select_text = self.font_small.render(tower_info, True, color)
            self.screen.blit(select_text, (10, self.SCREEN_HEIGHT - 25))

        # Action feedback text
        if 'timer' in self.last_action_feedback and self.last_action_feedback['timer'] > 0:
            feedback_text = self.font_small.render(self.last_action_feedback['text'], True, self.COLOR_TEXT)
            pos = (self.SCREEN_WIDTH // 2 - feedback_text.get_width() // 2, self.SCREEN_HEIGHT - 25)
            self.screen.blit(feedback_text, pos)
            self.last_action_feedback['timer'] -= 1

        # Game Over / Win message
        if self.game_over:
            if self.win:
                msg = "YOU WIN!"
                color = self.COLOR_WIN
            else:
                msg = "GAME OVER"
                color = self.COLOR_LOSE
            end_text = self.font_large.render(msg, True, color)
            pos = (self.SCREEN_WIDTH // 2 - end_text.get_width() // 2, self.SCREEN_HEIGHT // 2 - end_text.get_height() // 2)
            self.screen.blit(end_text, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gold": self.gold,
            "wave": self.wave_number,
            "base_health": self.base_health,
        }

    def close(self):
        pygame.quit()

    # --- Helper Classes ---

    class Enemy:
        ENEMY_TYPES = {
            'normal': {'health': 10, 'speed': 1.0, 'color': (200, 0, 0), 'radius': 8, 'gold': 2, 'damage': 5},
            'strong': {'health': 30, 'speed': 0.8, 'color': (139, 0, 0), 'radius': 10, 'gold': 5, 'damage': 10},
            'fast': {'health': 8, 'speed': 2.0, 'color': (255, 100, 100), 'radius': 6, 'gold': 3, 'damage': 3},
        }

        def __init__(self, np_random, enemy_type, wave_num):
            stats = self.ENEMY_TYPES[enemy_type]
            self.max_health = stats['health'] + (wave_num // 2) * 2
            self.health = self.max_health
            self.speed = stats['speed'] + wave_num * 0.05
            self.color = stats['color']
            self.radius = stats['radius']
            self.gold_value = stats['gold']
            self.damage_to_base = stats['damage']
            
            self.path_index = 0
            self.pos = np.array(GameEnv.PATH[0], dtype=float)
            self.distance_on_segment = 0.0

        def update(self, path):
            if self.path_index >= len(path) - 1:
                return True # Reached end

            start = np.array(path[self.path_index])
            end = np.array(path[self.path_index + 1])
            segment_vec = end - start
            segment_len = np.linalg.norm(segment_vec)

            self.distance_on_segment += self.speed
            if self.distance_on_segment >= segment_len:
                self.distance_on_segment -= segment_len
                self.path_index += 1
                if self.path_index >= len(path) - 1:
                    return True # Reached end
                start = np.array(path[self.path_index])
                end = np.array(path[self.path_index + 1])
                segment_vec = end - start
                segment_len = np.linalg.norm(segment_vec)

            if segment_len > 0:
                self.pos = start + (segment_vec / segment_len) * self.distance_on_segment
            
            return False

        def draw(self, screen):
            x, y = int(self.pos[0]), int(self.pos[1])
            pygame.gfxdraw.filled_circle(screen, x, y, self.radius, self.color)
            pygame.gfxdraw.aacircle(screen, x, y, self.radius, tuple(c//2 for c in self.color))
            
            # Health bar
            if self.health < self.max_health:
                health_pct = self.health / self.max_health
                bar_width = self.radius * 2
                bar_y = y - self.radius - 5
                pygame.draw.rect(screen, (80,0,0), (x - bar_width/2, bar_y, bar_width, 3))
                pygame.draw.rect(screen, (0,200,0), (x - bar_width/2, bar_y, bar_width * health_pct, 3))


    class Tower:
        def __init__(self, np_random, pos, tower_type):
            self.np_random = np_random
            self.pos = pos
            self.type = tower_type
            self.cooldown = 0
            self.target = None

        def update(self, enemies):
            self.cooldown = max(0, self.cooldown - 1)
            
            # Find new target if needed
            if self.target is None or self.target.health <= 0 or \
               math.hypot(self.target.pos[0] - self.pos[0], self.target.pos[1] - self.pos[1]) > self.type['range']:
                self.target = None
                in_range = [e for e in enemies if math.hypot(e.pos[0] - self.pos[0], e.pos[1] - self.pos[1]) <= self.type['range']]
                if in_range:
                    self.target = min(in_range, key=lambda e: math.hypot(e.pos[0] - self.pos[0], e.pos[1] - self.pos[1]))
            
            # Fire if ready and has target
            if self.target and self.cooldown == 0:
                self.cooldown = self.type['fire_rate']
                # sfx: tower_shoot
                return [GameEnv.Projectile(self.np_random, self.pos, self.target, self.type)]
            return None

        def draw(self, screen):
            x, y = self.pos
            color = self.type['color']
            
            # Base
            pygame.draw.circle(screen, tuple(c//3 for c in color), (x,y), 12)
            pygame.draw.circle(screen, tuple(c//2 for c in color), (x,y), 10)
            
            # Turret
            p1 = (x - 6, y + 6)
            p2 = (x + 6, y + 6)
            p3 = (x, y - 8)
            
            # Rotate turret towards target
            if self.target:
                angle = math.atan2(self.target.pos[1] - y, self.target.pos[0] - x) - math.pi/2
                points = [p1, p2, p3]
                rotated_points = []
                for px, py in points:
                    dx, dy = px - x, py - y
                    new_x = x + dx * math.cos(angle) - dy * math.sin(angle)
                    new_y = y + dx * math.sin(angle) + dy * math.cos(angle)
                    rotated_points.append((new_x, new_y))
                pygame.gfxdraw.aapolygon(screen, rotated_points, color)
                pygame.gfxdraw.filled_polygon(screen, rotated_points, color)
            else: # Default orientation
                pygame.gfxdraw.aapolygon(screen, [p1, p2, p3], color)
                pygame.gfxdraw.filled_polygon(screen, [p1, p2, p3], color)

    class Projectile:
        def __init__(self, np_random, start_pos, target, tower_type):
            self.np_random = np_random
            self.pos = np.array(start_pos, dtype=float)
            self.target = target
            self.speed = tower_type['projectile_speed']
            self.damage = tower_type['damage']
            self.aoe = tower_type['aoe']
            self.color = (255, 255, 255)

        def update(self, enemies):
            if self.target not in enemies: # Target died
                return True # Mark for deletion

            direction = self.target.pos - self.pos
            dist = np.linalg.norm(direction)

            if dist < self.speed:
                self.pos = self.target.pos
                if self.aoe == 0: # Direct damage
                    self.target.health -= self.damage
                return True
            
            self.pos += (direction / dist) * self.speed
            return False

        def draw(self, screen):
            pygame.draw.circle(screen, self.color, (int(self.pos[0]), int(self.pos[1])), 3)

    class Particle:
        def __init__(self, np_random, pos, color, duration, size, speed, gravity=0):
            self.np_random = np_random
            angle = self.np_random.uniform(0, 2 * math.pi)
            s = self.np_random.uniform(0.5, 1.0) * speed
            self.vel = np.array([math.cos(angle) * s, math.sin(angle) * s])
            self.pos = np.array(pos, dtype=float)
            self.color = color
            self.duration = duration
            self.max_duration = duration
            self.size = size
            self.gravity = gravity

        def update(self):
            self.pos += self.vel
            self.vel[1] += self.gravity
            self.duration -= 1
            return self.duration <= 0

        def draw(self, screen):
            life_pct = self.duration / self.max_duration
            current_size = int(self.size * life_pct)
            if current_size > 0:
                pygame.draw.rect(screen, self.color, (int(self.pos[0]), int(self.pos[1]), current_size, current_size))


# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This part requires a display and is for human testing.
    # It will not run in a headless environment.
    try:
        import os
        os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "dummy", etc.
        pygame.display.init()
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption("Tower Defense")
        
        obs, info = env.reset()
        done = False
        
        while not done:
            movement, space, shift = 0, 0, 0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Render the observation to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            # Since auto_advance is False, we need a small delay to make it playable
            pygame.time.wait(50) 
            
        print(f"Game Over. Final Score: {info['score']}, Wave: {info['wave']}")
        
    except pygame.error as e:
        print(f"Pygame display could not be initialized ({e}). Running headless test.")
        # --- Headless Test ---
        # Re-initialize with the default headless driver
        env = GameEnv()
        obs, info = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            done = terminated or truncated
        print(f"Headless test finished in {step_count} steps.")
        print(f"Final Info: {info}")

    env.close()