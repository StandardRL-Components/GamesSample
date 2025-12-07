
# Generated: 2025-08-28T05:48:29.258083
# Source Brief: brief_02733.md
# Brief Index: 2733

        
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


# Helper classes for game entities
class Crop:
    def __init__(self, pos, crop_type):
        self.pos = pos
        self.type = crop_type
        self.health = 100
        self.max_health = 100
        self.growth = 0
        self.is_mature = False

class Alien:
    def __init__(self, pos, speed, np_random):
        self.pos = list(pos)
        self.health = 25
        self.max_health = 25
        self.speed = speed
        self.target_crop = None
        self.np_random = np_random
        self.angle = self.np_random.uniform(0, 2 * math.pi)
        self.anim_offset = self.np_random.uniform(0, 2 * math.pi)

    def move(self):
        if self.target_crop:
            target_pos = self.target_crop.pos
            dx = target_pos[0] - self.pos[0]
            dy = target_pos[1] - self.pos[1]
            dist = math.hypot(dx, dy)
            if dist > 0:
                self.pos[0] += self.speed * dx / dist
                self.pos[1] += self.speed * dy / dist
        else: # Wander if no crops
            self.pos[0] += math.cos(self.angle) * self.speed * 0.5
            self.pos[1] += math.sin(self.angle) * self.speed * 0.5
            if self.np_random.random() < 0.02:
                self.angle += self.np_random.uniform(-0.5, 0.5)


class Turret:
    def __init__(self, pos, turret_type):
        self.pos = pos
        self.type = turret_type
        self.cooldown = 0
        self.target_alien = None

class Projectile:
    def __init__(self, start_pos, target_alien, proj_type):
        self.pos = list(start_pos)
        self.target = target_alien
        self.type = proj_type
        self.speed = proj_type['speed']
        self.damage = proj_type['damage']

class Particle:
    def __init__(self, pos, color, life, size, velocity):
        self.pos = list(pos)
        self.color = color
        self.life = life
        self.max_life = life
        self.size = size
        self.velocity = velocity

    def update(self):
        self.pos[0] += self.velocity[0]
        self.pos[1] += self.velocity[1]
        self.life -= 1
        return self.life > 0

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrows to move cursor. Space to interact (plant/harvest). Shift to cycle buildable item."
    )

    game_description = (
        "Cultivate alien crops, defend against waves of pests, and amass a fortune as a Space Farmer."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 40
        self.GRID_W, self.GRID_H = self.WIDTH // self.GRID_SIZE, self.HEIGHT // self.GRID_SIZE
        self.MAX_STEPS = 5000
        self.WIN_GOLD = 1000

        # Colors
        self.COLOR_BG = (15, 10, 25)
        self.COLOR_GRID = (30, 20, 50)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 255)
        self.COLOR_ALIEN = (180, 50, 255)
        self.COLOR_PROJECTILE = (0, 255, 255)

        # Entity definitions
        self.BUILDABLES = {
            "Starbloom": {"type": "crop", "cost": 25, "growth_time": 400, "yield": 60, "color": (0, 255, 100)},
            "Pulse Turret": {"type": "turret", "cost": 75, "range": 120, "fire_rate": 30, "proj": {"speed": 8, "damage": 10}}
        }
        self.BUILDABLE_KEYS = list(self.BUILDABLES.keys())

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.gold = 0
        self.game_over_reason = None # "win", "loss", "timeout"
        self.cursor_grid_pos = [0, 0]
        self.farm_plots = {}
        self.aliens = []
        self.defenses = {}
        self.projectiles = []
        self.particles = []
        self.wave_timer = 0
        self.wave_number = 0
        self.alien_spawn_count = 0
        self.alien_base_speed = 0.0
        self.selected_buildable_idx = 0
        self.prev_action = np.array([0, 0, 0])
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.gold = 150
        self.game_over_reason = None
        self.cursor_grid_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.farm_plots = {}
        self.aliens = []
        self.defenses = {}
        self.projectiles = []
        self.particles = []
        self.wave_timer = 600
        self.wave_number = 0
        self.alien_spawn_count = 3
        self.alien_base_speed = 0.8
        self.selected_buildable_idx = 0
        self.prev_action = np.array([0, 0, 0])

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        terminated = False

        # --- 1. Handle Player Input ---
        self._handle_input(movement, space_held, shift_held)
        
        # --- Event-based rewards from input ---
        # This is a proxy for detecting events. We'll set flags inside _handle_input
        # and check them here. For simplicity, we calculate rewards inside the methods.
        reward_from_action, event_info = self._process_actions(action)
        reward += reward_from_action

        # --- 2. Update Game State ---
        self._update_waves()
        self._update_crops()
        self._update_defenses()
        self._update_aliens()
        reward_from_aliens = self._update_projectiles()
        reward += reward_from_aliens
        self._update_particles()
        
        # Continuous survival reward
        reward += 0.001 * len(self.farm_plots)

        # --- 3. Check for Termination ---
        if self.gold >= self.WIN_GOLD:
            terminated = True
            self.game_over_reason = "VICTORY!"
            reward += 100
        
        num_planted_crops = len(self.farm_plots)
        if num_planted_crops > 0 and all(c.health <= 0 for c in self.farm_plots.values()):
            terminated = True
            self.game_over_reason = "DEFEAT"
            reward -= 100

        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over_reason = "TIME OUT"

        self.score += reward
        self.prev_action = action

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        if movement == 1: self.cursor_grid_pos[1] -= 1  # Up
        if movement == 2: self.cursor_grid_pos[1] += 1  # Down
        if movement == 3: self.cursor_grid_pos[0] -= 1  # Left
        if movement == 4: self.cursor_grid_pos[0] += 1  # Right
        self.cursor_grid_pos[0] = np.clip(self.cursor_grid_pos[0], 0, self.GRID_W - 1)
        self.cursor_grid_pos[1] = np.clip(self.cursor_grid_pos[1], 0, self.GRID_H - 1)

    def _process_actions(self, action):
        reward = 0
        event_info = {}
        
        space_pressed = action[1] == 1 and self.prev_action[1] == 0
        shift_pressed = action[2] == 1 and self.prev_action[2] == 0

        if shift_pressed:
            self.selected_buildable_idx = (self.selected_buildable_idx + 1) % len(self.BUILDABLE_KEYS)
            # sfx: UI_CYCLE

        if space_pressed:
            grid_pos = tuple(self.cursor_grid_pos)
            
            # Harvest action
            if grid_pos in self.farm_plots and self.farm_plots[grid_pos].is_mature:
                crop = self.farm_plots.pop(grid_pos)
                item_def = self.BUILDABLES[crop.type]
                self.gold += item_def['yield']
                reward += 10 # Major reward for harvesting
                self._create_particles(crop.pos, (255, 223, 0), 20, count=15)
                # sfx: HARVEST
            
            # Plant/Build action
            elif grid_pos not in self.farm_plots and grid_pos not in self.defenses:
                build_key = self.BUILDABLE_KEYS[self.selected_buildable_idx]
                item_def = self.BUILDABLES[build_key]
                if self.gold >= item_def['cost']:
                    self.gold -= item_def['cost']
                    world_pos = (grid_pos[0] * self.GRID_SIZE + self.GRID_SIZE // 2, 
                                 grid_pos[1] * self.GRID_SIZE + self.GRID_SIZE // 2)

                    if item_def['type'] == 'crop':
                        self.farm_plots[grid_pos] = Crop(world_pos, build_key)
                        reward += 1 # Reward for planting
                        # sfx: PLANT
                    elif item_def['type'] == 'turret':
                        self.defenses[grid_pos] = Turret(world_pos, build_key)
                        reward += 2 # Reward for building defense
                        # sfx: BUILD
        return reward, event_info

    def _update_waves(self):
        self.wave_timer -= 1
        if self.wave_timer <= 0:
            self.wave_number += 1
            self.wave_timer = 500 # Time between waves
            
            for _ in range(self.alien_spawn_count):
                edge = self.np_random.integers(4)
                if edge == 0: x, y = self.np_random.integers(self.WIDTH), -20 # Top
                elif edge == 1: x, y = self.np_random.integers(self.WIDTH), self.HEIGHT + 20 # Bottom
                elif edge == 2: x, y = -20, self.np_random.integers(self.HEIGHT) # Left
                else: x, y = self.WIDTH + 20, self.np_random.integers(self.HEIGHT) # Right
                self.aliens.append(Alien((x, y), self.alien_base_speed, self.np_random))

            if self.wave_number % 2 == 0: # Every 2 waves, increase difficulty
                self.alien_spawn_count = min(20, self.alien_spawn_count + 1)
                self.alien_base_speed = min(2.5, self.alien_base_speed + 0.1)

    def _update_crops(self):
        for crop in self.farm_plots.values():
            if not crop.is_mature:
                crop.growth += 1
                if crop.growth >= self.BUILDABLES[crop.type]['growth_time']:
                    crop.is_mature = True
                    # sfx: CROP_MATURE

    def _update_defenses(self):
        for turret in self.defenses.values():
            if turret.cooldown > 0:
                turret.cooldown -= 1
            else:
                # Find target
                turret.target_alien = None
                min_dist = self.BUILDABLES[turret.type]['range'] ** 2
                for alien in self.aliens:
                    dist_sq = (alien.pos[0] - turret.pos[0])**2 + (alien.pos[1] - turret.pos[1])**2
                    if dist_sq < min_dist:
                        min_dist = dist_sq
                        turret.target_alien = alien
                
                if turret.target_alien:
                    # Fire projectile
                    proj_def = self.BUILDABLES[turret.type]['proj']
                    self.projectiles.append(Projectile(turret.pos, turret.target_alien, proj_def))
                    turret.cooldown = self.BUILDABLES[turret.type]['fire_rate']
                    # sfx: TURRET_FIRE

    def _update_aliens(self):
        reward_from_damage = 0
        living_crops = [c for c in self.farm_plots.values() if c.health > 0]
        
        for alien in self.aliens:
            # Update target
            if not alien.target_crop or alien.target_crop.health <= 0:
                if living_crops:
                    alien.target_crop = min(living_crops, key=lambda c: math.hypot(c.pos[0] - alien.pos[0], c.pos[1] - alien.pos[1]))
                else:
                    alien.target_crop = None
            
            alien.move()

            # Check for collision with crops
            if alien.target_crop:
                dist = math.hypot(alien.pos[0] - alien.target_crop.pos[0], alien.pos[1] - alien.target_crop.pos[1])
                if dist < self.GRID_SIZE // 2:
                    alien.target_crop.health -= 5
                    reward_from_damage -= 0.5 # Penalty for crop damage
                    self._create_particles(alien.target_crop.pos, (255, 0, 0), 10, count=3)
                    # sfx: CROP_DAMAGE
                    # For simplicity, alien is destroyed on contact to prevent clumping
                    alien.health = 0 

        self.aliens = [a for a in self.aliens if a.health > 0]
        return reward_from_damage

    def _update_projectiles(self):
        reward = 0
        for p in self.projectiles[:]:
            if p.target.health <= 0: # Target is dead
                self.projectiles.remove(p)
                continue
            
            dx = p.target.pos[0] - p.pos[0]
            dy = p.target.pos[1] - p.pos[1]
            dist = math.hypot(dx, dy)

            if dist < p.speed:
                p.target.health -= p.damage
                if p.target.health <= 0:
                    reward += 5 # Reward for destroying an alien
                    self._create_particles(p.target.pos, self.COLOR_ALIEN, 20, count=20)
                    # sfx: ALIEN_DEATH
                else:
                    self._create_particles(p.target.pos, (255, 100, 100), 10, count=5)
                    # sfx: ALIEN_HIT
                self.projectiles.remove(p)
            else:
                p.pos[0] += p.speed * dx / dist
                p.pos[1] += p.speed * dy / dist
        return reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.update()]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_grid()
        self._render_entities()
        self._render_effects()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "gold": self.gold, "wave": self.wave_number}

    def _render_background(self):
        # Create static stars on first call
        if not hasattr(self, 'stars'):
            self.stars = []
            for _ in range(100):
                self.stars.append(
                    (
                        (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT)),
                        self.np_random.integers(1, 3),
                        self.np_random.integers(50, 150)
                    )
                )
        for pos, size, brightness in self.stars:
            pygame.gfxdraw.pixel(self.screen, pos[0], pos[1], (brightness, brightness, brightness))

    def _render_grid(self):
        for x in range(0, self.WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _render_entities(self):
        # Crops
        for crop in self.farm_plots.values():
            item_def = self.BUILDABLES[crop.type]
            if crop.is_mature:
                color = item_def['color']
                size = 14
                pygame.draw.circle(self.screen, (255,255,255), (int(crop.pos[0]), int(crop.pos[1])), size + 3) # Glow
            else:
                growth_ratio = crop.growth / item_def['growth_time']
                color = tuple(int(c * growth_ratio) for c in item_def['color'])
                size = int(5 + 9 * growth_ratio)
            pygame.draw.circle(self.screen, color, (int(crop.pos[0]), int(crop.pos[1])), size)
            # Health bar
            if crop.health < crop.max_health:
                health_ratio = crop.health / crop.max_health
                bar_w = self.GRID_SIZE * 0.8
                bar_h = 5
                bar_x = crop.pos[0] - bar_w / 2
                bar_y = crop.pos[1] - self.GRID_SIZE / 2
                pygame.draw.rect(self.screen, (255,0,0), (bar_x, bar_y, bar_w, bar_h))
                pygame.draw.rect(self.screen, (0,255,0), (bar_x, bar_y, bar_w * health_ratio, bar_h))
        
        # Defenses
        for turret in self.defenses.values():
            item_def = self.BUILDABLES[turret.type]
            pos = (int(turret.pos[0]), int(turret.pos[1]))
            pygame.draw.rect(self.screen, (0, 150, 255), (pos[0]-10, pos[1]-10, 20, 20))
            pygame.draw.rect(self.screen, (200, 220, 255), (pos[0]-8, pos[1]-8, 16, 16))
            if turret.cooldown > item_def['fire_rate'] - 5: # Firing flash
                 pygame.draw.circle(self.screen, self.COLOR_PROJECTILE, pos, 12, 2)

        # Aliens
        for alien in self.aliens:
            pos = (int(alien.pos[0]), int(alien.pos[1]))
            anim_sin = math.sin(self.steps * 0.2 + alien.anim_offset)
            size = int(10 + 2 * anim_sin)
            pygame.draw.circle(self.screen, (255,255,255), pos, size + 2) # Glow
            pygame.draw.circle(self.screen, self.COLOR_ALIEN, pos, size)

    def _render_effects(self):
        # Projectiles
        for p in self.projectiles:
            pos = (int(p.pos[0]), int(p.pos[1]))
            pygame.draw.circle(self.screen, (255,255,255), pos, 5)
            pygame.draw.circle(self.screen, self.COLOR_PROJECTILE, pos, 3)

        # Particles
        for p in self.particles:
            alpha = p.life / p.max_life
            color = (int(p.color[0] * alpha), int(p.color[1] * alpha), int(p.color[2] * alpha))
            size = int(p.size * alpha)
            if size > 0:
                pygame.draw.circle(self.screen, color, (int(p.pos[0]), int(p.pos[1])), size)

    def _render_ui(self):
        # Cursor
        cursor_x = self.cursor_grid_pos[0] * self.GRID_SIZE
        cursor_y = self.cursor_grid_pos[1] * self.GRID_SIZE
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, (cursor_x, cursor_y, self.GRID_SIZE, self.GRID_SIZE), 2)
        
        # Top-left UI (Gold)
        gold_text = self.font_small.render(f"GOLD: {self.gold}", True, self.COLOR_TEXT)
        self.screen.blit(gold_text, (10, 10))
        
        # Selected buildable
        build_key = self.BUILDABLE_KEYS[self.selected_buildable_idx]
        item_def = self.BUILDABLES[build_key]
        build_text = self.font_small.render(f"SELECT: {build_key} (Cost: {item_def['cost']})", True, self.COLOR_TEXT)
        self.screen.blit(build_text, (10, 35))

        # Top-right UI (Wave Timer)
        wave_str = f"WAVE {self.wave_number+1} IN: {self.wave_timer // 30}" if self.wave_timer > 0 else f"WAVE {self.wave_number} ACTIVE"
        wave_text = self.font_small.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))

        # Game Over Message
        if self.game_over_reason:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0,0))
            
            end_text = self.font_large.render(self.game_over_reason, True, (255, 255, 255))
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _create_particles(self, pos, color, life, size, count=1):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            velocity = (math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append(Particle(pos, color, life, size, velocity))

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

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    total_reward = 0
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Space Farmer")
    clock = pygame.time.Clock()

    action = env.action_space.sample()
    action.fill(0)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        # Map keys to MultiDiscrete action space
        action.fill(0) # Reset actions
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Gold: {info['gold']}, Steps: {info['steps']}")
            obs, info = env.reset() # Auto-reset
        
        clock.tick(30) # Run at 30 FPS

    pygame.quit()