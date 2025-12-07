
# Generated: 2025-08-27T14:52:53.492577
# Source Brief: brief_00818.md
# Brief Index: 818

        
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


# --- Helper Classes for Game Entities ---

class Particle:
    """A simple class for a cosmetic particle effect."""
    def __init__(self, pos, color, life, vel):
        self.pos = list(pos)
        self.color = color
        self.life = life
        self.initial_life = life
        self.vel = vel

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.life -= 1

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.initial_life))
            color = self.color + (alpha,)
            radius = int(2 * (self.life / self.initial_life))
            pygame.gfxdraw.filled_circle(surface, int(self.pos[0]), int(self.pos[1]), max(1, radius), color)


class Enemy:
    """Represents an enemy unit."""
    def __init__(self, path, speed=1.5, health=100):
        self.path = path
        self.path_index = 0
        self.pos = list(self.path[0])
        self.speed = speed
        self.max_health = health
        self.health = health

    def update(self):
        if self.path_index >= len(self.path) - 1:
            return True  # Reached the end

        target_pos = self.path[self.path_index + 1]
        dist = math.dist(self.pos, target_pos)

        if dist < self.speed:
            self.pos = list(target_pos)
            self.path_index += 1
        else:
            angle = math.atan2(target_pos[1] - self.pos[1], target_pos[0] - self.pos[0])
            self.pos[0] += math.cos(angle) * self.speed
            self.pos[1] += math.sin(angle) * self.speed
        return False

    def draw(self, surface):
        # Body
        size = 12
        rect = pygame.Rect(self.pos[0] - size / 2, self.pos[1] - size / 2, size, size)
        pygame.draw.rect(surface, (255, 64, 64), rect, border_radius=2)
        pygame.draw.rect(surface, (255, 128, 128), rect, width=1, border_radius=2)
        
        # Health bar
        bar_width = 20
        bar_height = 4
        bar_x = self.pos[0] - bar_width / 2
        bar_y = self.pos[1] - size - 5
        health_ratio = self.health / self.max_health
        pygame.draw.rect(surface, (60, 20, 20), (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(surface, (255, 64, 64), (bar_x, bar_y, bar_width * health_ratio, bar_height))

class Tower:
    """Represents a tower."""
    def __init__(self, pos, tower_type, np_random):
        self.pos = pos
        self.type = tower_type
        self.np_random = np_random
        self.target = None
        
        if tower_type == 1: # Fast, low damage
            self.range = 80
            self.damage = 10
            self.fire_rate = 10 # frames per shot
            self.color = (64, 128, 255)
        else: # Slow, high damage
            self.range = 100
            self.damage = 35
            self.fire_rate = 45
            self.color = (192, 64, 255)
        
        self.cooldown = 0
        self.placement_animation = 20

    def update(self, enemies):
        if self.cooldown > 0:
            self.cooldown -= 1
        
        if self.placement_animation > 0:
            self.placement_animation -= 1

        # Find a new target if current is dead or out of range
        if self.target and (self.target.health <= 0 or math.dist(self.pos, self.target.pos) > self.range):
            self.target = None

        if not self.target:
            in_range = [e for e in enemies if math.dist(self.pos, e.pos) <= self.range]
            if in_range:
                self.target = in_range[0] # Target the first enemy in range

    def can_fire(self):
        return self.cooldown == 0 and self.target is not None

    def fire(self):
        self.cooldown = self.fire_rate
        # sfx: tower_shoot_sound
        projectile_speed = 8
        return Projectile(self.pos, self.target, self.damage, projectile_speed)

    def draw(self, surface):
        radius = 12
        if self.placement_animation > 0:
            anim_prog = self.placement_animation / 20
            anim_radius = int(radius * (1 + anim_prog * 2))
            alpha = int(128 * (1 - anim_prog))
            pygame.gfxdraw.filled_circle(surface, int(self.pos[0]), int(self.pos[1]), anim_radius, self.color + (alpha,))

        pygame.gfxdraw.filled_circle(surface, int(self.pos[0]), int(self.pos[1]), radius, self.color)
        pygame.gfxdraw.aacircle(surface, int(self.pos[0]), int(self.pos[1]), radius, (200, 200, 255))
        # Draw a small inner detail
        pygame.gfxdraw.filled_circle(surface, int(self.pos[0]), int(self.pos[1]), 4, (255, 255, 255))


class Projectile:
    """Represents a projectile fired from a tower."""
    def __init__(self, pos, target, damage, speed):
        self.pos = list(pos)
        self.target = target
        self.damage = damage
        self.speed = speed
        self.trail = []

    def update(self):
        if self.target.health <= 0:
            return "miss" # Target is already dead

        dist = math.dist(self.pos, self.target.pos)
        self.trail.append(list(self.pos))
        if len(self.trail) > 5:
            self.trail.pop(0)

        if dist < self.speed:
            self.target.health -= self.damage
            # sfx: enemy_hit_sound
            return "hit"
        else:
            angle = math.atan2(self.target.pos[1] - self.pos[1], self.target.pos[0] - self.pos[0])
            self.pos[0] += math.cos(angle) * self.speed
            self.pos[1] += math.sin(angle) * self.speed
            return "flying"

    def draw(self, surface):
        # Draw trail
        for i, p in enumerate(self.trail):
            alpha = int(255 * (i / len(self.trail)))
            pygame.gfxdraw.filled_circle(surface, int(p[0]), int(p[1]), 1, (255, 255, 255, alpha))
        # Draw head
        pygame.gfxdraw.filled_circle(surface, int(self.pos[0]), int(self.pos[1]), 3, (255, 255, 255))
        pygame.gfxdraw.aacircle(surface, int(self.pos[0]), int(self.pos[1]), 3, (255, 255, 255))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to select a build location. Press space to build a fast, low-damage tower. "
        "Hold shift to build a slow, high-damage tower."
    )

    game_description = (
        "A minimalist tower defense game. Place towers to defend your base (green square) from waves of enemies (red squares). "
        "Survive all waves to win."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.width, self.height = 640, 400
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        # --- Game Constants ---
        self.COLOR_BG = (32, 32, 40)
        self.COLOR_PATH = (64, 64, 80)
        self.COLOR_BASE = (64, 255, 128)
        self.COLOR_ZONE = (48, 48, 64)
        self.COLOR_SELECT = (128, 255, 255)
        self.MAX_STEPS = 30 * 60 # 60 seconds at 30fps

        self.ENEMY_PATH = [
            (-20, 50), (150, 50), (150, 300), (490, 300), (490, 50), (self.width + 20, 50)
        ]
        self.TOWER_ZONES = [
            (80, 175), (225, 175), (415, 175), (560, 175)
        ]
        self.BASE_POS = (self.width - 40, 30)
        self.BASE_SIZE = (40, 40)
        
        self.TOTAL_ENEMIES = 20
        self.BASE_SPAWN_RATE = 90 # frames
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = None # "VICTORY" or "DEFEAT"
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.placed_towers = {i: None for i in range(len(self.TOWER_ZONES))}
        self.selected_zone_idx = 0
        
        self.enemies_spawned = 0
        self.enemies_killed = 0
        self.current_spawn_rate = self.BASE_SPAWN_RATE
        self.spawn_timer = self.current_spawn_rate

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1

        action_reward = self._handle_input(action)
        reward += action_reward
        
        game_event_reward = self._update_game_state()
        reward += game_event_reward
        
        self.score += reward

        terminated = self._check_termination()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Update selection based on movement
        if movement != 0:
            # sfx: ui_blip_sound
            current_x, _ = self.TOWER_ZONES[self.selected_zone_idx]
            if movement == 3: # left
                next_idx = (self.selected_zone_idx - 1 + len(self.TOWER_ZONES)) % len(self.TOWER_ZONES)
            elif movement == 4: # right
                next_idx = (self.selected_zone_idx + 1) % len(self.TOWER_ZONES)
            # Up/down are not used to avoid ambiguity
            else:
                next_idx = self.selected_zone_idx
            self.selected_zone_idx = next_idx

        # Place tower
        action_reward = 0
        if space_held or shift_held:
            if self.placed_towers[self.selected_zone_idx] is None:
                # sfx: place_tower_sound
                tower_type = 2 if shift_held else 1
                pos = self.TOWER_ZONES[self.selected_zone_idx]
                new_tower = Tower(pos, tower_type, self.np_random)
                self.towers.append(new_tower)
                self.placed_towers[self.selected_zone_idx] = new_tower
            else:
                action_reward = -0.1 # Small penalty for trying to build on an existing tower
        return action_reward

    def _update_game_state(self):
        reward = 0
        
        # 1. Spawn Enemies
        self.spawn_timer -= 1
        if self.spawn_timer <= 0 and self.enemies_spawned < self.TOTAL_ENEMIES:
            # sfx: enemy_spawn_sound
            self.enemies.append(Enemy(self.ENEMY_PATH, speed=1.2 + self.enemies_killed * 0.05))
            self.enemies_spawned += 1
            self.spawn_timer = self.current_spawn_rate

        # 2. Update Towers & Fire Projectiles
        for tower in self.towers:
            tower.update(self.enemies)
            if tower.can_fire():
                self.projectiles.append(tower.fire())

        # 3. Update Projectiles
        new_projectiles = []
        for p in self.projectiles:
            result = p.update()
            if result == "hit":
                reward += 0.1
                self._create_particles(p.pos, (255, 255, 255), 5)
            elif result == "flying":
                new_projectiles.append(p)
        self.projectiles = new_projectiles

        # 4. Update Enemies
        surviving_enemies = []
        for enemy in self.enemies:
            if enemy.health <= 0:
                # sfx: enemy_destroy_sound
                reward += 1.0
                self.enemies_killed += 1
                self._create_particles(enemy.pos, (255, 64, 64), 20)
                # Update spawn rate every 5 kills
                if self.enemies_killed % 5 == 0:
                    self.current_spawn_rate = max(30, self.BASE_SPAWN_RATE - (self.enemies_killed // 5) * 15)
                continue
            
            if enemy.update(): # Reached base
                self.game_over = True
                self.win_state = "DEFEAT"
                reward -= 100
                # sfx: base_destroyed_sound
                self._create_particles((self.BASE_POS[0] + self.BASE_SIZE[0]/2, self.BASE_POS[1] + self.BASE_SIZE[1]/2), self.COLOR_BASE, 50)
            else:
                surviving_enemies.append(enemy)
        self.enemies = surviving_enemies
        
        # 5. Update particles
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()
            
        return reward

    def _check_termination(self):
        if self.game_over:
            return True
        if self.enemies_killed >= self.TOTAL_ENEMIES:
            self.game_over = True
            self.win_state = "VICTORY"
            self.score += 100 # Add final bonus directly to score
            # sfx: victory_sound
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win_state = "DEFEAT" # Time ran out
            self.score -= 100
            return True
        return False

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            life = self.np_random.integers(10, 20)
            self.particles.append(Particle(pos, color, life, vel))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_path()
        self._render_base()
        self._render_placement_zones()
        
        for particle in self.particles: particle.draw(self.screen)
        for tower in self.towers: tower.draw(self.screen)
        for proj in self.projectiles: proj.draw(self.screen)
        for enemy in self.enemies: enemy.draw(self.screen)
        
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_path(self):
        for i in range(len(self.ENEMY_PATH) - 1):
            p1 = self.ENEMY_PATH[i]
            p2 = self.ENEMY_PATH[i+1]
            pygame.draw.line(self.screen, self.COLOR_PATH, p1, p2, 10)
    
    def _render_base(self):
        base_rect = pygame.Rect(self.BASE_POS, self.BASE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect, border_radius=4)

    def _render_placement_zones(self):
        for i, pos in enumerate(self.TOWER_ZONES):
            radius = 25
            # Draw glow for selected zone
            if i == self.selected_zone_idx:
                glow_radius = int(radius * 1.5)
                s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, self.COLOR_SELECT + (50,), (glow_radius, glow_radius), glow_radius)
                self.screen.blit(s, (pos[0] - glow_radius, pos[1] - glow_radius))
                pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), radius, self.COLOR_SELECT)
            
            # Draw the zone itself
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), radius, self.COLOR_ZONE)

    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, (224, 224, 224))
        self.screen.blit(score_text, (self.width - score_text.get_width() - 10, 10))
        
        enemies_text = self.font_small.render(f"ENEMIES: {self.enemies_killed}/{self.TOTAL_ENEMIES}", True, (224, 224, 224))
        self.screen.blit(enemies_text, (10, 10))
        
    def _render_game_over(self):
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        color = self.COLOR_BASE if self.win_state == "VICTORY" else (255, 64, 64)
        text = self.font_large.render(self.win_state, True, color)
        text_rect = text.get_rect(center=(self.width / 2, self.height / 2))
        self.screen.blit(text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "enemies_killed": self.enemies_killed,
            "towers_placed": len(self.towers)
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # Game loop
    while running:
        # --- Event Handling ---
        movement, space, shift = 0, 0, 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}. Total reward: {total_reward}")
            # In a real scenario, you might wait for a key press to reset
            # obs, info = env.reset()
            # total_reward = 0

        # --- Rendering ---
        # The observation is already the rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Timing ---
        clock.tick(30) # Match the intended FPS of the environment
        
    env.close()