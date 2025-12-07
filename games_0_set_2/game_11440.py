import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:59:29.167173
# Source Brief: brief_01440.md
# Brief Index: 1440
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper classes for game entities

class Particle:
    """A simple particle for visual effects."""
    def __init__(self, x, y, color, life, size_range=(2, 5), speed_range=(1, 3)):
        self.x = x
        self.y = y
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(*speed_range)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.life = life
        self.max_life = life
        self.color = color
        self.size = random.uniform(*size_range)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.max_life))
            color_with_alpha = self.color + (alpha,)
            pygame.draw.circle(surface, color_with_alpha, (int(self.x), int(self.y)), int(self.size))

class Projectile:
    """A projectile fired by the player."""
    def __init__(self, x, y, target_x, target_y, speed, color, damage):
        self.x = x
        self.y = y
        angle = math.atan2(target_y - y, target_x - x)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.color = color
        self.damage = damage
        self.size = 4

    def update(self):
        self.x += self.vx
        self.y += self.vy

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.size)
        pygame.gfxdraw.aacircle(surface, int(self.x), int(self.y), self.size, self.color)

class Enemy:
    """An enemy that moves towards the tower."""
    def __init__(self, x, y, speed, health, size, color, target_pos):
        self.x = x
        self.y = y
        self.speed = speed
        self.health = health
        self.max_health = health
        self.size = size
        self.color = color
        self.target_x, self.target_y = target_pos

    def update(self):
        angle = math.atan2(self.target_y - self.y, self.target_x - self.x)
        self.x += math.cos(angle) * self.speed
        self.y += math.sin(angle) * self.speed

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.size)
        # Health bar
        if self.health < self.max_health:
            bar_width = self.size * 2
            bar_height = 5
            health_pct = self.health / self.max_health
            fill_width = int(bar_width * health_pct)
            pygame.draw.rect(surface, (255, 0, 0), (int(self.x - self.size), int(self.y - self.size - 10), bar_width, bar_height))
            pygame.draw.rect(surface, (0, 255, 0), (int(self.x - self.size), int(self.y - self.size - 10), fill_width, bar_height))

class Clone:
    """A player-created clone to brew potions."""
    def __init__(self, x, y, effects):
        self.x = x
        self.y = y
        self.size = 12
        self.color = (200, 200, 255)
        self.target_station = None
        self.state = "idle" # idle, moving, brewing
        self.effects = effects.copy() # Inherit effects

    def update(self):
        if self.state == "moving" and self.target_station:
            target_x, target_y = self.target_station.pos
            dist = math.hypot(target_x + 20 - self.x, target_y + 20 - self.y)
            if dist < 5:
                self.x, self.y = target_x + 20, target_y + 20
                self.state = "brewing"
                self.target_station.start_brewing(self)
            else:
                angle = math.atan2(target_y + 20 - self.y, target_x + 20 - self.x)
                self.x += math.cos(angle) * 3
                self.y += math.sin(angle) * 3

    def assign_task(self, station):
        self.target_station = station
        self.state = "moving"

    def draw(self, surface):
        # Draw clone
        pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), self.size, self.color)
        pygame.gfxdraw.aacircle(surface, int(self.x), int(self.y), self.size, (255, 255, 255))
        # Draw state indicator
        if self.state == "brewing":
            pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y - self.size - 5), 3, (255, 255, 0))


class BrewingStation:
    """A station for brewing potions."""
    def __init__(self, pos, potion_type):
        self.pos = pos
        self.potion_type = potion_type
        self.progress = 0
        self.brewing_time = 150 # steps
        self.occupant = None
        self.color = (100, 50, 150)

    def start_brewing(self, clone):
        if not self.occupant:
            self.occupant = clone
            self.progress = 0

    def update(self):
        if self.occupant and self.progress < self.brewing_time:
            self.progress += 1
            return False # Not finished
        elif self.progress >= self.brewing_time:
            return True # Finished
        return False

    def collect_potion(self):
        if self.progress >= self.brewing_time:
            self.progress = 0
            if self.occupant:
                self.occupant.state = "idle"
            self.occupant = None
            return self.potion_type
        return None

    def draw(self, surface, font):
        pygame.draw.rect(surface, self.color, (*self.pos, 40, 40), border_radius=5)
        if self.progress > 0:
            # Progress bar
            bar_width = 40
            bar_height = 5
            fill_width = int(bar_width * (self.progress / self.brewing_time))
            pygame.draw.rect(surface, (50, 50, 50), (self.pos[0], self.pos[1] + 45, bar_width, bar_height))
            color = (0, 255, 255) if self.progress < self.brewing_time else (0, 255, 0)
            pygame.draw.rect(surface, color, (self.pos[0], self.pos[1] + 45, fill_width, bar_height))
        
        # Potion type icon
        icon_char = self.potion_type[0].upper()
        text = font.render(icon_char, True, (255, 255, 255))
        text_rect = text.get_rect(center=(self.pos[0] + 20, self.pos[1] + 20))
        surface.blit(text, text_rect)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your magic tower from waves of enemies. Brew powerful potions to enhance your abilities and create clones to help with your alchemical tasks."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to collect a finished potion from a station. "
        "Press shift to create or direct a clone to an empty brewing station."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    MAX_STEPS = 1000
    TOWER_MAX_HEALTH = 200

    # Colors
    COLOR_BG = (15, 10, 25)
    COLOR_PLAYER = (50, 255, 150)
    COLOR_TOWER = (180, 180, 200)
    COLOR_UI_TEXT = (220, 220, 220)
    POTION_COLORS = {
        "size": (70, 70, 255), # Blue
        "speed": (255, 255, 0), # Yellow
        "firerate": (255, 70, 70), # Red
    }
    POTION_RECIPES = ["size", "firerate", "speed"]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        self.render_mode = render_mode

        # Game state variables are initialized in reset()
        self.reset()
        
        # self.validate_implementation() # Uncomment for self-check

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.step_reward = 0

        # Tower
        self.tower_health = self.TOWER_MAX_HEALTH
        self.tower_pos = (self.WIDTH // 2, self.HEIGHT // 2)

        # Player
        self.player_pos = np.array([self.WIDTH // 2, self.HEIGHT - 50], dtype=float)
        self.player_effects = {}
        self.player_shoot_cooldown = 0
        self.last_space_held = 0
        self.last_shift_held = 0

        # Entities
        self.clones = []
        self.enemies = []
        self.projectiles = []
        self.particles = []

        # Difficulty
        self.enemy_spawn_timer = 0
        self.base_spawn_rate = 60 # Ticks
        self.enemy_speed_multiplier = 1.0
        
        # Potions
        self.unlocked_recipes = [self.POTION_RECIPES[0]]
        self.brewing_stations = [
            BrewingStation((50, 50), self.POTION_RECIPES[0]),
            BrewingStation((self.WIDTH // 2 - 20, 50), self.POTION_RECIPES[1]),
            BrewingStation((self.WIDTH - 90, 50), self.POTION_RECIPES[2]),
        ]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.step_reward = 0

        self._handle_input(action)
        self._update_game_state()
        
        terminated = self.tower_health <= 0 or self.steps >= self.MAX_STEPS
        if terminated:
            self.game_over = True
            if self.tower_health > 0:
                self.step_reward += 50 # Survival bonus
            else:
                self.step_reward -= 100 # Tower destroyed penalty
        
        self.score += self.step_reward

        return (
            self._get_observation(),
            self.step_reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action
        
        # --- Player Movement ---
        player_speed = 3 + self.player_effects.get("speed_bonus", 0)
        if movement == 1: self.player_pos[1] -= player_speed
        elif movement == 2: self.player_pos[1] += player_speed
        elif movement == 3: self.player_pos[0] -= player_speed
        elif movement == 4: self.player_pos[0] += player_speed
        
        # Clamp player position
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.WIDTH)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.HEIGHT)

        # --- Drink Potion Action (Space) ---
        if space_held and not self.last_space_held:
            for station in self.brewing_stations:
                if station.progress >= station.brewing_time:
                    dist = math.hypot(self.player_pos[0] - (station.pos[0]+20), self.player_pos[1] - (station.pos[1]+20))
                    if dist < 50:
                        potion_type = station.collect_potion()
                        if potion_type:
                            self._apply_potion_effect(potion_type)
                            # sfx: potion drink
                            break
        self.last_space_held = space_held

        # --- Clone Action (Shift) ---
        if shift_held and not self.last_shift_held:
            unoccupied_stations = [s for s in self.brewing_stations if not s.occupant]
            if not unoccupied_stations:
                return

            if len(self.clones) < 2:
                new_clone = Clone(self.player_pos[0], self.player_pos[1], self.player_effects)
                self.clones.append(new_clone)
                target_station = min(unoccupied_stations, key=lambda s: math.hypot(new_clone.x - s.pos[0], new_clone.y - s.pos[1]))
                new_clone.assign_task(target_station)
                target_station.occupant = new_clone
                # sfx: clone created
            else:
                idle_clones = [c for c in self.clones if c.state == "idle"]
                if idle_clones:
                    clone_to_direct = min(idle_clones, key=lambda c: math.hypot(self.player_pos[0] - c.x, self.player_pos[1] - c.y))
                    target_station = min(unoccupied_stations, key=lambda s: math.hypot(clone_to_direct.x - s.pos[0], clone_to_direct.y - s.pos[1]))
                    clone_to_direct.assign_task(target_station)
                    target_station.occupant = clone_to_direct
                    # sfx: clone directed

        self.last_shift_held = shift_held

    def _update_game_state(self):
        # Update potion effects
        for effect in list(self.player_effects.keys()):
            self.player_effects[effect] -= 1
            if self.player_effects[effect] <= 0:
                del self.player_effects[effect]

        # Update clones and brewing stations
        for clone in self.clones:
            clone.update()
        for station in self.brewing_stations:
            if station.update(): # Potion finished brewing
                self.step_reward += 1
                self._create_particles(station.pos[0]+20, station.pos[1]+20, (0, 255, 0), 10)
                # sfx: potion ready
        
        # Player shooting
        self.player_shoot_cooldown -= 1
        firerate_bonus = self.player_effects.get("firerate_bonus", 0)
        if self.player_shoot_cooldown <= 0 and self.enemies:
            target = min(self.enemies, key=lambda e: math.hypot(self.player_pos[0] - e.x, self.player_pos[1] - e.y))
            self.projectiles.append(Projectile(self.player_pos[0], self.player_pos[1], target.x, target.y, 8, self.COLOR_PLAYER, 1))
            self.player_shoot_cooldown = 20 - firerate_bonus
            # sfx: player shoot

        # Update projectiles and check collisions
        for p in self.projectiles[:]:
            p.update()
            if not (0 < p.x < self.WIDTH and 0 < p.y < self.HEIGHT):
                if p in self.projectiles: self.projectiles.remove(p)
                continue
            for e in self.enemies[:]:
                if math.hypot(p.x - e.x, p.y - e.y) < e.size + p.size:
                    e.health -= p.damage
                    self.step_reward += 0.1
                    self._create_particles(p.x, p.y, e.color, 5, size_range=(1,3), speed_range=(0.5, 2))
                    if p in self.projectiles: self.projectiles.remove(p)
                    if e.health <= 0:
                        self.step_reward += 5
                        self._create_particles(e.x, e.y, e.color, 20, size_range=(2,6))
                        if e in self.enemies: self.enemies.remove(e)
                        # sfx: enemy defeat
                    break

        # Update and spawn enemies
        self._spawn_enemies()
        for e in self.enemies[:]:
            e.update()
            if math.hypot(e.x - self.tower_pos[0], e.y - self.tower_pos[1]) < e.size + 30:
                damage = 10
                self.tower_health -= damage
                self.step_reward -= 0.1 * damage
                if e in self.enemies: self.enemies.remove(e)
                self._create_particles(self.tower_pos[0], self.tower_pos[1], (255, 0, 0), 15)
                # sfx: tower damage

        # Update particles
        for particle in self.particles[:]:
            particle.update()
            if particle.life <= 0:
                self.particles.remove(particle)

        # Unlock recipes
        if self.steps > 0 and self.steps % 200 == 0:
            if len(self.unlocked_recipes) < len(self.POTION_RECIPES):
                self.unlocked_recipes.append(self.POTION_RECIPES[len(self.unlocked_recipes)])
                # sfx: recipe unlocked
    
    def _spawn_enemies(self):
        if self.steps > 0 and self.steps % 50 == 0:
            self.base_spawn_rate = max(10, self.base_spawn_rate * 0.99)
        if self.steps > 0 and self.steps % 100 == 0:
            self.enemy_speed_multiplier = min(3.0, self.enemy_speed_multiplier + 0.1)

        self.enemy_spawn_timer -= 1
        if self.enemy_spawn_timer <= 0:
            self.enemy_spawn_timer = self.base_spawn_rate
            
            side = self.np_random.integers(0, 4)
            if side == 0: x, y = self.np_random.integers(0, self.WIDTH), -20
            elif side == 1: x, y = self.WIDTH + 20, self.np_random.integers(0, self.HEIGHT)
            elif side == 2: x, y = self.np_random.integers(0, self.WIDTH), self.HEIGHT + 20
            else: x, y = -20, self.np_random.integers(0, self.HEIGHT)
            
            if self.steps > 500 and self.np_random.random() < 0.3:
                # Fast, low health enemy
                speed = 2.0 * self.enemy_speed_multiplier
                health = 1
                size = 8
                color = (255, 100, 100)
            else:
                # Standard enemy
                speed = 1.0 * self.enemy_speed_multiplier
                health = 3
                size = 12
                color = (200, 50, 50)
            
            self.enemies.append(Enemy(x, y, speed, health, size, color, self.tower_pos))

    def _apply_potion_effect(self, potion_type):
        duration = 300 # 10 seconds at 30fps
        if potion_type == "size":
            self.player_effects["size_duration"] = duration
        elif potion_type == "speed":
            self.player_effects["speed_bonus"] = duration + 2
        elif potion_type == "firerate":
            self.player_effects["firerate_bonus"] = duration + 10
        
        self._create_particles(self.player_pos[0], self.player_pos[1], self.POTION_COLORS[potion_type], 30)
    
    def _create_particles(self, x, y, color, count, size_range=(2,5), speed_range=(1,3)):
        for _ in range(count):
            self.particles.append(Particle(x, y, color, self.np_random.integers(20, 41), size_range, speed_range))

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
            "tower_health": self.tower_health,
            "clones": len(self.clones),
            "active_effects": [k for k in self.player_effects.keys()]
        }

    def _render_game(self):
        # Draw tower
        pygame.gfxdraw.filled_circle(self.screen, self.tower_pos[0], self.tower_pos[1], 30, self.COLOR_TOWER)
        pygame.gfxdraw.aacircle(self.screen, self.tower_pos[0], self.tower_pos[1], 30, (255,255,255))
        
        # Draw brewing stations
        for i, station in enumerate(self.brewing_stations):
            if i < len(self.unlocked_recipes):
                station.draw(self.screen, self.font_small)

        # Draw enemies
        for enemy in self.enemies:
            enemy.draw(self.screen)

        # Draw projectiles
        for p in self.projectiles:
            p.draw(self.screen)
            
        # Draw clones
        for clone in self.clones:
            clone.draw(self.screen)

        # Draw Player
        player_size = 15 + (10 if "size_duration" in self.player_effects else 0)
        px, py = int(self.player_pos[0]), int(self.player_pos[1])
        # Glow effect
        glow_color = list(self.COLOR_PLAYER) + [50]
        pygame.gfxdraw.filled_circle(self.screen, px, py, int(player_size * 1.5), glow_color)
        # Player body
        pygame.gfxdraw.filled_circle(self.screen, px, py, player_size, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, px, py, player_size, (255,255,255))
        
        # Draw particles on a separate surface for alpha blending
        s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        for particle in self.particles:
            particle.draw(s)
        self.screen.blit(s, (0,0))

    def _render_ui(self):
        # Tower Health Bar
        bar_width = 200
        bar_height = 20
        health_pct = max(0, self.tower_health / self.TOWER_MAX_HEALTH)
        fill_width = int(bar_width * health_pct)
        pygame.draw.rect(self.screen, (50,0,0), (self.WIDTH//2 - bar_width//2, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, (0,200,0), (self.WIDTH//2 - bar_width//2, 10, fill_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (self.WIDTH//2 - bar_width//2, 10, bar_width, bar_height), 2)
        health_text = self.font_small.render(f"TOWER: {int(self.tower_health)}/{self.TOWER_MAX_HEALTH}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (self.WIDTH//2 - health_text.get_width()//2, 12))

        # Score and Steps
        score_text = self.font_large.render(f"REWARD: {self.score:.2f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, self.HEIGHT - 35))
        steps_text = self.font_small.render(f"TIME: {self.MAX_STEPS - self.steps}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - 120, self.HEIGHT - 25))

        # Potion Effects
        y_offset = 10
        for effect, duration in self.player_effects.items():
            effect_name = effect.replace("_bonus", "").replace("_duration", "")
            color = self.POTION_COLORS.get(effect_name, (200,200,200))
            pygame.draw.rect(self.screen, color, (10, y_offset, 10, 10))
            text = self.font_small.render(f"{effect_name.upper()} ({duration//30}s)", True, self.COLOR_UI_TEXT)
            self.screen.blit(text, (25, y_offset))
            y_offset += 20
        
        # Clone count
        clone_text = self.font_small.render(f"Clones: {len(self.clones)}/2", True, self.COLOR_UI_TEXT)
        self.screen.blit(clone_text, (10, y_offset))

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
        assert trunc is False
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # --- Manual Play Example ---
    env = GameEnv()
    obs, info = env.reset()
    
    # Pygame setup for rendering
    pygame.display.set_caption("Potion Tower Defense")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()

    running = True
    while running:
        # --- Action Selection ---
        movement = 0
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f"Game Over! Final Reward: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()

        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS

    env.close()