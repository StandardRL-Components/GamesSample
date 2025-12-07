
# Generated: 2025-08-27T16:28:41.738795
# Source Brief: brief_01236.md
# Brief Index: 1236

        
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
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper classes for game objects
class Enemy:
    def __init__(self, path, health, speed, size, value):
        self.path = path
        self.path_index = 0
        self.pos = np.array(self.path[0], dtype=float)
        self.max_health = health
        self.health = health
        self.speed = speed
        self.size = size
        self.value = value  # Resources given on defeat
        self.slow_effect_timer = 0

class Tower:
    def __init__(self, pos, tower_type, stats):
        self.pos = np.array(pos, dtype=float)
        self.type = tower_type
        self.stats = stats
        self.cooldown = 0
        self.target = None

class Projectile:
    def __init__(self, start_pos, target_enemy, speed, damage, color):
        self.pos = np.array(start_pos, dtype=float)
        self.target = target_enemy
        self.speed = speed
        self.damage = damage
        self.color = color

class Particle:
    def __init__(self, pos, velocity, size, lifetime, color):
        self.pos = np.array(pos, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.size = size
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.color = color

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to select placement slot/start button. SHIFT to cycle tower type. SPACE to place tower or start wave."
    )

    game_description = (
        "Defend your base from enemy waves by placing towers. Earn resources to build more powerful defenses and survive all 10 waves."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400

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
            self.font_s = pygame.font.SysFont("Consolas", 14)
            self.font_m = pygame.font.SysFont("Consolas", 20)
            self.font_l = pygame.font.SysFont("Consolas", 48)
        except pygame.error:
            self.font_s = pygame.font.Font(None, 18)
            self.font_m = pygame.font.Font(None, 24)
            self.font_l = pygame.font.Font(None, 52)


        # Game constants
        self.MAX_STEPS = 5000 # Increased for longer games
        self.MAX_WAVES = 10

        # Colors
        self.COLOR_BG = (25, 25, 40)
        self.COLOR_PATH = (40, 40, 60)
        self.COLOR_PATH_BORDER = (50, 50, 75)
        self.COLOR_BASE = (0, 150, 0)
        self.COLOR_BASE_BORDER = (0, 200, 0)
        self.COLOR_ENEMY = (200, 50, 50)
        self.COLOR_ENEMY_BORDER = (255, 100, 100)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_UI_ACCENT = (100, 100, 255)
        self.COLOR_HEALTH_GREEN = (0, 255, 0)
        self.COLOR_HEALTH_RED = (255, 0, 0)

        # Tower definitions
        self.TOWER_STATS = {
            0: {"name": "Gatling", "cost": 50, "range": 80, "damage": 5, "fire_rate": 5, "color": (80, 80, 255), "proj_speed": 10},
            1: {"name": "Cannon", "cost": 120, "range": 120, "damage": 40, "fire_rate": 30, "color": (255, 200, 0), "proj_speed": 8},
            2: {"name": "Frost", "cost": 80, "range": 100, "damage": 1, "fire_rate": 20, "color": (200, 0, 255), "proj_speed": 6, "slow": 0.5, "slow_duration": 60},
        }

        # Game path and placement
        self.path = [(50, 350), (150, 350), (150, 100), (450, 100), (450, 300), (590, 300)]
        self.placement_zones = [(100, 225), (225, 175), (375, 175), (500, 200)]
        self.base_pos = (590, 300)
        self.base_size = 20

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.base_health = 0
        self.resources = 0
        self.current_wave = 0
        self.game_phase = "" # 'placement' or 'wave'
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.wave_spawner = []
        self.wave_spawn_timer = 0
        
        # Action handling state
        self.selectable_items = self.placement_zones + ["start_button"]
        self.selected_item_index = 0
        self.selected_tower_type = 0
        self.prev_shift_held = False
        self.prev_space_held = False
        self.last_action_time = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.base_health = 100
        self.resources = 150
        self.current_wave = 0
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = deque(maxlen=200)

        self.selected_item_index = 0
        self.selected_tower_type = 0
        self.prev_shift_held = False
        self.prev_space_held = False

        self._start_placement_phase()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over or self.game_won:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = -0.01  # Time penalty

        # --- Action Handling ---
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        if self.game_phase == 'placement':
            reward += self._handle_placement_phase(movement, space_pressed, shift_pressed)
        
        # --- Game Logic Update (always runs, but content depends on phase) ---
        if self.game_phase == 'wave':
            wave_rewards = self._handle_wave_phase()
            reward += wave_rewards

            # Check for wave completion
            if not self.enemies and not self.wave_spawner:
                if self.current_wave >= self.MAX_WAVES:
                    self.game_won = True
                else:
                    self.resources += 100 + self.current_wave * 10
                    self._start_placement_phase()
        
        self.steps += 1
        
        # --- Termination ---
        terminated = False
        if self.base_health <= 0:
            self.game_over = True
            terminated = True
            reward -= 100
        if self.game_won:
            terminated = True
            reward += 100
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True # End as a loss if time runs out

        self.score += reward
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_placement_phase(self, movement, space_pressed, shift_pressed):
        # Handle selection cycling
        if movement in [3, 4]: # Left/Right
            now = pygame.time.get_ticks()
            if now - self.last_action_time > 150: # Debounce input
                direction = -1 if movement == 3 else 1
                self.selected_item_index = (self.selected_item_index + direction) % len(self.selectable_items)
                self.last_action_time = now
        
        # Handle tower type cycling
        if shift_pressed:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_STATS)

        # Handle confirmation action
        if space_pressed:
            selected_item = self.selectable_items[self.selected_item_index]
            if selected_item == "start_button":
                self._start_wave_phase()
            else: # It's a placement zone
                pos = selected_item
                can_afford = self.resources >= self.TOWER_STATS[self.selected_tower_type]["cost"]
                is_occupied = any(t.pos[0] == pos[0] and t.pos[1] == pos[1] for t in self.towers)
                
                if can_afford and not is_occupied:
                    self.resources -= self.TOWER_STATS[self.selected_tower_type]["cost"]
                    stats = self.TOWER_STATS[self.selected_tower_type]
                    self.towers.append(Tower(pos, self.selected_tower_type, stats))
                    # sfx: place_tower
        return 0

    def _handle_wave_phase(self):
        reward = 0
        # Spawn enemies
        self.wave_spawn_timer -= 1
        if self.wave_spawn_timer <= 0 and self.wave_spawner:
            enemy_data = self.wave_spawner.pop(0)
            self.enemies.append(Enemy(self.path, **enemy_data))
            self.wave_spawn_timer = 30 # Time between spawns

        # Update Towers
        for tower in self.towers:
            tower.cooldown = max(0, tower.cooldown - 1)
            if tower.cooldown == 0:
                # Find target
                potential_targets = [e for e in self.enemies if np.linalg.norm(tower.pos - e.pos) <= tower.stats["range"]]
                if potential_targets:
                    # Target enemy furthest along the path
                    tower.target = max(potential_targets, key=lambda e: e.path_index + np.linalg.norm(e.pos - e.path[e.path_index]))
                    
                    # Fire
                    tower.cooldown = tower.stats["fire_rate"]
                    if tower.type == 2: # Frost tower (AoE, no projectile)
                        for enemy in potential_targets:
                            enemy.slow_effect_timer = tower.stats["slow_duration"]
                            enemy.health -= tower.stats["damage"]
                            self._create_particles(enemy.pos, 5, (150, 150, 255))
                        # sfx: frost_attack
                    else: # Projectile towers
                        self.projectiles.append(Projectile(tower.pos, tower.target, tower.stats["proj_speed"], tower.stats["damage"], tower.stats["color"]))
                        # sfx: shoot_gatling / shoot_cannon
                        self._create_particles(tower.pos, 3, (255, 255, 100), count=3)


        # Update Projectiles
        for proj in self.projectiles[:]:
            if proj.target not in self.enemies:
                self.projectiles.remove(proj)
                continue
            
            direction = proj.target.pos - proj.pos
            dist = np.linalg.norm(direction)
            if dist < proj.speed:
                proj.pos = proj.target.pos
            else:
                proj.pos += (direction / dist) * proj.speed
            
            # Collision
            if np.linalg.norm(proj.pos - proj.target.pos) < proj.target.size:
                proj.target.health -= proj.damage
                reward += 0.1 # Hit reward
                self._create_particles(proj.pos, 10, proj.color)
                self.projectiles.remove(proj)
                # sfx: enemy_hit

        # Update Enemies
        for enemy in self.enemies[:]:
            if enemy.health <= 0:
                reward += 1.0 # Defeat reward
                self.resources += enemy.value
                self._create_particles(enemy.pos, 20, self.COLOR_ENEMY, count=15)
                self.enemies.remove(enemy)
                # sfx: enemy_explode
                continue

            # Movement
            if enemy.path_index < len(enemy.path) - 1:
                target_pos = np.array(enemy.path[enemy.path_index + 1])
                direction = target_pos - enemy.pos
                dist = np.linalg.norm(direction)
                
                current_speed = enemy.speed
                if enemy.slow_effect_timer > 0:
                    enemy.slow_effect_timer -= 1
                    current_speed *= self.TOWER_STATS[2]["slow"]

                if dist < current_speed:
                    enemy.pos = target_pos
                    enemy.path_index += 1
                else:
                    enemy.pos += (direction / dist) * current_speed
            else: # Reached base
                self.base_health -= 10
                reward -= 10 # Base damage penalty
                self.enemies.remove(enemy)
                self._create_particles(self.base_pos, 30, self.COLOR_BASE, count=20)
                # sfx: base_damage

        # Update Particles
        for p in list(self.particles):
            p.pos += p.velocity
            p.lifetime -= 1
            p.size = max(0, p.size - 0.1)
            if p.lifetime <= 0:
                self.particles.remove(p)
        
        return reward

    def _start_placement_phase(self):
        self.game_phase = 'placement'
        self.selected_item_index = 0
    
    def _start_wave_phase(self):
        self.game_phase = 'wave'
        self.current_wave += 1
        
        # Wave scaling
        num_enemies = 5 + self.current_wave * 2
        health = 50 + self.current_wave * 15
        speed = 1.0 + self.current_wave * 0.1
        value = 10 + self.current_wave
        
        self.wave_spawner = [{"health": health, "speed": speed, "size": 8, "value": value} for _ in range(num_enemies)]
        self.wave_spawn_timer = 0

    def _create_particles(self, pos, radius, color, count=10):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            velocity = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = random.randint(15, 30)
            size = random.uniform(radius/4, radius/2)
            self.particles.append(Particle(pos, velocity, size, lifetime, color))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw path
        for i in range(len(self.path) - 1):
            pygame.draw.line(self.screen, self.COLOR_PATH_BORDER, self.path[i], self.path[i+1], 24)
            pygame.draw.line(self.screen, self.COLOR_PATH, self.path[i], self.path[i+1], 20)
        
        # Draw base
        base_rect = pygame.Rect(self.base_pos[0] - self.base_size, self.base_pos[1] - self.base_size, self.base_size*2, self.base_size*2)
        pygame.draw.rect(self.screen, self.COLOR_BASE_BORDER, base_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect.inflate(-4, -4), border_radius=4)
        
        # Draw placement zones
        for pz in self.placement_zones:
            pygame.gfxdraw.aacircle(self.screen, int(pz[0]), int(pz[1]), 20, (60, 60, 80))

        # Draw towers
        for tower in self.towers:
            pygame.draw.circle(self.screen, (0,0,0), tower.pos, 16)
            pygame.draw.circle(self.screen, tower.stats["color"], tower.pos, 15)
            if tower.type == 0: # Gatling
                pygame.draw.circle(self.screen, (200, 200, 255), tower.pos, 5)
            elif tower.type == 1: # Cannon
                pygame.draw.rect(self.screen, (255, 255, 255), (tower.pos[0]-5, tower.pos[1]-5, 10, 10))
            elif tower.type == 2: # Frost
                pygame.gfxdraw.filled_trigon(self.screen, int(tower.pos[0]), int(tower.pos[1]-8), int(tower.pos[0]-8), int(tower.pos[1]+6), int(tower.pos[0]+8), int(tower.pos[1]+6), (255,150,255))

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p.lifetime / p.max_lifetime))
            color = (*p.color, alpha)
            temp_surf = pygame.Surface((p.size*2, p.size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p.size, p.size), p.size)
            self.screen.blit(temp_surf, (int(p.pos[0] - p.size), int(p.pos[1] - p.size)))

        # Draw projectiles
        for proj in self.projectiles:
            pygame.draw.circle(self.screen, (255,255,255), proj.pos, 4)
            pygame.draw.circle(self.screen, proj.color, proj.pos, 3)

        # Draw enemies
        for enemy in self.enemies:
            pos_i = (int(enemy.pos[0]), int(enemy.pos[1]))
            size_i = int(enemy.size)
            # Body
            pygame.draw.circle(self.screen, self.COLOR_ENEMY_BORDER, pos_i, size_i + 1)
            color = self.COLOR_ENEMY
            if enemy.slow_effect_timer > 0:
                color = (100, 100, 255) # Blue tint when slowed
            pygame.draw.circle(self.screen, color, pos_i, size_i)
            
            # Health bar
            health_pct = enemy.health / enemy.max_health
            bar_width = size_i * 2
            bar_pos = (pos_i[0] - size_i, pos_i[1] - size_i - 8)
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, (bar_pos[0], bar_pos[1], bar_width, 4))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (bar_pos[0], bar_pos[1], bar_width * health_pct, 4))
    
    def _render_ui(self):
        # Top UI Panel
        ui_panel = pygame.Surface((self.WIDTH, 40), pygame.SRCALPHA)
        ui_panel.fill((10, 10, 20, 200))
        self.screen.blit(ui_panel, (0, 0))
        pygame.draw.line(self.screen, self.COLOR_UI_ACCENT, (0, 40), (self.WIDTH, 40))

        # UI Text
        texts = [
            f"❤️ Health: {self.base_health}",
            f"💰 Res: {self.resources}",
            f"🌊 Wave: {self.current_wave}/{self.MAX_WAVES}",
            f"Score: {int(self.score)}"
        ]
        for i, text in enumerate(texts):
            self._draw_text(text, (10 + i * 150, 20), self.font_m, self.COLOR_TEXT, "midleft")

        # Placement phase UI
        if self.game_phase == 'placement':
            # Draw selector
            selected_item = self.selectable_items[self.selected_item_index]
            if selected_item == "start_button":
                pygame.draw.rect(self.screen, self.COLOR_UI_ACCENT, (self.WIDTH - 130, self.HEIGHT - 50, 120, 40), 2, 5)
            else: # Placement zone
                pos = selected_item
                # Show tower preview and range
                stats = self.TOWER_STATS[self.selected_tower_type]
                # Range circle
                range_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
                pygame.gfxdraw.aacircle(range_surf, pos[0], pos[1], stats["range"], (*stats["color"], 80))
                pygame.gfxdraw.filled_circle(range_surf, pos[0], pos[1], stats["range"], (*stats["color"], 40))
                self.screen.blit(range_surf, (0,0))
                # Tower preview
                pygame.draw.circle(self.screen, (*stats["color"], 150), pos, 15)
                # Selector box
                pygame.gfxdraw.box(self.screen, (pos[0]-22, pos[1]-22, 44, 44), (255,255,255,100))

            # Bottom UI Panel for placement
            bottom_panel = pygame.Surface((self.WIDTH, 60), pygame.SRCALPHA)
            bottom_panel.fill((10, 10, 20, 200))
            self.screen.blit(bottom_panel, (0, self.HEIGHT - 60))
            pygame.draw.line(self.screen, self.COLOR_UI_ACCENT, (0, self.HEIGHT - 60), (self.WIDTH, self.HEIGHT - 60))

            # Selected Tower Info
            stats = self.TOWER_STATS[self.selected_tower_type]
            tower_text = f"Selected: {stats['name']} | Cost: {stats['cost']} | Dmg: {stats['damage']} | Range: {stats['range']}"
            self._draw_text(tower_text, (20, self.HEIGHT - 30), self.font_m, self.COLOR_TEXT, "midleft")
            
            # Start Wave Button
            start_rect = pygame.Rect(self.WIDTH - 130, self.HEIGHT - 50, 120, 40)
            pygame.draw.rect(self.screen, (0, 80, 0) if self.current_wave < self.MAX_WAVES else (80,0,0), start_rect, border_radius=5)
            self._draw_text("START WAVE", start_rect.center, self.font_m, self.COLOR_TEXT, "center")

        # Game Over / Victory message
        if self.game_over:
            self._draw_text("GAME OVER", (self.WIDTH/2, self.HEIGHT/2), self.font_l, self.COLOR_ENEMY, "center")
        elif self.game_won:
            self._draw_text("VICTORY!", (self.WIDTH/2, self.HEIGHT/2), self.font_l, self.COLOR_UI_ACCENT, "center")

    def _draw_text(self, text, pos, font, color, align="center"):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        setattr(text_rect, align, pos)
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "resources": self.resources,
            "wave": self.current_wave,
            "game_phase": self.game_phase,
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


if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tower Defense Gym Environment")
    
    done = False
    clock = pygame.time.Clock()
    
    # Game loop for manual play
    while not done:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Wave: {info['wave']}")
            # Visual pause on game over
            for _ in range(90): # 3 seconds at 30fps
                screen.blit(pygame.transform.flip(pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2))), False, True), (0, 0))
                pygame.display.flip()
                clock.tick(30)
            obs, info = env.reset()

        # Render the observation to the display
        # Pygame uses (width, height) and coordinates from top-left.
        # Gym uses (height, width) and numpy arrays. Transpose and flip are needed.
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        surf = pygame.transform.flip(surf, False, True) # Flip vertically
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit to 30 FPS for smooth manual play

    env.close()