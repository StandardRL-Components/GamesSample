
# Generated: 2025-08-27T16:15:43.339802
# Source Brief: brief_01168.md
# Brief Index: 1168

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


# Set SDL to dummy to run headless
os.environ["SDL_VIDEODRIVER"] = "dummy"

# --- Helper Classes for Game Entities ---

class Enemy:
    def __init__(self, path, wave_modifier):
        self.path = path
        self.path_index = 0
        self.pos = np.array(self.path[0], dtype=float)
        self.max_health = 50 * wave_modifier
        self.health = self.max_health
        self.speed = 1.0 * min(1.5, wave_modifier) # Cap speed
        self.slow_timer = 0
        self.size = 8
        self.color = (220, 50, 50)
        self.gold_value = 10

    def move(self):
        if self.path_index >= len(self.path) - 1:
            return True  # Reached the end

        if self.slow_timer > 0:
            current_speed = self.speed * 0.5
            self.slow_timer -= 1
        else:
            current_speed = self.speed

        target = np.array(self.path[self.path_index + 1], dtype=float)
        direction = target - self.pos
        distance = np.linalg.norm(direction)

        if distance < current_speed:
            self.pos = target
            self.path_index += 1
        else:
            self.pos += (direction / distance) * current_speed
        
        return False

    def draw(self, surface):
        # Body
        x, y = int(self.pos[0]), int(self.pos[1])
        pygame.gfxdraw.filled_circle(surface, x, y, self.size, self.color)
        pygame.gfxdraw.aacircle(surface, x, y, self.size, (255, 150, 150))
        
        # Health bar
        if self.health < self.max_health:
            bar_width = 20
            bar_height = 4
            health_pct = self.health / self.max_health
            health_bar_x = x - bar_width // 2
            health_bar_y = y - self.size - bar_height - 3
            pygame.draw.rect(surface, (50, 0, 0), (health_bar_x, health_bar_y, bar_width, bar_height))
            pygame.draw.rect(surface, (255, 0, 0), (health_bar_x, health_bar_y, int(bar_width * health_pct), bar_height))

    def take_damage(self, amount):
        self.health -= amount
        return self.health <= 0

class Tower:
    def __init__(self, pos, tower_type_info):
        self.pos = pos
        self.type_info = tower_type_info
        self.range = tower_type_info['range']
        self.cooldown = 0
        self.target = None

    def find_target(self, enemies):
        # If current target is invalid, find a new one
        if self.target and (self.target.health <= 0 or np.linalg.norm(np.array(self.pos) - self.target.pos) > self.range):
            self.target = None

        if not self.target:
            closest_enemy = None
            min_dist = float('inf')
            for enemy in enemies:
                dist = np.linalg.norm(np.array(self.pos) - enemy.pos)
                if dist <= self.range and dist < min_dist:
                    min_dist = dist
                    closest_enemy = enemy
            self.target = closest_enemy

    def update(self, enemies):
        if self.cooldown > 0:
            self.cooldown -= 1
        self.find_target(enemies)
        if self.target and self.cooldown <= 0:
            self.cooldown = self.type_info['fire_rate']
            return Projectile(self.pos, self.target, self.type_info)
        return None

    def draw(self, surface):
        x, y = int(self.pos[0]), int(self.pos[1])
        size = 12
        # Base
        pygame.draw.rect(surface, self.type_info['color'], (x - size, y - size, size*2, size*2))
        pygame.draw.rect(surface, tuple(c*0.7 for c in self.type_info['color']), (x - size, y - size, size*2, size*2), 2)
        # Turret
        pygame.draw.circle(surface, (200, 200, 200), (x, y), size // 2)
        
class Projectile:
    def __init__(self, start_pos, target, tower_type_info):
        self.pos = np.array(start_pos, dtype=float)
        self.target = target
        self.type_info = tower_type_info
        self.speed = 10.0
        self.color = (255, 255, 255)

    def move(self):
        if self.target.health <= 0:
            return True, False # Reached target (it's dead), no hit

        direction = self.target.pos - self.pos
        distance = np.linalg.norm(direction)

        if distance < self.speed:
            return True, True # Reached target, hit
        
        self.pos += (direction / distance) * self.speed
        return False, False

    def draw(self, surface):
        x, y = int(self.pos[0]), int(self.pos[1])
        pygame.draw.circle(surface, self.color, (x, y), 3)

class Particle:
    def __init__(self, pos, vel, color, duration, size):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.color = color
        self.duration = duration
        self.max_duration = duration
        self.size = size

    def update(self):
        self.pos += self.vel
        self.duration -= 1
        return self.duration <= 0

    def draw(self, surface):
        alpha = int(255 * (self.duration / self.max_duration))
        color = (*self.color, alpha)
        temp_surf = pygame.Surface((self.size*2, self.size*2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, color, (self.size, self.size), int(self.size * (self.duration / self.max_duration)))
        surface.blit(temp_surf, (int(self.pos[0] - self.size), int(self.pos[1] - self.size)))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to select a tower slot. Press SHIFT to cycle tower types. Press SPACE to build."
    )

    game_description = (
        "A top-down tower defense game. Place towers to defend your base from waves of enemies. Survive 10 waves to win."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.width, self.height = 640, 400
        self.observation_space = Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        # --- Colors and Fonts ---
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_PATH = (40, 50, 60)
        self.COLOR_SLOT = (60, 70, 80)
        self.COLOR_BASE = (50, 200, 50)
        self.COLOR_TEXT = (220, 220, 220)
        self.UI_FONT = pygame.font.Font(None, 24)
        self.TITLE_FONT = pygame.font.Font(None, 48)
        self.PARTICLE_FONT = pygame.font.Font(None, 18)

        # --- Game Constants ---
        self.MAX_STEPS = 30 * 120 # 2 minutes at 30fps
        self.MAX_WAVES = 10
        self.INTER_WAVE_TIME = 30 * 5 # 5 seconds

        self.path_waypoints = [(0, 200), (100, 200), (100, 100), (540, 100), (540, 300), (100, 300), (100, 200), (self.width, 200)]
        self.tower_slots = [
            (150, 150), (250, 150), (350, 150), (450, 150),
            (150, 250), (250, 250), (350, 250), (450, 250)
        ]
        self.tower_types = [
            {'name': 'Machine Gun', 'cost': 50, 'range': 80, 'damage': 5, 'fire_rate': 10, 'color': (50, 150, 255), 'effect': None},
            {'name': 'Cannon', 'cost': 120, 'range': 120, 'damage': 25, 'fire_rate': 60, 'color': (255, 200, 50), 'effect': 'splash', 'splash_radius': 30},
            {'name': 'Slower', 'cost': 80, 'range': 100, 'damage': 2, 'fire_rate': 30, 'color': (200, 50, 255), 'effect': 'slow', 'slow_duration': 60},
        ]
        
        # Initialize state variables
        self.state_initialized = False
        self.reset()
        self.state_initialized = True
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.reward_this_step = 0.0

        self.base_health = 100
        self.gold = 100
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.current_wave = 0
        self.game_state = "INTER_WAVE" # or "WAVE_ACTIVE"
        self.wave_timer = self.INTER_WAVE_TIME

        self.selected_tower_slot_idx = 0
        self.selected_tower_type_idx = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False
        self.move_cooldown = 0
        
        if self.state_initialized:
            return self._get_observation(), self._get_info()
        else:
            # During __init__, we need to return dummy values before the first render
            return np.zeros(self.observation_space.shape, dtype=np.uint8), {}


    def step(self, action):
        self.reward_this_step = 0.0
        
        self._handle_input(action)

        if not self.game_over:
            self._update_game_logic()
        
        self.steps += 1
        terminated = self._check_termination()

        if terminated and not self.game_over:
            self.game_over = True
            if self.win:
                self.reward_this_step += 100.0
            else:
                self.reward_this_step -= 100.0
        
        self.score += self.reward_this_step
        
        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Tower Slot Selection ---
        if self.move_cooldown > 0:
            self.move_cooldown -= 1
        
        if movement != 0 and self.move_cooldown == 0:
            self.move_cooldown = 5 # 5-frame cooldown to prevent rapid selection
            current_row = self.selected_tower_slot_idx // 4
            current_col = self.selected_tower_slot_idx % 4

            if movement == 1: # Up
                if current_row > 0: self.selected_tower_slot_idx -= 4
            elif movement == 2: # Down
                if current_row < 1: self.selected_tower_slot_idx += 4
            elif movement == 3: # Left
                if current_col > 0: self.selected_tower_slot_idx -= 1
            elif movement == 4: # Right
                if current_col < 3: self.selected_tower_slot_idx += 1
        
        # --- Cycle Tower Type ---
        if shift_held and not self.prev_shift_held:
            self.selected_tower_type_idx = (self.selected_tower_type_idx + 1) % len(self.tower_types)
            # sfx: UI_cycle.wav

        # --- Place Tower ---
        if space_held and not self.prev_space_held:
            slot_pos = self.tower_slots[self.selected_tower_slot_idx]
            tower_info = self.tower_types[self.selected_tower_type_idx]
            is_slot_occupied = any(t.pos == slot_pos for t in self.towers)

            if not is_slot_occupied and self.gold >= tower_info['cost']:
                self.gold -= tower_info['cost']
                self.towers.append(Tower(slot_pos, tower_info))
                # sfx: build_tower.wav
                self._create_particles(slot_pos, 15, (255, 255, 100), 20)


        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _update_game_logic(self):
        # --- Handle Game State ---
        if self.game_state == "INTER_WAVE":
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self._spawn_wave()
        
        elif self.game_state == "WAVE_ACTIVE":
            if not self.enemies and self.current_wave > 0:
                self.game_state = "INTER_WAVE"
                self.wave_timer = self.INTER_WAVE_TIME
                self.reward_this_step += 1.0
                self.gold += 50 + self.current_wave * 10
                if self.current_wave >= self.MAX_WAVES:
                    self.win = True

        # --- Update Entities ---
        # Towers shoot projectiles
        for tower in self.towers:
            new_projectile = tower.update(self.enemies)
            if new_projectile:
                self.projectiles.append(new_projectile)
                # sfx: shoot.wav

        # Projectiles move and hit
        new_projectiles = []
        for p in self.projectiles:
            is_done, did_hit = p.move()
            if is_done:
                if did_hit:
                    self._handle_projectile_hit(p)
            else:
                new_projectiles.append(p)
        self.projectiles = new_projectiles
        
        # Enemies move
        new_enemies = []
        for enemy in self.enemies:
            if enemy.health > 0:
                if enemy.move(): # Reached base
                    self.base_health -= 10
                    self.reward_this_step -= 0.01
                    # sfx: base_damage.wav
                else:
                    new_enemies.append(enemy)
        self.enemies = new_enemies
        
        # Particles
        self.particles = [p for p in self.particles if not p.update()]

    def _handle_projectile_hit(self, proj):
        # sfx: hit_enemy.wav
        if proj.type_info['effect'] == 'splash':
            self._create_particles(proj.target.pos, 20, proj.type_info['color'], 25)
            # AoE damage
            for enemy in self.enemies:
                if np.linalg.norm(proj.target.pos - enemy.pos) <= proj.type_info['splash_radius']:
                    if enemy.take_damage(proj.type_info['damage']):
                        self._on_enemy_killed(enemy)
        else: # Single target
            self._create_particles(proj.target.pos, 5, (200, 200, 200), 15)
            if proj.type_info['effect'] == 'slow':
                proj.target.slow_timer = proj.type_info['slow_duration']
            
            if proj.target.take_damage(proj.type_info['damage']):
                self._on_enemy_killed(proj.target)

    def _on_enemy_killed(self, enemy):
        self.gold += enemy.gold_value
        self.reward_this_step += 0.1
        # sfx: enemy_die.wav
        self._create_particles(enemy.pos, 10, enemy.color, 20)

    def _spawn_wave(self):
        self.current_wave += 1
        if self.current_wave > self.MAX_WAVES:
            return
            
        self.game_state = "WAVE_ACTIVE"
        num_enemies = 5 + self.current_wave * 2
        wave_modifier = 1 + (self.current_wave - 1) * 0.1
        
        for i in range(num_enemies):
            enemy = Enemy(self.path_waypoints, wave_modifier)
            # Stagger spawn
            enemy.pos[0] -= i * 25 
            self.enemies.append(enemy)

    def _check_termination(self):
        return self.base_health <= 0 or self.win or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Path
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_waypoints, 30)
        
        # Base (end of path)
        base_pos = self.path_waypoints[-1]
        pygame.draw.rect(self.screen, self.COLOR_BASE, (base_pos[0]-10, base_pos[1]-10, 20, 20))

        # Tower Slots
        for i, pos in enumerate(self.tower_slots):
            pygame.draw.rect(self.screen, self.COLOR_SLOT, (pos[0]-15, pos[1]-15, 30, 30), border_radius=3)
        
        # Selected Tower Slot Highlight
        sel_pos = self.tower_slots[self.selected_tower_slot_idx]
        for i in range(5):
            alpha = 150 - i * 30
            pygame.draw.rect(self.screen, (255, 255, 100, alpha), (sel_pos[0]-18-i, sel_pos[1]-18-i, 36+i*2, 36+i*2), 1, border_radius=5)
        
        # Placed Towers
        for tower in self.towers:
            tower.draw(self.screen)

        # Enemies
        for enemy in self.enemies:
            enemy.draw(self.screen)

        # Projectiles
        for proj in self.projectiles:
            proj.draw(self.screen)

        # Particles
        for particle in self.particles:
            particle.draw(self.screen)

    def _render_ui(self):
        # Top bar background
        pygame.draw.rect(self.screen, (0,0,0,100), (0, 0, self.width, 35))
        
        # Gold
        gold_text = self.UI_FONT.render(f"GOLD: {self.gold}", True, (255, 223, 0))
        self.screen.blit(gold_text, (10, 10))

        # Base Health
        health_text = self.UI_FONT.render(f"BASE HP: {max(0, self.base_health)}/100", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (150, 10))

        # Wave
        wave_str = f"WAVE: {self.current_wave}/{self.MAX_WAVES}" if self.current_wave > 0 else "WAVE: 0/10"
        wave_text = self.UI_FONT.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (320, 10))

        # Score
        score_text = self.UI_FONT.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (480, 10))

        # Tower Selection UI
        ui_y = self.height - 50
        pygame.draw.rect(self.screen, (0,0,0,100), (0, ui_y - 10, self.width, 60))
        for i, t_type in enumerate(self.tower_types):
            ui_x = self.width // 2 - 150 + i * 100
            # Box
            box_color = (80, 80, 80) if i != self.selected_tower_type_idx else (255, 255, 255)
            pygame.draw.rect(self.screen, box_color, (ui_x, ui_y, 80, 40), 2, border_radius=5)
            # Tower icon
            pygame.draw.rect(self.screen, t_type['color'], (ui_x + 5, ui_y + 5, 30, 30))
            # Text
            name_text = self.PARTICLE_FONT.render(t_type['name'].split()[0], True, self.COLOR_TEXT)
            cost_text = self.PARTICLE_FONT.render(f"${t_type['cost']}", True, (255, 223, 0))
            self.screen.blit(name_text, (ui_x + 40, ui_y + 8))
            self.screen.blit(cost_text, (ui_x + 40, ui_y + 22))

        # Inter-wave timer
        if self.game_state == "INTER_WAVE" and not self.win:
            seconds = math.ceil(self.wave_timer / 30)
            timer_text = self.TITLE_FONT.render(f"Wave {self.current_wave + 1} starting in {seconds}...", True, self.COLOR_TEXT)
            text_rect = timer_text.get_rect(center=(self.width/2, self.height/2 - 50))
            self.screen.blit(timer_text, text_rect)

        # Game Over / Victory Message
        if self.game_over:
            s = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            
            message = "VICTORY!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            end_text = self.TITLE_FONT.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.width/2, self.height/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gold": self.gold,
            "base_health": self.base_health,
            "wave": self.current_wave,
        }

    def _create_particles(self, pos, count, color, duration):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 2.5)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            size = random.randint(2, 5)
            self.particles.append(Particle(pos, vel, color, duration, size))

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    # You will need to install pygame (`pip install pygame`)
    # and remove/comment out `os.environ["SDL_VIDEODRIVER"] = "dummy"`
    
    # To run, comment out the following line:
    # os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    try:
        pygame.display.init()
        screen = pygame.display.set_mode((env.width, env.height))
        pygame.display.set_caption("Tower Defense")
        clock = pygame.time.Clock()
        running = True
    except pygame.error:
        print("Pygame display not available. Run in headless mode.")
        running = False


    action = env.action_space.sample()
    action.fill(0)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        # Reset action
        action.fill(0)

        # Movement
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        # Actions
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            obs, info = env.reset() # Auto-reset
            pygame.time.wait(2000)

        # Render to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # 30 FPS

    env.close()