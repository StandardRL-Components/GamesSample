
# Generated: 2025-08-27T15:12:39.018076
# Source Brief: brief_00922.md
# Brief Index: 922

        
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

class Enemy:
    def __init__(self, path, health, speed, size, color, gold_value):
        self.path = path
        self.path_index = 0
        self.pos = np.array(self.path[0], dtype=float)
        self.max_health = health
        self.health = health
        self.speed = speed
        self.size = size
        self.color = color
        self.gold_value = gold_value
        self.damage_flash = 0

    def move(self):
        if self.path_index >= len(self.path) - 1:
            return True  # Reached the end

        target_pos = np.array(self.path[self.path_index + 1], dtype=float)
        direction = target_pos - self.pos
        distance = np.linalg.norm(direction)

        if distance < self.speed:
            self.pos = target_pos
            self.path_index += 1
        else:
            self.pos += (direction / distance) * self.speed
        
        return False

    def draw(self, surface):
        x, y = int(self.pos[0]), int(self.pos[1])
        # Main body
        color = self.color if self.damage_flash == 0 else (255, 255, 255)
        pygame.gfxdraw.filled_circle(surface, x, y, self.size, color)
        pygame.gfxdraw.aacircle(surface, x, y, self.size, (0,0,0))
        if self.damage_flash > 0:
            self.damage_flash -= 1
        # Health bar
        if self.health < self.max_health:
            bar_width = self.size * 2
            bar_height = 4
            health_pct = self.health / self.max_health
            fill_width = int(bar_width * health_pct)
            pygame.draw.rect(surface, (255,0,0), (x - self.size, y - self.size - 8, bar_width, bar_height))
            pygame.draw.rect(surface, (0,255,0), (x - self.size, y - self.size - 8, fill_width, bar_height))


class Tower:
    def __init__(self, pos, tower_type):
        self.pos = np.array(pos, dtype=float)
        self.type_info = tower_type
        self.range = tower_type['range']
        self.damage = tower_type['damage']
        self.fire_rate = tower_type['fire_rate']
        self.color = tower_type['color']
        self.cost = tower_type['cost']
        self.cooldown = 0
        self.target = None

    def find_target(self, enemies):
        # Target enemy furthest along the path
        best_target = None
        max_path_dist = -1

        for enemy in enemies:
            dist = np.linalg.norm(self.pos - enemy.pos)
            if dist <= self.range:
                # Combine path index and inverse distance to waypoint for progress
                progress = enemy.path_index
                if enemy.path_index + 1 < len(enemy.path):
                    dist_to_next = np.linalg.norm(enemy.pos - np.array(enemy.path[enemy.path_index + 1]))
                    progress += 1 - (dist_to_next / np.linalg.norm(np.array(enemy.path[enemy.path_index]) - np.array(enemy.path[enemy.path_index + 1])))

                if progress > max_path_dist:
                    max_path_dist = progress
                    best_target = enemy
        self.target = best_target

    def update(self, enemies):
        if self.cooldown > 0:
            self.cooldown -= 1
        self.find_target(enemies)

    def can_fire(self):
        return self.cooldown == 0 and self.target is not None

    def draw(self, surface):
        x, y = int(self.pos[0]), int(self.pos[1])
        # Base
        pygame.gfxdraw.filled_circle(surface, x, y, 12, (50, 50, 50))
        pygame.gfxdraw.aacircle(surface, x, y, 12, (0, 0, 0))
        # Turret
        pygame.gfxdraw.filled_trigon(surface, x-8, y+8, x+8, y+8, x, y-8, self.color)
        pygame.gfxdraw.aatrigon(surface, x-8, y+8, x+8, y+8, x, y-8, (0,0,0))


class Projectile:
    def __init__(self, start_pos, target_enemy, damage, speed, color):
        self.pos = np.array(start_pos, dtype=float)
        self.target = target_enemy
        self.damage = damage
        self.speed = speed
        self.color = color

    def move(self):
        if self.target is None or self.target.health <= 0:
            return True # Target lost

        direction = self.target.pos - self.pos
        distance = np.linalg.norm(direction)

        if distance < self.speed:
            self.pos = self.target.pos
            return True # Reached target
        else:
            self.pos += (direction / distance) * self.speed
        return False

    def draw(self, surface):
        x, y = int(self.pos[0]), int(self.pos[1])
        pygame.draw.circle(surface, self.color, (x, y), 3)


class Particle:
    def __init__(self, pos, color, life, start_size, end_size):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.random.randn(2) * 2
        self.color = color
        self.max_life = life
        self.life = life
        self.start_size = start_size
        self.end_size = end_size

    def update(self):
        self.pos += self.vel
        self.life -= 1
        return self.life <= 0

    def draw(self, surface):
        if self.life > 0:
            lerp = self.life / self.max_life
            current_size = int(self.start_size * lerp + self.end_size * (1 - lerp))
            if current_size > 0:
                x, y = int(self.pos[0]), int(self.pos[1])
                pygame.gfxdraw.filled_circle(surface, x, y, current_size, self.color)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the placement cursor. "
        "Press Shift to cycle tower types. Press Space to build a tower."
    )

    game_description = (
        "A top-down tower defense game. Place towers to defend your base "
        "from waves of enemies. Survive 15 waves to win."
    )

    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    MAX_STEPS = 30 * 60 * 2  # 2 minutes at 30fps
    MAX_WAVES = 15
    
    COLOR_BG = (20, 30, 40)
    COLOR_PATH = (50, 60, 70)
    COLOR_BASE = (0, 150, 50)
    COLOR_GRID = (255, 255, 255, 30)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_CURSOR_INVALID = (255, 0, 0)
    COLOR_TEXT = (220, 220, 220)

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
        self.font_s = pygame.font.SysFont("Consolas", 16)
        self.font_m = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_l = pygame.font.SysFont("Consolas", 48, bold=True)

        self._setup_game_parameters()
        self.reset()
        
        self.validate_implementation()

    def _setup_game_parameters(self):
        # Enemy Path
        self.PATH = [
            (-20, 100), (80, 100), (80, 300), (250, 300), (250, 50),
            (450, 50), (450, 350), (self.SCREEN_WIDTH + 20, 350)
        ]
        
        # Tower placement grid
        self.GRID_SPOTS = []
        self.occupied_spots = set()
        for y in [180, 220]:
            for x in range(120, 221, 50): self.GRID_SPOTS.append((x, y))
        for y in range(100, 301, 50):
            if y not in [150, 200, 250]: self.GRID_SPOTS.append((350, y))
        self.GRID_SPOTS_NP = np.array(self.GRID_SPOTS)

        # Tower Types
        self.TOWER_TYPES = [
            {'name': 'Gatling', 'cost': 50, 'damage': 5, 'range': 80, 'fire_rate': 10, 'color': (0, 150, 255)},
            {'name': 'Cannon', 'cost': 120, 'damage': 25, 'range': 120, 'fire_rate': 45, 'color': (255, 150, 0)},
            {'name': 'Sniper', 'cost': 200, 'damage': 100, 'range': 200, 'fire_rate': 90, 'color': (200, 50, 255)},
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.terminal_reward_given = False
        
        self.base_health = 100
        self.gold = 100
        
        self.current_wave = 0
        self.wave_spawning = False
        self.wave_spawn_timer = 0
        self.wave_enemy_idx = 0
        self.inter_wave_timer = 150 # Start first wave quickly
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.occupied_spots.clear()
        
        self.cursor_grid_idx = 0
        self.selected_tower_type_idx = 0
        
        self.last_shift_press = False
        self.last_space_press = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = -0.001 # Small penalty for time passing
        
        self._handle_input(movement, space_held, shift_held)
        
        if not self.game_over:
            # --- Game Logic Updates ---
            self._update_wave_spawning()
            
            enemy_reward, base_damage = self._update_enemies()
            reward += enemy_reward
            self.base_health -= base_damage
            
            self._update_towers()
            
            hit_reward = self._update_projectiles()
            reward += hit_reward
            
            self._update_particles()

            # --- State Checks ---
            if not self.wave_spawning and not self.enemies and self.current_wave > 0:
                if self.current_wave == self.MAX_WAVES:
                    self.game_won = True
                else:
                    self.inter_wave_timer += 1
                    if self.inter_wave_timer > 150: # 5 seconds
                        reward += 1.0 # Wave clear bonus
                        self._start_next_wave()
        
        self.steps += 1
        terminated = self._check_termination()
        
        if terminated and not self.terminal_reward_given:
            if self.game_won:
                reward += 100
            else:
                reward -= 100
            self.terminal_reward_given = True
            self.game_over = True
        
        self.score += reward
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # Cursor movement
        if movement != 0:
            current_pos = self.GRID_SPOTS[self.cursor_grid_idx]
            if movement == 1: # Up
                potential_targets = self.GRID_SPOTS_NP[self.GRID_SPOTS_NP[:, 1] < current_pos[1]]
            elif movement == 2: # Down
                potential_targets = self.GRID_SPOTS_NP[self.GRID_SPOTS_NP[:, 1] > current_pos[1]]
            elif movement == 3: # Left
                potential_targets = self.GRID_SPOTS_NP[self.GRID_SPOTS_NP[:, 0] < current_pos[0]]
            else: # Right
                potential_targets = self.GRID_SPOTS_NP[self.GRID_SPOTS_NP[:, 0] > current_pos[0]]
            
            if len(potential_targets) > 0:
                distances = np.linalg.norm(potential_targets - current_pos, axis=1)
                closest_idx = np.argmin(distances)
                new_pos = tuple(potential_targets[closest_idx])
                self.cursor_grid_idx = self.GRID_SPOTS.index(new_pos)

        # Cycle tower type
        if shift_held and not self.last_shift_press:
            self.selected_tower_type_idx = (self.selected_tower_type_idx + 1) % len(self.TOWER_TYPES)
        self.last_shift_press = shift_held

        # Place tower
        if space_held and not self.last_space_press:
            self._place_tower()
        self.last_space_press = space_held

    def _place_tower(self):
        pos = self.GRID_SPOTS[self.cursor_grid_idx]
        tower_type = self.TOWER_TYPES[self.selected_tower_type_idx]
        if pos not in self.occupied_spots and self.gold >= tower_type['cost']:
            self.gold -= tower_type['cost']
            self.towers.append(Tower(pos, tower_type))
            self.occupied_spots.add(pos)
            # SFX: build_tower

    def _start_next_wave(self):
        self.current_wave += 1
        self.wave_spawning = True
        self.wave_spawn_timer = 0
        self.wave_enemy_idx = 0
        self.inter_wave_timer = 0

    def _update_wave_spawning(self):
        if not self.wave_spawning: return

        self.wave_spawn_timer += 1
        num_enemies = 2 + self.current_wave * 2
        spawn_interval = max(5, 30 - self.current_wave)
        
        if self.wave_enemy_idx < num_enemies and self.wave_spawn_timer > spawn_interval:
            self.wave_spawn_timer = 0
            self.wave_enemy_idx += 1
            
            health = 20 * (1.05 ** self.current_wave)
            speed = 1.0 * (1.05 ** self.current_wave)
            size = 8
            color = (200, 50, 50)
            gold = 5 + self.current_wave
            
            self.enemies.append(Enemy(self.PATH, health, speed, size, color, gold))
            # SFX: enemy_spawn
        
        if self.wave_enemy_idx >= num_enemies:
            self.wave_spawning = False

    def _update_enemies(self):
        reward = 0
        base_damage = 0
        enemies_to_remove = []
        for enemy in self.enemies:
            if enemy.move():
                base_damage += 10 # Damage to base
                enemies_to_remove.append(enemy)
                # SFX: base_hit
        
        for enemy in enemies_to_remove:
            self.enemies.remove(enemy)
        return reward, base_damage

    def _update_towers(self):
        for tower in self.towers:
            tower.update(self.enemies)
            if tower.can_fire():
                proj = Projectile(tower.pos, tower.target, tower.damage, 10, tower.color)
                self.projectiles.append(proj)
                tower.cooldown = tower.fire_rate
                # SFX: tower_shoot

    def _update_projectiles(self):
        reward = 0
        projectiles_to_remove = []
        enemies_to_remove = []
        for proj in self.projectiles:
            if proj.target is None or proj.target in enemies_to_remove or proj.target.health <= 0:
                projectiles_to_remove.append(proj)
                continue

            if proj.move():
                proj.target.health -= proj.damage
                proj.target.damage_flash = 3
                
                for _ in range(10):
                    self.particles.append(Particle(proj.pos, proj.color, 15, 4, 0))
                
                projectiles_to_remove.append(proj)
                # SFX: enemy_hit
                
                if proj.target.health <= 0 and proj.target not in enemies_to_remove:
                    reward += 0.1
                    self.gold += proj.target.gold_value
                    enemies_to_remove.append(proj.target)
                    # SFX: enemy_die

        self.enemies = [e for e in self.enemies if e not in enemies_to_remove]
        self.projectiles = [p for p in self.projectiles if p not in projectiles_to_remove]
        return reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if not p.update()]

    def _check_termination(self):
        return self.base_health <= 0 or self.game_won or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Path
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.PATH, 40)
        
        # Base
        base_pos = self.PATH[-1]
        pygame.draw.rect(self.screen, self.COLOR_BASE, (base_pos[0]-20, base_pos[1]-20, 40, 40))

        # Grid and Cursor
        cursor_pos = self.GRID_SPOTS[self.cursor_grid_idx]
        selected_tower = self.TOWER_TYPES[self.selected_tower_type_idx]
        can_afford = self.gold >= selected_tower['cost']
        is_occupied = cursor_pos in self.occupied_spots
        
        for pos in self.GRID_SPOTS:
            pygame.gfxdraw.box(self.screen, (*pos, 20, 20), self.COLOR_GRID)
        
        # Draw cursor with range indicator
        cursor_color = self.COLOR_CURSOR if can_afford and not is_occupied else self.COLOR_CURSOR_INVALID
        pygame.gfxdraw.aacircle(self.screen, cursor_pos[0]+10, cursor_pos[1]+10, selected_tower['range'], (*cursor_color, 100))
        pygame.gfxdraw.filled_circle(self.screen, cursor_pos[0]+10, cursor_pos[1]+10, selected_tower['range'], (*cursor_color, 20))
        pygame.draw.rect(self.screen, cursor_color, (*cursor_pos, 20, 20), 2)

        # Entities
        for tower in self.towers: tower.draw(self.screen)
        for enemy in self.enemies: enemy.draw(self.screen)
        for proj in self.projectiles: proj.draw(self.screen)
        for particle in self.particles: particle.draw(self.screen)

    def _render_ui(self):
        # Gold
        gold_text = self.font_m.render(f"GOLD: {self.gold}", True, self.COLOR_TEXT)
        self.screen.blit(gold_text, (10, 10))
        
        # Base Health
        health_text = self.font_m.render(f"BASE HP: {max(0, self.base_health)}/100", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (self.SCREEN_WIDTH - health_text.get_width() - 10, 10))
        
        # Wave Info
        if self.game_won:
            wave_str = "VICTORY!"
        elif self.game_over:
             wave_str = f"DEFEATED ON WAVE {self.current_wave}"
        elif self.wave_spawning or self.enemies:
            wave_str = f"WAVE {self.current_wave} / {self.MAX_WAVES}"
        else:
            wave_str = f"NEXT WAVE IN {max(0, 5 - self.inter_wave_timer // 30)}..."
        
        wave_text = self.font_m.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH/2 - wave_text.get_width()/2, self.SCREEN_HEIGHT - 30))

        # Selected Tower Info
        tower_type = self.TOWER_TYPES[self.selected_tower_type_idx]
        name_text = self.font_s.render(f"Tower: {tower_type['name']}", True, self.COLOR_TEXT)
        cost_text = self.font_s.render(f"Cost: {tower_type['cost']}", True, self.COLOR_TEXT if self.gold >= tower_type['cost'] else (255, 100, 100))
        self.screen.blit(name_text, (10, 35))
        self.screen.blit(cost_text, (10, 55))
        
        # Game Over / Win message
        if self.game_over and self.terminal_reward_given:
            msg = "VICTORY!" if self.game_won else "GAME OVER"
            color = (100, 255, 100) if self.game_won else (255, 100, 100)
            end_text = self.font_l.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            pygame.draw.rect(self.screen, (0,0,0,150), text_rect.inflate(20, 20))
            self.screen.blit(end_text, text_rect)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gold": self.gold,
            "base_health": self.base_health,
            "wave": self.current_wave,
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        
        # Test brief constraints
        assert self.base_health <= 100
        assert self.gold >= 0
        assert self.current_wave <= self.MAX_WAVES
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    # Override auto_advance for human play
    env.auto_advance = False
    
    running = True
    terminated = False
    
    # Pygame setup for human play
    render_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Tower Defense")
    
    action = env.action_space.sample()
    action.fill(0)

    while running:
        # --- Action mapping for human play ---
        movement = 0 # none
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = np.array([movement, space, shift])

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Print info for debugging
            # print(f"Step: {info['steps']}, Score: {info['score']:.2f}, Reward: {reward:.2f}, Terminated: {terminated}")

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Limit to 30 FPS

    env.close()