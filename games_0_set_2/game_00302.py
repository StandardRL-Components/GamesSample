
# Generated: 2025-08-27T13:14:46.992442
# Source Brief: brief_00302.md
# Brief Index: 302

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import namedtuple
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper classes for game objects
Enemy = namedtuple("Enemy", ["pos", "health", "max_health", "speed", "path_index", "type", "slow_timer", "value"])
Tower = namedtuple("Tower", ["pos", "type", "cooldown", "target_id"])
Projectile = namedtuple("Projectile", ["pos", "velocity", "damage", "type", "blast_radius", "target_id"])
Particle = namedtuple("Particle", ["pos", "velocity", "color", "radius", "lifetime"])

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the placement cursor. "
        "Press 'Shift' to cycle through tower types. "
        "Press 'Space' to build the selected tower at the cursor's location."
    )

    game_description = (
        "Defend your base from waves of enemies by strategically placing defensive towers. "
        "Survive 10 waves to win. If your base's health reaches zero, you lose."
    )

    auto_advance = True

    # --- Constants ---
    # Game parameters
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = 40
    MAX_STEPS = 5000  # Increased to allow for 10 waves
    STARTING_BASE_HEALTH = 100
    STARTING_RESOURCES = 100
    WAVES_TO_WIN = 10
    
    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_GRID = (30, 40, 60, 100)
    COLOR_PATH = (40, 50, 75)
    COLOR_BASE = (0, 150, 100)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_TEXT = (220, 220, 240)
    
    TOWER_SPECS = {
        "SINGLE": {"cost": 30, "range": 100, "damage": 15, "fire_rate": 30, "color": (200, 50, 200), "proj_speed": 5, "blast": 0},
        "AOE": {"cost": 50, "range": 80, "damage": 10, "fire_rate": 60, "color": (220, 200, 50), "proj_speed": 4, "blast": 30},
        "SLOW": {"cost": 40, "range": 120, "damage": 2, "fire_rate": 45, "color": (50, 150, 250), "proj_speed": 6, "blast": 0},
    }
    TOWER_TYPES = list(TOWER_SPECS.keys())

    ENEMY_SPECS = {
        "NORMAL": {"health": 50, "speed": 1.0, "color": (220, 50, 50), "value": 5},
        "FAST": {"health": 35, "speed": 1.8, "color": (250, 120, 50), "value": 7},
        "HEAVY": {"health": 150, "speed": 0.7, "color": (180, 40, 100), "value": 10},
    }
    ENEMY_TYPES = list(ENEMY_SPECS.keys())

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 14, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 32, bold=True)
        
        self.grid_w = self.SCREEN_WIDTH // self.GRID_SIZE
        self.grid_h = self.SCREEN_HEIGHT // self.GRID_SIZE
        
        self.reset()

        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.base_health = self.STARTING_BASE_HEALTH
        self.resources = self.STARTING_RESOURCES
        
        self.wave_number = 0
        self.wave_in_progress = False
        self.wave_cooldown = 90  # Frames before first wave
        self.enemies_to_spawn = []
        self.spawn_timer = 0
        
        self.enemies = {}
        self.next_enemy_id = 0
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_pos = np.array([self.grid_w // 2, self.grid_h // 2])
        self.selected_tower_type_idx = 0
        self.last_space_held = False
        self.last_shift_held = False
        
        self._generate_path()
        self.occupied_cells = {tuple(p) for p in self.path_grid_coords}
        self.base_pos = pygame.Vector2(self.path_waypoints[-1])

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        
        self._handle_actions(movement, space_held, shift_held)
        
        reward += self._update_game_state()

        self.steps += 1
        terminated = self._check_termination()
        
        if terminated and not self.win:
            reward -= 100 # Penalty for losing
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_actions(self, movement, space_held, shift_held):
        # Move cursor
        if movement == 1 and self.cursor_pos[1] > 0: self.cursor_pos[1] -= 1
        elif movement == 2 and self.cursor_pos[1] < self.grid_h - 1: self.cursor_pos[1] += 1
        elif movement == 3 and self.cursor_pos[0] > 0: self.cursor_pos[0] -= 1
        elif movement == 4 and self.cursor_pos[0] < self.grid_w - 1: self.cursor_pos[0] += 1

        # Cycle tower type on key press
        if shift_held and not self.last_shift_held:
            self.selected_tower_type_idx = (self.selected_tower_type_idx + 1) % len(self.TOWER_TYPES)
        self.last_shift_held = shift_held

        # Place tower on key press
        if space_held and not self.last_space_held:
            self._place_tower()
        self.last_space_held = space_held

    def _update_game_state(self):
        reward = 0
        damage_dealt = False
        
        self._update_waves()
        self._spawn_enemies()
        
        reward_t, damage_t = self._update_towers()
        reward += reward_t
        damage_dealt |= damage_t
        
        reward_p, damage_p = self._update_projectiles()
        reward += reward_p
        damage_dealt |= damage_p
        
        reward += self._update_enemies()
        self._update_particles()
        
        if not damage_dealt:
            reward -= 0.01

        return reward

    def _update_waves(self):
        if not self.wave_in_progress and not self.game_over:
            if self.wave_cooldown > 0:
                self.wave_cooldown -= 1
            else:
                self.wave_number += 1
                if self.wave_number > self.WAVES_TO_WIN:
                    self.win = True
                    self.game_over = True
                    return
                self.wave_in_progress = True
                self.enemies_to_spawn = self._generate_wave()
                self.spawn_timer = 0

    def _generate_wave(self):
        wave = []
        num_enemies = 3 + self.wave_number * 2
        
        available_enemies = self.ENEMY_TYPES[:1 + self.wave_number // 2]
        
        for _ in range(num_enemies):
            enemy_type = self.np_random.choice(available_enemies)
            spec = self.ENEMY_SPECS[enemy_type].copy()
            spec["health"] *= (1.05 ** (self.wave_number - 1))
            spec["speed"] *= (1.02 ** ((self.wave_number - 1) // 3))
            wave.append((enemy_type, spec))
        return wave

    def _spawn_enemies(self):
        if self.enemies_to_spawn:
            self.spawn_timer -= 1
            if self.spawn_timer <= 0:
                enemy_type, spec = self.enemies_to_spawn.pop(0)
                new_enemy = Enemy(
                    pos=pygame.Vector2(self.path_waypoints[0]),
                    health=spec["health"], max_health=spec["health"],
                    speed=spec["speed"], path_index=0, type=enemy_type,
                    slow_timer=0, value=spec["value"]
                )
                self.enemies[self.next_enemy_id] = new_enemy
                self.next_enemy_id += 1
                self.spawn_timer = 30 # Spawn interval

    def _update_enemies(self):
        reward = 0
        to_remove = []
        for enemy_id, enemy in self.enemies.items():
            if enemy.path_index >= len(self.path_waypoints) - 1:
                self.base_health -= 10
                self.score -= 10
                reward -= 1
                to_remove.append(enemy_id)
                continue

            target_pos = pygame.Vector2(self.path_waypoints[enemy.path_index + 1])
            direction = (target_pos - enemy.pos).normalize()
            
            current_speed = enemy.speed
            if enemy.slow_timer > 0:
                current_speed *= 0.5
                enemy = enemy._replace(slow_timer=enemy.slow_timer - 1)

            enemy.pos.x += direction.x * current_speed
            enemy.pos.y += direction.y * current_speed

            if enemy.pos.distance_to(target_pos) < 5:
                enemy = enemy._replace(path_index=enemy.path_index + 1)
            
            self.enemies[enemy_id] = enemy

        for enemy_id in to_remove:
            del self.enemies[enemy_id]

        if self.wave_in_progress and not self.enemies and not self.enemies_to_spawn:
            self.wave_in_progress = False
            self.wave_cooldown = 180 # Time between waves
            reward += 100
            self.score += 100
        
        return reward

    def _update_towers(self):
        reward = 0
        damage_dealt = False
        for i, tower in enumerate(self.towers):
            tower = tower._replace(cooldown=max(0, tower.cooldown - 1))
            if tower.cooldown == 0:
                target, target_id = self._find_target(tower)
                if target:
                    tower = tower._replace(cooldown=self.TOWER_SPECS[tower.type]["fire_rate"], target_id=target_id)
                    self._fire_projectile(tower)
            self.towers[i] = tower
        return reward, damage_dealt

    def _find_target(self, tower):
        tower_spec = self.TOWER_SPECS[tower.type]
        tower_pos = pygame.Vector2(tower.pos[0] * self.GRID_SIZE + self.GRID_SIZE/2, tower.pos[1] * self.GRID_SIZE + self.GRID_SIZE/2)
        
        closest_enemy = None
        closest_dist = float('inf')
        closest_id = -1

        for enemy_id, enemy in self.enemies.items():
            dist = tower_pos.distance_to(enemy.pos)
            if dist <= tower_spec["range"] and dist < closest_dist:
                closest_dist = dist
                closest_enemy = enemy
                closest_id = enemy_id
        
        return closest_enemy, closest_id

    def _fire_projectile(self, tower):
        spec = self.TOWER_SPECS[tower.type]
        start_pos = pygame.Vector2(tower.pos[0] * self.GRID_SIZE + self.GRID_SIZE/2, tower.pos[1] * self.GRID_SIZE + self.GRID_SIZE/2)
        
        if tower.target_id not in self.enemies:
            return

        target_enemy = self.enemies[tower.target_id]
        direction = (target_enemy.pos - start_pos).normalize()
        velocity = direction * spec["proj_speed"]
        
        proj = Projectile(
            pos=start_pos, velocity=velocity, damage=spec["damage"],
            type=tower.type, blast_radius=spec["blast"], target_id=tower.target_id
        )
        self.projectiles.append(proj)
        # sfx: pew_sound()

    def _update_projectiles(self):
        reward = 0
        damage_dealt = False
        projectiles_to_keep = []
        for proj in self.projectiles:
            proj.pos.x += proj.velocity.x
            proj.pos.y += proj.velocity.y
            
            hit = False
            if proj.target_id in self.enemies:
                target_enemy = self.enemies[proj.target_id]
                if proj.pos.distance_to(target_enemy.pos) < 10:
                    hit = True
            elif proj.pos.x < 0 or proj.pos.x > self.SCREEN_WIDTH or proj.pos.y < 0 or proj.pos.y > self.SCREEN_HEIGHT:
                continue # Projectile went off-screen, discard

            if hit:
                # sfx: explosion_sound()
                r, d = self._handle_hit(proj, proj.target_id)
                reward += r
                damage_dealt |= d
            else:
                projectiles_to_keep.append(proj)
        
        self.projectiles = projectiles_to_keep
        return reward, damage_dealt

    def _handle_hit(self, proj, main_target_id):
        reward = 0
        damage_dealt = False
        
        if proj.blast_radius > 0: # AOE damage
            for enemy_id, enemy in list(self.enemies.items()):
                if proj.pos.distance_to(enemy.pos) <= proj.blast_radius:
                    r, d = self._damage_enemy(enemy_id, proj.damage, proj.type)
                    reward += r
                    damage_dealt |= d
        else: # Single target damage
            r, d = self._damage_enemy(main_target_id, proj.damage, proj.type)
            reward += r
            damage_dealt |= d
        
        self._create_explosion(proj.pos, proj.type)
        return reward, damage_dealt

    def _damage_enemy(self, enemy_id, damage, tower_type):
        if enemy_id not in self.enemies:
            return 0, False
        
        enemy = self.enemies[enemy_id]
        enemy = enemy._replace(health=enemy.health - damage)
        reward = 0.1 # Reward for any hit
        
        if tower_type == "SLOW":
            enemy = enemy._replace(slow_timer=90) # Slow for 3 seconds

        if enemy.health <= 0:
            reward += 1 # Reward for destroying
            self.score += enemy.value
            self.resources += enemy.value
            del self.enemies[enemy_id]
        else:
            self.enemies[enemy_id] = enemy
            
        return reward, True

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.lifetime > 0]
        for i, p in enumerate(self.particles):
            p.pos.x += p.velocity.x
            p.pos.y += p.velocity.y
            self.particles[i] = p._replace(
                lifetime=p.lifetime - 1,
                radius=max(0, p.radius * 0.95)
            )
            
    def _create_explosion(self, pos, tower_type):
        color = self.TOWER_SPECS[tower_type]["color"]
        num_particles = 10 if tower_type != "AOE" else 30
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            velocity = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            radius = self.np_random.uniform(2, 5)
            lifetime = self.np_random.integers(15, 30)
            self.particles.append(Particle(pos.copy(), velocity, color, radius, lifetime))

    def _generate_path(self):
        self.path_waypoints = []
        start_y = self.np_random.integers(self.GRID_SIZE * 2, self.SCREEN_HEIGHT - self.GRID_SIZE * 2)
        self.path_waypoints.append((-self.GRID_SIZE, start_y))
        
        x, y = 0, start_y
        px, py = -1, start_y / self.GRID_SIZE
        
        path_grid = set()
        
        while x < self.SCREEN_WIDTH - self.GRID_SIZE * 4:
            self.path_waypoints.append((x, y))
            gx, gy = x // self.GRID_SIZE, y // self.GRID_SIZE
            for i in range(max(0, gx-1), min(self.grid_w, gx+2)):
                for j in range(max(0, gy-1), min(self.grid_h, gy+2)):
                    path_grid.add((i,j))
            
            move_x = self.np_random.integers(self.GRID_SIZE * 2, self.GRID_SIZE * 5)
            x += move_x
            self.path_waypoints.append((x, y))
            
            move_y = self.np_random.integers(self.GRID_SIZE * 2, self.GRID_SIZE * 5)
            if y < self.SCREEN_HEIGHT / 2:
                y += move_y
            else:
                y -= move_y
            y = np.clip(y, self.GRID_SIZE, self.SCREEN_HEIGHT - self.GRID_SIZE)
        
        base_pos = (self.SCREEN_WIDTH - self.GRID_SIZE * 3, self.SCREEN_HEIGHT // 2)
        self.path_waypoints.append((x, y))
        self.path_waypoints.append((base_pos[0], y))
        self.path_waypoints.append(base_pos)
        
        self.path_grid_coords = set()
        for i in range(len(self.path_waypoints)-1):
            p1 = pygame.Vector2(self.path_waypoints[i])
            p2 = pygame.Vector2(self.path_waypoints[i+1])
            dist = p1.distance_to(p2)
            for d in range(0, int(dist), self.GRID_SIZE // 2):
                point = p1.lerp(p2, d / dist)
                gx, gy = int(point.x // self.GRID_SIZE), int(point.y // self.GRID_SIZE)
                self.path_grid_coords.add((gx, gy))

    def _place_tower(self):
        cx, cy = self.cursor_pos
        tower_type = self.TOWER_TYPES[self.selected_tower_type_idx]
        cost = self.TOWER_SPECS[tower_type]["cost"]

        if self.resources >= cost and (cx, cy) not in self.occupied_cells:
            self.resources -= cost
            new_tower = Tower(pos=(cx, cy), type=tower_type, cooldown=0, target_id=-1)
            self.towers.append(new_tower)
            self.occupied_cells.add((cx, cy))

    def _check_termination(self):
        if self.game_over: return True
        if self.base_health <= 0:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

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
            "base_health": self.base_health,
            "resources": self.resources,
            "wave": self.wave_number,
        }

    def _render_game(self):
        # Draw path
        if len(self.path_waypoints) > 1:
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_waypoints, self.GRID_SIZE)

        # Draw grid for empty, placeable cells
        for x in range(self.grid_w):
            for y in range(self.grid_h):
                if (x, y) not in self.occupied_cells:
                    rect = pygame.Rect(x * self.GRID_SIZE, y * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
                    pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # Draw base
        pygame.gfxdraw.box(self.screen, pygame.Rect(self.base_pos.x - 15, self.base_pos.y - 15, 30, 30), (*self.COLOR_BASE, 200))
        pygame.gfxdraw.rectangle(self.screen, pygame.Rect(self.base_pos.x - 15, self.base_pos.y - 15, 30, 30), self.COLOR_BASE)

        # Draw towers
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower.type]
            cx = tower.pos[0] * self.GRID_SIZE + self.GRID_SIZE // 2
            cy = tower.pos[1] * self.GRID_SIZE + self.GRID_SIZE // 2
            pygame.gfxdraw.filled_circle(self.screen, cx, cy, self.GRID_SIZE // 3, (*spec["color"], 100))
            pygame.gfxdraw.aacircle(self.screen, cx, cy, self.GRID_SIZE // 3, spec["color"])

        # Draw projectiles
        for proj in self.projectiles:
            color = self.TOWER_SPECS[proj.type]["color"]
            pygame.gfxdraw.filled_circle(self.screen, int(proj.pos.x), int(proj.pos.y), 4, color)
            pygame.gfxdraw.aacircle(self.screen, int(proj.pos.x), int(proj.pos.y), 4, (255,255,255))

        # Draw enemies
        for enemy in self.enemies.values():
            spec = self.ENEMY_SPECS[enemy.type]
            pos = (int(enemy.pos.x), int(enemy.pos.y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, spec["color"])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, (255,255,255))
            # Health bar
            health_pct = enemy.health / enemy.max_health
            pygame.draw.rect(self.screen, (50,50,50), (pos[0]-10, pos[1]-15, 20, 4))
            pygame.draw.rect(self.screen, (0,255,0), (pos[0]-10, pos[1]-15, int(20 * health_pct), 4))

        # Draw particles
        for p in self.particles:
            alpha_color = (*p.color, max(0, min(255, int(255 * (p.lifetime / 30.0)))))
            pygame.gfxdraw.filled_circle(self.screen, int(p.pos.x), int(p.pos.y), int(p.radius), alpha_color)

        # Draw cursor
        cx, cy = self.cursor_pos[0] * self.GRID_SIZE, self.cursor_pos[1] * self.GRID_SIZE
        rect = pygame.Rect(cx, cy, self.GRID_SIZE, self.GRID_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 2)
        
        # Draw tower range preview
        tower_type = self.TOWER_TYPES[self.selected_tower_type_idx]
        spec = self.TOWER_SPECS[tower_type]
        can_afford = self.resources >= spec["cost"]
        is_occupied = tuple(self.cursor_pos) in self.occupied_cells
        color = (0, 255, 0, 50) if can_afford and not is_occupied else (255, 0, 0, 50)
        pygame.gfxdraw.filled_circle(self.screen, cx + self.GRID_SIZE//2, cy + self.GRID_SIZE//2, spec["range"], color)

    def _render_ui(self):
        # Top panel
        panel_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, 30)
        pygame.gfxdraw.box(self.screen, panel_rect, (10, 15, 25, 200))
        
        # Health
        health_text = self.font_small.render(f"Base: {self.base_health}/{self.STARTING_BASE_HEALTH}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 8))
        
        # Resources
        res_text = self.font_small.render(f"Resources: {self.resources}", True, (255, 223, 0))
        self.screen.blit(res_text, (200, 8))
        
        # Wave
        wave_text = self.font_small.render(f"Wave: {self.wave_number}/{self.WAVES_TO_WIN}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (350, 8))

        # Score
        score_text = self.font_small.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (500, 8))

        # Selected Tower Info
        tower_type = self.TOWER_TYPES[self.selected_tower_type_idx]
        spec = self.TOWER_SPECS[tower_type]
        tower_info = self.font_small.render(f"Selected: {tower_type} (Cost: {spec['cost']})", True, spec["color"])
        self.screen.blit(tower_info, (10, self.SCREEN_HEIGHT - 20))
        
        if self.game_over:
            outcome_text = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            text_surf = self.font_large.render(outcome_text, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # To display the game, we need a Pygame window
    pygame.display.set_caption("Tower Defense")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    # Map keyboard keys to actions
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    total_reward = 0
    
    while not done:
        # Construct action from keyboard input
        movement = 0
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        for key, move_action in key_to_action.items():
            if keys[key]:
                movement = move_action
                break
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Control the frame rate of the manual play

    print(f"Game Over. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()