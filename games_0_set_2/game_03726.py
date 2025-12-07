
# Generated: 2025-08-28T00:13:54.356224
# Source Brief: brief_03726.md
# Brief Index: 3726

        
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
class Tower:
    def __init__(self, grid_pos, tower_type_info):
        self.grid_pos = grid_pos
        self.type = tower_type_info['name']
        self.range = tower_type_info['range']
        self.damage = tower_type_info['damage']
        self.fire_rate = tower_type_info['fire_rate']
        self.cooldown = 0
        self.target = None

class Enemy:
    def __init__(self, health, speed, value, path):
        self.path = path
        self.pos = list(path[0])  # Start at the beginning of the path
        self.health = health
        self.max_health = health
        self.speed = speed
        self.value = value
        self.waypoint_index = 1
        self.is_alive = True

    def move(self):
        if self.waypoint_index >= len(self.path):
            return True  # Reached the end

        target_waypoint = self.path[self.waypoint_index]
        direction = [target_waypoint[i] - self.pos[i] for i in range(2)]
        distance = math.hypot(*direction)

        if distance < self.speed:
            self.pos = list(target_waypoint)
            self.waypoint_index += 1
        else:
            normalized_direction = [d / distance for d in direction]
            self.pos[0] += normalized_direction[0] * self.speed
            self.pos[1] += normalized_direction[1] * self.speed
        return False

    def take_damage(self, amount):
        self.health -= amount
        if self.health <= 0:
            self.is_alive = False
        return not self.is_alive

class Projectile:
    def __init__(self, start_pos, target_enemy, speed, damage):
        self.pos = list(start_pos)
        self.target = target_enemy
        self.speed = speed
        self.damage = damage

    def move(self):
        if not self.target.is_alive:
            return True # Target is dead

        direction = [self.target.pos[i] - self.pos[i] for i in range(2)]
        distance = math.hypot(*direction)

        if distance < self.speed:
            return True # Reached target
        
        normalized_direction = [d / distance for d in direction]
        self.pos[0] += normalized_direction[0] * self.speed
        self.pos[1] += normalized_direction[1] * self.speed
        return False

class Particle:
    def __init__(self, pos, vel, life, start_color, end_color):
        self.pos = list(pos)
        self.vel = list(vel)
        self.life = life
        self.max_life = life
        self.start_color = start_color
        self.end_color = end_color

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.life -= 1
        return self.life <= 0

    def get_color(self):
        life_ratio = self.life / self.max_life
        r = self.start_color[0] * life_ratio + self.end_color[0] * (1 - life_ratio)
        g = self.start_color[1] * life_ratio + self.end_color[1] * (1 - life_ratio)
        b = self.start_color[2] * life_ratio + self.end_color[2] * (1 - life_ratio)
        return (int(r), int(g), int(b))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the placement cursor. "
        "Press Space to build the selected tower. Hold Shift to cycle tower types."
    )

    game_description = (
        "Defend your base from waves of invaders by strategically placing towers. "
        "Survive 20 waves to win."
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_W, GRID_H = 16, 12
    TILE_W_HALF, TILE_H_HALF = 20, 10
    ISO_OFFSET_X, ISO_OFFSET_Y = WIDTH // 2, 80
    MAX_STEPS = 3000
    MAX_WAVES = 20

    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_PATH = (40, 50, 60)
    COLOR_GRID = (25, 35, 45)
    COLOR_BASE = (0, 150, 50)
    COLOR_ENEMY = (220, 40, 40)
    COLOR_ENEMY_HEALTH = (50, 200, 50)
    COLOR_SELECTOR = (255, 255, 0)
    COLOR_TEXT = (220, 220, 220)
    COLOR_UI_BG = (30, 40, 55)
    COLOR_UI_ACCENT = (80, 120, 200)

    TOWER_TYPES = [
        {'name': 'Turret', 'cost': 50, 'range': 3.5, 'damage': 10, 'fire_rate': 20, 'color': (60, 140, 255)},
        {'name': 'Sniper', 'cost': 120, 'range': 6.0, 'damage': 50, 'fire_rate': 80, 'color': (255, 100, 255)},
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("consolas", 16, bold=True)
        self.font_m = pygame.font.SysFont("consolas", 20, bold=True)

        self._define_level()
        self.reset()
        self.validate_implementation()
    
    def _define_level(self):
        self.path_waypoints = [
            (-1, 5), (2, 5), (2, 2), (6, 2), (6, 8), (10, 8), (10, 1), (14, 1), (14, 6), (self.GRID_W, 6)
        ]
        self.placement_zones = set()
        for r in range(self.GRID_H):
            for c in range(self.GRID_W):
                self.placement_zones.add((c, r))
        
        # Remove path tiles from placement zones
        path_tiles = set()
        for i in range(len(self.path_waypoints) - 1):
            p1 = self.path_waypoints[i]
            p2 = self.path_waypoints[i+1]
            dx = 1 if p2[0] > p1[0] else -1 if p2[0] < p1[0] else 0
            dy = 1 if p2[1] > p1[1] else -1 if p2[1] < p1[1] else 0
            x, y = p1
            while (x, y) != (p2[0] + dx, p2[1] + dy):
                if 0 <= x < self.GRID_W and 0 <= y < self.GRID_H:
                    path_tiles.add((x, y))
                    if (x, y) in self.placement_zones:
                        self.placement_zones.remove((x, y))
                if (x,y) == p2: break
                x += dx
                y += dy
        self.base_pos = (self.GRID_W - 1, 6)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.gold = 150
        self.current_wave = 0
        self.wave_timer = 150 # Time before first wave
        self.wave_ongoing = False
        self.enemies_to_spawn = []

        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.selector_pos = (self.GRID_W // 2, self.GRID_H // 2)
        self.selected_tower_idx = 0

        self.prev_space_held = False
        self.prev_shift_held = False
        self.move_cooldown = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        
        # --- Handle Input ---
        self._handle_input(action)

        # --- Game Logic ---
        self._update_wave_manager()
        reward += self._update_towers()
        kill_reward = self._update_projectiles()
        reward += kill_reward
        self.gold += kill_reward * 5 # Gold for kills

        enemy_reached_base = self._update_enemies()
        self._update_particles()

        # --- Check Termination Conditions ---
        terminated = False
        if enemy_reached_base:
            self.game_over = True
            terminated = True
            reward = -100
        elif self.win:
            self.game_over = True
            terminated = True
            reward += 100
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            reward = -50
        
        if self.auto_advance:
            self.clock.tick(30)

        self.score += reward
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Selector movement with cooldown
        if self.move_cooldown > 0:
            self.move_cooldown -= 1
        elif movement != 0:
            x, y = self.selector_pos
            if movement == 1: y -= 1 # Up
            elif movement == 2: y += 1 # Down
            elif movement == 3: x -= 1 # Left
            elif movement == 4: x += 1 # Right
            self.selector_pos = (max(0, min(self.GRID_W - 1, x)), max(0, min(self.GRID_H - 1, y)))
            self.move_cooldown = 5

        # Cycle tower type on key press (rising edge)
        if shift_held and not self.prev_shift_held:
            self.selected_tower_idx = (self.selected_tower_idx + 1) % len(self.TOWER_TYPES)
        self.prev_shift_held = shift_held

        # Place tower on key press (rising edge)
        if space_held and not self.prev_space_held:
            self._place_tower()
        self.prev_space_held = space_held

    def _place_tower(self):
        tower_type = self.TOWER_TYPES[self.selected_tower_idx]
        if self.gold >= tower_type['cost']:
            if self.selector_pos in self.placement_zones:
                is_occupied = any(t.grid_pos == self.selector_pos for t in self.towers)
                if not is_occupied:
                    self.gold -= tower_type['cost']
                    self.towers.append(Tower(self.selector_pos, tower_type))
                    # sfx: place_tower.wav
                    self._create_effect(self._to_iso(*self.selector_pos), 15, (200, 200, 255))


    def _update_wave_manager(self):
        if self.game_over: return

        if not self.wave_ongoing:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self.current_wave += 1
                if self.current_wave > self.MAX_WAVES:
                    self.win = True
                    return
                self._generate_wave()
                self.wave_ongoing = True

        elif not self.enemies and not self.enemies_to_spawn:
            # Wave cleared
            self.wave_ongoing = False
            self.wave_timer = 300 # 10 seconds grace period
            # reward is handled in step() to be returned
            # self.score += 50
    
    def _generate_wave(self):
        num_enemies = 3 + self.current_wave * 2
        health = 50 * (1.05 ** (self.current_wave - 1))
        speed = 0.05 * (1.05 ** (self.current_wave - 1))
        value = 5
        for i in range(num_enemies):
            # Stagger spawn times
            self.enemies_to_spawn.append({'delay': i * 30, 'health': health, 'speed': speed, 'value': value})

    def _update_towers(self):
        reward = 0
        for tower in self.towers:
            if tower.cooldown > 0:
                tower.cooldown -= 1
                continue
            
            # Find a target
            possible_targets = []
            for enemy in self.enemies:
                dist = math.hypot(enemy.pos[0] - tower.grid_pos[0], enemy.pos[1] - tower.grid_pos[1])
                if dist <= tower.range:
                    possible_targets.append(enemy)
            
            if possible_targets:
                tower.target = possible_targets[0] # Target first enemy in list
                
                # Fire
                tower.cooldown = tower.fire_rate
                start_pos_iso = self._to_iso(*tower.grid_pos)
                self.projectiles.append(Projectile(start_pos_iso, tower.target, 5, tower.damage))
                # sfx: shoot.wav
                self._create_effect(start_pos_iso, 5, (255, 255, 150))
        return reward

    def _update_projectiles(self):
        kill_reward = 0
        for p in self.projectiles[:]:
            if p.move():
                if p.target.is_alive:
                    # sfx: hit.wav
                    is_kill = p.target.take_damage(p.damage)
                    self._create_effect(p.pos, 10, (255, 150, 50))
                    if is_kill:
                        # sfx: enemy_die.wav
                        kill_reward += 1
                        self._create_effect(self._to_iso(*p.target.pos), 25, (255, 50, 50))
                self.projectiles.remove(p)
        return kill_reward

    def _update_enemies(self):
        # Spawn new enemies from the wave list
        if self.wave_ongoing and self.enemies_to_spawn:
            for spawn_info in self.enemies_to_spawn[:]:
                spawn_info['delay'] -= 1
                if spawn_info['delay'] <= 0:
                    self.enemies.append(Enemy(spawn_info['health'], spawn_info['speed'], spawn_info['value'], self.path_waypoints))
                    self.enemies_to_spawn.remove(spawn_info)

        # Update existing enemies
        for enemy in self.enemies[:]:
            if not enemy.is_alive:
                self.enemies.remove(enemy)
                continue
            if enemy.move():
                # sfx: base_damage.wav
                self.enemies.remove(enemy)
                return True # Enemy reached base
        return False

    def _update_particles(self):
        for p in self.particles[:]:
            if p.update():
                self.particles.remove(p)

    def _create_effect(self, pos, num_particles, color):
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 2.5)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = random.randint(10, 25)
            self.particles.append(Particle(pos, vel, life, color, self.COLOR_BG))

    def _to_iso(self, x, y):
        iso_x = self.ISO_OFFSET_X + (x - y) * self.TILE_W_HALF
        iso_y = self.ISO_OFFSET_Y + (x + y) * self.TILE_H_HALF
        return int(iso_x), int(iso_y)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid and path
        for r in range(self.GRID_H):
            for c in range(self.GRID_W):
                p1 = self._to_iso(c, r)
                p2 = self._to_iso(c + 1, r)
                p3 = self._to_iso(c + 1, r + 1)
                p4 = self._to_iso(c, r + 1)
                is_path = any(math.hypot(c-wp[0], r-wp[1])<1 for wp in self.path_waypoints) # Quick check
                color = self.COLOR_PATH if is_path else self.COLOR_GRID
                pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3, p4], color)

        # Draw base
        base_iso = self._to_iso(*self.base_pos)
        pygame.gfxdraw.filled_circle(self.screen, base_iso[0], base_iso[1], 12, self.COLOR_BASE)
        pygame.gfxdraw.aacircle(self.screen, base_iso[0], base_iso[1], 12, tuple(c*0.8 for c in self.COLOR_BASE))

        # Draw towers
        for tower in self.towers:
            pos_iso = self._to_iso(*tower.grid_pos)
            tower_type = next(t for t in self.TOWER_TYPES if t['name'] == tower.type)
            pygame.gfxdraw.filled_circle(self.screen, pos_iso[0], pos_iso[1], 8, tower_type['color'])
            pygame.gfxdraw.aacircle(self.screen, pos_iso[0], pos_iso[1], int(tower.range * self.TILE_W_HALF), (*tower_type['color'], 20))


        # Draw enemies
        for enemy in self.enemies:
            pos_iso = self._to_iso(*enemy.pos)
            pygame.gfxdraw.filled_circle(self.screen, pos_iso[0], pos_iso[1], 6, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos_iso[0], pos_iso[1], 6, tuple(c*0.8 for c in self.COLOR_ENEMY))
            # Health bar
            health_ratio = enemy.health / enemy.max_health
            bar_w = 12
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (pos_iso[0] - bar_w/2, pos_iso[1] - 12, bar_w, 3))
            pygame.draw.rect(self.screen, self.COLOR_ENEMY_HEALTH, (pos_iso[0] - bar_w/2, pos_iso[1] - 12, bar_w * health_ratio, 3))

        # Draw projectiles and particles
        for p in self.projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(p.pos[0]), int(p.pos[1]), 3, (255, 255, 200))
        for p in self.particles:
            pygame.draw.circle(self.screen, p.get_color(), p.pos, int(p.life/p.max_life * 3 + 1))
        
        # Draw selector
        if not self.game_over:
            sel_iso = self._to_iso(*self.selector_pos)
            points = [
                self._to_iso(self.selector_pos[0], self.selector_pos[1]),
                self._to_iso(self.selector_pos[0] + 1, self.selector_pos[1]),
                self._to_iso(self.selector_pos[0] + 1, self.selector_pos[1] + 1),
                self._to_iso(self.selector_pos[0], self.selector_pos[1] + 1),
            ]
            pygame.draw.lines(self.screen, self.COLOR_SELECTOR, True, points, 2)


    def _render_ui(self):
        # Top bar
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (0, 0, self.WIDTH, 40))
        gold_text = self.font_m.render(f"GOLD: {self.gold}", True, (255, 223, 0))
        self.screen.blit(gold_text, (10, 10))

        if self.win:
            wave_str = "YOU WIN!"
        elif self.game_over:
            wave_str = "GAME OVER"
        else:
            wave_str = f"WAVE: {self.current_wave}/{self.MAX_WAVES}"
        wave_text = self.font_m.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))
        
        if not self.wave_ongoing and not self.game_over:
            timer_text = self.font_s.render(f"Next wave in: {self.wave_timer // 30 + 1}", True, self.COLOR_TEXT)
            self.screen.blit(timer_text, (self.WIDTH/2 - timer_text.get_width()/2, 12))

        # Bottom bar (tower selection)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (0, self.HEIGHT - 60, self.WIDTH, 60))
        
        start_x = self.WIDTH // 2 - (len(self.TOWER_TYPES) * 100) // 2
        for i, t_type in enumerate(self.TOWER_TYPES):
            box_rect = pygame.Rect(start_x + i * 110, self.HEIGHT - 55, 100, 50)
            
            is_selected = i == self.selected_tower_idx
            can_afford = self.gold >= t_type['cost']
            
            border_color = self.COLOR_UI_ACCENT if is_selected else self.COLOR_GRID
            pygame.draw.rect(self.screen, self.COLOR_PATH, box_rect)
            pygame.draw.rect(self.screen, border_color, box_rect, 2)

            name_color = self.COLOR_TEXT if can_afford else (150, 150, 150)
            name_surf = self.font_s.render(t_type['name'], True, name_color)
            self.screen.blit(name_surf, (box_rect.centerx - name_surf.get_width()//2, box_rect.y + 5))

            cost_color = (255, 223, 0) if can_afford else (150, 100, 0)
            cost_surf = self.font_s.render(f"Cost: {t_type['cost']}", True, cost_color)
            self.screen.blit(cost_surf, (box_rect.centerx - cost_surf.get_width()//2, box_rect.y + 25))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gold": self.gold,
            "wave": self.current_wave,
            "enemies": len(self.enemies),
            "towers": len(self.towers),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    
    terminated = False
    total_reward = 0
    
    # Mapping from Pygame keys to action space
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    while not terminated:
        movement_action = 0
        space_action = 0
        shift_action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        for key, move_val in key_map.items():
            if keys[key]:
                movement_action = move_val
                break # Only one movement at a time
        
        if keys[pygame.K_SPACE]:
            space_action = 1
        
        if keys[pygame.K_SHIFT] or keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1
            
        action = [movement_action, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}, Wave: {info['wave']}")
            pygame.time.wait(3000) # Pause before closing

    env.close()