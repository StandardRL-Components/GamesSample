import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper classes defined outside the environment for clarity
class Tower:
    def __init__(self, pos, spec):
        self.pos = pos
        self.spec = spec
        self.cooldown = 0
        self.target = None

class Enemy:
    def __init__(self, health, speed, gold_value, path_points):
        self.pos = list(path_points[0])
        self.max_health = health
        self.health = health
        self.speed = speed
        self.gold_value = gold_value
        self.path_points = path_points
        self.path_index = 0
        self.distance_traveled = 0
        self.is_alive = True

    def move(self):
        if self.path_index >= len(self.path_points) - 1:
            return True  # Reached the end

        target_pos = self.path_points[self.path_index + 1]
        direction = np.array(target_pos) - np.array(self.pos)
        distance = np.linalg.norm(direction)

        if distance < self.speed:
            self.pos = list(target_pos)
            self.path_index += 1
            self.distance_traveled += distance
        else:
            move_vec = (direction / distance) * self.speed
            self.pos[0] += move_vec[0]
            self.pos[1] += move_vec[1]
            self.distance_traveled += self.speed
            
        return False

class Projectile:
    def __init__(self, start_pos, target, damage, speed, color):
        self.pos = list(start_pos)
        self.target = target
        self.damage = damage
        self.speed = speed
        self.color = color

    def move(self):
        if not self.target.is_alive:
            return True # Target is gone
            
        direction = np.array(self.target.pos) - np.array(self.pos)
        distance = np.linalg.norm(direction)

        if distance < self.speed:
            return True  # Hit
        
        move_vec = (direction / distance) * self.speed
        self.pos[0] += move_vec[0]
        self.pos[1] += move_vec[1]
        return False

class Particle:
    def __init__(self, pos, vel, radius, color, lifespan):
        self.pos = list(pos)
        self.vel = list(vel)
        self.radius = radius
        self.color = color
        self.lifespan = lifespan
        self.life = lifespan

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.life -= 1
        return self.life <= 0

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to select a tower slot. Shift to cycle tower types. Space to place a tower."
    )

    game_description = (
        "A top-down tower defense game. Place towers to defend your base from waves of enemies."
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    MAX_STEPS = 30 * 60 * 2  # 2 minutes at 30fps
    MAX_WAVES = 20
    BASE_MAX_HEALTH = 100
    INITIAL_GOLD = 100
    PREP_PHASE_DURATION = 30 * 5  # 5 seconds

    # --- Colors ---
    COLOR_BG = (32, 32, 32)
    COLOR_PATH = (64, 64, 64)
    COLOR_SLOT = (80, 80, 80)
    COLOR_SLOT_HOVER = (255, 255, 0)
    COLOR_BASE = (0, 200, 0)
    COLOR_ENEMY = (255, 64, 64)
    COLOR_TEXT = (255, 255, 255)
    COLOR_GOLD = (255, 223, 0)
    
    TOWER_SPECS = [
        {"name": "Cannon", "cost": 25, "range": 80, "damage": 10, "fire_rate": 30, "color": (0, 170, 255), "proj_speed": 8},
        {"name": "Sniper", "cost": 60, "range": 150, "damage": 35, "fire_rate": 90, "color": (200, 0, 255), "proj_speed": 15},
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("Consolas", 16)
        self.font_m = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_l = pygame.font.SysFont("Consolas", 32, bold=True)
        
        self._define_world()
        
        # State must be initialized by reset() before validation is called,
        # as validation calls methods that depend on the initialized state.
        self.reset()
        self.validate_implementation()
    
    def _define_world(self):
        self.PATH_POINTS = [
            (-20, 200), (80, 200), (80, 80), (320, 80), 
            (320, 320), (560, 320), (560, 150), (self.WIDTH + 20, 150)
        ]
        self.BASE_POS = (self.WIDTH, 150)

        self.TOWER_SLOTS = []
        for x in [160, 240, 400, 480]:
            for y in [40, 120, 200, 280, 360]:
                 # Ensure slots are not on the path
                too_close = False
                for i in range(len(self.PATH_POINTS) - 1):
                    p1 = np.array(self.PATH_POINTS[i])
                    p2 = np.array(self.PATH_POINTS[i+1])
                    p3 = np.array((x, y))
                    d = np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1) if np.linalg.norm(p2-p1) != 0 else np.linalg.norm(p3-p1)
                    if d < 30:
                        too_close = True
                        break
                if not too_close:
                    self.TOWER_SLOTS.append((x, y))
        self.TOWER_SLOTS.sort(key=lambda p: (p[1], p[0])) # Sort top-to-bottom, left-to-right

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.base_health = self.BASE_MAX_HEALTH
        self.gold = self.INITIAL_GOLD
        
        self.current_wave_index = -1
        self.enemies_to_spawn = []
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_slot_idx = 0
        self.selected_tower_type_idx = 0
        
        self.last_space_held = False
        self.last_shift_held = False

        self._start_next_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.001 # Small reward for surviving

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.last_space_held
        shift_press = shift_held and not self.last_shift_held
        self.last_space_held, self.last_shift_held = space_held, shift_held
        
        # --- Handle Input ---
        if self.game_phase == "PREP":
            reward += self._handle_input(movement, space_press, shift_press)

        # --- Update Game State ---
        state_reward = self._update_game_state()
        reward += state_reward

        self.score += reward
        
        # --- Check Termination ---
        terminated = self.steps >= self.MAX_STEPS or self.base_health <= 0 or self.win
        if terminated and not self.game_over:
            self.game_over = True
            if self.win:
                reward += 100
            elif self.base_health <= 0:
                reward -= 100
        
        # Reward clipping as per brief
        if terminated:
            reward = np.clip(reward, -100, 100)
        else:
            reward = np.clip(reward, -10, 10)

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_press, shift_press):
        num_slots = len(self.TOWER_SLOTS)
        unique_cols = sorted(list(set(p[0] for p in self.TOWER_SLOTS)))
        num_cols = len(unique_cols)
        
        current_pos = self.TOWER_SLOTS[self.cursor_slot_idx]
        current_col_idx = unique_cols.index(current_pos[0])

        # Movement: 1=up, 2=down, 3=left, 4=right
        if movement == 1: self.cursor_slot_idx = (self.cursor_slot_idx - 1) % num_slots
        elif movement == 2: self.cursor_slot_idx = (self.cursor_slot_idx + 1) % num_slots
        elif movement == 3: 
            new_col_idx = (current_col_idx - 1 + num_cols) % num_cols
            target_x = unique_cols[new_col_idx]
            # Find closest slot in the new column
            slots_in_col = [i for i, pos in enumerate(self.TOWER_SLOTS) if pos[0] == target_x]
            self.cursor_slot_idx = min(slots_in_col, key=lambda i: abs(self.TOWER_SLOTS[i][1] - current_pos[1]))
        elif movement == 4:
            new_col_idx = (current_col_idx + 1) % num_cols
            target_x = unique_cols[new_col_idx]
            # Find closest slot in the new column
            slots_in_col = [i for i, pos in enumerate(self.TOWER_SLOTS) if pos[0] == target_x]
            self.cursor_slot_idx = min(slots_in_col, key=lambda i: abs(self.TOWER_SLOTS[i][1] - current_pos[1]))


        if shift_press:
            self.selected_tower_type_idx = (self.selected_tower_type_idx + 1) % len(self.TOWER_SPECS)
            # sound: "UI_cycle.wav"

        if space_press:
            spec = self.TOWER_SPECS[self.selected_tower_type_idx]
            pos = self.TOWER_SLOTS[self.cursor_slot_idx]
            is_occupied = any(t.pos == pos for t in self.towers)

            if not is_occupied and self.gold >= spec["cost"]:
                self.gold -= spec["cost"]
                self.towers.append(Tower(pos, spec))
                self._create_particles(pos, 15, spec["color"], 20, 2)
                # sound: "place_tower.wav"
                return 0.5 # Reward for successful placement
            else:
                # sound: "error.wav"
                return -0.1 # Penalty for invalid action
        return 0

    def _update_game_state(self):
        reward = 0
        
        # --- Phase Management ---
        if self.game_phase == "PREP":
            self.prep_phase_timer -= 1
            if self.prep_phase_timer <= 0:
                self.game_phase = "WAVE"
        elif self.game_phase == "WAVE":
            self._spawn_enemies_for_wave()
        
        # --- Update Entities ---
        # Particles
        self.particles = [p for p in self.particles if not p.update()]

        # Projectiles
        for p in self.projectiles[:]:
            if p.move():
                self.projectiles.remove(p)
                if p.target.is_alive: # If it hit
                    p.target.health -= p.damage
                    self._create_particles(p.target.pos, 5, (255,255,255), 10, 1)
                    # sound: "hit_enemy.wav"
        
        # Enemies
        for e in self.enemies[:]:
            if not e.is_alive: continue

            if e.health <= 0:
                reward += 1.0
                self.gold += e.gold_value
                e.is_alive = False
                self.enemies.remove(e)
                self._create_particles(e.pos, 30, self.COLOR_ENEMY, 30, 3)
                # sound: "enemy_die.wav"
                continue

            if e.move():
                self.base_health -= 10
                e.is_alive = False
                self.enemies.remove(e)
                self.base_health = max(0, self.base_health)
                reward -= 5.0
                # sound: "base_damage.wav"

        # Towers
        for t in self.towers:
            if t.cooldown > 0:
                t.cooldown -= 1
                continue
            
            # Find target: furthest enemy in range
            possible_targets = [e for e in self.enemies if np.linalg.norm(np.array(e.pos) - np.array(t.pos)) <= t.spec["range"]]
            if not possible_targets:
                t.target = None
                continue
            
            t.target = max(possible_targets, key=lambda e: e.distance_traveled)
            
            if t.target and t.cooldown <= 0:
                t.cooldown = t.spec["fire_rate"]
                proj_color = (255, 255, 100) if t.spec["name"] == "Cannon" else (220, 100, 255)
                self.projectiles.append(Projectile(t.pos, t.target, t.spec["damage"], t.spec["proj_speed"], proj_color))
                self._create_particles(t.pos, 3, (255,255,255), 5, 0.5, math.atan2(t.target.pos[1]-t.pos[1], t.target.pos[0]-t.pos[0]))
                # sound: "tower_fire.wav"

        # --- Wave End Check ---
        if self.game_phase == "WAVE" and not self.enemies and not self.enemies_to_spawn:
            reward += 10.0 # Wave clear bonus
            self.gold += 50 + self.current_wave_index * 5
            self._start_next_wave()

        return reward

    def _start_next_wave(self):
        self.current_wave_index += 1
        if self.current_wave_index >= self.MAX_WAVES:
            self.win = True
            return

        self.game_phase = "PREP"
        self.prep_phase_timer = self.PREP_PHASE_DURATION
        
        num_enemies = 3 + self.current_wave_index
        base_health = 10 + self.current_wave_index * 2
        health_multiplier = 1 + (self.current_wave_index // 5) * 0.25
        
        self.enemies_to_spawn = []
        for i in range(num_enemies):
            health = int(base_health * health_multiplier)
            speed = 1.0 + self.np_random.uniform(-0.1, 0.1) + (self.current_wave_index / self.MAX_WAVES) * 0.5
            gold = 2 + int(self.current_wave_index / 5)
            self.enemies_to_spawn.append({"health": health, "speed": speed, "gold": gold})
    
    def _spawn_enemies_for_wave(self):
        # Stagger spawns
        if self.enemies_to_spawn and self.steps % 15 == 0:
            spec = self.enemies_to_spawn.pop(0)
            self.enemies.append(Enemy(spec["health"], spec["speed"], spec["gold"], self.PATH_POINTS))
            
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
            "gold": self.gold,
            "wave": self.current_wave_index + 1,
            "base_health": self.base_health,
        }

    def _render_game(self):
        # Path
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.PATH_POINTS, 30)

        # Tower Slots
        for i, pos in enumerate(self.TOWER_SLOTS):
            color = self.COLOR_SLOT_HOVER if i == self.cursor_slot_idx and self.game_phase == "PREP" else self.COLOR_SLOT
            pygame.draw.rect(self.screen, color, (pos[0]-10, pos[1]-10, 20, 20), 1, border_radius=3)
            
        # Base (visual only)
        pygame.draw.rect(self.screen, self.COLOR_BASE, (self.WIDTH-10, self.BASE_POS[1]-15, 10, 30))

        # Towers
        for t in self.towers:
            pygame.draw.rect(self.screen, t.spec["color"], (t.pos[0]-8, t.pos[1]-8, 16, 16), border_radius=2)
            pygame.draw.rect(self.screen, self.COLOR_TEXT, (t.pos[0]-8, t.pos[1]-8, 16, 16), 1, border_radius=2)
            if self.cursor_slot_idx < len(self.TOWER_SLOTS) and t.pos == self.TOWER_SLOTS[self.cursor_slot_idx]:
                 pygame.gfxdraw.aacircle(self.screen, int(t.pos[0]), int(t.pos[1]), t.spec["range"], (255,255,255,100))

        # Enemies
        for e in self.enemies:
            size = int(8 + e.health/e.max_health * 4)
            pygame.gfxdraw.filled_circle(self.screen, int(e.pos[0]), int(e.pos[1]), size, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, int(e.pos[0]), int(e.pos[1]), size, self.COLOR_ENEMY)
            # Health bar
            health_pct = e.health / e.max_health
            bar_len = 20
            pygame.draw.rect(self.screen, (90,0,0), (e.pos[0]-bar_len/2, e.pos[1]-18, bar_len, 3))
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (e.pos[0]-bar_len/2, e.pos[1]-18, bar_len * health_pct, 3))

        # Projectiles
        for p in self.projectiles:
            direction = np.array(p.target.pos) - np.array(p.pos)
            distance = np.linalg.norm(direction)
            if distance == 0: continue
            move_vec = (direction / distance) * p.speed
            pygame.draw.line(self.screen, p.color, p.pos, (p.pos[0]-move_vec[0]*0.5, p.pos[1]-move_vec[1]*0.5), 3)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p.life / p.lifespan))
            color = p.color + (alpha,)
            s = pygame.Surface((p.radius*2, p.radius*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p.radius, p.radius), int(p.radius * (p.life/p.lifespan)))
            self.screen.blit(s, (p.pos[0]-p.radius, p.pos[1]-p.radius))
    
    def _render_ui(self):
        # Top Bar
        pygame.draw.rect(self.screen, (0,0,0,150), (0,0,self.WIDTH, 30))
        # Base Health
        health_text = self.font_m.render("BASE HEALTH", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (self.WIDTH/2 - health_text.get_width()/2, 5))
        health_pct = self.base_health / self.BASE_MAX_HEALTH
        health_bar_color = (int(255*(1-health_pct)), int(255*health_pct), 0)
        pygame.draw.rect(self.screen, (90,0,0), (self.WIDTH/2 - 100, 22, 200, 5))
        pygame.draw.rect(self.screen, health_bar_color, (self.WIDTH/2 - 100, 22, 200 * health_pct, 5))

        # Gold
        gold_text = self.font_m.render(f"GOLD: {self.gold}", True, self.COLOR_GOLD)
        self.screen.blit(gold_text, (self.WIDTH - gold_text.get_width() - 10, 5))

        # Wave
        wave_str = f"WAVE: {self.current_wave_index+1}/{self.MAX_WAVES}"
        wave_text = self.font_m.render(wave_str, True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 5))
        
        # Bottom Bar (Prep Phase)
        if self.game_phase == "PREP":
            pygame.draw.rect(self.screen, (0,0,0,150), (0, self.HEIGHT-40, self.WIDTH, 40))
            spec = self.TOWER_SPECS[self.selected_tower_type_idx]
            can_afford = self.gold >= spec['cost']
            cost_color = self.COLOR_TEXT if can_afford else self.COLOR_ENEMY
            
            info_str = f"Tower: {spec['name']} | Cost: {spec['cost']} | Dmg: {spec['damage']} | Rng: {spec['range']}"
            info_text = self.font_s.render(info_str, True, self.COLOR_TEXT)
            cost_text = self.font_s.render(f"{spec['cost']}", True, cost_color)
            
            self.screen.blit(info_text, (10, self.HEIGHT - 28))

            # Prep Timer
            timer_pct = self.prep_phase_timer / self.PREP_PHASE_DURATION
            timer_text = self.font_m.render("PREP PHASE", True, self.COLOR_TEXT)
            self.screen.blit(timer_text, (self.WIDTH/2 - timer_text.get_width()/2, self.HEIGHT - 32))
            pygame.draw.rect(self.screen, self.COLOR_SLOT, (self.WIDTH/2 - 100, self.HEIGHT - 12, 200, 5))
            pygame.draw.rect(self.screen, self.COLOR_SLOT_HOVER, (self.WIDTH/2 - 100, self.HEIGHT - 12, 200 * timer_pct, 5))

        # Game Over/Win Message
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_BASE if self.win else self.COLOR_ENEMY
            end_text = self.font_l.render(msg, True, color)
            s.blit(end_text, (self.WIDTH/2 - end_text.get_width()/2, self.HEIGHT/2 - end_text.get_height()/2))
            self.screen.blit(s, (0,0))

    def _create_particles(self, pos, count, color, lifespan, speed_mult, angle=None):
        for _ in range(count):
            if angle is None:
                rad = self.np_random.uniform(0, 2 * math.pi)
            else:
                rad = angle + self.np_random.uniform(-0.5, 0.5)
            speed = self.np_random.uniform(1, 3) * speed_mult
            vel = [math.cos(rad) * speed, math.sin(rad) * speed]
            radius = self.np_random.uniform(2, 5)
            self.particles.append(Particle(pos, vel, radius, color, lifespan))

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- To play manually ---
    # This requires a window, so we'll re-init pygame for display
    pygame.display.init()
    pygame.display.set_caption("Tower Defense")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    # Map pygame keys to the action space
    key_to_action = {
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
        for key, move_val in key_to_action.items():
            if keys[key]:
                movement_action = move_val
                break
        
        if keys[pygame.K_SPACE]:
            space_action = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1

        action = [movement_action, space_action, shift_action]
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Run at 30 FPS
        
    pygame.quit()