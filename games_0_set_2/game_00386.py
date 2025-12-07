
# Generated: 2025-08-27T13:30:04.236949
# Source Brief: brief_00386.md
# Brief Index: 386

        
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


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrows to move cursor, SPACE to build tower, SHIFT to cycle tower type. "
        "Survive 10 waves!"
    )

    game_description = (
        "Defend your base from waves of enemies by strategically placing towers. "
        "Earn money by defeating enemies to build more defenses."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_m = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_l = pygame.font.SysFont("Consolas", 32, bold=True)

        # --- Game Constants ---
        self.MAX_STEPS = 5000
        self.MAX_WAVES = 10
        self.INTERMISSION_TIME = 150  # 5 seconds at 30fps

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_PATH = (40, 50, 70)
        self.COLOR_BASE = (60, 180, 75)
        self.COLOR_UI_BG = (30, 35, 55)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_PLACEMENT_SPOT = (60, 180, 75, 50) # RGBA

        # --- Entity Definitions ---
        self.TOWER_TYPES = {
            "GUN": {
                "cost": 100, "range": 80, "cooldown": 10, "damage": 10,
                "color": (255, 215, 0), "proj_speed": 8
            },
            "SLOW": {
                "cost": 75, "range": 60, "cooldown": 30, "damage": 2,
                "color": (0, 191, 255), "proj_speed": 6, "slow_effect": 0.5, "slow_duration": 60
            }
        }
        self.TOWER_TYPE_NAMES = list(self.TOWER_TYPES.keys())

        # --- Game Layout ---
        self.path_waypoints = [
            (-20, 100), (100, 100), (100, 300), (300, 300), (300, 100),
            (500, 100), (500, 300), (self.screen_width + 20, 300)
        ]
        self.base_pos = (self.screen_width - 40, 300)
        self.tower_spots = [
            (100, 200), (200, 300), (200, 100),
            (400, 100), (400, 300), (500, 200)
        ]
        # Pre-calculate navigation for cursor
        self.spot_navigation = self._calculate_spot_navigation()

        self.reset()
        # self.validate_implementation() # Uncomment for testing

    def _calculate_spot_navigation(self):
        nav = {i: {"up": i, "down": i, "left": i, "right": i} for i in range(len(self.tower_spots))}
        for i, pos1 in enumerate(self.tower_spots):
            # Find closest in each direction
            dists = {"up": float('inf'), "down": float('inf'), "left": float('inf'), "right": float('inf')}
            for j, pos2 in enumerate(self.tower_spots):
                if i == j: continue
                dx, dy = pos2[0] - pos1[0], pos2[1] - pos1[1]
                dist = math.hypot(dx, dy)
                angle = math.atan2(-dy, dx) # -dy because pygame y is inverted
                
                if -math.pi/4 < angle <= math.pi/4 and dist < dists["right"]: # Right
                    dists["right"] = dist; nav[i]["right"] = j
                elif math.pi/4 < angle <= 3*math.pi/4 and dist < dists["up"]: # Up
                    dists["up"] = dist; nav[i]["up"] = j
                elif 3*math.pi/4 < angle or angle <= -3*math.pi/4 and dist < dists["left"]: # Left
                    dists["left"] = dist; nav[i]["left"] = j
                elif -3*math.pi/4 < angle <= -math.pi/4 and dist < dists["down"]: # Down
                    dists["down"] = dist; nav[i]["down"] = j
        return nav

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.last_terminal_reward = 0

        self.base_health = 100
        self.money = 150 # Start with enough for one tower
        self.current_wave = 0
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.game_phase = "intermission" # Start in placement phase
        self.phase_timer = self.INTERMISSION_TIME
        self.enemies_to_spawn = []
        
        self.cursor_index = 0
        self.selected_tower_type_idx = 0
        
        self.space_was_held = False
        self.shift_was_held = False

        self._start_next_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), self.last_terminal_reward, True, False, self._get_info()
            
        reward = -0.01  # Time penalty
        self.steps += 1
        
        # --- 1. Handle Input ---
        reward += self._handle_input(action)

        # --- 2. Update Game Logic ---
        self.phase_timer -= 1
        
        if self.game_phase == "wave":
            if not self.enemies and not self.enemies_to_spawn:
                # Wave cleared
                reward += 10
                self.score += 100
                if self.current_wave >= self.MAX_WAVES:
                    self.win = True
                else:
                    self.game_phase = "intermission"
                    self.phase_timer = self.INTERMISSION_TIME
            else:
                if self.phase_timer <= 0:
                    self._spawn_enemy()
                    self.phase_timer = max(10, 30 - self.current_wave) # Spawn faster on later waves

        elif self.game_phase == "intermission":
            if self.phase_timer <= 0:
                self._start_next_wave()
                self.game_phase = "wave"

        reward += self._update_towers()
        reward += self._update_projectiles()
        reward += self._update_enemies()
        self._update_particles()

        # --- 3. Check Termination ---
        terminated = False
        if self.base_health <= 0:
            self.game_over = True
            terminated = True
            self.last_terminal_reward = -100
            reward += self.last_terminal_reward
            self.score -= 1000
        elif self.win:
            self.game_over = True
            terminated = True
            self.last_terminal_reward = 100
            reward += self.last_terminal_reward
            self.score += 5000
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True
            self.last_terminal_reward = -50
            reward += self.last_terminal_reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # Movement: Navigate cursor
        if movement > 0:
            if movement == 1: self.cursor_index = self.spot_navigation[self.cursor_index]["up"]
            elif movement == 2: self.cursor_index = self.spot_navigation[self.cursor_index]["down"]
            elif movement == 3: self.cursor_index = self.spot_navigation[self.cursor_index]["left"]
            elif movement == 4: self.cursor_index = self.spot_navigation[self.cursor_index]["right"]

        # Shift (Cycle Tower Type) - on press
        if shift_held and not self.shift_was_held:
            self.selected_tower_type_idx = (self.selected_tower_type_idx + 1) % len(self.TOWER_TYPES)
        
        # Space (Build Tower) - on press
        if space_held and not self.space_was_held:
            spot_pos = self.tower_spots[self.cursor_index]
            is_occupied = any(t['pos'] == spot_pos for t in self.towers)
            
            tower_name = self.TOWER_TYPE_NAMES[self.selected_tower_type_idx]
            tower_cost = self.TOWER_TYPES[tower_name]["cost"]

            if not is_occupied and self.money >= tower_cost:
                self.money -= tower_cost
                self.towers.append({
                    "pos": spot_pos, "type": tower_name, "cooldown": 0, "target": None
                })
                # sfx: build_tower.wav
                for _ in range(20):
                    self._create_particle(spot_pos, color=(255,255,255), lifespan=15, size=self.np_random.integers(1,4))

        self.space_was_held = space_held
        self.shift_was_held = shift_held
        return reward

    def _start_next_wave(self):
        self.current_wave += 1
        if self.current_wave > self.MAX_WAVES: return

        num_enemies = 2 + self.current_wave * 2
        base_health = 10 + self.current_wave * 5
        base_speed = 1.0 + self.current_wave * 0.1

        for i in range(num_enemies):
            health = int(base_health * self.np_random.uniform(0.9, 1.1))
            speed = base_speed * self.np_random.uniform(0.9, 1.1)
            self.enemies_to_spawn.append({'health': health, 'speed': speed})
        
        self.phase_timer = 30 # Time until first enemy of wave spawns

    def _spawn_enemy(self):
        if not self.enemies_to_spawn: return
        
        spec = self.enemies_to_spawn.pop(0)
        self.enemies.append({
            "pos": list(self.path_waypoints[0]),
            "max_health": spec['health'],
            "health": spec['health'],
            "speed": spec['speed'],
            "path_index": 1,
            "dist_on_segment": 0,
            "slow_timer": 0
        })

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            # Apply slow effect
            current_speed = enemy['speed']
            if enemy['slow_timer'] > 0:
                enemy['slow_timer'] -= 1
                current_speed *= 0.5
            
            # Move along path
            if enemy['path_index'] >= len(self.path_waypoints):
                self.base_health -= enemy['health'] / 2 # Enemy reached base
                self._create_particle(self.base_pos, count=30, color=(255,0,0), lifespan=30, size=3)
                self.enemies.remove(enemy)
                reward -= 5 # Penalty for letting an enemy through
                # sfx: base_damage.wav
                continue
            
            p1 = self.path_waypoints[enemy['path_index'] - 1]
            p2 = self.path_waypoints[enemy['path_index']]
            segment_vec = (p2[0] - p1[0], p2[1] - p1[1])
            segment_len = math.hypot(*segment_vec)
            
            if segment_len > 0:
                enemy['dist_on_segment'] += current_speed
                progress = enemy['dist_on_segment'] / segment_len
                
                if progress >= 1.0:
                    enemy['path_index'] += 1
                    enemy['dist_on_segment'] = 0
                else:
                    enemy['pos'][0] = p1[0] + segment_vec[0] * progress
                    enemy['pos'][1] = p1[1] + segment_vec[1] * progress
        return reward

    def _update_towers(self):
        reward = 0
        for tower in self.towers:
            if tower['cooldown'] > 0:
                tower['cooldown'] -= 1
                continue

            tower_spec = self.TOWER_TYPES[tower['type']]
            
            # Find target
            possible_targets = []
            for enemy in self.enemies:
                dist = math.hypot(enemy['pos'][0] - tower['pos'][0], enemy['pos'][1] - tower['pos'][1])
                if dist <= tower_spec['range']:
                    possible_targets.append(enemy)
            
            if possible_targets:
                # Target enemy furthest along the path
                target = max(possible_targets, key=lambda e: (e['path_index'], e['dist_on_segment']))
                tower['target'] = target
                tower['cooldown'] = tower_spec['cooldown']
                
                self.projectiles.append({
                    "pos": list(tower['pos']),
                    "type": tower['type'],
                    "target": target,
                    "speed": tower_spec['proj_speed'],
                    "damage": tower_spec['damage']
                })
                # sfx: shoot.wav
        return reward

    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            if proj['target'] not in self.enemies:
                self.projectiles.remove(proj)
                continue

            target_pos = proj['target']['pos']
            proj_pos = proj['pos']
            
            # Move towards target
            angle = math.atan2(target_pos[1] - proj_pos[1], target_pos[0] - proj_pos[0])
            proj_pos[0] += math.cos(angle) * proj['speed']
            proj_pos[1] += math.sin(angle) * proj['speed']
            
            # Check collision
            if math.hypot(proj_pos[0] - target_pos[0], proj_pos[1] - target_pos[1]) < 8:
                enemy = proj['target']
                enemy['health'] -= proj['damage']
                reward += 0.1 # Hit reward
                
                # Apply slow effect if applicable
                if proj['type'] == 'SLOW':
                    tower_spec = self.TOWER_TYPES['SLOW']
                    enemy['slow_timer'] = tower_spec['slow_duration']

                self._create_particle(proj_pos, count=5, color=(200,200,200), lifespan=10)
                
                if enemy['health'] <= 0:
                    self.money += int(enemy['max_health'] / 4)
                    self.score += int(enemy['max_health'])
                    reward += 1 # Kill reward
                    self._create_particle(enemy['pos'], count=25, color=(255, 80, 80), lifespan=20, size=2)
                    self.enemies.remove(enemy)
                    # sfx: enemy_die.wav
                
                self.projectiles.remove(proj)
        return reward

    def _create_particle(self, pos, count=1, color=(255,255,255), lifespan=10, size=1, speed_mult=2):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({'pos': list(pos), 'vel': vel, 'lifespan': lifespan, 'color': color, 'size': size})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95
            p['vel'][1] *= 0.95
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.current_wave, "money": self.money, "base_health": self.base_health}
    
    def _draw_text(self, text, pos, font, color, center=False):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    def _render_game(self):
        # Path
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_waypoints, 30)

        # Base
        base_rect = pygame.Rect(0, 0, 40, 40)
        base_rect.center = self.base_pos
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect, border_radius=5)
        # Base Health Bar
        if self.base_health > 0:
            health_pct = self.base_health / 100
            bar_w = 40
            bar_h = 5
            pygame.draw.rect(self.screen, (255,0,0), (self.base_pos[0]-bar_w/2, self.base_pos[1]-30, bar_w, bar_h))
            pygame.draw.rect(self.screen, self.COLOR_BASE, (self.base_pos[0]-bar_w/2, self.base_pos[1]-30, bar_w * health_pct, bar_h))

        # Tower Placement Spots & Cursor
        for i, pos in enumerate(self.tower_spots):
            is_occupied = any(t['pos'] == pos for t in self.towers)
            if not is_occupied:
                pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), 15, self.COLOR_PLACEMENT_SPOT)
                pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 15, self.COLOR_PLACEMENT_SPOT)
        
        # Cursor
        cursor_pos = self.tower_spots[self.cursor_index]
        pulse = abs(math.sin(self.steps * 0.1)) * 5
        pygame.draw.circle(self.screen, self.COLOR_CURSOR, cursor_pos, 18 + pulse, 2)

        # Towers
        for tower in self.towers:
            spec = self.TOWER_TYPES[tower['type']]
            pos = tower['pos']
            p1 = (pos[0], pos[1] - 12)
            p2 = (pos[0] - 10, pos[1] + 6)
            p3 = (pos[0] + 10, pos[1] + 6)
            pygame.gfxdraw.aatrigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), spec['color'])
            pygame.gfxdraw.filled_trigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), spec['color'])

        # Projectiles
        for proj in self.projectiles:
            color = self.TOWER_TYPES[proj['type']]['color']
            pygame.draw.circle(self.screen, color, proj['pos'], 4)
            pygame.draw.circle(self.screen, (255,255,255), proj['pos'], 2)

        # Enemies
        for enemy in self.enemies:
            pos = (int(enemy['pos'][0]), int(enemy['pos'][1]))
            size = 8
            color = (220, 50, 50) if enemy['slow_timer'] == 0 else (150, 50, 200)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, color)
            # Health bar
            health_pct = max(0, enemy['health'] / enemy['max_health'])
            bar_w = 16
            bar_h = 3
            pygame.draw.rect(self.screen, (255,0,0), (pos[0]-bar_w/2, pos[1]-15, bar_w, bar_h))
            pygame.draw.rect(self.screen, (0,255,0), (pos[0]-bar_w/2, pos[1]-15, bar_w * health_pct, bar_h))

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 20))
            if alpha > 0:
                color = (*p['color'], alpha)
                surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
                pygame.draw.circle(surf, color, (p['size'], p['size']), p['size'])
                self.screen.blit(surf, (p['pos'][0] - p['size'], p['pos'][1] - p['size']))

    def _render_ui(self):
        # UI Background
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (0, 0, self.screen_width, 40))
        
        # Wave Info
        self._draw_text(f"Wave: {self.current_wave}/{self.MAX_WAVES}", (10, 10), self.font_m, self.COLOR_TEXT)
        
        # Money
        self._draw_text(f"$ {self.money}", (self.screen_width - 100, 10), self.font_m, (255, 215, 0))

        # Game Phase / Timer
        if self.game_phase == "intermission":
            time_left = f"{self.phase_timer/30:.1f}s"
            self._draw_text(f"Next wave in: {time_left}", (self.screen_width/2, 10), self.font_m, self.COLOR_TEXT, center=True)
        elif self.game_over:
            msg = "VICTORY!" if self.win else "GAME OVER"
            color = (0, 255, 0) if self.win else (255, 0, 0)
            self._draw_text(msg, (self.screen_width/2, self.screen_height/2), self.font_l, color, center=True)
        
        # Selected Tower UI
        tower_name = self.TOWER_TYPE_NAMES[self.selected_tower_type_idx]
        tower_spec = self.TOWER_TYPES[tower_name]
        cost_color = (255, 215, 0) if self.money >= tower_spec['cost'] else (255, 50, 50)
        self._draw_text(f"Build: {tower_name}", (150, 5), self.font_s, self.COLOR_TEXT)
        self._draw_text(f"Cost: ${tower_spec['cost']}", (150, 20), self.font_s, cost_color)
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("✓ Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # For human play
    import sys

    env = GameEnv(render_mode="rgb_array")
    env.reset()
    
    # Set up Pygame window for display
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0
    
    # --- Key mapping for human play ---
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    while running:
        # --- Action gathering ---
        movement = 0
        space = 0
        shift = 0

        keys = pygame.key.get_pressed()
        for key, move_action in key_map.items():
            if keys[key]:
                movement = move_action
                break # Prioritize one movement key
        
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
        
        action = [movement, space, shift]

        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset on 'r'
                    total_reward = 0
                    env.reset()
                if event.key == pygame.K_ESCAPE:
                    running = False

        # --- Step environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Game auto-pauses on termination in the env logic, so we just wait for reset/quit
        
        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()
    sys.exit()