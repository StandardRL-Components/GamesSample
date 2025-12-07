
# Generated: 2025-08-28T06:21:44.409615
# Source Brief: brief_05878.md
# Brief Index: 5878

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to place a tower. Press Shift to cycle tower types."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of enemies by strategically placing towers along the path."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 5000 # Increased for longer games
    WIN_WAVE = 10

    # Colors
    COLOR_BG = (20, 25, 30)
    COLOR_PATH = (40, 50, 60)
    COLOR_BASE = (0, 150, 50)
    COLOR_BASE_DAMAGED = (200, 150, 0)
    COLOR_ENEMY = (200, 50, 50)
    COLOR_ENEMY_HEALTH = (50, 200, 50)
    COLOR_TEXT = (220, 220, 220)
    COLOR_GOLD = (255, 200, 0)
    COLOR_CURSOR = (255, 255, 255, 100)
    COLOR_CURSOR_INVALID = (255, 0, 0, 100)
    
    TOWER_SPECS = [
        {
            "name": "Gun Turret",
            "cost": 100,
            "range": 80,
            "damage": 5,
            "fire_rate": 10, # steps per shot
            "color": (0, 150, 255),
            "proj_speed": 10,
            "proj_color": (100, 200, 255)
        },
        {
            "name": "Cannon",
            "cost": 250,
            "range": 120,
            "damage": 25,
            "fire_rate": 40,
            "color": (255, 150, 0),
            "proj_speed": 7,
            "proj_color": (255, 180, 50)
        }
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("sans-serif", 18)
        self.font_large = pygame.font.SysFont("sans-serif", 24)
        
        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 0
        self.max_base_health = 100
        self.gold = 0
        self.wave_number = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.path = []
        self.tower_spots = []
        self.cursor_pos = [0, 0]
        self.selected_tower_type = 0
        self.inter_wave_timer = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.screen_shake = 0
        
        self.reset()
        
        # Run validation check
        # self.validate_implementation() # Comment out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.base_health = self.max_base_health
        self.gold = 250
        self.wave_number = 0
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self._generate_map()
        
        self.cursor_pos = [self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2]
        self.selected_tower_type = 0
        
        self.inter_wave_timer = 150 # Time before first wave
        self.last_space_held = False
        self.last_shift_held = False
        self.screen_shake = 0

        return self._get_observation(), self._get_info()

    def _generate_map(self):
        self.path = []
        padding = 50
        y = self.np_random.integers(padding, self.SCREEN_HEIGHT - padding)
        self.path.append((0, y))
        
        x = 0
        num_segments = 5
        for i in range(num_segments):
            x = int((i + 1) * (self.SCREEN_WIDTH / (num_segments + 1)))
            y = self.np_random.integers(padding, self.SCREEN_HEIGHT - padding)
            self.path.append((x, y))
        
        self.path.append((self.SCREEN_WIDTH, y))
        self.base_pos = (self.SCREEN_WIDTH - 20, y)

        # Generate tower spots
        self.tower_spots = []
        for _ in range(15):
            spot_x = self.np_random.integers(padding, self.SCREEN_WIDTH - padding)
            spot_y = self.np_random.integers(padding, self.SCREEN_HEIGHT - padding)
            
            # Ensure spot is not too close to the path
            min_dist = float('inf')
            for i in range(len(self.path) - 1):
                p1 = np.array(self.path[i])
                p2 = np.array(self.path[i+1])
                p = np.array([spot_x, spot_y])
                
                l2 = np.sum((p1-p2)**2)
                if l2 == 0:
                    dist = np.linalg.norm(p - p1)
                else:
                    t = max(0, min(1, np.dot(p - p1, p2 - p1) / l2))
                    projection = p1 + t * (p2 - p1)
                    dist = np.linalg.norm(p - projection)
                min_dist = min(min_dist, dist)

            if min_dist > 30: # Minimum distance from path
                self.tower_spots.append((spot_x, spot_y))

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.01  # Cost of living
        self.steps += 1
        
        # --- Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_input(movement, space_held, shift_held)
        
        # --- Game Logic ---
        reward += self._update_towers()
        reward += self._update_projectiles()
        wave_cleared_reward = self._update_enemies()
        reward += wave_cleared_reward

        # --- Wave Management ---
        if not self.enemies and self.wave_number > 0 and self.wave_number <= self.WIN_WAVE:
            if self.inter_wave_timer > 0:
                self.inter_wave_timer -= 1
            else: # Start next wave
                if wave_cleared_reward > 0: # Only if we just cleared a wave
                    self.wave_number += 1
                    if self.wave_number <= self.WIN_WAVE:
                        self._spawn_wave()
                        self.inter_wave_timer = 150 # Reset for next inter-wave period
        elif self.wave_number == 0 and self.inter_wave_timer > 0:
            self.inter_wave_timer -= 1
            if self.inter_wave_timer == 0:
                self.wave_number = 1
                self._spawn_wave()

        self._update_particles()
        if self.screen_shake > 0:
            self.screen_shake -= 1

        # --- Termination ---
        terminated = False
        if self.base_health <= 0:
            reward -= 100
            terminated = True
            self.game_over = True
        elif self.wave_number > self.WIN_WAVE and not self.enemies:
            reward += 100
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _handle_input(self, movement, space_held, shift_held):
        # Cursor movement
        cursor_speed = 10
        if movement == 1: self.cursor_pos[1] -= cursor_speed
        elif movement == 2: self.cursor_pos[1] += cursor_speed
        elif movement == 3: self.cursor_pos[0] -= cursor_speed
        elif movement == 4: self.cursor_pos[0] += cursor_speed
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.SCREEN_WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.SCREEN_HEIGHT)
        
        # Cycle tower type (on key press)
        if shift_held and not self.last_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
            # sfx: ui_switch
        
        # Place tower (on key press)
        if space_held and not self.last_space_held:
            spec = self.TOWER_SPECS[self.selected_tower_type]
            if self.gold >= spec["cost"]:
                # Find the closest valid tower spot
                closest_spot, min_dist = None, float('inf')
                for spot in self.tower_spots:
                    dist = math.hypot(spot[0] - self.cursor_pos[0], spot[1] - self.cursor_pos[1])
                    if dist < min_dist and dist < 20: # Must be close to a spot
                        # Check if spot is already occupied
                        is_occupied = False
                        for tower in self.towers:
                            if tower['pos'] == spot:
                                is_occupied = True
                                break
                        if not is_occupied:
                            closest_spot = spot
                            min_dist = dist
                
                if closest_spot:
                    self.gold -= spec["cost"]
                    self.towers.append({
                        "pos": closest_spot,
                        "type": self.selected_tower_type,
                        "cooldown": 0,
                        "target": None
                    })
                    # sfx: place_tower
                    self._create_particles(closest_spot, 20, self.COLOR_GOLD)
        
        self.last_space_held = space_held
        self.last_shift_held = shift_held

    def _spawn_wave(self):
        num_enemies = 3 + self.wave_number * 2
        base_health = 20 + self.wave_number * 5
        base_speed = 1.0 + self.wave_number * 0.1
        base_value = 10 + self.wave_number * 2

        for i in range(num_enemies):
            spawn_offset = -i * 25
            self.enemies.append({
                "pos": [spawn_offset, self.path[0][1]],
                "health": base_health,
                "max_health": base_health,
                "speed": base_speed * self.np_random.uniform(0.9, 1.1),
                "path_index": 0,
                "value": base_value,
                "id": self.np_random.integers(1, 1_000_000)
            })
        # sfx: wave_start

    def _update_towers(self):
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower['type']]
            if tower['cooldown'] > 0:
                tower['cooldown'] -= 1
                continue

            # Find a new target if current is dead or out of range
            if tower['target']:
                if tower['target'] not in self.enemies or \
                   math.hypot(tower['pos'][0] - tower['target']['pos'][0], tower['pos'][1] - tower['target']['pos'][1]) > spec['range']:
                    tower['target'] = None

            if not tower['target']:
                # Find closest enemy in range
                closest_enemy, min_dist = None, spec['range']
                for enemy in self.enemies:
                    dist = math.hypot(tower['pos'][0] - enemy['pos'][0], tower['pos'][1] - enemy['pos'][1])
                    if dist <= min_dist:
                        min_dist = dist
                        closest_enemy = enemy
                tower['target'] = closest_enemy
            
            # Fire if target is available
            if tower['target']:
                self.projectiles.append({
                    "pos": list(tower['pos']),
                    "target_id": tower['target']['id'],
                    "damage": spec['damage'],
                    "speed": spec['proj_speed'],
                    "color": spec['proj_color']
                })
                tower['cooldown'] = spec['fire_rate']
                # sfx: tower_shoot
                self._create_particles(tower['pos'], 5, spec['color'], 1, 3) # Muzzle flash
        return 0

    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            target_enemy = next((e for e in self.enemies if e['id'] == proj['target_id']), None)
            
            if not target_enemy:
                self.projectiles.remove(proj)
                continue

            # Move towards target
            target_pos = target_enemy['pos']
            direction = math.atan2(target_pos[1] - proj['pos'][1], target_pos[0] - proj['pos'][0])
            proj['pos'][0] += math.cos(direction) * proj['speed']
            proj['pos'][1] += math.sin(direction) * proj['speed']

            # Check for hit
            if math.hypot(proj['pos'][0] - target_pos[0], proj['pos'][1] - target_pos[1]) < 10:
                target_enemy['health'] -= proj['damage']
                reward += 0.1
                # sfx: enemy_hit
                self._create_particles(proj['pos'], 10, proj['color'])
                self.projectiles.remove(proj)
        return reward

    def _update_enemies(self):
        wave_cleared_reward = 0
        for enemy in self.enemies[:]:
            # Check for death
            if enemy['health'] <= 0:
                self.gold += enemy['value']
                wave_cleared_reward += 1.0
                # sfx: enemy_die
                self._create_particles(enemy['pos'], 30, self.COLOR_ENEMY, 2, 5)
                self.enemies.remove(enemy)
                continue

            # Move along path
            if enemy['path_index'] < len(self.path) - 1:
                target_pos = self.path[enemy['path_index'] + 1]
                direction = math.atan2(target_pos[1] - enemy['pos'][1], target_pos[0] - enemy['pos'][0])
                enemy['pos'][0] += math.cos(direction) * enemy['speed']
                enemy['pos'][1] += math.sin(direction) * enemy['speed']

                if math.hypot(enemy['pos'][0] - target_pos[0], enemy['pos'][1] - target_pos[1]) < enemy['speed']:
                    enemy['path_index'] += 1
            else: # Reached base
                self.base_health -= 10
                self.screen_shake = 10
                # sfx: base_damage
                self._create_particles(self.base_pos, 50, self.COLOR_BASE_DAMAGED, 3, 7)
                self.enemies.remove(enemy)

        # Check if wave is cleared
        if not self.enemies and self.wave_number > 0 and self.wave_number <= self.WIN_WAVE:
             if self.inter_wave_timer > 0: # Ensure we don't give reward every step of inter-wave
                wave_cleared_reward += 10
                self.gold += 50 + self.wave_number * 10
                # sfx: wave_clear

        return wave_cleared_reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _create_particles(self, pos, count, color, min_speed=1, max_speed=4):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(min_speed, max_speed)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "lifespan": self.np_random.integers(10, 30),
                "color": color,
                "size": self.np_random.integers(1, 4)
            })

    def _get_observation(self):
        # Apply screen shake
        render_offset_x = 0
        render_offset_y = 0
        if self.screen_shake > 0:
            render_offset_x = self.np_random.integers(-5, 6)
            render_offset_y = self.np_random.integers(-5, 6)
        
        # Create a temporary surface for rendering to apply the shake
        temp_surf = self.screen.copy()
        temp_surf.fill(self.COLOR_BG)
        
        self._render_game(temp_surf)
        self._render_ui(temp_surf)

        self.screen.fill(self.COLOR_BG)
        self.screen.blit(temp_surf, (render_offset_x, render_offset_y))

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self, surface):
        # Draw path
        for i in range(len(self.path) - 1):
            pygame.draw.line(surface, self.COLOR_PATH, self.path[i], self.path[i+1], 20)
        
        # Draw tower spots
        for spot in self.tower_spots:
            is_occupied = any(tower['pos'] == spot for tower in self.towers)
            color = (100, 100, 100, 50) if is_occupied else (255, 255, 255, 50)
            pygame.gfxdraw.aacircle(surface, int(spot[0]), int(spot[1]), 10, color)

        # Draw base
        base_rect = pygame.Rect(self.base_pos[0]-15, self.base_pos[1]-15, 30, 30)
        pygame.draw.rect(surface, self.COLOR_BASE, base_rect, border_radius=3)
        
        # Draw towers and ranges
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower['type']]
            pos = (int(tower['pos'][0]), int(tower['pos'][1]))
            pygame.draw.circle(surface, spec['color'], pos, 12)
            pygame.draw.circle(surface, (0,0,0), pos, 8)
            # Draw a small shape to differentiate
            if tower['type'] == 1:
                 pygame.draw.rect(surface, spec['color'], (pos[0]-3, pos[1]-3, 6, 6))

        # Draw projectiles
        for proj in self.projectiles:
            pos = (int(proj['pos'][0]), int(proj['pos'][1]))
            pygame.draw.circle(surface, proj['color'], pos, 4)
            pygame.gfxdraw.aacircle(surface, pos[0], pos[1], 4, (255,255,255,150))

        # Draw enemies
        for enemy in self.enemies:
            pos = (int(enemy['pos'][0]), int(enemy['pos'][1]))
            pygame.draw.circle(surface, self.COLOR_ENEMY, pos, 8)
            # Health bar
            health_ratio = enemy['health'] / enemy['max_health']
            bar_width = 16
            pygame.draw.rect(surface, (100,0,0), (pos[0] - bar_width/2, pos[1] - 15, bar_width, 3))
            pygame.draw.rect(surface, self.COLOR_ENEMY_HEALTH, (pos[0] - bar_width/2, pos[1] - 15, bar_width * health_ratio, 3))
            
        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifespan'] / 20.0))))
            color = (*p['color'], alpha)
            temp_p_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.rect(temp_p_surf, color, (0,0, p['size'], p['size']))
            surface.blit(temp_p_surf, (int(p['pos'][0] - p['size']/2), int(p['pos'][1] - p['size']/2)), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self, surface):
        # Cursor and range indicator
        spec = self.TOWER_SPECS[self.selected_tower_type]
        cursor_color = self.COLOR_CURSOR
        if self.gold < spec['cost']:
            cursor_color = self.COLOR_CURSOR_INVALID
        
        pygame.gfxdraw.aacircle(surface, int(self.cursor_pos[0]), int(self.cursor_pos[1]), spec['range'], cursor_color)
        pygame.gfxdraw.filled_circle(surface, int(self.cursor_pos[0]), int(self.cursor_pos[1]), spec['range'], cursor_color)
        pygame.draw.circle(surface, (255,255,255), (int(self.cursor_pos[0]), int(self.cursor_pos[1])), 3)

        # Top-left: Wave info
        wave_text = f"Wave: {self.wave_number}/{self.WIN_WAVE}"
        if self.wave_number == 0 or (not self.enemies and self.wave_number <= self.WIN_WAVE):
            wave_text += f" (Next in {self.inter_wave_timer // 30}s)"
        
        text_surf = self.font_large.render(wave_text, True, self.COLOR_TEXT)
        surface.blit(text_surf, (10, 10))
        
        # Top-right: Base Health
        health_text = self.font_large.render(f"Base Health: {max(0, self.base_health)}", True, self.COLOR_TEXT)
        surface.blit(health_text, (self.SCREEN_WIDTH - health_text.get_width() - 10, 10))
        # Health bar
        bar_w, bar_h = 150, 15
        health_ratio = max(0, self.base_health / self.max_base_health)
        pygame.draw.rect(surface, (100,0,0), (self.SCREEN_WIDTH - bar_w - 10, 40, bar_w, bar_h))
        pygame.draw.rect(surface, self.COLOR_BASE, (self.SCREEN_WIDTH - bar_w - 10, 40, bar_w * health_ratio, bar_h))

        # Bottom-left: Gold
        gold_text = self.font_large.render(f"Gold: {self.gold}", True, self.COLOR_GOLD)
        surface.blit(gold_text, (10, self.SCREEN_HEIGHT - gold_text.get_height() - 10))

        # Bottom-right: Tower Selection
        spec = self.TOWER_SPECS[self.selected_tower_type]
        tower_name = self.font_small.render(f"{spec['name']}", True, self.COLOR_TEXT)
        tower_cost = self.font_small.render(f"Cost: {spec['cost']}", True, self.COLOR_GOLD)
        surface.blit(tower_name, (self.SCREEN_WIDTH - tower_name.get_width() - 10, self.SCREEN_HEIGHT - 50))
        surface.blit(tower_cost, (self.SCREEN_WIDTH - tower_cost.get_width() - 10, self.SCREEN_HEIGHT - 30))

        # Game Over / Win Text
        if self.game_over:
            msg = "GAME OVER"
            color = self.COLOR_ENEMY
            if self.wave_number > self.WIN_WAVE:
                msg = "YOU WIN!"
                color = self.COLOR_GOLD
            
            end_text = self.font_large.render(msg, True, color)
            pos = (self.SCREEN_WIDTH/2 - end_text.get_width()/2, self.SCREEN_HEIGHT/2 - end_text.get_height()/2)
            surface.blit(end_text, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gold": self.gold,
            "wave": self.wave_number,
            "base_health": self.base_health,
            "enemies_left": len(self.enemies)
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    # Set up Pygame window for human play
    pygame.display.set_caption("Tower Defense")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Render to the display ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # Wait for 'R' to reset
            waiting_for_reset = True
            while waiting_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting_for_reset = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        waiting_for_reset = False
                clock.tick(30)

        clock.tick(30) # Limit frame rate for human play

    env.close()