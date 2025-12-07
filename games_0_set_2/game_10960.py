import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:13:27.337578
# Source Brief: brief_00960.md
# Brief Index: 960
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your core from incoming fractal enemies. Clone their unique shapes and strategically deploy them as defensive barriers."
    )
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to clone an enemy's shape and shift to deploy a cloned defense at the cursor's location."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    
    # Colors
    COLOR_BG = (10, 15, 25)
    COLOR_ENEMY = (255, 50, 80)
    COLOR_PLAYER = (60, 150, 255)
    COLOR_CORE = (255, 255, 255)
    COLOR_ENERGY = (50, 255, 150)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_CURSOR = (255, 200, 0)

    # Game Parameters
    CORE_POS = np.array([WIDTH / 2, HEIGHT / 2])
    CORE_RADIUS = 15
    INITIAL_CORE_HEALTH = 100.0
    
    INITIAL_CLONE_CHARGES = 5
    INITIAL_TELEPORT_CHARGES = 3
    CHARGE_REGEN_RATE = 120 # frames per charge
    
    INITIAL_FRACTAL_SPAWN_RATE = 0.015
    FRACTAL_SPAWN_RATE_INCREASE = 0.00005
    INITIAL_FRACTAL_SPEED = 0.8
    FRACTAL_SPEED_INCREASE_INTERVAL = 100
    FRACTAL_SPEED_INCREASE_AMOUNT = 0.05
    
    ENERGY_SPAWN_RATE = 0.01
    ENERGY_PICKUP_VALUE = 50
    UPGRADE_CLONE_EFFICIENCY_COST = 250
    UPGRADE_TELEPORT_RANGE_COST = 500
    
    CURSOR_SPEED = 6.0
    INITIAL_TELEPORT_RANGE = 150.0
    
    MAX_EPISODE_STEPS = 1000

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
        self.font_ui = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_msg = pygame.font.SysFont("sans-serif", 24, bold=True)
        
        self.render_mode = render_mode
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # This will be initialized in reset
        self.np_random = None

    def _initialize_state_variables(self):
        self.core_health = self.INITIAL_CORE_HEALTH
        self.fractals = []
        self.defenses = []
        self.cloned_segments_pending = []
        self.energy_pickups = []
        self.particles = []
        
        self.clone_charges = self.INITIAL_CLONE_CHARGES
        self.teleport_charges = self.INITIAL_TELEPORT_CHARGES
        self.charge_regen_timer = 0
        
        self.energy = 0
        self.upgrades = {"clone_efficiency": False, "teleport_range": False}
        
        self.fractal_spawn_rate = self.INITIAL_FRACTAL_SPAWN_RATE
        self.fractal_speed = self.INITIAL_FRACTAL_SPEED
        
        self.teleport_target_pos = self.CORE_POS.copy()
        self.teleport_range = self.INITIAL_TELEPORT_RANGE

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self._initialize_state_variables()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        
        # 1. Handle player actions
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        self._handle_movement(movement)
        if space_pressed: reward += self._handle_clone()
        if shift_pressed: reward += self._handle_teleport()

        # 2. Update game state
        self._update_difficulty()
        self._update_charges()
        self._spawn_entities()
        
        self._update_fractals()
        self._update_particles()
        
        # 3. Handle interactions and collect rewards
        reward += self._handle_collisions()
        reward += self._handle_energy_collection()
        reward += self._check_for_upgrades()

        # 4. Check for termination
        terminated = self.core_health <= 0 or self.steps >= self.MAX_EPISODE_STEPS
        truncated = False # Not using truncation based on time limit
        if terminated:
            self.game_over = True
            if self.steps >= self.MAX_EPISODE_STEPS:
                reward += 100  # Survival bonus
            if self.core_health <= 0:
                reward -= 100  # Core destroyed penalty
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    # --- Action Handling ---
    def _handle_movement(self, movement):
        direction = np.array([0.0, 0.0])
        if movement == 1: direction[1] = -1 # Up
        elif movement == 2: direction[1] = 1  # Down
        elif movement == 3: direction[0] = -1 # Left
        elif movement == 4: direction[0] = 1  # Right
        
        self.teleport_target_pos += direction * self.CURSOR_SPEED
        
        # Clamp cursor to teleport range
        to_cursor = self.teleport_target_pos - self.CORE_POS
        dist = np.linalg.norm(to_cursor)
        if dist > self.teleport_range:
            self.teleport_target_pos = self.CORE_POS + (to_cursor / dist) * self.teleport_range

    def _handle_clone(self):
        if self.clone_charges >= 1:
            self.clone_charges -= 1
            # sfx: clone_created.wav
            num_to_clone = 2 if self.upgrades["clone_efficiency"] else 1
            for _ in range(num_to_clone):
                if self.fractals:
                    # Clone the shape of the oldest fractal
                    template = self.fractals[0]
                    shape = template['shape']
                    radius = template['radius']
                else:
                    # Default shape if no fractals exist
                    shape, radius = self._generate_fractal_shape(level=1, angle=0, length=10)
                
                self.cloned_segments_pending.append({'shape': shape, 'radius': radius})
            return 0.01 # Small reward for taking a valid action
        return 0

    def _handle_teleport(self):
        if self.teleport_charges >= 1 and self.cloned_segments_pending:
            self.teleport_charges -= 1
            segment = self.cloned_segments_pending.pop(0)
            self.defenses.append({
                'pos': self.teleport_target_pos.copy(),
                'shape': segment['shape'],
                'radius': segment['radius'],
                'angle': 0,
                'health': 100 # Defenses can take a hit
            })
            # sfx: teleport.wav
            return 0.01 # Small reward for deploying a defense
        return 0

    # --- Game State Updates ---
    def _update_difficulty(self):
        self.fractal_spawn_rate = self.INITIAL_FRACTAL_SPAWN_RATE + self.steps * self.FRACTAL_SPAWN_RATE_INCREASE
        if self.steps > 0 and self.steps % self.FRACTAL_SPEED_INCREASE_INTERVAL == 0:
            self.fractal_speed += self.FRACTAL_SPEED_INCREASE_AMOUNT

    def _update_charges(self):
        self.charge_regen_timer += 1
        if self.charge_regen_timer >= self.CHARGE_REGEN_RATE:
            self.charge_regen_timer = 0
            if self.clone_charges < self.INITIAL_CLONE_CHARGES: self.clone_charges += 1
            if self.teleport_charges < self.INITIAL_TELEPORT_CHARGES: self.teleport_charges += 1

    def _spawn_entities(self):
        # Spawn Fractals
        if self.np_random.random() < self.fractal_spawn_rate:
            self._spawn_fractal()
        # Spawn Energy
        if self.np_random.random() < self.ENERGY_SPAWN_RATE:
            self._spawn_energy_pickup()

    def _update_fractals(self):
        for f in self.fractals:
            f['pos'] += f['vel'] * self.fractal_speed
            f['angle'] += f['rot_speed']

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    # --- Collision & Event Logic ---
    def _handle_collisions(self):
        reward = 0
        
        # Fractals vs Defenses
        destroyed_fractals = set()
        destroyed_defenses = set()
        for i, f in enumerate(self.fractals):
            for j, d in enumerate(self.defenses):
                dist = np.linalg.norm(f['pos'] - d['pos'])
                if dist < f['radius'] + d['radius']:
                    destroyed_fractals.add(i)
                    destroyed_defenses.add(j)
                    self._spawn_particles(f['pos'], self.COLOR_ENEMY, 20)
                    self._spawn_particles(d['pos'], self.COLOR_PLAYER, 10)
                    reward += 0.1 # Reward for destroying a fractal
                    # sfx: defense_hit.wav
        
        self.fractals = [f for i, f in enumerate(self.fractals) if i not in destroyed_fractals]
        self.defenses = [d for j, d in enumerate(self.defenses) if j not in destroyed_defenses]
        
        # Fractals vs Core
        remaining_fractals = []
        for f in self.fractals:
            dist = np.linalg.norm(f['pos'] - self.CORE_POS)
            if dist < f['radius'] + self.CORE_RADIUS:
                self.core_health -= 10
                self._spawn_particles(self.CORE_POS, self.COLOR_CORE, 30)
                # sfx: core_hit.wav
            else:
                remaining_fractals.append(f)
        self.fractals = remaining_fractals
        
        return reward

    def _handle_energy_collection(self):
        reward = 0
        remaining_pickups = []
        for p in self.energy_pickups:
            dist = np.linalg.norm(p['pos'] - self.teleport_target_pos)
            if dist < p['radius'] + 10: # Cursor radius is 10
                self.energy += self.ENERGY_PICKUP_VALUE
                reward += 1.0
                self._spawn_particles(p['pos'], self.COLOR_ENERGY, 15)
                # sfx: energy_pickup.wav
            else:
                remaining_pickups.append(p)
        self.energy_pickups = remaining_pickups
        return reward
    
    def _check_for_upgrades(self):
        reward = 0
        if not self.upgrades["clone_efficiency"] and self.energy >= self.UPGRADE_CLONE_EFFICIENCY_COST:
            self.upgrades["clone_efficiency"] = True
            reward += 5.0
            # sfx: upgrade_unlocked.wav
        if not self.upgrades["teleport_range"] and self.energy >= self.UPGRADE_TELEPORT_RANGE_COST:
            self.upgrades["teleport_range"] = True
            self.teleport_range *= 2
            reward += 5.0
            # sfx: upgrade_unlocked.wav
        return reward

    # --- Entity Creation ---
    def _spawn_fractal(self):
        edge = self.np_random.integers(4)
        if edge == 0: pos = np.array([self.np_random.uniform(0, self.WIDTH), -20.0]) # Top
        elif edge == 1: pos = np.array([self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 20.0]) # Bottom
        elif edge == 2: pos = np.array([-20.0, self.np_random.uniform(0, self.HEIGHT)]) # Left
        else: pos = np.array([self.WIDTH + 20.0, self.np_random.uniform(0, self.HEIGHT)]) # Right
        
        direction = (self.CORE_POS - pos) + self.np_random.uniform(-20, 20, 2)
        vel = direction / np.linalg.norm(direction)
        
        level = self.np_random.integers(2, 4)
        base_angle = math.atan2(vel[1], vel[0])
        base_length = self.np_random.uniform(15, 25)
        shape, radius = self._generate_fractal_shape(level, base_angle, base_length)
        
        self.fractals.append({
            'pos': pos, 'vel': vel, 'shape': shape, 'radius': radius,
            'angle': 0, 'rot_speed': self.np_random.uniform(-0.05, 0.05)
        })

    def _generate_fractal_shape(self, level, angle, length, branch_angle=math.pi/4, length_decay=0.7):
        points = []
        q = [(np.array([0.0, 0.0]), angle, length, level)]
        max_dist = 0
        
        while q:
            start_pos, current_angle, current_length, current_level = q.pop(0)
            if current_level == 0: continue
            
            end_pos = start_pos + np.array([math.cos(current_angle), math.sin(current_angle)]) * current_length
            points.append((start_pos, end_pos))
            max_dist = max(max_dist, np.linalg.norm(start_pos), np.linalg.norm(end_pos))
            
            q.append((end_pos, current_angle - branch_angle, current_length * length_decay, current_level - 1))
            q.append((end_pos, current_angle + branch_angle, current_length * length_decay, current_level - 1))
            
        return points, max_dist

    def _spawn_energy_pickup(self):
        pos = self.np_random.uniform([50, 50], [self.WIDTH - 50, self.HEIGHT - 50])
        self.energy_pickups.append({'pos': pos, 'radius': 10})

    def _spawn_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': pos.copy(), 'vel': vel, 'life': self.np_random.integers(15, 30),
                'color': color, 'max_life': 30
            })

    # --- Rendering ---
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_particles()
        self._render_teleport_range()
        self._render_energy_pickups()
        self._render_defenses()
        self._render_fractals()
        self._render_core()
        self._render_cloned_segments()
        self._render_cursor()

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            radius = int(3 * (p['life'] / p['max_life']))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, color)

    def _render_teleport_range(self):
        color = (*self.COLOR_PLAYER, 30)
        pygame.gfxdraw.filled_circle(self.screen, int(self.CORE_POS[0]), int(self.CORE_POS[1]), int(self.teleport_range), color)
        pygame.gfxdraw.aacircle(self.screen, int(self.CORE_POS[0]), int(self.CORE_POS[1]), int(self.teleport_range), (*self.COLOR_PLAYER, 60))

    def _render_energy_pickups(self):
        for p in self.energy_pickups:
            self._draw_glow_circle(self.screen, p['pos'], p['radius'], self.COLOR_ENERGY, 0.5)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), p['radius'], self.COLOR_ENERGY)
            pygame.gfxdraw.aacircle(self.screen, int(p['pos'][0]), int(p['pos'][1]), p['radius'], self.COLOR_ENERGY)

    def _render_defenses(self):
        for d in self.defenses:
            self._draw_fractal_shape(d['pos'], d['angle'], d['shape'], self.COLOR_PLAYER, 2)

    def _render_fractals(self):
        for f in self.fractals:
            self._draw_fractal_shape(f['pos'], f['angle'], f['shape'], self.COLOR_ENEMY, 2)

    def _render_core(self):
        # Health bar as a background ring
        health_ratio = max(0, self.core_health / self.INITIAL_CORE_HEALTH)
        health_color = (int(255 * (1-health_ratio)), int(255 * health_ratio), 50)
        for i in range(5):
            alpha = 100 - i * 20
            pygame.gfxdraw.aacircle(self.screen, int(self.CORE_POS[0]), int(self.CORE_POS[1]), self.CORE_RADIUS + i, (*health_color, alpha))
        
        self._draw_glow_circle(self.screen, self.CORE_POS, self.CORE_RADIUS, self.COLOR_CORE, 0.6)
        pygame.gfxdraw.filled_circle(self.screen, int(self.CORE_POS[0]), int(self.CORE_POS[1]), self.CORE_RADIUS, self.COLOR_CORE)
        pygame.gfxdraw.aacircle(self.screen, int(self.CORE_POS[0]), int(self.CORE_POS[1]), self.CORE_RADIUS, self.COLOR_CORE)

    def _render_cloned_segments(self):
        for i, seg in enumerate(self.cloned_segments_pending[:5]):
            pos = self.CORE_POS + np.array([-30, -20 + i*10])
            self._draw_fractal_shape(pos, 0, seg['shape'], self.COLOR_PLAYER, 1, scale=0.3)

    def _render_cursor(self):
        pos = self.teleport_target_pos
        self._draw_glow_circle(self.screen, pos, 8, self.COLOR_CURSOR, 0.7)
        for i in range(4):
            angle = i * math.pi / 2 + (self.steps * 0.1)
            start = pos + np.array([math.cos(angle), math.sin(angle)]) * 5
            end = pos + np.array([math.cos(angle), math.sin(angle)]) * 10
            pygame.draw.aaline(self.screen, self.COLOR_CURSOR, start, end, 2)

    def _render_ui(self):
        # Health
        health_text = self.font_ui.render(f"CORE: {int(self.core_health)}%", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (10, 10))
        # Charges
        clone_text = self.font_ui.render(f"CLONES: {self.clone_charges}", True, self.COLOR_PLAYER if self.clone_charges > 0 else self.COLOR_ENEMY)
        self.screen.blit(clone_text, (10, 30))
        teleport_text = self.font_ui.render(f"TELEPORTS: {self.teleport_charges}", True, self.COLOR_PLAYER if self.teleport_charges > 0 else self.COLOR_ENEMY)
        self.screen.blit(teleport_text, (10, 50))
        # Energy
        energy_color = self.COLOR_ENERGY if not all(self.upgrades.values()) else self.COLOR_CURSOR
        energy_text = self.font_ui.render(f"ENERGY: {self.energy}", True, energy_color)
        self.screen.blit(energy_text, (10, 70))
        # Upgrades
        clone_upgrade_text = self.font_ui.render(f"x2 CLONE", True, self.COLOR_PLAYER if self.upgrades["clone_efficiency"] else (80,80,80))
        self.screen.blit(clone_upgrade_text, (self.WIDTH - 100, 10))
        range_upgrade_text = self.font_ui.render(f"x2 RANGE", True, self.COLOR_PLAYER if self.upgrades["teleport_range"] else (80,80,80))
        self.screen.blit(range_upgrade_text, (self.WIDTH - 100, 30))
        # Steps
        steps_text = self.font_ui.render(f"TIME: {self.steps}/{self.MAX_EPISODE_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.WIDTH/2 - steps_text.get_width()/2, 10))

    # --- Drawing Helpers ---
    def _draw_fractal_shape(self, center_pos, angle, shape_points, color, width, scale=1.0):
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        for p1, p2 in shape_points:
            p1_rot = np.dot(p1 * scale, rotation_matrix)
            p2_rot = np.dot(p2 * scale, rotation_matrix)
            
            start = center_pos + p1_rot
            end = center_pos + p2_rot
            
            pygame.draw.aaline(self.screen, color, start, end, width)

    def _draw_glow_circle(self, surface, center, radius, color, intensity=0.5):
        for i in range(int(radius * 2), 0, -2):
            alpha = int(255 * (1 - (i / (radius * 2))) * intensity)
            pygame.gfxdraw.aacircle(surface, int(center[0]), int(center[1]), radius + i, (*color, alpha))

    # --- Gymnasium Interface Helpers ---
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "core_health": self.core_health,
            "energy": self.energy,
            "clone_charges": self.clone_charges,
            "teleport_charges": self.teleport_charges,
        }
        
    def close(self):
        pygame.quit()

# --- Example Usage ---
if __name__ == '__main__':
    # The original code had a validation function that is not part of the Gym API
    # and causes issues when run outside the initial context. It has been removed.
    # The main execution block is for human play and demonstration.
    
    # Check if a display is available for human play
    try:
        pygame.display.init()
        pygame.font.init()
        human_play = True
        # Re-set the video driver if we're in human mode
        os.environ["SDL_VIDEODRIVER"] = "x11" 
    except pygame.error:
        human_play = False
        print("No display available. Running in headless mode.")

    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    if human_play:
        screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        pygame.display.set_caption("Fractal Defender")
        clock = pygame.time.Clock()
        
        done = False
        total_reward = 0
        
        while not done:
            # --- Human Input Mapping ---
            movement = 0 # None
            space = 0
            shift = 0
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
            
            space_pressed_this_frame = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        space = 1
                    if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                        shift = 1
            
            action = [movement, space, shift]
            
            # --- Gym Step ---
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # --- Pygame Rendering ---
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
                done = True
                
            clock.tick(GameEnv.FPS)
            
    else: # Headless mode execution example
        done = False
        for _ in range(2000):
            if done:
                break
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        print(f"Headless run finished. Final info: {info}")

    env.close()