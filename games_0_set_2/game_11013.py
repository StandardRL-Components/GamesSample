import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:20:59.950452
# Source Brief: brief_01013.md
# Brief Index: 1013
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    game_description = (
        "Defend your central core from waves of incoming projectiles. Aim and launch your orbiting polygonal defenders to intercept threats and survive as long as possible."
    )
    user_guide = (
        "Controls: Use ↑↓ arrow keys to adjust power and ←→ to aim. Press space to launch your selected defender and shift to cycle between available ammo types."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 10000
        
        # --- Colors ---
        self.COLOR_BG = (15, 18, 32)
        self.COLOR_CORE = (0, 191, 255)
        self.COLOR_CORE_GLOW = (0, 191, 255, 50)
        self.COLOR_ORBIT_PATH = (255, 255, 255, 20)
        self.COLOR_PLAYER_BASE = (57, 255, 20)
        self.COLOR_ENEMY_BASE = (255, 20, 57)
        self.COLOR_AIM_LINE = (255, 204, 0)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_BAR = (57, 255, 20)
        self.COLOR_HEALTH_BAR_BG = (255, 20, 57, 128)

        # --- Core ---
        self.CORE_POS = pygame.Vector2(self.WIDTH // 2, self.HEIGHT // 2)
        self.CORE_RADIUS = 25
        self.MAX_CORE_HEALTH = 100

        # --- Slingshot ---
        self.SLINGSHOT_POWER_MIN = 2
        self.SLINGSHOT_POWER_MAX = 15
        self.SLINGSHOT_POWER_STEP = 0.5
        self.SLINGSHOT_ANGLE_STEP = 3 # degrees

        # --- Game Object Templates ---
        self.SHAPE_TEMPLATES = [
            {'name': 'Triangle', 'mass': 10, 'health': 20, 'radius': 10, 'sides': 3, 'color': self.COLOR_PLAYER_BASE},
            {'name': 'Square', 'mass': 15, 'health': 30, 'radius': 12, 'sides': 4, 'color': (255, 165, 0)}, # Orange
            {'name': 'Pentagon', 'mass': 20, 'health': 40, 'radius': 14, 'sides': 5, 'color': (173, 216, 230)}, # Light Blue
        ]
        self.PROJECTILE_TEMPLATES = [
            {'health': 10, 'radius': 6, 'color': self.COLOR_ENEMY_BASE},
            {'health': 25, 'radius': 8, 'color': (255, 0, 255)}, # Magenta
        ]
        
        # --- Rewards ---
        self.REWARD_DESTROY_PROJECTILE = 0.1
        self.REWARD_WAVE_COMPLETE = 1.0
        self.REWARD_BONUS_WAVE_COMPLETE = 50.0
        self.REWARD_CORE_HIT = -1.0
        self.REWARD_DEATH = -50.0

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.render_mode = render_mode
        if self.render_mode == "human":
            self.human_screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Orbital Defender")

        # --- State Variables ---
        self.state_vars_initialized = False # Defer init to reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.core_health = self.MAX_CORE_HEALTH
        self.wave_number = 1
        self.wave_progress_timer = 0
        self.projectile_spawn_timer = 0
        
        self.slingshot_angle_deg = 0
        self.slingshot_power = (self.SLINGSHOT_POWER_MAX + self.SLINGSHOT_POWER_MIN) / 2
        
        self.projectiles = []
        self.particles = []
        
        self.unlocked_shape_types = [0]
        self.orbiting_shapes = []
        self.launched_shapes = []
        self._initialize_orbiting_shapes()
        self.selected_orbit_slot = 0

        self.prev_space_held = True # prevent launch on first frame
        self.prev_shift_held = True # prevent cycle on first frame
        
        self.state_vars_initialized = True
        
        return self._get_observation(), self._get_info()

    def _initialize_orbiting_shapes(self):
        initial_orbits = [
            {'radius': 80, 'speed': 0.02, 'type_idx': 0},
            {'radius': 120, 'speed': 0.015, 'type_idx': 0},
            {'radius': 160, 'speed': 0.01, 'type_idx': 0},
        ]
        for i, orbit_info in enumerate(initial_orbits):
            self.orbiting_shapes.append({
                'slot': i,
                'orbit_radius': orbit_info['radius'],
                'orbit_angle': self.np_random.uniform(0, 2 * math.pi),
                'orbit_speed': orbit_info['speed'],
                'shape_type_idx': orbit_info['type_idx'],
                'is_respawning': False,
                'respawn_timer': 0,
                'pos': pygame.Vector2(0,0)
            })

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.step_reward = 0
        
        self._handle_input(action)
        self._update_wave_logic()
        self._spawn_projectiles()
        self._update_entities()
        self._handle_collisions()
        self._cleanup_entities()

        self.steps += 1
        terminated = self._check_termination()
        
        if terminated and not self.game_over:
            self.step_reward += self.REWARD_DEATH
            self.game_over = True
            # sfx: game_over_sound

        self.prev_space_held = (action[1] == 1)
        self.prev_shift_held = (action[2] == 1)
        
        if self.render_mode == "human":
            self.human_screen.blit(self.screen, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

        return (
            self._get_observation(),
            self.step_reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if movement == 1: # Up
            self.slingshot_power = min(self.SLINGSHOT_POWER_MAX, self.slingshot_power + self.SLINGSHOT_POWER_STEP)
        elif movement == 2: # Down
            self.slingshot_power = max(self.SLINGSHOT_POWER_MIN, self.slingshot_power - self.SLINGSHOT_POWER_STEP)
        elif movement == 3: # Left
            self.slingshot_angle_deg = (self.slingshot_angle_deg - self.SLINGSHOT_ANGLE_STEP) % 360
        elif movement == 4: # Right
            self.slingshot_angle_deg = (self.slingshot_angle_deg + self.SLINGSHOT_ANGLE_STEP) % 360
        
        # Cycle selected shape on SHIFT press (rising edge)
        if shift_held and not self.prev_shift_held:
            self.selected_orbit_slot = (self.selected_orbit_slot + 1) % len(self.orbiting_shapes)
            # sfx: cycle_weapon_sound

        # Launch shape on SPACE press (rising edge)
        if space_held and not self.prev_space_held:
            slot = self.orbiting_shapes[self.selected_orbit_slot]
            if not slot['is_respawning']:
                self._launch_shape(self.selected_orbit_slot)
                # sfx: launch_shape_sound

    def _launch_shape(self, slot_idx):
        slot = self.orbiting_shapes[slot_idx]
        template = self.SHAPE_TEMPLATES[slot['shape_type_idx']]
        
        angle_rad = math.radians(self.slingshot_angle_deg)
        velocity = pygame.Vector2(math.cos(angle_rad), -math.sin(angle_rad)) * self.slingshot_power
        
        self.launched_shapes.append({
            'pos': slot['pos'].copy(),
            'vel': velocity,
            'health': template['health'],
            'type_idx': slot['shape_type_idx'],
            'angle': 0,
            'rot_speed': 0.1
        })
        
        slot['is_respawning'] = True
        slot['respawn_timer'] = self.FPS * 3 # 3 second respawn

    def _update_wave_logic(self):
        self.wave_progress_timer += 1
        wave_duration = self.FPS * 20 # 20 seconds per wave
        
        if self.wave_progress_timer >= wave_duration:
            self.wave_progress_timer = 0
            self.wave_number += 1
            self.score += 10 * (self.wave_number -1)
            self.step_reward += self.REWARD_WAVE_COMPLETE
            # sfx: wave_complete_sound
            
            if self.wave_number % 5 == 0:
                self.step_reward += self.REWARD_BONUS_WAVE_COMPLETE
                # sfx: bonus_wave_sound

            if self.wave_number % 3 == 0:
                new_shape_idx = (self.wave_number // 3)
                if new_shape_idx < len(self.SHAPE_TEMPLATES) and new_shape_idx not in self.unlocked_shape_types:
                    self.unlocked_shape_types.append(new_shape_idx)
                    # sfx: unlock_sound
    
    def _spawn_projectiles(self):
        spawn_rate_increase = self.wave_number * 0.01
        base_spawn_interval = self.FPS * 3
        spawn_interval = max(self.FPS * 0.2, base_spawn_interval / (1 + spawn_rate_increase))

        self.projectile_spawn_timer += 1
        if self.projectile_spawn_timer >= spawn_interval:
            self.projectile_spawn_timer = 0
            
            edge = self.np_random.integers(0, 4)
            if edge == 0: pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), -20)
            elif edge == 1: pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 20)
            elif edge == 2: pos = pygame.Vector2(-20, self.np_random.uniform(0, self.HEIGHT))
            else: pos = pygame.Vector2(self.WIDTH + 20, self.np_random.uniform(0, self.HEIGHT))
            
            direction = (self.CORE_POS - pos).normalize()
            speed = 1.0 + self.wave_number * 0.05
            
            projectile_type_idx = 0
            if self.wave_number >= 5:
                # Introduce harder projectiles, with increasing probability
                if self.np_random.random() < min(0.5, (self.wave_number - 4) * 0.05):
                    projectile_type_idx = 1

            template = self.PROJECTILE_TEMPLATES[projectile_type_idx]
            self.projectiles.append({
                'pos': pos,
                'vel': direction * speed,
                'health': template['health'],
                'type_idx': projectile_type_idx
            })

    def _update_entities(self):
        # Update orbiting shapes
        for slot in self.orbiting_shapes:
            if slot['is_respawning']:
                slot['respawn_timer'] -= 1
                if slot['respawn_timer'] <= 0:
                    slot['is_respawning'] = False
                    # Cycle to newly unlocked shape type if available
                    if len(self.unlocked_shape_types) > 1:
                       slot['shape_type_idx'] = (slot['shape_type_idx'] + 1) % len(self.unlocked_shape_types)

            else:
                slot['orbit_angle'] += slot['orbit_speed']
                slot['pos'].x = self.CORE_POS.x + slot['orbit_radius'] * math.cos(slot['orbit_angle'])
                slot['pos'].y = self.CORE_POS.y + slot['orbit_radius'] * math.sin(slot['orbit_angle'])

        # Update launched shapes
        for shape in self.launched_shapes:
            shape['pos'] += shape['vel']
            shape['angle'] += shape['rot_speed']

        # Update projectiles
        for p in self.projectiles:
            p['pos'] += p['vel']
        
        # Update particles
        for particle in self.particles:
            particle['pos'] += particle['vel']
            particle['lifespan'] -= 1

    def _handle_collisions(self):
        # Projectile vs Launched Shape
        for p in self.projectiles[:]:
            p_template = self.PROJECTILE_TEMPLATES[p['type_idx']]
            for s in self.launched_shapes[:]:
                s_template = self.SHAPE_TEMPLATES[s['type_idx']]
                dist = p['pos'].distance_to(s['pos'])
                if dist < p_template['radius'] + s_template['radius']:
                    # sfx: impact_sound
                    damage = min(p['health'], s['health'])
                    p['health'] -= damage
                    s['health'] -= damage
                    
                    self._create_particle_burst(p['pos'], p_template['color'], 15, 2.0)
                    self._create_particle_burst(s['pos'], s_template['color'], 10, 1.5)
                    
                    if p['health'] <= 0:
                        self.step_reward += self.REWARD_DESTROY_PROJECTILE
                        self.score += 1

        # Projectile vs Core
        for p in self.projectiles[:]:
            p_template = self.PROJECTILE_TEMPLATES[p['type_idx']]
            dist = p['pos'].distance_to(self.CORE_POS)
            if dist < p_template['radius'] + self.CORE_RADIUS:
                # sfx: core_hit_sound
                self.core_health -= p['health']
                self.step_reward += self.REWARD_CORE_HIT
                self._create_particle_burst(p['pos'], self.COLOR_CORE, 30, 3.0)
                p['health'] = 0 # Mark for removal
    
    def _cleanup_entities(self):
        self.projectiles = [p for p in self.projectiles if p['health'] > 0 and self.screen.get_rect().collidepoint(p['pos'])]
        self.launched_shapes = [s for s in self.launched_shapes if s['health'] > 0 and self.screen.get_rect().inflate(50, 50).collidepoint(s['pos'])]
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _create_particle_burst(self, pos, color, count, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, max_speed)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifespan': self.np_random.integers(15, 31),
                'color': color,
                'radius': self.np_random.uniform(1, 3)
            })

    def _check_termination(self):
        return self.core_health <= 0 or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        if not self.state_vars_initialized:
            return np.zeros(self.observation_space.shape, dtype=np.uint8)
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "core_health": self.core_health,
            "unlocked_shapes": len(self.unlocked_shape_types),
        }

    def _render_game(self):
        # Render orbit paths
        for slot in self.orbiting_shapes:
            pygame.gfxdraw.aacircle(self.screen, int(self.CORE_POS.x), int(self.CORE_POS.y), int(slot['orbit_radius']), self.COLOR_ORBIT_PATH)

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 30))
            color = p['color'] + (alpha,)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), color)

        # Render Core
        pygame.gfxdraw.filled_circle(self.screen, int(self.CORE_POS.x), int(self.CORE_POS.y), self.CORE_RADIUS + 5, self.COLOR_CORE_GLOW)
        pygame.gfxdraw.aacircle(self.screen, int(self.CORE_POS.x), int(self.CORE_POS.y), self.CORE_RADIUS, self.COLOR_CORE)
        pygame.gfxdraw.filled_circle(self.screen, int(self.CORE_POS.x), int(self.CORE_POS.y), self.CORE_RADIUS, self.COLOR_CORE)

        # Render projectiles
        for p in self.projectiles:
            template = self.PROJECTILE_TEMPLATES[p['type_idx']]
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), template['radius'], template['color'])
            pygame.gfxdraw.aacircle(self.screen, int(p['pos'].x), int(p['pos'].y), template['radius'], template['color'])

        # Render launched shapes
        for s in self.launched_shapes:
            self._render_polygon(s['pos'], self.SHAPE_TEMPLATES[s['type_idx']], s['angle'])

        # Render orbiting shapes
        for slot in self.orbiting_shapes:
            if not slot['is_respawning']:
                self._render_polygon(slot['pos'], self.SHAPE_TEMPLATES[slot['shape_type_idx']], slot['orbit_angle'] * 3)

        # Render slingshot aimer
        selected_slot = self.orbiting_shapes[self.selected_orbit_slot]
        if not selected_slot['is_respawning']:
            start_pos = selected_slot['pos']
            angle_rad = math.radians(self.slingshot_angle_deg)
            length = self.slingshot_power * 10
            end_pos = start_pos + pygame.Vector2(math.cos(angle_rad), -math.sin(angle_rad)) * length
            pygame.draw.aaline(self.screen, self.COLOR_AIM_LINE, start_pos, end_pos, 2)
    
    def _render_polygon(self, pos, template, rotation_angle):
        points = []
        for i in range(template['sides']):
            angle = (i / template['sides']) * 2 * math.pi + rotation_angle
            x = pos.x + template['radius'] * math.cos(angle)
            y = pos.y + template['radius'] * math.sin(angle)
            points.append((int(x), int(y)))
        
        pygame.gfxdraw.aapolygon(self.screen, points, template['color'])
        pygame.gfxdraw.filled_polygon(self.screen, points, template['color'])

    def _render_ui(self):
        # Health Bar
        bar_width = 100
        bar_height = 10
        bar_x = self.CORE_POS.x - bar_width // 2
        bar_y = self.CORE_POS.y + self.CORE_RADIUS + 15
        health_ratio = max(0, self.core_health / self.MAX_CORE_HEALTH)
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (bar_x, bar_y, bar_width * health_ratio, bar_height), border_radius=3)

        # Score Text
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        # Wave Text
        wave_text = self.font_large.render(f"WAVE: {self.wave_number}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (10, 10))

        # Selected Shape Text
        selected_slot = self.orbiting_shapes[self.selected_orbit_slot]
        shape_name = "Respawning..."
        if not selected_slot['is_respawning']:
            shape_name = self.SHAPE_TEMPLATES[selected_slot['shape_type_idx']]['name']
        ammo_text = self.font_small.render(f"AMMO: {shape_name}", True, self.COLOR_UI_TEXT)
        self.screen.blit(ammo_text, (10, self.HEIGHT - ammo_text.get_height() - 10))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        obs, _ = self.reset()
        assert obs.shape == (400, 640, 3)
        assert obs.dtype == np.uint8
        
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

if __name__ == '__main__':
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    done = False
    
    # Manual play loop
    print("\n--- Manual Control ---")
    print("  Arrows: Aim/Power")
    print("  Space: Launch")
    print("  Shift: Cycle Ammo")
    print("  Q: Quit")
    
    while not done:
        # Default action is no-op
        action = [0, 0, 0] # [movement, space, shift]
        
        # Pygame event handling for manual control
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]: done = True
        if keys[pygame.K_UP]: action[0] = 1
        if keys[pygame.K_DOWN]: action[0] = 2
        if keys[pygame.K_LEFT]: action[0] = 3
        if keys[pygame.K_RIGHT]: action[0] = 4
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
            
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Wave: {info['wave']}, Score: {info['score']}, Reward: {reward:.2f}")

        if terminated or truncated:
            print("Game Over!")
            done = True
            
    env.close()