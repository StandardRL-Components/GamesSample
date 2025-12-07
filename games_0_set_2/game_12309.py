import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:46:56.064914
# Source Brief: brief_02309.md
# Brief Index: 2309
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    GameEnv: Defend your base from waves of colorful enemies by trapping them
    in bubbles. Chain captures for massive score multipliers.
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Defend your base from waves of colorful enemies by trapping them in bubbles. "
        "Chain captures for massive score multipliers."
    )
    user_guide = (
        "Controls: ←→ to aim the launcher. Press space to fire a single bubble and "
        "shift to fire a larger area bubble."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        self.WIDTH, self.HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_combo = pygame.font.SysFont("monospace", 28, bold=True)

        # --- Game Constants ---
        self.MAX_STEPS = 5000
        self.LAUNCHER_POS = (self.WIDTH // 2, self.HEIGHT - 20)
        self.LAUNCHER_TURN_RATE = 2.5 # degrees per step
        self.MIN_ANGLE, self.MAX_ANGLE = 10, 170
        self.BUBBLE_SPEED = 6.0
        self.COMBO_TIMEOUT = 90 # steps (3 seconds at 30fps)

        # --- Color Palette ---
        self.COLOR_BG = (26, 26, 46) # Dark blue/purple
        self.COLOR_LAUNCHER_BASE = (70, 70, 90)
        self.COLOR_LAUNCHER_TURRET = (200, 200, 220)
        self.COLOR_TEXT = (255, 255, 255)
        self.ENEMY_COLORS = [(255, 89, 114), (255, 166, 0), (52, 235, 146)] # Magenta, Orange, Green
        self.COMBO_COLORS = [
            (100, 100, 255), (0, 150, 255), (0, 255, 255), (0, 255, 150),
            (150, 255, 0), (255, 255, 0), (255, 150, 0), (255, 0, 0)
        ]

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.launcher_angle = 90.0
        self.enemies = []
        self.bubbles = []
        self.particles = []
        self.combo_multiplier = 1
        self.last_capture_step = -self.COMBO_TIMEOUT
        self.prev_space_held = False
        self.prev_shift_held = False
        self.base_enemy_speed = 0.8
        self.enemy_spawn_count = 3
        
        # This is called here to set up the RNG, but the state will be
        # properly reset by the first call to reset() by the environment wrapper.
        self.reset()
        

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.launcher_angle = 90.0
        
        self.enemies = []
        self.bubbles = []
        self.particles = []
        
        self.combo_multiplier = 1
        self.last_capture_step = -self.COMBO_TIMEOUT

        self.prev_space_held = False
        self.prev_shift_held = False

        self.base_enemy_speed = 0.8
        self.enemy_spawn_count = 3

        self._spawn_enemies()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held_raw, shift_held_raw = action
        space_held = space_held_raw == 1
        shift_held = shift_held_raw == 1

        self.steps += 1
        reward = 0.0

        # --- Handle Input ---
        self._handle_movement(movement)
        self._handle_shooting(space_held, shift_held)
        
        # --- Update Game State ---
        self._update_bubbles()
        enemy_reward, enemies_at_bottom = self._update_enemies()
        reward += enemy_reward
        self._update_particles()
        
        # --- Handle Collisions & Captures ---
        capture_reward = self._handle_collisions()
        reward += capture_reward
        
        # --- Handle Combo Logic ---
        if self.steps - self.last_capture_step > self.COMBO_TIMEOUT or enemies_at_bottom > 0:
            if self.combo_multiplier > 1:
                # sfx: combo_break
                self._spawn_particles(self.WIDTH - 120, 55, 20, (255, 50, 50), p_type='burst')
            self.combo_multiplier = 1

        # --- Spawn new wave if needed ---
        if not self.enemies:
            self._spawn_enemies()

        # --- Difficulty Progression ---
        if self.steps > 0 and self.steps % 500 == 0:
            self.base_enemy_speed = min(3.0, self.base_enemy_speed + 0.1)
            self.enemy_spawn_count = min(10, self.enemy_spawn_count + 1)
            # sfx: level_up

        # --- Termination Condition ---
        terminated = self.steps >= self.MAX_STEPS
        truncated = False # Not using truncation based on time limit
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_movement(self, movement):
        if movement == 3: # Left
            self.launcher_angle += self.LAUNCHER_TURN_RATE
        elif movement == 4: # Right
            self.launcher_angle -= self.LAUNCHER_TURN_RATE
        self.launcher_angle = np.clip(self.launcher_angle, self.MIN_ANGLE, self.MAX_ANGLE)

    def _handle_shooting(self, space_held, shift_held):
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held

        if space_pressed:
            self._spawn_bubble(b_type='single')
            # sfx: shoot_single_bubble
        if shift_pressed:
            self._spawn_bubble(b_type='area')
            # sfx: shoot_area_bubble

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _spawn_bubble(self, b_type):
        angle_rad = math.radians(self.launcher_angle)
        vel = (math.cos(angle_rad) * self.BUBBLE_SPEED, -math.sin(angle_rad) * self.BUBBLE_SPEED)
        pos = list(self.LAUNCHER_POS)
        
        if b_type == 'single':
            max_radius = 30
            lifetime = 120
        else: # area
            max_radius = 50
            lifetime = 90

        self.bubbles.append({
            'pos': pos, 'vel': vel, 'radius': 5, 'max_radius': max_radius,
            'lifetime': lifetime, 'state': 'expanding', 'trapped_enemy_color': None,
            'pulse': 0.0
        })

    def _spawn_enemies(self):
        for _ in range(self.enemy_spawn_count):
            enemy_type = self.np_random.choice(['straight', 'zigzag'])
            speed_mult = 1.0 if enemy_type == 'straight' else 1.2
            color_idx = 0 if enemy_type == 'straight' else 1
            self.enemies.append({
                'pos': [self.np_random.uniform(20, self.WIDTH - 20), self.np_random.uniform(-50, -20)],
                'speed': self.base_enemy_speed * speed_mult,
                'type': enemy_type,
                'color': self.ENEMY_COLORS[color_idx],
                'size': 12 if enemy_type == 'straight' else 10,
                'zigzag_phase': self.np_random.uniform(0, 2 * math.pi)
            })

    def _update_bubbles(self):
        for b in reversed(self.bubbles):
            # Movement
            b['pos'][0] += b['vel'][0]
            b['pos'][1] += b['vel'][1]
            
            # Pulse effect
            b['pulse'] += 0.2
            
            # State update
            if b['state'] == 'expanding':
                b['radius'] += 1.5
                if b['radius'] >= b['max_radius']:
                    b['radius'] = b['max_radius']
                    b['state'] = 'active'
            
            # Lifetime and boundary check
            b['lifetime'] -= 1
            if b['lifetime'] <= 0 or not (0 < b['pos'][0] < self.WIDTH and 0 < b['pos'][1] < self.HEIGHT):
                if b['trapped_enemy_color'] is None:
                    # sfx: bubble_pop
                    self._spawn_particles(b['pos'][0], b['pos'][1], 15, (200, 200, 255), p_type='burst')
                self.bubbles.remove(b)

    def _update_enemies(self):
        reward = 0.0
        enemies_at_bottom = 0
        for e in reversed(self.enemies):
            if e['type'] == 'straight':
                e['pos'][1] += e['speed']
            elif e['type'] == 'zigzag':
                e['pos'][1] += e['speed']
                e['pos'][0] += math.sin(e['pos'][1] * 0.05 + e['zigzag_phase']) * 2.0

            if e['pos'][1] > self.HEIGHT - 15:
                # sfx: enemy_escape
                self._spawn_particles(e['pos'][0], self.HEIGHT, 20, (255, 50, 50), p_type='fail')
                self.enemies.remove(e)
                enemies_at_bottom += 1
        
        return reward, enemies_at_bottom

    def _handle_collisions(self):
        reward = 0.0
        for e in reversed(self.enemies):
            for b in self.bubbles:
                if b['state'] == 'active' and b['trapped_enemy_color'] is None:
                    dist = math.hypot(e['pos'][0] - b['pos'][0], e['pos'][1] - b['pos'][1])
                    if dist < b['radius'] - e['size'] / 2:
                        # sfx: enemy_capture
                        b['trapped_enemy_color'] = e['color']
                        b['vel'] = (0, -0.2) # Float upwards slowly
                        b['lifetime'] = 60 # Reset lifetime for captured bubble
                        
                        self.combo_multiplier += 1
                        self.last_capture_step = self.steps
                        
                        self.score += 1 * self.combo_multiplier
                        reward += 0.1 + (1.0 * self.combo_multiplier)
                        
                        self._spawn_particles(b['pos'][0], b['pos'][1], 30, e['color'], p_type='implode')
                        self.enemies.remove(e)
                        break # Move to next enemy
        return reward

    def _update_particles(self):
        for p in reversed(self.particles):
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.particles.remove(p)

    def _spawn_particles(self, x, y, count, color, p_type):
        for _ in range(count):
            if p_type == 'burst':
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
                vel = (math.cos(angle) * speed, math.sin(angle) * speed)
                lifetime = self.np_random.integers(15, 31)
            elif p_type == 'implode':
                angle = self.np_random.uniform(0, 2 * math.pi)
                start_dist = self.np_random.uniform(20, 40)
                pos = [x + math.cos(angle) * start_dist, y + math.sin(angle) * start_dist]
                vel = [-(pos[0] - x) / 15, -(pos[1] - y) / 15]
                lifetime = 15
            elif p_type == 'fail':
                vel = (self.np_random.uniform(-1, 1), self.np_random.uniform(-3, -1))
                lifetime = self.np_random.integers(20, 41)
            
            self.particles.append({
                'pos': [x, y] if p_type != 'implode' else pos,
                'vel': vel,
                'lifetime': lifetime,
                'max_lifetime': lifetime,
                'color': color,
                'size': self.np_random.uniform(1, 4)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "combo": self.combo_multiplier}

    def _render_game(self):
        self._render_enemies()
        self._render_bubbles()
        self._render_launcher()
        self._render_particles()

    def _render_launcher(self):
        # Base
        pygame.draw.rect(self.screen, self.COLOR_LAUNCHER_BASE, (self.LAUNCHER_POS[0] - 40, self.HEIGHT - 20, 80, 20))
        pygame.draw.circle(self.screen, self.COLOR_LAUNCHER_BASE, (self.LAUNCHER_POS[0], self.HEIGHT - 15), 25)

        # Turret
        angle_rad = math.radians(self.launcher_angle)
        end_x = self.LAUNCHER_POS[0] + math.cos(angle_rad) * 35
        end_y = self.LAUNCHER_POS[1] - math.sin(angle_rad) * 35
        pygame.draw.line(self.screen, self.COLOR_LAUNCHER_TURRET, self.LAUNCHER_POS, (int(end_x), int(end_y)), 8)
        pygame.draw.circle(self.screen, self.COLOR_LAUNCHER_TURRET, self.LAUNCHER_POS, 10)

    def _render_enemies(self):
        for e in self.enemies:
            pos = (int(e['pos'][0]), int(e['pos'][1]))
            size = int(e['size'])
            color = e['color']
            
            # Glow effect
            glow_radius = int(size * 1.5)
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*color, 50), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surf, (pos[0] - glow_radius, pos[1] - glow_radius))

            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, color)

    def _render_bubbles(self):
        for b in self.bubbles:
            pos = (int(b['pos'][0]), int(b['pos'][1]))
            radius = int(b['radius'])
            
            if b['trapped_enemy_color']:
                # Trapped bubble
                color = b['trapped_enemy_color']
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, (*color, 80))
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, (*color, 180))
            else:
                # Normal bubble
                pulse_offset = math.sin(b['pulse']) * 2
                outer_radius = int(radius + pulse_offset)
                
                # Rainbow combo border
                combo_idx = min(self.combo_multiplier - 1, len(self.COMBO_COLORS) - 1)
                border_color = self.COMBO_COLORS[combo_idx] if self.combo_multiplier > 1 else (255, 255, 255)

                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], outer_radius, (255, 255, 255, 30))
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], outer_radius, (*border_color, 128))
                
                # Inner shimmer
                inner_radius = max(0, int(radius - 5 + pulse_offset))
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], inner_radius, (255, 255, 255, 60))

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['lifetime'] / p['max_lifetime']))
            color = (*p['color'], alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            size = int(p['size'])
            
            # Use a rect for a chunkier particle feel
            part_rect = pygame.Rect(pos[0] - size//2, pos[1] - size//2, size, size)
            shape_surf = pygame.Surface(part_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(shape_surf, color, shape_surf.get_rect())
            self.screen.blit(shape_surf, part_rect.topleft)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Combo Multiplier
        if self.combo_multiplier > 1:
            combo_idx = min(self.combo_multiplier - 2, len(self.COMBO_COLORS) - 1)
            color = self.COMBO_COLORS[combo_idx]
            combo_text = self.font_combo.render(f"x{self.combo_multiplier}", True, color)
            
            # Pulsating size effect
            scale = 1.0 + 0.1 * math.sin((self.steps - self.last_capture_step) * 0.3)
            scaled_font = pygame.font.SysFont("monospace", int(28 * scale), bold=True)
            combo_text = scaled_font.render(f"x{self.combo_multiplier}", True, color)

            text_rect = combo_text.get_rect(center=(self.WIDTH - 70, 40))
            self.screen.blit(combo_text, text_rect)

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == "__main__":
    # The following code is for human play and requires a graphical display.
    # It will not run in a headless environment.
    os.environ.unsetenv("SDL_VIDEODRIVER")

    env = GameEnv(render_mode="rgb_array")
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Bubble Trap")
    
    obs, info = env.reset()
    done = False
    
    while not done:
        # Action mapping for human play
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Handle window closing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        env.clock.tick(30) # Limit to 30 FPS

    env.close()
    print(f"Game Over! Final Score: {info['score']}")