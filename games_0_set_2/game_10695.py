import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import deque

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a physics-based puzzle game.
    The player sets the launch parameters for 5 projectiles and then launches them
    simultaneously to destroy a grid of targets. Projectiles can hit multiple
    targets, gaining power with each one they destroy.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Set the launch angle and power for five projectiles, then watch them fly "
        "simultaneously to destroy all targets in a physics-based puzzle."
    )
    user_guide = (
        "Controls: Use ←→ arrows to change angle and ↑↓ arrows to change power. "
        "Press space to confirm a projectile's settings and move to the next. "
        "Hold shift to reset the current projectile's aim."
    )
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1000
    MAX_AIMING_STEPS = 500
    NUM_PROJECTILES = 5
    NUM_TARGETS = 25
    TARGET_GRID_SIZE = 5

    # Colors
    COLOR_BG = (10, 20, 40)
    COLOR_PLATFORM = (100, 100, 120)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_VALUE = (180, 255, 180)
    COLOR_UI_ACCENT = (255, 100, 100)
    COLOR_TARGET = (255, 50, 50)
    COLOR_TARGET_GLOW = (255, 50, 50, 50)
    COLOR_PROJECTILE = (100, 200, 255)
    COLOR_PROJECTILE_GLOW = (100, 200, 255, 80)
    COLOR_TRAJECTORY = (100, 255, 100, 150)
    COLOR_EXPLOSION = [(255, 200, 50), (255, 150, 50), (255, 100, 50)]

    # Physics & Gameplay
    GRAVITY = 0.1
    LAUNCHER_POS = (60, 350)
    PROJECTILE_RADIUS = 6
    TARGET_RADIUS = 10
    TARGET_HEALTH = 10.0
    POWER_RANGE = (1.0, 15.0)
    ANGLE_RANGE = (10, 170)
    POWER_STEP = 0.2
    ANGLE_STEP = 1.0
    DEFAULT_POWER = 8.0
    DEFAULT_ANGLE = 90.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_small = pygame.font.SysFont("Consolas", 16)
            self.font_large = pygame.font.SysFont("Consolas", 24)
        except pygame.error:
            self.font_small = pygame.font.Font(None, 20)
            self.font_large = pygame.font.Font(None, 30)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.reward_this_step = 0
        self.phase = 'AIMING'

        self.projectile_params = [{
            'power': self.DEFAULT_POWER,
            'angle': self.DEFAULT_ANGLE
        } for _ in range(self.NUM_PROJECTILES)]
        self.current_projectile_index = 0

        self._setup_targets()

        self.active_projectiles = []
        self.particles = []

        self.last_space_held = False
        self.last_shift_held = False

        return self._get_observation(), self._get_info()

    def _setup_targets(self):
        self.targets = []
        grid_w = self.SCREEN_WIDTH * 0.6
        grid_h = self.SCREEN_HEIGHT * 0.5
        start_x = self.SCREEN_WIDTH * 0.35
        start_y = 50
        spacing_x = grid_w / (self.TARGET_GRID_SIZE - 1)
        spacing_y = grid_h / (self.TARGET_GRID_SIZE - 1)

        for i in range(self.TARGET_GRID_SIZE):
            for j in range(self.TARGET_GRID_SIZE):
                self.targets.append({
                    'pos': np.array([start_x + j * spacing_x, start_y + i * spacing_y]),
                    'alive': True,
                    'health': self.TARGET_HEALTH
                })

    def step(self, action):
        self.reward_this_step = 0
        self.steps += 1
        terminated = False

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held
        self.last_space_held, self.last_shift_held = space_held, shift_held

        if self.phase == 'AIMING':
            self._handle_aiming_phase(movement, space_pressed, shift_pressed)
        elif self.phase == 'LAUNCH':
            self._handle_launch_phase()

        terminated = self._check_termination()
        
        if not terminated:
            self.reward_this_step -= 0.001 # Small penalty for each step to encourage efficiency

        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False,
            self._get_info()
        )

    def _handle_aiming_phase(self, movement, space_pressed, shift_pressed):
        params = self.projectile_params[self.current_projectile_index]

        if movement == 1: params['power'] += self.POWER_STEP # Up
        if movement == 2: params['power'] -= self.POWER_STEP # Down
        if movement == 3: params['angle'] -= self.ANGLE_STEP # Left
        if movement == 4: params['angle'] += self.ANGLE_STEP # Right

        params['power'] = np.clip(params['power'], self.POWER_RANGE[0], self.POWER_RANGE[1])
        params['angle'] = np.clip(params['angle'], self.ANGLE_RANGE[0], self.ANGLE_RANGE[1])

        if shift_pressed:
            # SFX: Reset sound
            params['power'] = self.DEFAULT_POWER
            params['angle'] = self.DEFAULT_ANGLE

        if space_pressed:
            # SFX: UI Confirm sound
            self.current_projectile_index += 1
            if self.current_projectile_index >= self.NUM_PROJECTILES:
                self._start_launch_phase()
                return

        if self.steps > self.MAX_AIMING_STEPS:
            self._start_launch_phase()
            return

        self._calculate_aiming_reward()

    def _start_launch_phase(self):
        if self.phase == 'LAUNCH': return
        self.phase = 'LAUNCH'
        # SFX: Multiple launch sounds
        for i, params in enumerate(self.projectile_params):
            angle_rad = math.radians(params['angle'])
            vel = np.array([
                params['power'] * math.cos(angle_rad),
                -params['power'] * math.sin(angle_rad) # Pygame Y is inverted
            ])
            self.active_projectiles.append({
                'pos': np.array(self.LAUNCHER_POS, dtype=float),
                'vel': vel,
                'force': params['power'],
                'trail': deque(maxlen=20),
                'id': i
            })

    def _calculate_aiming_reward(self):
        current_angle = self.projectile_params[self.current_projectile_index]['angle']
        living_targets = [t for t in self.targets if t['alive']]
        if not living_targets:
            return

        target_angles = [
            math.degrees(math.atan2(
                -(t['pos'][1] - self.LAUNCHER_POS[1]),
                t['pos'][0] - self.LAUNCHER_POS[0]
            ))
            for t in living_targets
        ]

        if not target_angles: return

        angle_diffs = [abs(current_angle - ta) for ta in target_angles]
        min_diff = min(angle_diffs)
        
        aim_quality = (180.0 - min_diff) / 180.0
        self.reward_this_step += aim_quality * 0.01

    def _handle_launch_phase(self):
        if not self.active_projectiles and not self.particles:
            return

        projectiles_to_remove = []
        for proj in self.active_projectiles:
            proj['trail'].append(proj['pos'].copy())
            proj['vel'][1] += self.GRAVITY
            proj['pos'] += proj['vel']

            for target in self.targets:
                if not target['alive']: continue
                dist = np.linalg.norm(proj['pos'] - target['pos'])
                if dist < self.PROJECTILE_RADIUS + self.TARGET_RADIUS:
                    # SFX: Impact sound
                    damage = proj['force']
                    target['health'] -= damage
                    proj['force'] *= 0.8 # Projectile loses some force on impact

                    if target['health'] <= 0:
                        target['alive'] = False
                        self.score += 1
                        self.reward_this_step += 1.0
                        proj['force'] += 2.0 # Absorb mass from destroyed target
                        self._create_explosion(target['pos'])
                        # SFX: Explosion sound

            if not (0 < proj['pos'][0] < self.SCREEN_WIDTH and proj['pos'][1] < self.SCREEN_HEIGHT + 50):
                projectiles_to_remove.append(proj)
        
        self.active_projectiles = [p for p in self.active_projectiles if p not in projectiles_to_remove]

        particles_to_remove = []
        for p in self.particles:
            p['vel'][1] += self.GRAVITY * 0.2
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                particles_to_remove.append(p)
        self.particles = [p for p in self.particles if p not in particles_to_remove]

    def _create_explosion(self, pos):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            color_index = self.np_random.integers(0, len(self.COLOR_EXPLOSION))
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifespan': self.np_random.integers(20, 41),
                'color': self.COLOR_EXPLOSION[color_index],
                'radius': self.np_random.uniform(1, 4)
            })

    def _check_termination(self):
        targets_left = sum(1 for t in self.targets if t['alive'])
        
        if targets_left == 0:
            self.reward_this_step += 100.0 # Victory bonus
            # SFX: Victory Jingle
            return True

        if self.phase == 'LAUNCH' and not self.active_projectiles and not self.particles:
            self.reward_this_step -= 10.0 # Penalty for failure
            return True

        if self.steps >= self.MAX_STEPS:
            self.reward_this_step -= 10.0 # Penalty for timeout
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
            "targets_left": sum(1 for t in self.targets if t['alive']),
            "phase": self.phase,
            "current_projectile": self.current_projectile_index
        }

    def _render_game(self):
        pygame.draw.rect(self.screen, self.COLOR_PLATFORM, (self.LAUNCHER_POS[0] - 20, self.LAUNCHER_POS[1], 40, 20))
        pygame.draw.circle(self.screen, self.COLOR_PLATFORM, (int(self.LAUNCHER_POS[0]), int(self.LAUNCHER_POS[1])), 10)

        for target in self.targets:
            if target['alive']:
                pos_int = (int(target['pos'][0]), int(target['pos'][1]))
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.TARGET_RADIUS + 3, self.COLOR_TARGET_GLOW)
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.TARGET_RADIUS, self.COLOR_TARGET)
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.TARGET_RADIUS, self.COLOR_TARGET)

        if self.phase == 'AIMING' and self.current_projectile_index < self.NUM_PROJECTILES:
            self._render_aiming_trajectory()

        for proj in self.active_projectiles:
            if len(proj['trail']) > 1:
                for i, p in enumerate(proj['trail']):
                    alpha = int(255 * (i / (len(proj['trail']) - 1)))
                    color = (*self.COLOR_PROJECTILE, alpha)
                    radius = int(self.PROJECTILE_RADIUS * (i / len(proj['trail'])))
                    if radius > 0:
                        pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), radius, color)

            pos_int = (int(proj['pos'][0]), int(proj['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.PROJECTILE_RADIUS + 4, self.COLOR_PROJECTILE_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.PROJECTILE_RADIUS, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.PROJECTILE_RADIUS, self.COLOR_PROJECTILE)

        for p in self.particles:
            alpha = max(0, int(255 * (p['lifespan'] / 40.0)))
            color = (*p['color'], alpha)
            pos_int = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(p['radius']), color)

    def _render_aiming_trajectory(self):
        params = self.projectile_params[self.current_projectile_index]
        angle_rad = math.radians(params['angle'])
        vel = np.array([params['power'] * math.cos(angle_rad), -params['power'] * math.sin(angle_rad)])
        pos = np.array(self.LAUNCHER_POS, dtype=float)

        for i in range(100):
            vel[1] += self.GRAVITY
            new_pos = pos + vel
            if i % 3 == 0:
                if 0 < pos[0] < self.SCREEN_WIDTH and 0 < pos[1] < self.SCREEN_HEIGHT:
                    pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), 2, self.COLOR_TRAJECTORY)
            pos = new_pos

    def _render_ui(self):
        def draw_text(text, font, color, pos, shadow_color=(0,0,0)):
            text_surf = font.render(text, True, color)
            shadow_surf = font.render(text, True, shadow_color)
            self.screen.blit(shadow_surf, (pos[0]+1, pos[1]+1))
            self.screen.blit(text_surf, pos)

        draw_text(f"Score: {self.score}", self.font_large, self.COLOR_UI_VALUE, (10, 10))
        targets_left = sum(1 for t in self.targets if t['alive'])
        draw_text(f"Targets Left: {targets_left}", self.font_large, self.COLOR_UI_TEXT, (10, 40))

        if self.phase == 'AIMING' and self.current_projectile_index < self.NUM_PROJECTILES:
            draw_text(f"Aiming Projectile: {self.current_projectile_index + 1}/{self.NUM_PROJECTILES}", self.font_small, self.COLOR_UI_ACCENT, (120, 330))
            params = self.projectile_params[self.current_projectile_index]
            draw_text(f"Power: {params['power']:.1f}", self.font_small, self.COLOR_UI_TEXT, (120, 350))
            draw_text(f"Angle: {params['angle']:.1f}", self.font_small, self.COLOR_UI_TEXT, (120, 370))
        elif self.phase == 'LAUNCH':
            draw_text("LAUNCHING...", self.font_large, self.COLOR_UI_ACCENT, (self.SCREEN_WIDTH // 2 - 80, self.SCREEN_HEIGHT - 40))

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # To see the game, we need to create a real display.
    try:
        real_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Physics Puzzle")
        os.environ["SDL_VIDEODRIVER"] = "x11"
    except pygame.error:
        print("Could not create display. Running headlessly.")
        real_screen = None

    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if real_screen:
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            real_screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']}")
            total_reward = 0
            obs, info = env.reset()
            if real_screen:
                pygame.time.wait(1000)

        clock.tick(30)
        
    env.close()