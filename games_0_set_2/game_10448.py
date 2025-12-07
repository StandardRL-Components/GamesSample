import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:24:18.882625
# Source Brief: brief_00448.md
# Brief Index: 448
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    Gymnasium environment for a skill-based ricochet game.

    The player controls the launch angle and power of a projectile.
    The goal is to hit all three targets by bouncing the projectile off
    reflective surfaces. Hitting a target provides a speed boost.

    The game prioritizes visual quality and satisfying "game feel".
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Control the launch angle and power of a projectile to hit all targets by ricocheting off walls."
    user_guide = "Controls: Use ↑↓ arrow keys to adjust power and ←→ to adjust angle. Press space to launch and shift to reset your aim."
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    WORLD_WIDTH = 100
    WORLD_HEIGHT = 100 * (SCREEN_HEIGHT / SCREEN_WIDTH) # Maintain aspect ratio

    COLOR_BG = (25, 28, 36)
    COLOR_WALL = (100, 110, 130)
    COLOR_TARGET = (255, 80, 80)
    COLOR_TARGET_HIT = (80, 80, 80)
    COLOR_PROJECTILE_CORE = (255, 255, 255)
    COLOR_PROJECTILE_GLOW = (255, 220, 50)
    COLOR_LAUNCHER = (180, 180, 200)
    COLOR_AIM_GUIDE = (255, 255, 255, 100)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_SHADOW = (10, 10, 10)

    MAX_STEPS = 1000
    PROJECTILE_RADIUS = 2.0 # World units
    TARGET_RADIUS = 4.0 # World units

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_big = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- World to Screen Scaling ---
        self.scale_x = self.SCREEN_WIDTH / self.WORLD_WIDTH
        self.scale_y = self.SCREEN_HEIGHT / self.WORLD_HEIGHT

        # --- Game Entities (defined once) ---
        self.walls = self._define_walls()
        self.initial_targets = self._define_targets()

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_state = 'aiming'
        self.launcher_pos = (self.WORLD_WIDTH / 2, self.WORLD_HEIGHT - 5)
        self.launch_angle = -math.pi / 2
        self.launch_power = 50.0
        self.projectile = None
        self.targets = []
        self.particles = []
        self.projectile_trail = deque(maxlen=10)
        self.prev_space_held = False
        self.prev_shift_held = False

        self.reset()

    def _define_walls(self):
        """Defines the reflective surfaces as pygame Rects in world coordinates."""
        return [
            # Outer boundaries
            pygame.Rect(0, 0, self.WORLD_WIDTH, 2), # Top
            pygame.Rect(0, 0, 2, self.WORLD_HEIGHT), # Left
            pygame.Rect(self.WORLD_WIDTH - 2, 0, 2, self.WORLD_HEIGHT), # Right
            # Inner obstacles
            pygame.Rect(20, 20, 60, 2),
            pygame.Rect(49, 40, 2, 25),
        ]

    def _define_targets(self):
        """Defines initial target positions in world coordinates."""
        return [
            {'pos': (25, 10), 'radius': self.TARGET_RADIUS},
            {'pos': (75, 10), 'radius': self.TARGET_RADIUS},
            {'pos': (50, 32), 'radius': self.TARGET_RADIUS},
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_state = 'aiming'
        self.launch_angle = -math.pi / 2
        self.launch_power = 50.0
        self.projectile = None
        self.targets = [{'pos': t['pos'], 'radius': t['radius'], 'active': True} for t in self.initial_targets]
        self.particles = []
        self.projectile_trail.clear()
        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        
        if self.game_state == 'aiming':
            self._update_aiming_phase(movement, space_held, shift_held)
        elif self.game_state == 'flying':
            reward = self._update_flying_phase()

        self._update_particles()
        
        all_targets_hit = all(not t['active'] for t in self.targets)
        
        if all_targets_hit:
            reward += 100.0
            self.score += 100.0
            self.game_over = True
            # sfx: win_sound
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        if self.steps >= self.MAX_STEPS and not all_targets_hit:
            reward -= 1.0 # Small penalty for timeout

        self.steps += 1
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_aiming_phase(self, movement, space_held, shift_held):
        """Handle player input during the aiming phase."""
        # Adjust angle
        if movement == 3: # Left
            self.launch_angle -= 0.04
        elif movement == 4: # Right
            self.launch_angle += 0.04
        self.launch_angle = max(-math.pi + 0.1, min(-0.1, self.launch_angle))

        # Adjust power
        if movement == 1: # Up
            self.launch_power = min(100.0, self.launch_power + 1.5)
        elif movement == 2: # Down
            self.launch_power = max(10.0, self.launch_power - 1.5)

        # Reset aim
        if shift_held and not self.prev_shift_held:
            self.launch_angle = -math.pi / 2
            self.launch_power = 50.0
            # sfx: reset_aim

        # Launch
        if space_held and not self.prev_space_held:
            self._launch_projectile()
            # sfx: launch

    def _launch_projectile(self):
        """Creates the projectile and changes the game state."""
        self.game_state = 'flying'
        power_scalar = self.launch_power / 100.0 * 1.5 # Scaled for game feel
        velocity = (
            math.cos(self.launch_angle) * power_scalar,
            math.sin(self.launch_angle) * power_scalar
        )
        self.projectile = {
            'pos': np.array(self.launcher_pos, dtype=float),
            'prev_pos': np.array(self.launcher_pos, dtype=float),
            'vel': np.array(velocity, dtype=float),
            'radius': self.PROJECTILE_RADIUS
        }
        self.projectile_trail.append(self.projectile['pos'].copy())


    def _update_flying_phase(self):
        """Handle projectile physics and collisions."""
        if not self.projectile:
            return 0.0
        
        # Store previous position for collision checks
        self.projectile['prev_pos'] = self.projectile['pos'].copy()
        
        # Update position
        self.projectile['pos'] += self.projectile['vel']
        self.projectile_trail.append(self.projectile['pos'].copy())

        # Check collisions
        bounce_reward = self._handle_wall_collisions()
        target_reward = self._handle_target_collisions()
        
        # Check out of bounds
        px, py = self.projectile['pos']
        if not (0 < px < self.WORLD_WIDTH and 0 < py < self.WORLD_HEIGHT):
            self.game_over = True
            # sfx: lose_sound
            return -10.0 # Penalty for going out of bounds
            
        return bounce_reward + target_reward

    def _handle_wall_collisions(self):
        """Check and resolve collisions with walls."""
        reward = 0.0
        px, py = self.projectile['pos']
        pr = self.projectile['radius']
        proj_rect = pygame.Rect(px - pr, py - pr, pr * 2, pr * 2)

        for wall in self.walls:
            if proj_rect.colliderect(wall):
                prev_px, prev_py = self.projectile['prev_pos']
                prev_proj_rect = pygame.Rect(prev_px - pr, prev_py - pr, pr * 2, pr * 2)

                # Horizontal collision
                if prev_proj_rect.right <= wall.left or prev_proj_rect.left >= wall.right:
                    self.projectile['vel'][0] *= -1
                    self.projectile['pos'][0] = self.projectile['prev_pos'][0] # Prevent sticking
                    self._create_particles(self.projectile['pos'], 15, self.COLOR_PROJECTILE_GLOW, 0.5, 20)
                    reward += 0.1
                    # sfx: bounce
                
                # Vertical collision
                if prev_proj_rect.bottom <= wall.top or prev_proj_rect.top >= wall.bottom:
                    self.projectile['vel'][1] *= -1
                    self.projectile['pos'][1] = self.projectile['prev_pos'][1] # Prevent sticking
                    if reward == 0.0: # Avoid double reward for corner hits
                        self._create_particles(self.projectile['pos'], 15, self.COLOR_PROJECTILE_GLOW, 0.5, 20)
                        reward += 0.1
                        # sfx: bounce
        return reward

    def _handle_target_collisions(self):
        """Check and resolve collisions with targets."""
        reward = 0.0
        for target in self.targets:
            if target['active']:
                dist_sq = np.sum((self.projectile['pos'] - np.array(target['pos']))**2)
                if dist_sq < (self.projectile['radius'] + target['radius'])**2:
                    target['active'] = False
                    self.projectile['vel'] *= 1.1 # Speed boost
                    reward += 10.0
                    self.score += 10.0
                    self._create_particles(target['pos'], 40, self.COLOR_TARGET, 1.0, 30, is_shockwave=True)
                    # sfx: target_hit
        return reward
        
    def _update_particles(self):
        """Update position and life of all particles."""
        particles_to_keep = []
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] > 0:
                particles_to_keep.append(p)
        self.particles = particles_to_keep

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _world_to_screen(self, pos):
        """Converts world coordinates to screen coordinates."""
        return int(pos[0] * self.scale_x), int(pos[1] * self.scale_y)

    # --- Rendering Methods ---

    def _render_game(self):
        self._render_walls()
        self._render_targets()
        self._render_launcher()
        if self.game_state == 'aiming':
            self._render_aim_guide()
        if self.projectile:
            self._render_projectile()
        self._render_particles()
        
        if self.game_over:
            self._render_end_message()

    def _render_walls(self):
        for wall in self.walls:
            screen_rect = pygame.Rect(
                wall.x * self.scale_x, wall.y * self.scale_y,
                wall.w * self.scale_x, wall.h * self.scale_y
            )
            pygame.draw.rect(self.screen, self.COLOR_WALL, screen_rect)

    def _render_targets(self):
        for target in self.targets:
            pos_screen = self._world_to_screen(target['pos'])
            radius_screen = int(target['radius'] * self.scale_x)
            color = self.COLOR_TARGET if target['active'] else self.COLOR_TARGET_HIT
            pygame.gfxdraw.filled_circle(self.screen, pos_screen[0], pos_screen[1], radius_screen, color)
            pygame.gfxdraw.aacircle(self.screen, pos_screen[0], pos_screen[1], radius_screen, color)

    def _render_launcher(self):
        pos_screen = self._world_to_screen(self.launcher_pos)
        size = int(2 * self.scale_x)
        pygame.draw.rect(self.screen, self.COLOR_LAUNCHER, (pos_screen[0] - size, pos_screen[1] - size, size * 2, size * 2))

    def _render_aim_guide(self):
        """Renders a predictive trajectory line."""
        sim_pos = np.array(self.launcher_pos, dtype=float)
        power_scalar = self.launch_power / 100.0 * 1.5
        sim_vel = np.array([math.cos(self.launch_angle) * power_scalar, math.sin(self.launch_angle) * power_scalar])
        
        points = []
        for _ in range(150): # Simulate steps
            sim_pos += sim_vel
            points.append(self._world_to_screen(sim_pos))
            
            # Simple wall bounce simulation for the guide
            pr = self.PROJECTILE_RADIUS
            sim_rect = pygame.Rect(sim_pos[0] - pr, sim_pos[1] - pr, pr * 2, pr * 2)
            for wall in self.walls:
                if sim_rect.colliderect(wall):
                    # This is a simplified check; just reflect based on wall orientation
                    if wall.width > wall.height: # Horizontal wall
                        sim_vel[1] *= -1
                    else: # Vertical wall
                        sim_vel[0] *= -1
                    break # Only one bounce per step
            if not (0 < sim_pos[0] < self.WORLD_WIDTH and 0 < sim_pos[1] < self.WORLD_HEIGHT):
                break
        
        if len(points) > 1:
            pygame.draw.aalines(self.screen, self.COLOR_AIM_GUIDE, False, points)

    def _render_projectile(self):
        # Motion blur trail
        for i, pos in enumerate(self.projectile_trail):
            alpha = int(50 * (i / len(self.projectile_trail)))
            if alpha > 0:
                pos_screen = self._world_to_screen(pos)
                radius = int(self.projectile['radius'] * self.scale_x)
                s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(s, radius, radius, radius, (*self.COLOR_PROJECTILE_GLOW, alpha))
                self.screen.blit(s, (pos_screen[0] - radius, pos_screen[1] - radius))

        # Main projectile
        pos_screen = self._world_to_screen(self.projectile['pos'])
        radius_screen = int(self.projectile['radius'] * self.scale_x)
        pygame.gfxdraw.filled_circle(self.screen, pos_screen[0], pos_screen[1], radius_screen, self.COLOR_PROJECTILE_GLOW)
        pygame.gfxdraw.aacircle(self.screen, pos_screen[0], pos_screen[1], radius_screen, self.COLOR_PROJECTILE_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, pos_screen[0], pos_screen[1], radius_screen // 2, self.COLOR_PROJECTILE_CORE)


    def _render_particles(self):
        for p in self.particles:
            pos_screen = self._world_to_screen(p['pos'])
            alpha = p['alpha_start'] * (p['life'] / p['life_max'])
            color = (*p['color'], int(alpha))
            
            if p.get('is_shockwave', False):
                radius = p['radius_start'] * (1.0 - (p['life'] / p['life_max']))
                if radius > 1:
                    pygame.gfxdraw.aacircle(self.screen, pos_screen[0], pos_screen[1], int(radius), color)
            else:
                pygame.draw.circle(self.screen, color, pos_screen, max(1, int(p['life'] / 10)))


    def _render_ui(self):
        # Render Power and Angle during aiming
        if self.game_state == 'aiming':
            power_text = f"Power: {int(self.launch_power)}"
            angle_deg = -math.degrees(self.launch_angle) - 90
            angle_text = f"Angle: {angle_deg:.1f}"
            
            self._draw_text(power_text, (10, self.SCREEN_HEIGHT - 30))
            self._draw_text(angle_text, (10, self.SCREEN_HEIGHT - 55))

        # Render Score
        score_text = f"SCORE: {int(self.score)}"
        self._draw_text(score_text, (10, 10))

    def _render_end_message(self):
        win = all(not t['active'] for t in self.targets)
        message = "VICTORY!" if win else "GAME OVER"
        color = (100, 255, 100) if win else (255, 100, 100)
        
        text_surf = self.font_big.render(message, True, color)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text_surf, text_rect)

    def _draw_text(self, text, pos):
        text_surf = self.font_ui.render(text, True, self.COLOR_UI_TEXT)
        shadow_surf = self.font_ui.render(text, True, self.COLOR_UI_SHADOW)
        self.screen.blit(shadow_surf, (pos[0] + 1, pos[1] + 1))
        self.screen.blit(text_surf, pos)

    # --- Utility Methods ---

    def _create_particles(self, pos, count, color, speed_range, life_range, is_shockwave=False):
        """Generates particles at a given position."""
        for _ in range(count):
            if is_shockwave:
                self.particles.append({
                    'pos': np.array(pos, dtype=float),
                    'vel': np.array([0,0]), # Shockwaves don't move
                    'life': life_range,
                    'life_max': life_range,
                    'color': color,
                    'alpha_start': 255,
                    'radius_start': 5 * self.scale_x,
                    'is_shockwave': True,
                })
            else:
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(speed_range * 0.5, speed_range)
                life = random.randint(life_range // 2, life_range)
                self.particles.append({
                    'pos': np.array(pos, dtype=float),
                    'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                    'life': life,
                    'life_max': life,
                    'color': color,
                    'alpha_start': 200,
                })
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
    # --- Manual Play Testing ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Un-comment the line below to run with a visible display
    # os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Ricochet Puzzle Environment")
    clock = pygame.time.Clock()
    
    terminated = False
    
    # Action state
    movement = 0
    space_held = 0
    shift_held = 0
    
    print("\n--- Controls ---")
    print("Arrow Keys: Adjust Angle/Power")
    print("Spacebar:   Launch Projectile")
    print("Shift:      Reset Aim")
    print("R:          Reset Environment")
    print("Q:          Quit")
    print("----------------\n")
    
    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    terminated = True
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                # Update actions based on key presses
                if event.key == pygame.K_SPACE: space_held = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_held = 1
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE: space_held = 0
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_held = 0
                
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        else: movement = 0

        action = [movement, space_held, shift_held]
        obs, reward, term, trunc, info = env.step(action)
        
        if term:
            print(f"Episode Finished. Score: {info['score']}, Steps: {info['steps']}")
        
        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(60) # Run at 60 FPS for smooth manual play
        
    env.close()