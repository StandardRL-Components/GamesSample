import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:18:04.827262
# Source Brief: brief_00930.md
# Brief Index: 930
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent launches a transforming projectile
    to destroy fragile structures within a time limit.

    **Visuals:**
    - Clean, geometric aesthetic with a dark, gridded background.
    - Interactive elements (projectile, structures) are bright and high-contrast.
    - Explosions are rendered with a satisfying particle effect.
    - UI is clear and non-intrusive.

    **Gameplay:**
    - The agent controls the launch angle of a projectile from a fixed cannon.
    - The projectile can be in one of two forms: Sphere or Arrow.
    - Sphere: Faster, larger impact radius.
    - Arrow: Slower, smaller impact radius, less affected by gravity.
    - The goal is to destroy all 15 structures within 1000 steps (approx. 33 seconds).

    **Action Space `MultiDiscrete([5, 2, 2])`:**
    - `action[0]` (Movement): Adjusts the launch angle.
        - 0: No-op
        - 1: Up
        - 2: Down
        - 3: Left (fine-tune up)
        - 4: Right (fine-tune down)
    - `action[1]` (Space): Launches the projectile.
        - 0: Released
        - 1: Held (triggers launch on press)
    - `action[2]` (Shift): No effect.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    game_description = (
        "Launch transforming projectiles from a cannon to destroy all fragile structures before time runs out."
    )
    user_guide = (
        "Use the ↑↓ arrow keys to aim the cannon and ←→ to fine-tune your aim. Press space to launch a projectile."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1000 # Approx. 33 seconds at 30 FPS
    NUM_STRUCTURES = 15
    GRAVITY = 0.08

    # --- Colors ---
    COLOR_BG = (20, 20, 40)
    COLOR_GRID = (30, 30, 60)
    COLOR_LAUNCHER = (150, 150, 170)
    COLOR_AIM = (255, 255, 255, 150)
    COLOR_STRUCTURE = (255, 80, 80)
    COLOR_STRUCTURE_BORDER = (200, 50, 50)
    COLOR_SPHERE = (80, 255, 80)
    COLOR_ARROW = (80, 200, 255)
    COLOR_EXPLOSION = [(255, 255, 100), (255, 200, 50), (255, 100, 0)]
    COLOR_TEXT = (240, 240, 240)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        if self.render_mode == "human":
            self.human_screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            pygame.display.set_caption("Projectile Transformer")
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.launch_angle = 0.0
        self.launcher_pos = (70, self.SCREEN_HEIGHT - 70)
        self.projectile_form = 'sphere'
        self.projectiles = []
        self.structures = []
        self.particles = []
        self.last_space_held = False
        
        # --- Self-Validation ---
        # Note: We call reset() inside the validation function, so no need to call it before.
        # self.validate_implementation() # Commented out for production

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.launch_angle = -45.0
        self.projectile_form = 'sphere'
        self.projectiles = []
        self.particles = []
        self.last_space_held = False
        self._generate_structures()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()
            
        reward = 0.0
        
        self._handle_input(action)
        self._update_projectiles()
        self._update_particles()
        
        collision_reward, destroyed_count = self._check_collisions()
        reward += collision_reward
        self.score += destroyed_count

        self.steps += 1
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        
        if terminated:
            self.game_over = True

        if self.render_mode == "human":
            self._render_frame()

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # Adjust angle
        if movement == 1: self.launch_angle -= 1.0   # Up
        if movement == 2: self.launch_angle += 1.0   # Down
        if movement == 3: self.launch_angle -= 0.25  # Left (fine up)
        if movement == 4: self.launch_angle += 0.25  # Right (fine down)
        self.launch_angle = np.clip(self.launch_angle, -90, 0)

        # Launch on space press (rising edge)
        if space_held and not self.last_space_held:
            self._launch_projectile()
            # SFX: Launch sound (whoosh or thwump)
        self.last_space_held = space_held

    def _launch_projectile(self):
        angle_rad = math.radians(self.launch_angle)
        if self.projectile_form == 'sphere':
            speed = 10.0
            radius = 6
            proj_type = 'sphere'
        else: # arrow
            speed = 7.0
            radius = 3
            proj_type = 'arrow'
        
        velocity = [speed * math.cos(angle_rad), speed * math.sin(angle_rad)]
        
        self.projectiles.append({
            'pos': list(self.launcher_pos),
            'vel': velocity,
            'type': proj_type,
            'radius': radius,
            'age': 0
        })
        
        # Switch form for the *next* launch
        self.projectile_form = 'arrow' if self.projectile_form == 'sphere' else 'sphere'

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            p['vel'][1] += self.GRAVITY
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['age'] += 1
            
            # Remove if off-screen or too old
            if not (0 < p['pos'][0] < self.SCREEN_WIDTH and -50 < p['pos'][1] < self.SCREEN_HEIGHT + 50) or p['age'] > 300:
                self.projectiles.remove(p)

    def _update_particles(self):
        for part in self.particles[:]:
            part['pos'][0] += part['vel'][0]
            part['pos'][1] += part['vel'][1]
            part['vel'][1] += self.GRAVITY * 0.5
            part['lifespan'] -= 1
            if part['lifespan'] <= 0:
                self.particles.remove(part)

    def _check_collisions(self):
        reward = 0.0
        destroyed_count = 0
        for p in self.projectiles[:]:
            for s in self.structures[:]:
                dist = math.hypot(p['pos'][0] - s['pos'][0], p['pos'][1] - s['pos'][1])
                if dist < p['radius'] + s['radius']:
                    self.structures.remove(s)
                    self.projectiles.remove(p)
                    self._create_explosion(s['pos'])
                    reward += 1.1 # +1 for destruction, +0.1 for damage
                    destroyed_count += 1
                    # SFX: Explosion sound (crunchy pop)
                    break # Projectile is gone, stop checking it
        return reward, destroyed_count

    def _check_termination(self):
        terminated = False
        terminal_reward = 0.0
        
        if not self.structures: # Win condition
            terminated = True
            terminal_reward = 100.0
        elif self.steps >= self.MAX_STEPS: # Loss condition
            terminated = True
            terminal_reward = -100.0
            
        return terminated, terminal_reward

    def _get_observation(self):
        self._render_frame()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_frame(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_structures()
        self._render_launcher()
        self._render_projectiles()
        self._render_particles()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        if self.render_mode == "human":
            self.human_screen.blit(self.screen, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "structures_remaining": len(self.structures),
            "time_left": (self.MAX_STEPS - self.steps) / self.metadata["render_fps"]
        }

    # --- Rendering Helpers ---
    def _render_background(self):
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_launcher(self):
        # Base
        pygame.draw.circle(self.screen, self.COLOR_LAUNCHER, self.launcher_pos, 15)
        pygame.draw.circle(self.screen, self.COLOR_BG, self.launcher_pos, 10)
        
        # Aiming line
        angle_rad = math.radians(self.launch_angle)
        end_x = self.launcher_pos[0] + 50 * math.cos(angle_rad)
        end_y = self.launcher_pos[1] + 50 * math.sin(angle_rad)
        pygame.draw.line(self.screen, self.COLOR_AIM, self.launcher_pos, (int(end_x), int(end_y)), 2)

    def _render_structures(self):
        for s in self.structures:
            pos = (int(s['pos'][0]), int(s['pos'][1]))
            radius = int(s['radius'])
            pygame.draw.circle(self.screen, self.COLOR_STRUCTURE_BORDER, pos, radius)
            pygame.draw.circle(self.screen, self.COLOR_STRUCTURE, pos, radius - 2)

    def _render_projectiles(self):
        for p in self.projectiles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            if p['type'] == 'sphere':
                # Glow effect
                glow_radius = int(p['radius'] * 1.8)
                s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*self.COLOR_SPHERE, 50), (glow_radius, glow_radius), glow_radius)
                self.screen.blit(s, (pos[0] - glow_radius, pos[1] - glow_radius))
                # Core circle
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], p['radius'], self.COLOR_SPHERE)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], p['radius'], self.COLOR_SPHERE)
            else: # arrow
                angle_rad = math.atan2(p['vel'][1], p['vel'][0])
                l, w = 12, 4
                points = [
                    (pos[0] + l * math.cos(angle_rad), pos[1] + l * math.sin(angle_rad)),
                    (pos[0] + w * math.cos(angle_rad - math.pi/2), pos[1] + w * math.sin(angle_rad - math.pi/2)),
                    (pos[0] - l/2 * math.cos(angle_rad), pos[1] - l/2 * math.sin(angle_rad)),
                    (pos[0] + w * math.cos(angle_rad + math.pi/2), pos[1] + w * math.sin(angle_rad + math.pi/2))
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ARROW)
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ARROW)

    def _render_particles(self):
        for part in self.particles:
            life_ratio = part['lifespan'] / part['max_life']
            radius = int(part['radius'] * life_ratio)
            if radius > 0:
                color_idx = min(2, int((1.0 - life_ratio) * 3))
                color = self.COLOR_EXPLOSION[color_idx]
                pos = (int(part['pos'][0]), int(part['pos'][1]))
                pygame.draw.circle(self.screen, color, pos, radius)

    def _render_ui(self):
        # Time remaining
        time_left = (self.MAX_STEPS - self.steps) / self.metadata["render_fps"]
        time_text = self.font_small.render(f"TIME: {max(0, time_left):.1f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (10, 10))

        # Structures remaining
        struct_text = self.font_small.render(f"TARGETS: {len(self.structures)}/{self.NUM_STRUCTURES}", True, self.COLOR_TEXT)
        self.screen.blit(struct_text, (self.SCREEN_WIDTH - struct_text.get_width() - 10, 10))
        
        # Current projectile form indicator
        form_text = self.font_small.render("NEXT:", True, self.COLOR_TEXT)
        self.screen.blit(form_text, (10, self.SCREEN_HEIGHT - 30))
        if self.projectile_form == 'sphere':
            pygame.draw.circle(self.screen, self.COLOR_SPHERE, (80, self.SCREEN_HEIGHT - 22), 8)
        else:
            pygame.draw.polygon(self.screen, self.COLOR_ARROW, [(75, self.SCREEN_HEIGHT - 22), (90, self.SCREEN_HEIGHT - 22), (82.5, self.SCREEN_HEIGHT - 32)])

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        if not self.structures:
            msg = "MISSION COMPLETE"
            color = self.COLOR_SPHERE
        else:
            msg = "TIME UP"
            color = self.COLOR_STRUCTURE
            
        text = self.font_large.render(msg, True, color)
        text_rect = text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
        self.screen.blit(text, text_rect)

    # --- Game Setup Helpers ---
    def _generate_structures(self):
        self.structures = []
        target_area = pygame.Rect(self.SCREEN_WIDTH * 0.4, 50, self.SCREEN_WIDTH * 0.55, self.SCREEN_HEIGHT - 100)
        min_dist = 40 # Minimum distance between structure centers

        for _ in range(self.NUM_STRUCTURES):
            attempts = 0
            while attempts < 100:
                pos = [
                    self.np_random.uniform(target_area.left, target_area.right),
                    self.np_random.uniform(target_area.top, target_area.bottom)
                ]
                radius = self.np_random.uniform(8, 15)
                
                # Check for overlap with existing structures
                is_overlapping = False
                for s in self.structures:
                    if math.hypot(pos[0] - s['pos'][0], pos[1] - s['pos'][1]) < radius + s['radius'] + min_dist:
                        is_overlapping = True
                        break
                
                if not is_overlapping:
                    self.structures.append({'pos': pos, 'radius': radius})
                    break
                attempts += 1

    def _create_explosion(self, position):
        num_particles = self.np_random.integers(20, 30)
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            lifespan = self.np_random.integers(20, 40)
            self.particles.append({
                'pos': list(position),
                'vel': [speed * math.cos(angle), speed * math.sin(angle)],
                'lifespan': lifespan,
                'max_life': lifespan,
                'radius': self.np_random.uniform(2, 5)
            })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset and observation space
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), f"Obs shape is {obs.shape}"
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows for human playtesting
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    done = False
    
    print("\n--- Human Playtest Controls ---")
    print("Arrows: Aim")
    print("Space:  Launch Projectile")
    print("R:      Reset Environment")
    print("Q:      Quit")
    print("-----------------------------\n")

    while not done:
        # Default action is no-op
        action = [0, 0, 0] # [movement, space, shift]
        
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        
        # Allow resetting and quitting
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    print("Environment Reset.")
                if event.key == pygame.K_q:
                    done = True
            if event.type == pygame.QUIT:
                done = True

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Done: {terminated}")
            if terminated:
                print(f"Episode Finished. Final Score: {info['score']}. Resetting in 3 seconds...")
                pygame.time.wait(3000)
                obs, info = env.reset()

    env.close()