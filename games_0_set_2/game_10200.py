import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:58:59.022427
# Source Brief: brief_00200.md
# Brief Index: 200
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player must balance three interconnected pendulums.

    The goal is to survive for 45 seconds without any pendulum bob hitting the
    edge of the screen. The player can select one of the three pendulums and
    apply a small angular impulse to adjust its swing.

    Visuals are minimalist and elegant, focusing on the physics simulation.
    The pendulums become more vibrant as the score (survival time) increases.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement/Selection (0=none, 1=select left, 2=select middle, 3=select right, 4=deselect)
    - actions[1]: Increment Angle (0=released, 1=apply positive impulse)
    - actions[2]: Decrement Angle (0=released, 1=apply negative impulse)

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game state.

    Reward Structure:
    - +0.01 per step for survival.
    - +100 for winning (surviving 45 seconds).
    - -10 for losing (a pendulum hits the screen edge).
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    game_description = (
        "Balance three pendulums for 45 seconds by applying small impulses. "
        "A pendulum hitting the screen edge ends the game."
    )
    user_guide = (
        "Use the ←, ↓, and → arrow keys to select a pendulum. "
        "Press space to nudge it right and shift to nudge it left."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.MAX_STEPS = 45 * self.FPS
        self.PIVOT = (self.WIDTH // 2, 80)
        self.GRAVITY = 0.015
        self.DAMPING = 0.999
        self.ADJUST_STRENGTH = math.radians(1.0) # 1 degree adjustment

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_selection = pygame.font.SysFont("monospace", 16, bold=True)

        # --- Visuals ---
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (30, 50, 80)
        self.COLOR_UI = (255, 255, 255)
        self.COLOR_PENDULUM_BASE = (100, 100, 110)
        self.COLOR_PENDULUM_TARGETS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 80, 255)    # Blue
        ]
        self.COLOR_PARTICLE = (220, 220, 255)
        self.COLOR_SELECTION_GLOW = (255, 255, 100)

        # --- State Variables (initialized in reset) ---
        self.pendulums = []
        self.particles = []
        self.selected_pendulum_idx = None
        self.steps = 0
        self.score = 0.0
        self.timer_steps = 0
        self.terminated = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.timer_steps = self.MAX_STEPS
        self.terminated = False
        self.selected_pendulum_idx = None
        self.particles.clear()

        # Initialize pendulums with small random starting angles
        self.pendulums = [
            {'length': 150, 'angle': self.np_random.uniform(-0.1, 0.1), 'velocity': 0.0},
            {'length': 180, 'angle': self.np_random.uniform(-0.1, 0.1), 'velocity': 0.0},
            {'length': 120, 'angle': self.np_random.uniform(-0.1, 0.1), 'velocity': 0.0},
        ]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.terminated:
            # If the game is over, return the final state without updates
            return self._get_observation(), 0, True, False, self._get_info()

        self._handle_input(action)
        self._update_physics()
        self._update_particles()

        self.steps += 1
        self.timer_steps -= 1

        reward, terminated = self._calculate_reward_and_termination()
        self.score += reward # Accumulate score from per-step rewards
        self.terminated = terminated

        # Truncated is always False as per the brief
        truncated = self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            self.terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Action 0: no-op, Actions 1-3: select pendulum, Action 4: deselect
        if movement == 1: self.selected_pendulum_idx = 0
        elif movement == 2: self.selected_pendulum_idx = 1
        elif movement == 3: self.selected_pendulum_idx = 2
        elif movement == 4: self.selected_pendulum_idx = None

        if self.selected_pendulum_idx is not None:
            adjusted = False
            if space_held:
                # # Sound effect placeholder: sfx_adjust_up.wav
                self.pendulums[self.selected_pendulum_idx]['angle'] += self.ADJUST_STRENGTH
                adjusted = True
            if shift_held:
                # # Sound effect placeholder: sfx_adjust_down.wav
                self.pendulums[self.selected_pendulum_idx]['angle'] -= self.ADJUST_STRENGTH
                adjusted = True
            
            if adjusted:
                self._spawn_particles(5, self.PIVOT)


    def _update_physics(self):
        for p in self.pendulums:
            # Simplified pendulum physics
            acceleration = -(self.GRAVITY / p['length']) * math.sin(p['angle'])
            p['velocity'] += acceleration
            p['velocity'] *= self.DAMPING
            p['angle'] += p['velocity']

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1

    def _calculate_reward_and_termination(self):
        # Check for failure: pendulum hits edge
        for p in self.pendulums:
            bob_x = self.PIVOT[0] + p['length'] * math.sin(p['angle'])
            bob_y = self.PIVOT[1] + p['length'] * math.cos(p['angle'])
            if not (0 < bob_x < self.WIDTH and 0 < bob_y < self.HEIGHT):
                # # Sound effect placeholder: sfx_fail.wav
                return -10.0, True

        # Check for victory: timer runs out
        if self.timer_steps <= 0:
            # # Sound effect placeholder: sfx_win.wav
            return 100.0, True

        # Continuous reward for surviving
        return 0.01, False

    def _get_observation(self):
        self._render_background()
        self._render_pendulums()
        self._render_particles()
        self._render_ui()

        # Convert to numpy array in the required format
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Draw a vertical gradient
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_pendulums(self):
        # Max score before win is MAX_STEPS * 0.01
        max_score = self.MAX_STEPS * 0.01
        color_factor = min(1.0, self.score / max_score if max_score > 0 else 0)

        # Render selection glow
        if self.selected_pendulum_idx is not None:
            glow_radius = int(15 + 5 * math.sin(self.steps * 0.2))
            # Use gfxdraw for a smooth circle
            pygame.gfxdraw.filled_circle(self.screen, self.PIVOT[0], self.PIVOT[1], glow_radius, (*self.COLOR_SELECTION_GLOW, 50))
            pygame.gfxdraw.aacircle(self.screen, self.PIVOT[0], self.PIVOT[1], glow_radius, (*self.COLOR_SELECTION_GLOW, 100))

        for i, p in enumerate(self.pendulums):
            bob_x = self.PIVOT[0] + p['length'] * math.sin(p['angle'])
            bob_y = self.PIVOT[1] + p['length'] * math.cos(p['angle'])
            
            # Interpolate color based on score
            color = self._interpolate_color(self.COLOR_PENDULUM_BASE, self.COLOR_PENDULUM_TARGETS[i], color_factor)
            
            # Draw pendulum rod (antialiased)
            pygame.draw.aaline(self.screen, color, self.PIVOT, (bob_x, bob_y), 2)
            
            # Draw pendulum bob (antialiased)
            bob_radius = 12
            pygame.gfxdraw.filled_circle(self.screen, int(bob_x), int(bob_y), bob_radius, color)
            pygame.gfxdraw.aacircle(self.screen, int(bob_x), int(bob_y), bob_radius, color)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifespan'] / p['max_lifespan']))))
            color = (*self.COLOR_PARTICLE, alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), color)

    def _render_ui(self):
        # Score display
        score_text = f"SCORE: {self.score:.2f}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI)
        self.screen.blit(score_surf, (15, 10))

        # Timer display
        time_left = max(0, self.timer_steps / self.FPS)
        timer_text = f"TIME: {time_left:.2f}"
        timer_surf = self.font_ui.render(timer_text, True, self.COLOR_UI)
        self.screen.blit(timer_surf, (self.WIDTH - timer_surf.get_width() - 15, 10))
        
        # Selection display
        if self.selected_pendulum_idx is not None:
            selection_text = f"SELECTED: {['LEFT', 'MIDDLE', 'RIGHT'][self.selected_pendulum_idx]}"
            selection_surf = self.font_selection.render(selection_text, True, self.COLOR_SELECTION_GLOW)
            self.screen.blit(selection_surf, (self.PIVOT[0] - selection_surf.get_width() // 2, self.PIVOT[1] + 25))


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer_seconds": max(0, self.timer_steps / self.FPS),
        }

    def _spawn_particles(self, num_particles, position):
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 2.5)
            lifespan = random.randint(20, 40)
            self.particles.append({
                'pos': list(position),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'lifespan': lifespan,
                'max_lifespan': lifespan,
                'radius': random.uniform(1, 3)
            })

    def _interpolate_color(self, color1, color2, factor):
        r = int(color1[0] + (color2[0] - color1[0]) * factor)
        g = int(color1[1] + (color2[1] - color1[1]) * factor)
        b = int(color1[2] + (color2[2] - color1[2]) * factor)
        return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to run the environment directly for testing.
    # It demonstrates human play and random agent interaction.
    
    # --- Human Play Mode ---
    print("\n--- Human Play Mode ---")
    print(GameEnv.user_guide)
    print("  - Q: Quit")

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a separate display for rendering if not running headless
    if os.environ.get("SDL_VIDEODRIVER", "") != "dummy":
        display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Pendulum Balance")
    else:
        # To prevent the __main__ block from crashing in a headless environment
        print("Running in headless mode. No visual output will be shown.")
        display_screen = None

    
    running = True
    terminated = False
    
    while running:
        # Default action is no-op
        action = [0, 0, 0] # [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                # Map arrow keys to selection as per user_guide
                if event.key == pygame.K_LEFT: action[0] = 1 # Select Left
                if event.key == pygame.K_DOWN: action[0] = 2 # Select Middle
                if event.key == pygame.K_RIGHT: action[0] = 3 # Select Right
                # A key to deselect might be useful, e.g., 'd'
                if event.key == pygame.K_d: action[0] = 4 # Deselect
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            
            if display_screen:
                # Display the observation from the environment
                surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                display_screen.blit(surf, (0, 0))
                pygame.display.flip()

            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']:.2f}")
                # Wait a bit before resetting
                if display_screen: pygame.time.wait(2000)
                obs, info = env.reset()
                terminated = False

        if display_screen:
            env.clock.tick(env.FPS)

    env.close()